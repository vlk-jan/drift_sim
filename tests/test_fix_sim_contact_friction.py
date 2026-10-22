"""Contact force continuity and friction stability at rest (helhest_v2, torch)."""

import math

import pytest
import torch

from models.simulator import Simulator
from simulator.engine import DPhysicsEngine
from simulator.utils import (
    PhysicsEngineConfig,
    PhysicsState,
    WorldConfig,
    num_sim_steps,
    robot_model_config,
)

N, RES = 128, 0.1
MC = (N - 1) * RES / 2


def _rollout(T, mu, slope_deg=0.0, omega=0.0):
    sim = Simulator(robot="helhest_v2", device="cpu", T_horizon=T)
    sim.update_world_config(
        res=RES, max_coord=MC, side_range=(-MC, MC), fwd_range=(-MC, MC)
    )
    z = math.tan(math.radians(slope_deg)) * torch.linspace(-MC, MC, N)
    props = {
        "support": z.view(1, 1, -1).expand(1, N, N).clone(),
        "friction": torch.full((1, N, N), mu),
    }
    ctrl = torch.tensor(omega, dtype=torch.float32).expand(1, 2, 3).clone()
    with torch.no_grad():
        poses, _, xd, _ = sim.predict_trajectory(
            props, ctrl, torch.tensor([[0.0, T]], dtype=torch.float64)
        )
    return poses[0, :, :3, 3], xd[0], poses[0, -1, :3, :3]


def test_parked_robot_does_not_drift():
    """Was a period-2 friction limit cycle drifting ~2 mm/s."""
    pos, _, _ = _rollout(3.0, mu=1.0)
    drift = (pos[-1, :2] - pos[500, :2]).norm().item()  # after 1 s settling
    assert drift < 1e-4, f"parked robot drifted {drift * 1e3:.3f} mm in 2 s"


@pytest.mark.parametrize(
    "slope,mu,holds", [(10, 1.0, True), (10, 0.1, False), (20, 0.3, False)]
)
def test_slope_holds_or_slides(slope, mu, holds):
    pos, _, _ = _rollout(2.0, mu=mu, slope_deg=slope)
    speed = (pos[-1, :2] - pos[500, :2]).norm().item() / 1.0
    if holds:
        # Regularised creep mu F_N tanh(v / 5 mm/s) = m g sin: 0.9 mm/s.
        assert speed < 0.9e-3, f"{slope} deg, mu {mu}: creeps {speed * 1e3:.2f} mm/s"
    else:
        assert speed > 0.3, f"{slope} deg, mu {mu}: only {speed:.3f} m/s"


def test_straight_drive_speed():
    _, xd, _ = _rollout(2.0, mu=1.0, omega=1 / 0.334)
    v = xd[-250:, :2].norm(dim=-1).mean().item()
    assert abs(v - 1.0) < 0.02, f"omega = 1/r drove at {v:.3f} m/s"


def test_pivot_turn_is_not_capped():
    """The dragged rear wheel slides at slip ratio ~0.8 under drive torque; the
    impulse cap (gravity only) once clipped it on 998/1000 steps, -0.654 rad."""
    _, _, R = _rollout(2.0, mu=0.8, omega=[0.5 / 0.334, -0.5 / 0.334, 0.0])
    yaw = math.atan2(R[1, 0].item(), R[0, 0].item())
    assert abs(yaw + 0.578) < 0.01, f"pivot yawed {yaw:.3f} rad, law gives -0.578"


def test_contact_force_is_continuous_when_a_point_touches():
    """A second ring point touching must not lower the total normal force.

    Under the old robot-wide `/ count` normalisation the force fell ~35 %.
    """
    rm = robot_model_config("helhest_v2")
    eng = DPhysicsEngine(PhysicsEngineConfig(com_transfer=False), rm)
    world = WorldConfig(
        z_grid=torch.zeros(1, 64, 64),
        grid_res=0.1,
        max_coord=3.15,
        k_stiffness=rm.contact_stiffness,
        k_friction=0.0,
    )
    pitch = math.radians(10.75)  # two ring points of each front wheel nearly level
    q = torch.tensor([[math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]])

    def fz(z):
        s = PhysicsState.dummy(batch_size=1, device="cpu")
        s.x, s.q, s.wheel_omega = torch.tensor([[0.0, 0.0, z]]), q, torch.zeros(1, 3)
        return (eng(s, torch.zeros(1, 3), world).xdd[0, 2].item() + 9.81) * rm.mass

    # Second-lowest ring point of the front wheels, relative to the origin.
    R = torch.tensor(
        [
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)],
        ]
    )
    ring_z = (eng._wheel_pos_rel[0] @ R.T)[:, 2].sort().values
    z_touch = -ring_z[1].item()
    above, below = fz(z_touch + 1e-6), fz(z_touch - 1e-6)
    assert below >= above and below - above < 1.0, (above, below)


def test_step_count_covers_horizon():
    assert num_sim_steps(1.0, 2e-3) == 500
    assert num_sim_steps(torch.tensor(1.0), 2e-3) == 500
    assert num_sim_steps(1.001, 2e-3) == 501
    assert num_sim_steps(torch.tensor(0.3), 2e-3) == 150  # float32 0.3 is 0.30000001
    assert num_sim_steps(torch.tensor(1.1), 2e-3) == 550
