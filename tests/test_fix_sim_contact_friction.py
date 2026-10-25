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
    """Without the friction cut a period-2 limit cycle drifts a parked robot ~2 mm/s."""
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
    """The dragged rear wheel slides under drive torque; a gravity-only impulse
    cap clips it on 998/1000 steps (-0.654 rad).

    Reference -0.578: the dt -> 0 limit of this rollout at the committed
    constants (k 2.1e5, damping_alpha 2.0; dt 5e-4 and 2e-4 give -0.5782, dt
    2e-3 adds -0.0005 of Euler error). At k 1.42e5 with 32 rim points it is
    -0.590: three points per wheel sit under the 7.3 mm sag, a wider patch that
    scrubs more; at 4.9 mm sag one point carries the wheel at rest, as at 16
    points (-0.578, which no cap rule moves). Gating the cap on slip < 2
    slip_ref_ratio gives -0.598 with omega_z chatter (std 3e-3) at 16 points.
    """
    _, xd, R = _rollout(2.0, mu=0.8, omega=[0.5 / 0.334, -0.5 / 0.334, 0.0])
    yaw = math.atan2(R[1, 0].item(), R[0, 0].item())
    assert abs(yaw + 0.578) < 0.003, f"pivot yawed {yaw:.4f} rad, law gives -0.578"


@pytest.mark.parametrize("ramp", [0.5, 0.05])
@pytest.mark.parametrize("turn", [0.0, 0.5])
def test_braked_robot_settles(ramp, turn):
    """Drive at 3 rad/s, ramp the wheels to 0: the robot must come to rest.

    With the cap gated on slip < slip_ref_ratio the stopped wheels sit at slip
    0.1-0.2, where the uncapped law is ~3x too stiff for explicit Euler: the
    velocity flips sign every step (late std 8e-3 m/s, ~1.8 mm / 0.5 s).
    """
    sim = Simulator(robot="helhest_v2", device="cpu", T_horizon=2.0)
    sim.update_world_config(
        res=RES, max_coord=MC, side_range=(-MC, MC), fwd_range=(-MC, MC)
    )
    w = [3.0 + turn, 3.0 - turn, 3.0]
    ctrl = torch.tensor([[w, w, [0.0] * 3, [0.0] * 3]])
    ts = torch.tensor([[0.0, 0.5, 0.5 + ramp, 2.0]], dtype=torch.float64)
    props = {"support": torch.zeros(1, N, N), "friction": torch.full((1, N, N), 0.8)}
    with torch.no_grad():
        poses, _, xd, om = sim.predict_trajectory(props, ctrl, ts)
    std = max(xd[0, -100:, :2].std(0).max().item(), om[0, -100:, 2].std().item())
    drift = (poses[0, -1, :2, 3] - poses[0, -250, :2, 3]).norm().item()
    assert std < 1e-4, f"late velocity std {std:.2e} -- friction chatters"
    assert drift < 1e-5, f"stopped robot drifted {drift * 1e3:.4f} mm in 0.5 s"


def test_contact_force_is_continuous_when_a_point_touches():
    """A second ring point touching must not lower the total normal force.

    A robot-wide `/ count` normalisation would drop the force ~35 %.
    """
    rm = robot_model_config("helhest_v2")
    eng = DPhysicsEngine(PhysicsEngineConfig(com_transfer=False), rm)
    world = WorldConfig(
        z_grid=torch.zeros(1, 64, 64),
        grid_res=0.1,
        max_coord=3.15,
        k_friction=0.0,
    )
    pitch = math.radians(10.75)  # two ring points of each front wheel nearly level
    q = torch.tensor([[math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0]])

    def fz(z):
        s = PhysicsState.dummy(batch_size=1, device="cpu")
        s.x, s.q, s.wheel_omega = torch.tensor([[0.0, 0.0, z]]), q, torch.zeros(1, 3)
        return (eng(s, world).xdd[0, 2].item() + 9.81) * rm.mass

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
    with pytest.raises(ValueError):
        num_sim_steps(0.0, 2e-3)


def test_initial_state_rejects_unused_keys():
    """Wheel speeds are the controls: an initial "wheel_omega" would be ignored."""
    sim = Simulator(robot="helhest_v2", device="cpu", T_horizon=0.1)
    sim.update_world_config(
        res=RES, max_coord=MC, side_range=(-MC, MC), fwd_range=(-MC, MC)
    )
    props = {"support": torch.zeros(1, N, N), "friction": torch.full((1, N, N), 0.8)}
    with pytest.raises(ValueError, match="wheel_omega"):
        sim.predict_trajectory(
            props,
            torch.zeros(1, 2, 3),
            torch.tensor([[0.0, 0.1]], dtype=torch.float64),
            initial_state={"wheel_omega": torch.zeros(1, 3)},
        )


def test_float32_epoch_stamps_are_promoted():
    """float32 Unix stamps (~128 s spacing) must not collapse the window: the
    wrapper casts to float64 before building the timeline, like the warp one."""
    sim = Simulator(robot="helhest_v2", device="cpu", T_horizon=0.2)
    sim.update_world_config(
        res=RES, max_coord=MC, side_range=(-MC, MC), fwd_range=(-MC, MC)
    )
    props = {"support": torch.zeros(1, N, N), "friction": torch.full((1, N, N), 0.8)}
    # Exactly representable in float32, so both dtypes carry the same stamps.
    t0 = 1_757_690_112.0
    ts64 = torch.tensor([[t0, t0 + 256.0]], dtype=torch.float64)
    ctrl = torch.tensor([[[3.0, 3.0, 3.0], [-3.0, -3.0, -3.0]]])
    with torch.no_grad():
        p64, t_64, _, _ = sim.predict_trajectory(props, ctrl, ts64)
        p32, t_32, _, _ = sim.predict_trajectory(props, ctrl, ts64.float())
    assert t_32.dtype == torch.float64 and torch.equal(t_32, t_64)
    assert torch.equal(p32, p64)
