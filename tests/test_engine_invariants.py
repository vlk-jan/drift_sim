"""Physical sanity invariants of the differentiable spring-contact engine."""

import torch

from models.simulator import Simulator

GRID_N = 64
RES = 0.1
MAX_COORD = GRID_N * RES / 2  # ±3.2 m


def _flat_props(mu, k=4e5):
    return {
        "support": torch.zeros(1, GRID_N, GRID_N),
        "friction": torch.full((1, GRID_N, GRID_N), mu),
        "stiffness": torch.full((1, GRID_N, GRID_N), k),
    }


def _make_sim(t_horizon, dt=0.002):
    sim = Simulator(
        robot="helhest", batch_size=1, device="cpu", T_horizon=t_horizon, dt=dt
    )
    sim.update_world_config(
        res=RES,
        max_coord=MAX_COORD,
        side_range=(-MAX_COORD, MAX_COORD),
        fwd_range=(-MAX_COORD, MAX_COORD),
    )
    sim.eval()
    return sim


def _rollout(sim, props, wheel_omega, t_horizon):
    ctrl = torch.full((1, 2, 3), wheel_omega)
    ctrl_ts = torch.tensor([[100.0, 100.0 + t_horizon]])
    with torch.no_grad():
        poses, _, _, _ = sim.predict_trajectory(props, ctrl, ctrl_ts)
    return poses[0, :, :3, 3]


def test_static_equilibrium_on_flat_ground():
    """Zero wheel speed on flat ground: the robot must stay put."""
    t_horizon = 0.5
    sim = _make_sim(t_horizon)
    pos = _rollout(sim, _flat_props(mu=0.6), wheel_omega=0.0, t_horizon=t_horizon)

    horiz_drift = (pos[-1, :2] - pos[0, :2]).norm().item()
    z_drift = (pos[-1, 2] - pos[0, 2]).abs().item()
    assert horiz_drift < 0.05, f"robot drifted {horiz_drift:.3f} m at rest"
    assert z_drift < 0.05, f"robot sank/bounced {z_drift:.3f} m at rest"


def test_friction_monotonic_travel():
    """Same wheel commands from rest: higher friction must travel farther."""
    t_horizon = 1.0
    sim = _make_sim(t_horizon)

    def travel(mu):
        pos = _rollout(sim, _flat_props(mu=mu), wheel_omega=3.0, t_horizon=t_horizon)
        return (pos[-1, :2] - pos[0, :2]).norm().item()

    d_low, d_high = travel(0.1), travel(0.8)
    assert d_high > d_low + 0.05, (
        f"friction 0.8 travelled {d_high:.3f} m vs {d_low:.3f} m at friction 0.1"
    )
