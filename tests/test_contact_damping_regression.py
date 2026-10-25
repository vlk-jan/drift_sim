"""Contact damping stays stable at production constants: 2 s rough terrain, dt 2 ms, both engines.

The depth-proportional damping is explicit in the velocity; without the per-point
damping-impulse cap the adjoint blows up past ~56 mm / alpha of penetration while
the forward stays finite (see simulator/engine.py::_calculate_spring_force). The
start is pressed PRESS_M below its seat, past that depth at alpha 2.0: uncapped,
the gradient norm is 7e10 here (torch); capped, 1e4.
"""
import pytest
import torch

DT, N, RES, MC = 2e-3, 128, 0.1, 6.35
PRESS_M = 0.03


def _support_grad(engine, robot, steps, noise):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    support = (noise * torch.randn(1, N, N, generator=gen)).requires_grad_(True)
    friction = torch.full((1, N, N), 0.6, requires_grad=True)
    if engine == "warp":
        from models.simulator_wp import Simulator
    else:
        from models.simulator import Simulator
    sim = Simulator(robot=robot, device="cpu", T_horizon=steps * DT, dt=DT)
    sim.update_world_config(res=RES, max_coord=MC, side_range=(-MC, MC), fwd_range=(-MC, MC))
    seat = sim.seat_on_terrain
    sim.seat_on_terrain = lambda x, q, wc: seat(x, q, wc) - torch.tensor([0.0, 0.0, PRESS_M])
    ctrl = torch.full((1, 200, 3), 3.0)
    ts = torch.linspace(0.0, steps * DT, 200, dtype=torch.float64).unsqueeze(0)
    traj, _, xd, om = sim.predict_trajectory({"support": support, "friction": friction}, ctrl, ts)
    traj[:, :, :3, 3].pow(2).sum().backward()
    assert torch.isfinite(xd).all() and torch.isfinite(om).all()
    assert not bool((xd.abs() >= 49.999).any() or (om.abs() >= 49.999).any()), "velocity clamp hit"
    return support.grad


@pytest.mark.parametrize("engine", ["torch", "warp"])
@pytest.mark.parametrize("robot,noise_m", [("helhest_v2", 0.10)])
def test_contact_damping_gradient_is_bounded(engine, robot, noise_m):
    grad = _support_grad(engine, robot, 1000, noise_m)
    assert torch.isfinite(grad).all(), f"{engine} {robot}: non-finite gradient"
    norm = grad.norm()
    assert norm < 1e6, f"{engine} {robot} {noise_m:.2f} m: gnorm={norm:.3e}"
