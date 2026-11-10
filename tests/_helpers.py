"""Factories shared by several test modules (not a test file itself).

tests/ has no __init__.py, so pytest puts this directory on sys.path and
`from _helpers import ...` works from any test module.
"""

import torch
import torch.nn.functional as F


def make_sim(
    t_horizon=0.05,
    dt=0.005,
    *,
    engine="torch",
    device="cpu",
    res=0.1,
    max_coord=1.6,
    com_transfer=True,
    eval_mode=True,
    robot="helhest_v2",
):
    """Build a torch or warp Simulator over a square res/max_coord world."""
    if engine == "warp":
        from drift_sim.simulator_wp import Simulator
    else:
        from drift_sim.simulator import Simulator

    sim = Simulator(
        robot=robot,
        device=device,
        T_horizon=t_horizon,
        dt=dt,
        com_transfer=com_transfer,
    )
    sim.update_world_config(
        res=res,
        max_coord=max_coord,
        side_range=(-max_coord, max_coord),
        fwd_range=(-max_coord, max_coord),
    )
    if eval_mode:
        sim.eval()
    return sim


def flat_props(mu, n):
    """Flat props over an n x n grid in simulator layout [y, x]."""
    return {
        "support": torch.zeros(1, n, n),
        "friction": torch.full((1, n, n), mu),
    }


def cos(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def ratio(a, b):
    return (a.norm() / b.norm().clamp_min(1e-30)).item()


def sample_height(world_config, x, y):
    """Replicate engine._sample_terrain's grid_sample for the z channel."""
    xy = torch.tensor([[[[x, y]]]]).float() - world_config.grid_center
    xy_norm = (xy / world_config.max_coord).clamp(-1.0, 1.0)
    sampled = F.grid_sample(
        world_config._sample_grid,
        xy_norm.reshape(1, 1, 1, 2),
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )
    return sampled[0, 0, 0, 0].item()
