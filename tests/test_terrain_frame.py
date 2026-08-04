"""Terrain BEV-frame reconciliation between perception and the simulator.

The perception stack authors BEV grids as [row=X, col=Y] (utils.pcd.get_height_map
bins (x, y) in that order; the WaffleIron cell projection and the property heads
inherit it), but simulator.utils.WorldConfig / F.grid_sample use torch's "xy"
indexing, [row=Y, col=X]. Every rollout caller must therefore transpose support
and friction into the simulator layout before predict_trajectory --
utils.experiment.apply_terrain_props (adaptation/eval), overfit_single.py, and
utils.trainer (base training) all do ``.transpose(1, 2)``. Handing the grids
through untransposed makes the engine sample the terrain x<->y swapped -- a +x
ramp felt as a +y ramp -- which silently scrambles the terrain<->trajectory
coupling the method trains on (the base-training bug fixed 2026-08-04). This
test pins the convention so the trap cannot come back unnoticed.
"""

import torch
import torch.nn.functional as F

from simulator.utils import WorldConfig


def _sample_height(world_config, x, y):
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


def _x_ramp_heightmap(n, res):
    """A support grid that rises purely along world +x, in the perception
    heightmap layout [row=X, col=Y] (height_map[i, j] increases with i)."""
    c = (n - 1) / 2.0
    ii = torch.arange(n).float()
    return (ii[:, None] - c) * res * torch.ones(1, n)  # (n, n)


def test_untransposed_grid_is_sampled_x_swapped():
    # Documents the trap: passing the heightmap-frame grid straight to
    # WorldConfig makes the engine feel the +x ramp as a +y ramp.
    n, res = 41, 0.1
    c = (n - 1) / 2.0
    wc = WorldConfig(z_grid=_x_ramp_heightmap(n, res)[None], grid_res=res, max_coord=c * res)
    assert abs(_sample_height(wc, 1.0, 0.0)) < 0.05        # +x reads flat
    assert abs(_sample_height(wc, 0.0, 1.0) - 1.0) < 0.05  # ramp leaked to +y


def test_transpose_1_2_puts_grid_in_simulator_frame():
    # The reconciliation every rollout caller performs: transpose(1, 2) before
    # building the WorldConfig so a +x ramp is sampled along +x.
    n, res = 41, 0.1
    c = (n - 1) / 2.0
    z = _x_ramp_heightmap(n, res)[None].transpose(1, 2)  # what trainer/apply_terrain_props do
    wc = WorldConfig(z_grid=z, grid_res=res, max_coord=c * res)
    assert abs(_sample_height(wc, 1.0, 0.0) - 1.0) < 0.05  # +x is high
    assert abs(_sample_height(wc, 0.0, 1.0)) < 0.05        # +y is flat
