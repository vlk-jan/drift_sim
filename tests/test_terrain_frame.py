"""Terrain BEV-frame reconciliation between perception and the simulator.

Perception authors BEV grids as [row=X, col=Y]; WorldConfig / F.grid_sample use
"xy" indexing, [row=Y, col=X]. Every rollout caller must ``.transpose(1, 2)``
support and friction first, or the engine feels a +x ramp as a +y ramp and
silently scrambles the terrain-trajectory coupling.
"""

import torch
from _helpers import sample_height

from simulator.utils import WorldConfig


def _x_ramp_heightmap(n, res):
    """Support grid rising along world +x, in heightmap layout [row=X, col=Y]."""
    c = (n - 1) / 2.0
    ii = torch.arange(n).float()
    return (ii[:, None] - c) * res * torch.ones(1, n)  # (n, n)


def test_untransposed_grid_is_sampled_x_swapped():
    # The trap: an untransposed heightmap-frame grid feels +x as +y.
    n, res = 41, 0.1
    c = (n - 1) / 2.0
    wc = WorldConfig(
        z_grid=_x_ramp_heightmap(n, res)[None], grid_res=res, max_coord=c * res
    )
    assert abs(sample_height(wc, 1.0, 0.0)) < 0.05  # +x reads flat
    assert abs(sample_height(wc, 0.0, 1.0) - 1.0) < 0.05  # ramp leaked to +y


def test_transpose_1_2_puts_grid_in_simulator_frame():
    # What every rollout caller does: transpose(1, 2) before building WorldConfig.
    n, res = 41, 0.1
    c = (n - 1) / 2.0
    z = _x_ramp_heightmap(n, res)[None].transpose(
        1, 2
    )  # what trainer/apply_terrain_props do
    wc = WorldConfig(z_grid=z, grid_res=res, max_coord=c * res)
    assert abs(sample_height(wc, 1.0, 0.0) - 1.0) < 0.05  # +x is high
    assert abs(sample_height(wc, 0.0, 1.0)) < 0.05  # +y is flat
