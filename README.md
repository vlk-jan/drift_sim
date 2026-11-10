# drift_sim

Differentiable simulator of the HELHEST skid-steer robot (three wheels: left,
right, rear) driving over a terrain given as dense maps. You give it the terrain
properties and the wheel controls, and it returns the simulated SE(3) trajectory,
differentiable with respect to the maps.

Two engines run the same contact law: a torch one (`Simulator`) and a Warp one
(`WarpSimulator`, the default in DRIFT, about 12x faster). `tests/test_engine_parity.py`
holds them to the same poses and gradients, so any physics change must go into both.

## Install

```bash
uv add "drift_sim @ git+ssh://git@github.com/vlk-jan/drift_sim.git"   # or: pip install git+ssh://...
uv add "drift_sim[viz] @ ..."                                         # + SimView scene helpers
```

PyTorch comes from PyPI by default. To get a CPU-only build or a specific CUDA build,
install torch from the PyTorch index first. `warp-lang` is pinned to 1.14.0 because a
new version can change the simulated trajectories.

## Use

```python
import torch

from drift_sim import WarpSimulator  # or Simulator (torch engine)

# 64 x 64 maps, 0.1 m cells; the robot starts at the grid centre facing +x.
B, N, res = 2, 64, 0.1
half = N * res / 2

sim = WarpSimulator(robot="helhest_v2_median", device="cpu", T_horizon=2.0)
sim.update_world_config(res=res, max_coord=half - res / 2)

properties = {
    "support": torch.zeros(B, N, N),          # terrain height (m), indexed [y, x]
    "friction": torch.full((B, N, N), 0.8),   # friction coefficient per cell
}
# Wheel angular velocities (rad/s), columns [left, right, rear], at timestamps (s).
vel_ts = torch.linspace(0.0, 2.0, 21, dtype=torch.float64).expand(B, -1)
vel_ctrl = torch.full((B, 21, 3), 2.0)
vel_ctrl[1, :, 0] = 1.0  # second robot: slower left wheel, turns left

poses, ts, lin_vel, ang_vel = sim.predict_trajectory(properties, vel_ctrl, vel_ts)
# poses (B, K, 4, 4), ts (B, K) float64, velocities (B, K, 3); K = T_horizon / dt
```

Conventions:

- **Maps.** `(B, H, W)` tensors indexed `[y, x]`, with x forward and y left. The
  grid centre is the world origin. `max_coord` is the distance from the centre to
  the centre of the outermost cell, i.e. `half_extent - res / 2`.
- **Start pose.** By default the start pose is a plane fitted under the three
  wheels and then settled onto the tyres' static sag. To start from a known pose,
  pass `initial_state={"x": (B, 3), "q": (B, 4) wxyz}`; it may also contain `xd`
  and `omega`.
- **Controls.** Linearly interpolated onto the physics step (`dt`, default 2 ms)
  and held at their last value past the end. `vel_ts` must be float64: float32
  cannot represent absolute Unix timestamps.
- **Robots.** All robot constants (mass, inertia, centre of gravity, wheel
  geometry, tyre stiffness) live in `drift_sim/utils.py::robot_model_config`;
  `ROBOT_MODELS` lists the robots it supports.
- **Engine settings.** `PhysicsEngineConfig` holds the time step, gravity and
  slip-law constants.
- **Gradients.** They flow to `support` and `friction` when those tensors have
  `requires_grad`.

Code comments that cite `utils/`, `scripts/`, `datasets/` or `dataset_tools/` refer
to DRIFT, the project this simulator was developed in.

## Test

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q tests
```
