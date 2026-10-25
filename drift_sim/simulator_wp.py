from typing import Optional

import torch

from models.simulator import (
    _apply_initial_state_override,
    _build_world_config,
    _update_world_config,
)
from simulator.engine import DPhysicsEngine
from simulator.engine_wp import WarpStaticEngine
from simulator.utils import (
    PhysicsEngineConfig,
    PhysicsState,
    WorldConfig,
    find_init_rob_position,
    num_sim_steps,
    robot_model_config,
    seat_on_terrain,
)
from utils.geometry import quat_to_rotmat


class Simulator(torch.nn.Module):
    def __init__(
        self,
        robot="helhest_v2_median",
        device=None,
        T_horizon=1.0,
        dt=2e-3,
        com_transfer=True,
    ):
        """Warp simulator wrapper; T_horizon must match models/simulator.py."""
        super().__init__()

        robot_model = robot_model_config(robot)

        self.robot_model = robot_model

        self.T_horizon = T_horizon
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.engine = WarpStaticEngine(robot_model, device)
        self.engine_config = PhysicsEngineConfig(dt=dt, com_transfer=com_transfer)
        self._geom = DPhysicsEngine(self.engine_config, robot_model).to(device)
        self.res = None
        self.max_coord = None
        self.side_range = None
        self.fwd_range = None

    def update_world_config(
        self, res=None, max_coord=None, side_range=None, fwd_range=None
    ):
        """Update the world configuration parameters."""
        _update_world_config(
            self,
            res=res,
            max_coord=max_coord,
            side_range=side_range,
            fwd_range=fwd_range,
        )

    @torch.no_grad()
    def fit_plane_pose(self, world_config: WorldConfig, batch_size: int):
        """Fit the initial pose to the 3-wheel plane; see models/simulator.py."""
        return find_init_rob_position(
            world_config, self.robot_model, batch_size, self.device
        )

    def seat_on_terrain(self, x, q, world_config: WorldConfig):
        """Seat the robot on its static sag. See simulator/utils.py."""
        return seat_on_terrain(self._geom, x, q, world_config)

    def predict_trajectory(
        self,
        properties,
        vel_ctrl,
        vel_ts,
        initial_state: Optional[dict] = None,
        T_horizon: Optional[float] = None,
        include_start: bool = False,
    ):
        """Roll out a trajectory with the Warp engine: four kernel launches per
        physics step on a wp.Tape, replayed from a captured graph (see
        simulator/engine_wp.py).

        Parameters
        ----------
        properties : dict
            Terrain properties (support, friction).
        vel_ctrl : torch.Tensor
            Wheel velocity controls, (B, Steps, 3).
        vel_ts : torch.Tensor
            Absolute Unix timestamps, (B, Steps); each element has its own window.
        initial_state : dict, optional
            May contain only "x", "q", "xd", "omega".
        include_start : bool, optional
            Also return the start state first; see models/simulator.py.

        Returns:
        -------
        transformations : torch.Tensor
            Robot poses, (B, K, 4, 4), K = SimSteps + int(include_start).
        timestamps : torch.Tensor
            float64, (B, K). Pose i is the post-step state stamped at the
            step's end, so the first stamp is vel_ts[b, 0] + dt unless
            include_start.
        xd_out : torch.Tensor
            Linear velocity, (B, K, 3).
        omega_out : torch.Tensor
            Angular velocity, (B, K, 3).
        """
        world_config = _build_world_config(self, properties)

        actual_bs = vel_ctrl.shape[0]
        state = PhysicsState.dummy(batch_size=actual_bs, device=self.device)
        # Detached on purpose; see models/simulator.py::Simulator.fit_plane_pose.
        # Skipped when the caller seeds both x and q (the segmented loss does).
        if not {"x", "q"} <= set(initial_state or ()):
            with torch.no_grad():
                state.x, state.q = self.fit_plane_pose(world_config, actual_bs)

        # Seeds segmented-loss segments at their GT start pose.
        state = _apply_initial_state_override(state, initial_state, self.device)

        # Starts at rest on the static sag: a deep start launches the robot at
        # t=0, a start at zero depth sinks onto the sag.
        state.x = self.seat_on_terrain(state.x, state.q, world_config)

        sim_dt = self.engine_config.dt
        horizon = self.T_horizon if T_horizon is None else T_horizon
        num_steps = num_sim_steps(horizon, sim_dt)

        # Build the query timeline per element: windows are far apart in Unix time,
        # and losses match poses by nearest timestamp, so a shared timeline would
        # silently zero the gradient of every element but the first.
        B, S = vel_ts.shape
        # float64 is mandatory: float32 spacing at ~1.76e9 s is ~128 s.
        vel_ts = vel_ts.to(torch.float64).contiguous()
        step_offsets = (
            torch.arange(num_steps, device=vel_ts.device, dtype=torch.float64) * sim_dt
        )
        # Query at each sim step's start -- the control that drives that step.
        query_times = vel_ts[:, :1] + step_offsets  # (B, num_steps)

        idx = torch.searchsorted(vel_ts, query_times, right=True) - 1
        idx = idx.clamp(0, S - 2)
        batch_idx = torch.arange(B, device=vel_ts.device).unsqueeze(1)

        t0 = vel_ts[batch_idx, idx]
        t1 = vel_ts[batch_idx, idx + 1]
        v0 = vel_ctrl[batch_idx, idx]  # (B, num_steps, 3)
        v1 = vel_ctrl[batch_idx, idx + 1]

        alpha = ((query_times - t0) / (t1 - t0)).unsqueeze(-1)
        # Clamped like the torch engine (hold, don't invent commands); collate pads
        # ctrl_ts by repeating the last stamp, so padded 0/0 hits nan_to_num.
        alpha = torch.nan_to_num(alpha, 0.0).clamp(0.0, 1.0)

        # Dense velocities, (SimSteps, B, 3).
        dense_controls = (v0 + (v1 - v0) * alpha).permute(1, 0, 2).to(torch.float32)

        traj_x, traj_q, traj_xd, traj_omega = self.engine.run_trajectory(
            state, dense_controls, world_config, self.engine_config
        )
        if include_start:
            traj_x, traj_q, traj_xd, traj_omega = (
                torch.cat([s.unsqueeze(0), t])
                for s, t in zip(
                    (state.x, state.q, state.xd, state.omega),
                    (traj_x, traj_q, traj_xd, traj_omega),
                )
            )

        # engine_wp returns wxyz quaternions.
        R = quat_to_rotmat(traj_q)  # (SimSteps, B, 3, 3)
        x = traj_x  # (SimSteps, B, 3)

        T = (
            torch.eye(4, device=self.device)
            .view(1, 1, 4, 4)
            .repeat(x.shape[0], actual_bs, 1, 1)
        )
        T[:, :, :3, :3] = R
        T[:, :, :3, 3] = x

        T = T.permute(1, 0, 2, 3)  # (B, SimSteps, 4, 4)
        xd_out = traj_xd.permute(1, 0, 2)  # (B, SimSteps, 3)
        omega_out = traj_omega.permute(1, 0, 2)  # (B, SimSteps, 3)

        # Poses are post-step states stamped at the step's end: engine_wp drops the
        # seeded step 0, so its pose i is the state after step i+1; include_start
        # puts the start, stamped vel_ts[:, 0], in front.
        timestamps = query_times + sim_dt  # (B, SimSteps), float64
        if include_start:
            timestamps = torch.cat([vel_ts[:, :1], timestamps], dim=1)

        return T, timestamps, xd_out, omega_out
