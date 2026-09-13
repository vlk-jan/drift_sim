from typing import Optional

import torch

from models.simulator import (
    _apply_initial_state_override,
    _build_world_config,
    _fit_plane_pose,
    _update_world_config,
)
from simulator.engine import DPhysicsEngine
from simulator.engine_wp import WarpStaticEngine
from simulator.utils import (
    PhysicsEngineConfig,
    PhysicsState,
    WorldConfig,
    raise_to_clearance,
    robot_model_config,
)
from utils.geometry import quat_to_rotmat


class Simulator(torch.nn.Module):
    def __init__(
        self,
        robot="helhest",
        device=None,
        T_horizon=1.0,
        dt=2e-3,
        com_transfer=False,
    ):
        """Warp wrapper. See models/simulator.py for T_horizon (was 5.0 here,
        1.0 there, and the trainer's single-segment path passes neither).
        """
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
            self, res=res, max_coord=max_coord, side_range=side_range, fwd_range=fwd_range
        )

    @torch.no_grad()
    def fit_plane_pose(self, world_config: WorldConfig, batch_size: int):
        """Initial pose from the 3-point per-wheel plane fit. See models/simulator.py."""
        return _fit_plane_pose(self.robot_model, self.device, world_config, batch_size)

    @torch.no_grad()
    def raise_to_clearance(self, x, q, world_config: WorldConfig):
        """Lift the robot clear of the terrain. See simulator/utils.py."""
        return raise_to_clearance(self._geom, x, q, world_config)

    def predict_trajectory(
        self,
        properties,
        vel_ctrl,
        vel_ts,
        initial_state: Optional[dict] = None,
        T_horizon: Optional[float] = None,
    ):
        """
        Calculates trajectory using the Fused Warp Kernel with Velocity Control.

        Parameters
        ----------
        properties : dict
            Terrain properties (geom, friction).
        vel_ctrl : torch.Tensor
            Control inputs for wheel velocities. Shape (B, Steps, 3).
        vel_ts : torch.Tensor
            Timestamps associated with the control inputs. Shape (B, Steps).
            Absolute (Unix) seconds, and every batch element is its own window
            with its own start time — controls are resampled per element.
        initial_state : dict, optional
            Initial state for the simulator. Can contain "xd", "omega", and
                "wheel_omega".

        Returns:
        -------
        transformations : torch.Tensor
            Relative robot motion. Shape (B, SimSteps, 4, 4).
        timestamps : torch.Tensor
            Timestamps for each simulation step, float64, shape (B, SimSteps).
            Pose i is the post-step state, stamped at the step's end, so element
            b's first stamp is vel_ts[b, 0] + dt and stamps are spaced by the
            engine's dt.
        xd_out : torch.Tensor
            Linear velocity output. Shape (B, SimSteps, 3).
        omega_out : torch.Tensor
            Angular velocity output. Shape (B, SimSteps, 3).
        """
        # 1. Setup World
        world_config = _build_world_config(self, properties)

        # 2. Setup Initial State
        actual_bs = vel_ctrl.shape[0]
        state = PhysicsState.dummy(batch_size=actual_bs, device=self.device)
        # Detached: see models/simulator.py's predict_trajectory for why this 3-point
        # (per-wheel) plane fit must not be differentiable.
        with torch.no_grad():
            state.x, state.q = self.fit_plane_pose(world_config, actual_bs)

        # "x"/"q" override the terrain-fitted pose. The segmented trajectory loss
        # (utils/trainer.py) relies on this to seed each segment at the GT pose
        # closest to its own start timestamp; skipping it would silently restart
        # every segment at the fitted origin pose.
        state = _apply_initial_state_override(state, initial_state, self.device)

        # The plane fit only makes the wheel-triangle tangent; without this the
        # wheel circumference points start inside rough terrain and the contact
        # springs launch the robot at t=0.
        state.x = self.raise_to_clearance(state.x, state.q, world_config)

        # 3. Interpolate Controls
        sim_dt = self.engine_config.dt
        horizon = self.T_horizon if T_horizon is None else T_horizon
        # round, not truncate: 1.0/1e-3 lands on 999.9999... in binary floating
        # point, which silently made warp rollouts one step shorter than torch's
        num_steps = int(round(horizon / sim_dt))

        # Every batch element is its own data window, and the windows are seconds
        # to minutes apart in absolute Unix time — so the query timeline has to be
        # built per element. Interpolating the whole batch on element 0's timeline
        # (and returning it for everyone) put every element b>0 entirely outside
        # its own control span, and since utils/losses.py matches GT poses to
        # predicted ones by nearest timestamp, all of that element's GT collapsed
        # onto the first or last predicted pose — no crash, just ~zero gradient
        # from all but one window of every batch. This mirrors what the torch
        # engine's _prepare_inputs has always done.
        B, S = vel_ts.shape
        # float64 is mandatory, not tidiness: these are absolute Unix timestamps
        # (~1.76e9), where float32 spacing is ~128 s.
        vel_ts = vel_ts.to(torch.float64).contiguous()
        step_offsets = (
            torch.arange(num_steps, device=vel_ts.device, dtype=torch.float64) * sim_dt
        )
        # Query at each sim step's START — the control that drives that step.
        query_times = vel_ts[:, :1] + step_offsets  # (B, num_steps)

        idx = torch.searchsorted(vel_ts, query_times, right=True) - 1
        idx = idx.clamp(0, S - 2)
        batch_idx = torch.arange(B, device=vel_ts.device).unsqueeze(1)

        t0 = vel_ts[batch_idx, idx]
        t1 = vel_ts[batch_idx, idx + 1]
        v0 = vel_ctrl[batch_idx, idx]  # (B, num_steps, 3)
        v1 = vel_ctrl[batch_idx, idx + 1]

        alpha = ((query_times - t0) / (t1 - t0)).unsqueeze(-1)
        # Clamped, not extrapolated: a horizon longer than the control span used
        # to keep projecting the last slope outwards, inventing wheel speeds that
        # were never commanded. The torch engine holds the last value instead, and
        # the collate pads ctrl_ts by repeating the final timestamp, so padded
        # segments divide 0/0 and land on the nan_to_num branch.
        alpha = torch.nan_to_num(alpha, 0.0).clamp(0.0, 1.0)

        # Dense VELOCITIES (Match PyTorch kinematic driving), (SimSteps, B, 3)
        dense_controls = (v0 + (v1 - v0) * alpha).permute(1, 0, 2).to(torch.float32)

        # 4. RUN SUPER KERNEL
        traj_x, traj_q, traj_xd, traj_omega = self.engine.run_trajectory(
            state, dense_controls, world_config, self.engine_config
        )

        # 5. Convert to Absolute Pose Matrices
        # traj_q is now WXYZ (corrected by engine_wp), so quat_to_rotmat works
        R = quat_to_rotmat(traj_q)  # (SimSteps, B, 3, 3)
        x = traj_x  # (SimSteps, B, 3)

        # T is initially (SimSteps, B, 4, 4)
        T = (
            torch.eye(4, device=self.device)
            .view(1, 1, 4, 4)
            .repeat(num_steps, actual_bs, 1, 1)
        )
        T[:, :, :3, :3] = R
        T[:, :, :3, 3] = x

        # Permute to (B, SimSteps, 4, 4)
        T = T.permute(1, 0, 2, 3)
        xd_out = traj_xd.permute(1, 0, 2)  # (B, SimSteps, 3)
        omega_out = traj_omega.permute(1, 0, 2)  # (B, SimSteps, 3)

        # Poses are the post-step states, stamped at the step's END: engine_wp
        # drops the seeded step 0, so output pose i is the state after step i+1.
        timestamps = query_times + sim_dt  # (B, SimSteps), float64

        return T, timestamps, xd_out, omega_out
