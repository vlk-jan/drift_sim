from typing import Optional
import torch

from simulator.numerical import quat_to_rotmat
from simulator.engine import DPhysicsEngine
from simulator.utils import (
    find_init_rob_position,
    PhysicsEngineConfig,
    RobotModelConfig,
    PhysicsState,
    WorldConfig,
)


class Simulator(torch.nn.Module):
    def __init__(
        self,
        robot="helhest",
        batch_size=1,
        device=None,
        T_horizon=1.0,
        dt=0.01,
    ):
        super().__init__()

        if robot == "helhest":
            robot_model = RobotModelConfig(
                mass=93.8,
                cog=torch.tensor([0.145, 0.009, 0.0]),
                wheel_radii=torch.tensor([0.4685, 0.4685, 0.4685]),
                wheel_centers=torch.tensor(
                    [[0.4, 0.52, 0.0], [0.4, -0.52, 0.0], [-0.5, 0.0, 0.0]]
                ),
                wheel_inertias=torch.tensor([0.2, 0.2, 0.2]),
                inertia_tensor=torch.tensor(
                    [[8.65, -0.18, 0.0], [-0.18, 10.64, 0.0], [0.0, 0.0, 19.29]]
                ),
            )
        elif robot == "marv":
            robot_model = RobotModelConfig(
                mass=67.8,
                wheel_radii=torch.tensor([0.12, 0.12, 0.12]),
                wheel_centers=torch.tensor(
                    [[0.245, 0.2025, 0.0], [0.245, -0.2025, 0.0], [-0.245, 0.0, 0.0]]
                ),
                wheel_inertias=torch.tensor([0.1, 0.1, 0.1]),
                inertia_tensor=torch.diag(torch.tensor([15.0, 15.0, 30.0])),
            )
        else:
            raise ValueError(f"Unsupported robot: {robot}")

        self.robot_model = robot_model

        engine_config = PhysicsEngineConfig(num_robots=batch_size, dt=dt, soft_contact_sigma=0.01)
        self.res = None
        self.max_coord = None
        self.side_range = None
        self.fwd_range = None

        self.batch_size = batch_size
        self.T_horizon = torch.tensor(T_horizon)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.engine = DPhysicsEngine(engine_config, robot_model)
        self.engine.to(device)
        self.engine.eval()

    def update_world_config(
        self, res=None, max_coord=None, side_range=None, fwd_range=None
    ):
        """Update the world configuration parameters."""
        if res is not None:
            self.res = res
        if max_coord is not None:
            self.max_coord = max_coord
        if side_range is not None:
            self.side_range = side_range
        if fwd_range is not None:
            self.fwd_range = fwd_range

    def _prepare_inputs(self, vel_ctrl, vel_ts, T_horizon=None):
        """
        Prepare control inputs for simulation. This may include resampling or interpolation such
        that each input has the same number of simulation steps and these steps align with the
        physics engine's time step.

        Parameters
        ----------
        vel_ctrl : torch.Tensor
            Control inputs for wheel velocities. Shape (B, Steps, 3).
        vel_ts : torch.Tensor
            Timestamps associated with the control inputs. Shape (B, Steps).
        T_horizon : float or torch.Tensor, optional
            Simulation horizon. Defaults to self.T_horizon.

        Returns
        -------
        new_ctrl : torch.Tensor
            Resampled control inputs. Shape (B, SimSteps, 3).
        new_ts : torch.Tensor
            Resampled timestamps. Shape (B, SimSteps).
        """
        dt = self.engine.get_dt()
        device = vel_ctrl.device
        B, S = vel_ts.shape

        if T_horizon is None:
            T_horizon = self.T_horizon
        if type(T_horizon) is float:
            T_horizon = torch.tensor(T_horizon, device=device)

        # 1. Define target timeline
        # Create target timestamps: [t_start, t_start + dt, ..., t_end]
        t_start = vel_ts[:, 0].unsqueeze(1)
        num_steps = int(torch.ceil(T_horizon / dt).item()) + 1
        target_ts = t_start + (torch.arange(num_steps, device=device) * dt).unsqueeze(
            0
        ).expand(B, -1)

        # 2. Prepare for Vectorized Search
        # target_ts_batch: (B, NumTargets)
        target_ts_batch = target_ts.contiguous()
        vel_ts = vel_ts.contiguous()

        # 3. Find indices in the original timeline
        idx_right = torch.searchsorted(vel_ts, target_ts_batch, right=True)

        # Clamp indices to valid range for interpolation [0, S-2]
        # idx_right corresponds to 'next' element, so we want idx_left to be idx_right - 1
        idx_left = (idx_right - 1).clamp(0, S - 2)
        idx_right = idx_left + 1

        # 4. Gather values for interpolation
        # Helper to index (B, S, 3) using (B, NumTargets) indices
        batch_indices = (
            torch.arange(B, device=device).unsqueeze(1).expand(-1, num_steps)
        )

        # Values: (B, NumTargets, 3)
        v_left = vel_ctrl[batch_indices, idx_left]
        v_right = vel_ctrl[batch_indices, idx_right]

        t_left = vel_ts[batch_indices, idx_left]
        t_right = vel_ts[batch_indices, idx_right]

        # 5. Interpolate
        denom = (t_right - t_left).clamp(min=1e-6)
        alpha = (target_ts_batch - t_left) / denom

        # Clamp alpha to [0, 1] to avoid extrapolation issues
        alpha = alpha.clamp(0.0, 1.0).unsqueeze(-1)

        new_ctrl = v_left + (v_right - v_left) * alpha

        return new_ctrl, target_ts_batch

    def predict_trajectory(
        self,
        properties: dict,
        vel_ctrl: torch.Tensor,
        vel_ts: torch.Tensor,
        initial_state: Optional[dict] = None,
        T_horizon: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute a list of SO3 transformation matrices for the given wheel velocities.

        Parameters
        ----------
        properties : dict
            A dictionary containing terrain properties:
            - "height_map": torch.Tensor of shape (B, H, W) representing heightmap (used as fallback when "support" is absent).
            - "friction": torch.Tensor of shape (B, H, W) or a scalar representing friction coefficients.
            - "stiffness": torch.Tensor of shape (B, H, W) or a scalar representing stiffness coefficients.
        vel_ctrl : torch.Tensor
            Control inputs for wheel velocities. Shape (B, Steps, 3) where 3 is the number of wheels.
        vel_ts : torch.Tensor
            Timestamps associated with the control inputs. Shape (B, Steps).
        initial_state : dict, optional
            Initial state for the simulator. Can contain "x" (position, shape B×3),
            "q" (quaternion [w,x,y,z], shape B×4), "xd", "omega", and "wheel_omega".
            When "x" and "q" are provided they override the terrain-fitted initial pose,
            enabling warm-starting from an arbitrary pose (e.g., a GT segment boundary).
        T_horizon : float, optional
            Simulation time horizon. Defaults to self.T_horizon. Pass a shorter value
            when simulating a sub-segment of the full trajectory.

        Returns:
        -------
        transformations : torch.Tensor
            A stack of SE(3) transformation matrices representing relative robot motion.
            Shape (B, SimSteps, 4, 4).
        timestamps : torch.Tensor
            The timestamps for each simulation step. Shape (B, SimSteps).
        """
        actual_bs = vel_ctrl.shape[0]

        if "support" not in properties.keys():
            properties["support"] = properties["height_map"]

        world_config = WorldConfig(
            z_grid=properties["support"],
            grid_res=self.res,
            max_coord=self.max_coord,
            side_range=self.side_range,
            fwd_range=self.fwd_range,
            k_friction=properties["friction"],
            k_stiffness=properties["stiffness"],
        )
        state = PhysicsState.dummy(batch_size=actual_bs, device=self.device)
        state.x, state.q = find_init_rob_position(
            world_config, self.robot_model, actual_bs, self.device
        )

        if initial_state is not None:
            if "x" in initial_state:
                state.x = initial_state["x"].to(self.device)
            if "q" in initial_state:
                state.q = initial_state["q"].to(self.device)
            if "xd" in initial_state:
                state.xd = initial_state["xd"].to(self.device)
            if "omega" in initial_state:
                state.omega = initial_state["omega"].to(self.device)
            if "wheel_omega" in initial_state:
                state.wheel_omega = initial_state["wheel_omega"].to(self.device)

        vel_ctrl, vel_ts = self._prepare_inputs(vel_ctrl, vel_ts, T_horizon=T_horizon)

        # Precompute accelerations (B, SimSteps-1, 3)
        d_vel = vel_ctrl[:, 1:] - vel_ctrl[:, :-1]
        d_ts = (vel_ts[:, 1:] - vel_ts[:, :-1]).clamp(min=1e-6).unsqueeze(-1)
        accels = d_vel / d_ts

        dt = self.engine.get_dt()
        num_steps = vel_ts.shape[-1] - 1

        poses = torch.empty(
            num_steps, actual_bs, 4, 4, device=self.device, dtype=torch.float32
        )
        xd_out = torch.empty(
            num_steps, actual_bs, 3, device=self.device, dtype=torch.float32
        )
        omega_out = torch.empty(
            num_steps, actual_bs, 3, device=self.device, dtype=torch.float32
        )
        # Match dtype of vel_ts (double) so timestamp comparisons in trajectory_loss are exact
        timestamps = torch.empty(
            num_steps, actual_bs, device=self.device, dtype=vel_ts.dtype
        )

        ts = vel_ts[:, 0].clone()

        for idx in range(num_steps):
            # Update state wheel velocities to match interpolated input
            state.wheel_omega = vel_ctrl[:, idx]

            # Use precomputed acceleration
            curr_acceleration = accels[:, idx]

            # Forward with acceleration control
            dstate = self.engine.forward(state, curr_acceleration, world_config)
            state = self.engine.update_state(state, dstate)

            poses[idx, :, :3, :3] = quat_to_rotmat(state.q)
            poses[idx, :, :3, 3] = state.x
            xd_out[idx] = state.xd
            omega_out[idx] = state.omega
            timestamps[idx] = ts

            ts = ts + dt

        poses = poses.permute(1, 0, 2, 3)  # (B, SimSteps, 4, 4)
        xd_out = xd_out.permute(1, 0, 2)  # (B, SimSteps, 3)
        omega_out = omega_out.permute(1, 0, 2)  # (B, SimSteps, 3)
        timestamps = timestamps.permute(1, 0)  # (B, SimSteps)

        return poses, timestamps, xd_out, omega_out
