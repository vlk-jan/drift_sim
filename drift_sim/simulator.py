from typing import Optional

import torch

from simulator.engine import DPhysicsEngine
from simulator.numerical import quat_to_rotmat
from simulator.utils import (
    PhysicsEngineConfig,
    PhysicsState,
    RobotModelConfig,
    WorldConfig,
    find_init_rob_position,
    raise_to_clearance,
)


class Simulator(torch.nn.Module):
    def __init__(
        self,
        robot="helhest",
        batch_size=1,
        device=None,
        T_horizon=1.0,
        dt=2e-3,  # see PhysicsEngineConfig.dt / scripts/analysis/sweep_dt.py
    ):
        super().__init__()

        if robot == "helhest":
            robot_model = RobotModelConfig(
                mass=93.8,
                cog=torch.tensor([0.145, 0.009, 0.0]),
                wheel_radii=torch.tensor([0.3685, 0.3685, 0.3685]),
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

        engine_config = PhysicsEngineConfig(
            num_robots=batch_size, dt=dt, soft_contact_sigma=0.01
        )
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
        Prepare control inputs for simulation. This may include resampling or
            interpolation such
        that each input has the same number of simulation steps and these steps align
            with the
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
        # vel_ts is float64 by REQUIREMENT, not accident: these are absolute
        # Unix timestamps (~1.76e9 s), where float32 spacing is ~128 s — far
        # coarser than the 2 ms sim step. Everything derived from it below
        # (target_ts, alpha, and therefore new_ctrl) inherits float64; do not
        # downcast the timeline.
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
        # idx_right corresponds to 'next' element, so we want idx_left to be idx_right -
        # 1
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

    @torch.no_grad()
    def fit_plane_pose(
        self, world_config: WorldConfig, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initial pose from the 3-point (per-wheel) plane fit through the height field.

        Detached: a single bad cell under one wheel swings the fit a lot. Left
        differentiable, gradient descent exploits it as a cheap shortcut to fix
        rotational trajectory error by locally saturating height_residual under a
        wheel instead of learning real terrain/friction physics — the initial pose
        must be a non-differentiable function of the terrain snapshot; gradient
        into height_residual should only flow through the actual contact dynamics
        of the rollout.
        """
        return find_init_rob_position(
            world_config, self.robot_model, batch_size, self.device
        )

    @torch.no_grad()
    def raise_to_clearance(
        self, x: torch.Tensor, q: torch.Tensor, world_config: WorldConfig
    ) -> torch.Tensor:
        """Raise the robot until all wheel circumference sample points clear the terrain.

        fit_plane_pose only ensures the wheel-triangle plane is tangent; on
        sloped/curved terrain the 16 circumference sample points per wheel can
        penetrate, producing enormous initial spring forces that launch the robot
        upward at t=0. Returns the raised position (input `x` is not mutated —
        it may be a view into GT batch data).

        Each iteration applies -dh (vertical penetration depth), which removes
        fraction nz of the true geometric penetration per step — so convergence is
        geometric. The direct -dh/nz formula would be exact in one step but
        amplifies tiny dh at near-vertical faces (nz≈0) into a huge over-lift.
        5 iterations give <3% residual even on 60° slopes while never
        over-shooting.

        Implementation lives in simulator/utils.py so the warp wrapper seats the
        robot identically; this stays as the public entry point that analysis
        scripts already call.
        """
        return raise_to_clearance(self.engine, x, q, world_config)

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
            - "support": torch.Tensor of shape (B, H, W) representing the support
                surface (height_map - height_residual).
            - "friction": torch.Tensor of shape (B, H, W) or a scalar representing
                friction coefficients.
            - "stiffness": optional (B, H, W) or scalar; if absent, the fixed
              WorldConfig constant is used (stiffness is not predicted).
        vel_ctrl : torch.Tensor
            Control inputs for wheel velocities. Shape (B, Steps, 3) where 3 is the
                number of wheels.
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
            Pose i is the post-step state, stamped at the step's end, so element
            b's first stamp is vel_ts[b, 0] + dt (same convention as
            models/simulator_wp.py).
        """
        actual_bs = vel_ctrl.shape[0]

        world_config = WorldConfig(
            z_grid=properties["support"],
            grid_res=self.res,
            max_coord=self.max_coord,
            side_range=self.side_range,
            fwd_range=self.fwd_range,
            k_friction=properties["friction"],
            # Stiffness is not a predicted property
            k_stiffness=properties.get("stiffness", 400_000.0),
        )
        state = PhysicsState.dummy(batch_size=actual_bs, device=self.device)
        state.x, state.q = self.fit_plane_pose(world_config, actual_bs)

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

        state.x = self.raise_to_clearance(state.x, state.q, world_config)

        vel_ctrl, vel_ts = self._prepare_inputs(vel_ctrl, vel_ts, T_horizon=T_horizon)

        # Precompute accelerations (B, SimSteps-1, 3)
        d_vel = vel_ctrl[:, 1:] - vel_ctrl[:, :-1]
        d_ts = (vel_ts[:, 1:] - vel_ts[:, :-1]).clamp(min=1e-6).unsqueeze(-1)
        accels = d_vel / d_ts

        dt = self.engine.get_dt()
        num_steps = vel_ts.shape[-1] - 1

        # Collect per-step outputs as lists (preserves gradient graph through stack)
        R_list = []
        t_list = []
        xd_out_list = []
        omega_out_list = []
        timestamps_list = []

        ts = vel_ts[:, 0].clone()

        for idx in range(num_steps):
            state.wheel_omega = vel_ctrl[:, idx]
            curr_acceleration = accels[:, idx]

            dstate = self.engine.forward(state, curr_acceleration, world_config)
            state = self.engine.update_state(state, dstate)

            # poses are the post-step states, stamped at the step's END
            ts = ts + dt

            R_list.append(quat_to_rotmat(state.q))  # (B, 3, 3)
            t_list.append(state.x)  # (B, 3)
            xd_out_list.append(state.xd)
            omega_out_list.append(state.omega)
            timestamps_list.append(ts)

        # Build final tensors — SE(3) construction done once outside the loop
        R_all = torch.stack(R_list, dim=1)  # (B, SimSteps, 3, 3)
        t_all = torch.stack(t_list, dim=1)  # (B, SimSteps, 3)
        top_part = torch.cat(
            [R_all, t_all.unsqueeze(-1)], dim=-1
        )  # (B, SimSteps, 3, 4)
        bottom = torch.zeros(actual_bs, num_steps, 1, 4, device=self.device)
        bottom[:, :, 0, 3] = 1.0
        poses = torch.cat([top_part, bottom], dim=2)  # (B, SimSteps, 4, 4)

        xd_out = torch.stack(xd_out_list, dim=1)  # (B, SimSteps, 3)
        omega_out = torch.stack(omega_out_list, dim=1)  # (B, SimSteps, 3)
        timestamps = torch.stack(timestamps_list, dim=1)  # (B, SimSteps)

        return poses, timestamps, xd_out, omega_out
