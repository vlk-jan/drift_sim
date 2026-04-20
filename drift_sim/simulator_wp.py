from typing import Optional
import torch

from simulator.engine_wp import WarpStaticEngine
from simulator.numerical import quat_to_rotmat
from simulator.utils import find_init_rob_position
from simulator.utils import PhysicsEngineConfig
from simulator.utils import PhysicsState
from simulator.utils import RobotModelConfig
from simulator.utils import WorldConfig


class Simulator(torch.nn.Module):
    def __init__(
        self,
        robot="helhest",
        batch_size=1,
        device=None,
        T_horizon=5.0,
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

        self.batch_size = batch_size
        self.T_horizon = torch.tensor(T_horizon)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        MAX_STEPS = 100_000
        self.engine = WarpStaticEngine(robot_model, batch_size, MAX_STEPS, device)
        self.engine_config = PhysicsEngineConfig(num_robots=batch_size, dt=dt)
        self.res = None
        self.max_coord = None
        self.side_range = None
        self.fwd_range = None

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

    def predict_trajectory(
        self, properties, vel_ctrl, vel_ts, initial_state: Optional[dict] = None
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
        initial_state : dict, optional
            Initial state for the simulator. Can contain "xd", "omega", and "wheel_omega".

        Returns:
        -------
        transformations : torch.Tensor
            Relative robot motion. Shape (B, SimSteps, 4, 4).
        timestamps : torch.Tensor
            Timestamps for each simulation step. Shape (B, SimSteps).
        xd_out : torch.Tensor
            Linear velocity output. Shape (B, SimSteps, 3).
        omega_out : torch.Tensor
            Angular velocity output. Shape (B, SimSteps, 3).
        """
        # 1. Setup World
        world_config = WorldConfig(
            z_grid=properties["support"],
            grid_res=self.res,
            max_coord=self.max_coord,
            side_range=self.side_range,
            fwd_range=self.fwd_range,
            k_friction=properties["friction"],
            k_stiffness=properties["stiffness"],
        )

        # 2. Setup Initial State
        actual_bs = vel_ctrl.shape[0]
        state = PhysicsState.dummy(batch_size=actual_bs, device=self.device)
        state.x, state.q = find_init_rob_position(
            world_config, self.robot_model, actual_bs, self.device
        )

        if initial_state is not None:
            if "xd" in initial_state:
                state.xd = initial_state["xd"].to(self.device)
            if "omega" in initial_state:
                state.omega = initial_state["omega"].to(self.device)
            if "wheel_omega" in initial_state:
                state.wheel_omega = initial_state["wheel_omega"].to(self.device)

        # 3. Interpolate Controls
        actual_bs = vel_ctrl.shape[0]
        ref_ts = vel_ts[0]  # (Steps,)

        # Permute vel_ctrl to (Steps, B, 3) for efficient indexing
        vel_ctrl_permuted = vel_ctrl.permute(1, 0, 2)

        sim_dt = self.engine_config.dt
        num_steps = int(self.T_horizon / sim_dt)

        query_times = torch.linspace(
            ref_ts[0], ref_ts[0] + self.T_horizon, num_steps, device=self.device
        )

        idx = torch.searchsorted(ref_ts, query_times, right=True) - 1
        idx = idx.clamp(0, len(ref_ts) - 2)

        t0 = ref_ts[idx]
        t1 = ref_ts[idx + 1]

        # v0, v1 will be (SimSteps, B, 3)
        v0 = vel_ctrl_permuted[idx]
        v1 = vel_ctrl_permuted[idx + 1]

        alpha = ((query_times - t0) / (t1 - t0)).unsqueeze(-1).unsqueeze(-1)
        alpha = torch.nan_to_num(alpha, 0.0)

        # Dense VELOCITIES (Match PyTorch kinematic driving)
        dense_controls = v0 + (v1 - v0) * alpha  # (SimSteps, B, 3)
        dense_controls = dense_controls.to(torch.float32)

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

        # Expand timestamps to (B, SimSteps)
        query_times_batch = query_times.unsqueeze(0).expand(actual_bs, -1)

        return T, query_times_batch, xd_out, omega_out
