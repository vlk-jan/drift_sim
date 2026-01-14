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
        self, robot="helhest", grid_res=0.1, max_coord=6.4, batch_size=1, device=None
    ):
        super().__init__()

        if robot == "helhest":
            robot_model = RobotModelConfig(
                mass=93.8,
                cog=torch.tensor([0.145, 0.009, 0.0]),
                wheel_radii=torch.tensor([0.355, 0.355, 0.355]),
                wheel_centers=torch.tensor(
                    [[0.365, 0.365, 0.0], [0.365, -0.365, 0.0], [-0.365, 0.0, 0.0]]
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

        self.grid_res = grid_res
        self.max_coord = max_coord
        self.batch_size = batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        MAX_STEPS = 1000
        self.fast_engine = WarpStaticEngine(robot_model, batch_size, MAX_STEPS, device)
        self.engine_config = PhysicsEngineConfig(num_robots=batch_size)

    def predict_trajectory(self, properties, vel_ctrl, vel_ts):
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

        Returns:
        -------
        transformations : torch.Tensor
            Relative robot motion. Shape (B, SimSteps, 4, 4).
        timestamps : torch.Tensor
            Timestamps for each simulation step. Shape (B, SimSteps).
        """
        # 1. Setup World
        world_config = WorldConfig(
            z_grid=properties["geom"],
            grid_res=self.grid_res,
            max_coord=self.max_coord,
            k_friction=properties["friction"],
        )

        # 2. Setup Initial State
        state = PhysicsState.dummy(batch_size=self.batch_size, device=self.device)
        state.x, state.q = find_init_rob_position(
            world_config, self.robot_model, self.batch_size, self.device
        )

        # 3. Interpolate Controls
        ref_ts = vel_ts[0]  # (Steps,)

        # Permute vel_ctrl to (Steps, B, 3) for efficient indexing
        vel_ctrl_permuted = vel_ctrl.permute(1, 0, 2)

        sim_dt = self.engine_config.dt
        total_time = ref_ts[-1] - ref_ts[0]
        num_steps = int(total_time / sim_dt)

        query_times = torch.linspace(
            ref_ts[0], ref_ts[-1], num_steps, device=self.device
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

        # 4. RUN SUPER KERNEL
        traj_x, traj_q = self.fast_engine.run_trajectory(
            state, dense_controls, world_config, self.engine_config
        )

        # 5. Convert to Transformation Matrices
        # traj_q is now WXYZ (corrected by engine_wp), so quat_to_rotmat works
        R = quat_to_rotmat(traj_q)

        init_R = quat_to_rotmat(state.q).unsqueeze(0)
        init_x = state.x.unsqueeze(0)

        all_R = torch.cat([init_R, R], dim=0)
        all_x = torch.cat([init_x, traj_x], dim=0)

        R_prev = all_R[:-1]
        R_curr = all_R[1:]
        R_rel = R_prev.transpose(-1, -2) @ R_curr

        x_prev = all_x[:-1]
        x_curr = all_x[1:]
        rel_trans = x_curr - x_prev

        # T is initially (SimSteps, B, 4, 4)
        T = (
            torch.eye(4, device=self.device)
            .view(1, 1, 4, 4)
            .repeat(num_steps, self.batch_size, 1, 1)
        )
        T[:, :, :3, :3] = R_rel
        T[:, :, :3, 3] = rel_trans

        # Permute to (B, SimSteps, 4, 4)
        T = T.permute(1, 0, 2, 3)

        # Expand timestamps to (B, SimSteps)
        query_times_batch = query_times.unsqueeze(0).expand(self.batch_size, -1)

        return T, query_times_batch
