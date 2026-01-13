import torch

from simulator.numerical import quat_to_rotmat
from simulator.physics_engine import DPhysicsEngine
from simulator.physics_engine import find_init_rob_position
from models.engine_wp import WarpStaticEngine
from simulator.utils import PhysicsEngineConfig
from simulator.utils import PhysicsState
from simulator.utils import RobotModelConfig
from simulator.world_config import WorldConfig


class Simulator(torch.nn.Module):
    def __init__(self, robot="helhest", grid_res=0.1, max_coord=6.4, batch_size=1, device=None):
        super().__init__()

        if robot == "helhest":
            robot_model = RobotModelConfig(
                mass=90.0,
                wheel_radii=torch.tensor([0.5, 0.5, 0.5]),
                wheel_centers=torch.tensor([[0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, 0.0, 0.0]]),
                wheel_inertias=torch.tensor([0.1, 0.1, 0.1]),
                inertia_tensor=torch.diag(torch.tensor([10.0, 10.0, 20.0])),
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
        self.engine.eval()

    def predict_trajectory(self, properties, vel_ctrl, vel_ts):
        """
        Calculates trajectory using the Fused Warp Kernel.
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

        # 3. Interpolate Controls (Pre-processing in PyTorch)
        # We need to turn the sparse vel_ctrl/vel_ts into a dense (Steps, B, 3) tensor
        # This part runs in Python/Torch but it's vectorizable and fast.

        sim_dt = self.engine_config.dt
        total_time = vel_ts[-1] - vel_ts[0]
        num_steps = int(total_time / sim_dt)

        # Create dense time steps
        query_times = torch.linspace(vel_ts[0], vel_ts[-1], num_steps, device=self.device)

        # Vectorized Linear Interpolation for Controls
        # Find indices of timestamps just before query_times
        # (Assuming vel_ts is sorted)
        idx = torch.searchsorted(vel_ts, query_times, right=True) - 1
        idx = idx.clamp(0, len(vel_ts) - 2)

        t0 = vel_ts[idx]
        t1 = vel_ts[idx + 1]
        v0 = vel_ctrl[idx]
        v1 = vel_ctrl[idx + 1]

        alpha = ((query_times - t0) / (t1 - t0)).unsqueeze(-1).unsqueeze(-1)  # (T, 1, 1)
        alpha = torch.nan_to_num(alpha, 0.0)

        # Dense Controls: (NumSteps, Batch, 3)
        dense_controls = v0 + (v1 - v0) * alpha

        # Acceleration is derivative of velocity?
        # Your simulator takes *acceleration* as input to forward().
        # If vel_ctrl is TARGET velocity, your logic in original code was:
        # `curr_acceleration = (v1 - v0) / dt`.
        # So we need dense ACCELERATION.

        dense_accel = (v1 - v0) / (t1 - t0).unsqueeze(-1).unsqueeze(-1)
        dense_accel = torch.nan_to_num(dense_accel, 0.0)  # Handle dt=0

        # 4. RUN SUPER KERNEL
        traj_x, traj_q = self.fast_engine.run_trajectory(
            state, dense_accel, world_config, self.engine_config
        )

        # 5. Convert to Transformation Matrices (Post-processing)
        # traj_x: (Steps, B, 3)
        # traj_q: (Steps, B, 4)

        # We need relative transformations? Or absolute?
        # Your original code returned stack of SE3 matrices.

        # Vectorized SE3 construction
        # (We can do this in pure Torch extremely fast)
        R = quat_to_rotmat(traj_q)  # (Steps, B, 3, 3)

        T = (
            torch.eye(4, device=self.device)
            .view(1, 1, 4, 4)
            .repeat(num_steps, self.batch_size, 1, 1)
        )
        T[:, :, :3, :3] = R
        T[:, :, :3, 3] = traj_x

        # Your original code returned RELATIVE transforms (step-to-step)
        # `rel_trans = state.x - prev_x`
        # `R_rel = R_prev.T @ R_curr`

        # Let's compute that vectorized:
        T_abs = T
        T_prev = torch.cat([T_abs[:1], T_abs[:-1]], dim=0)  # Shift by 1

        # T_rel = inv(T_prev) @ T_curr
        # Since T is SE3, inv(T) is cheap, but torch.linalg.inv is fine for batch
        T_rel = torch.linalg.inv(T_prev) @ T_abs

        # Fix first step (which is identity or delta from init)
        T_rel[0] = torch.eye(4, device=self.device)

        return T_rel, query_times
