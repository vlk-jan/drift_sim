import torch

from simulator.numerical import quat_to_rotmat
from simulator.physics_engine import DPhysicsEngine, find_init_rob_position
from simulator.utils import (
    PhysicsEngineConfig,
    RobotModelConfig,
    PhysicsState,
)
from simulator.world_config import WorldConfig


class Simulator(torch.nn.Module):
    def __init__(
        self, robot="helhest", grid_res=0.1, max_coord=6.4, batch_size=1, device=None
    ):
        super().__init__()

        if robot == "helhest":
            robot_model = RobotModelConfig(
                mass=90.0,
                wheel_radii=torch.tensor([0.5, 0.5, 0.5]),
                wheel_centers=torch.tensor(
                    [[0.5, 0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, 0.0, 0.0]]
                ),
                wheel_inertias=torch.tensor([0.1, 0.1, 0.1]),
                inertia_tensor=torch.diag(torch.tensor([10.0, 10.0, 20.0])),
            )
        else:
            raise ValueError(f"Unsupported robot: {robot}")

        self.robot_model = robot_model

        engine_config = PhysicsEngineConfig(num_robots=batch_size)

        self.grid_res = grid_res
        self.max_coord = max_coord
        self.batch_size = batch_size
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.engine = DPhysicsEngine(engine_config, robot_model, device)
        self.engine.eval()

    def predict_trajectory(
        self,
        properties: dict,
        vel_ctrl: torch.Tensor,
        vel_ts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute a list of SO3 transformation matrices for the given wheel velocities.

        Parameters
        ----------
        properties : dict
            A dictionary containing terrain properties:
            - "geom": torch.Tensor of shape (B, H, W) representing heightmap.
            - "friction": torch.Tensor of shape (B, H, W) or scalar representing friction coefficients.
        vel_ctrl : torch.Tensor
            Control inputs for wheel velocities. Shape (Steps, B, 3) where 3 is the number of wheels.
        vel_ts : torch.Tensor
            Timestamps associated with the control inputs. Shape (Steps,).

        Returns:
        -------
        transformations : torch.Tensor
            A stack of SE(3) transformation matrices representing relative robot motion.
            Shape (SimSteps, B, 4, 4).
        timestamps : torch.Tensor
            The timestamps for each simulation step. Shape (SimSteps,).
        """

        world_config = WorldConfig(
            z_grid=properties["geom"],
            grid_res=self.grid_res,
            max_coord=self.max_coord,
            k_friction=properties["friction"],
        )
        state = PhysicsState.dummy(batch_size=self.batch_size, device=self.device)
        state.x, state.q = find_init_rob_position(
            world_config, self.robot_model, self.batch_size, self.device
        )

        start_time = vel_ts[0].item()
        stop_time = vel_ts[-1].item()
        ts = float(start_time)

        transformations = []
        timestamps = []
        prev_x = state.x.clone()
        prev_q = state.q.clone()

        curr_idx = 0

        while ts <= stop_time:
            while curr_idx + 1 < len(vel_ctrl) and ts >= vel_ts[curr_idx + 1]:
                curr_idx += 1

            # Linear Interpolation of Velocity
            if curr_idx + 1 < len(vel_ctrl):
                t0 = vel_ts[curr_idx]
                t1 = vel_ts[curr_idx + 1]
                v0 = vel_ctrl[curr_idx]
                v1 = vel_ctrl[curr_idx + 1]

                # Avoid division by zero
                dt = max(t1 - t0, 1e-6)
                ratio = (ts - t0) / dt

                curr_velocities = v0 + (v1 - v0) * ratio
                curr_acceleration = (v1 - v0) / dt
            else:
                # End of trajectory, hold last velocity, zero acceleration
                curr_velocities = vel_ctrl[curr_idx]
                curr_acceleration = torch.zeros_like(curr_velocities)

            # Update state wheel velocities to match interpolated input
            state.wheel_omega = curr_velocities

            # Forward with acceleration control
            dstate = self.engine.forward(state, curr_acceleration, world_config)
            state = self.engine.update_state(state, dstate)

            rel_trans = state.x - prev_x

            R_prev = quat_to_rotmat(prev_q)
            R_curr = quat_to_rotmat(state.q)
            R_rel = R_prev.transpose(-1, -2) @ R_curr

            T = (
                torch.eye(4, device=self.device)
                .unsqueeze(0)
                .repeat(self.batch_size, 1, 1)
            )
            T[:, :3, :3] = R_rel
            T[:, :3, 3] = rel_trans

            transformations.append(T)
            timestamps.append(ts)

            prev_x = state.x.clone()
            prev_q = state.q.clone()
            ts += self.engine.get_dt()

        return torch.stack(transformations), torch.tensor(timestamps)
