from typing import Optional
import torch

from simulator.engine import DPhysicsEngine
from simulator.engine_wp import WarpStaticEngine
from simulator.numerical import quat_to_rotmat
from simulator.utils import find_init_rob_position
from simulator.utils import PhysicsEngineConfig
from simulator.utils import PhysicsState
from simulator.utils import raise_to_clearance
from simulator.utils import RobotModelConfig
from simulator.utils import WorldConfig


class Simulator(torch.nn.Module):
    def __init__(
        self,
        robot="helhest",
        batch_size=1,
        device=None,
        T_horizon=5.0,
        # Matches simulator/engine.py. The old 0.01 default put omega*dt near the
        # semi-implicit-Euler stability edge at the stiffnesses actually used
        # (k=4e5, m=93.8 -> omega ~ 65 rad/s), and made warp/torch rollouts
        # incomparable. 2e-3 is the swept optimum — see PhysicsEngineConfig.dt.
        dt=2e-3,
        clip_grad_norm=None,
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

        self.batch_size = batch_size
        self.T_horizon = torch.tensor(T_horizon)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        MAX_STEPS = 100_000
        self.engine = WarpStaticEngine(
            robot_model, batch_size, MAX_STEPS, device, clip_grad_norm=clip_grad_norm
        )
        self.engine_config = PhysicsEngineConfig(num_robots=batch_size, dt=dt)
        # Torch engine instance kept purely for start-pose geometry (wheel points
        # and terrain sampling under no_grad). It never simulates — it exists so
        # that both wrappers seat the robot with the identical code path, which is
        # what makes tests/test_engine_parity.py a meaningful comparison.
        self._geom = DPhysicsEngine(self.engine_config, robot_model).to(device)
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

    @torch.no_grad()
    def fit_plane_pose(self, world_config: WorldConfig, batch_size: int):
        """Initial pose from the 3-point per-wheel plane fit. See models/simulator.py."""
        return find_init_rob_position(
            world_config, self.robot_model, batch_size, self.device
        )

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
            # Stiffness is not a predicted property; fall back to the fixed
            # WorldConfig constant when no map is supplied.
            k_stiffness=properties.get("stiffness", 400_000.0),
        )

        # 2. Setup Initial State
        actual_bs = vel_ctrl.shape[0]
        state = PhysicsState.dummy(batch_size=actual_bs, device=self.device)
        # Detached: see models/simulator.py's predict_trajectory for why this 3-point
        # (per-wheel) plane fit must not be differentiable.
        with torch.no_grad():
            state.x, state.q = self.fit_plane_pose(world_config, actual_bs)

        if initial_state is not None:
            # "x"/"q" override the terrain-fitted pose. The segmented trajectory
            # loss (utils/trainer.py) relies on this to seed each segment at the
            # GT pose closest to its own start timestamp; ignoring them here
            # would silently restart every segment at the fitted origin pose.
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

        # The plane fit only makes the wheel-triangle tangent; without this the
        # wheel circumference points start inside rough terrain and the contact
        # springs launch the robot at t=0. The torch wrapper has always done this
        # — the warp one used to skip it, so the two engines seated differently on
        # every non-flat window.
        state.x = self.raise_to_clearance(state.x, state.q, world_config)

        # 3. Interpolate Controls
        actual_bs = vel_ctrl.shape[0]
        ref_ts = vel_ts[0]  # (Steps,)

        # Permute vel_ctrl to (Steps, B, 3) for efficient indexing
        vel_ctrl_permuted = vel_ctrl.permute(1, 0, 2)

        sim_dt = self.engine_config.dt
        horizon = float(self.T_horizon if T_horizon is None else T_horizon)
        # round, not truncate: 1.0/1e-3 lands on 999.9999... in binary floating
        # point, which silently made warp rollouts one step shorter than torch's
        num_steps = int(round(horizon / sim_dt))

        # float64 is mandatory, not tidiness: these are absolute Unix timestamps
        # (~1.76e9), where float32 spacing is ~128 s. Building the query grid at the
        # default dtype collapsed all `num_steps` samples onto a single value tens
        # of seconds off, so every GT pose timestamp-matched to predicted index 0
        # and every GT-referenced metric silently measured nothing.
        query_times = torch.linspace(
            ref_ts[0],
            ref_ts[0] + horizon,
            num_steps,
            device=self.device,
            dtype=torch.float64,
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
