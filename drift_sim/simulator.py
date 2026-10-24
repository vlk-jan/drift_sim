from typing import Optional

import torch

from simulator.engine import DPhysicsEngine
from simulator.utils import (
    PhysicsEngineConfig,
    PhysicsState,
    WorldConfig,
    find_init_rob_position,
    num_sim_steps,
    raise_to_clearance,
    robot_model_config,
)
from utils.geometry import quat_to_rotmat


def _update_world_config(
    sim, res=None, max_coord=None, side_range=None, fwd_range=None
):
    """Update a Simulator wrapper's world-configuration parameters in place."""
    if res is not None:
        sim.res = res
    if max_coord is not None:
        sim.max_coord = max_coord
    if side_range is not None:
        sim.side_range = side_range
    if fwd_range is not None:
        sim.fwd_range = fwd_range


def _build_world_config(sim, properties: dict) -> WorldConfig:
    """Build a WorldConfig from the sim's grid geometry and predicted properties."""
    return WorldConfig(
        z_grid=properties["support"],
        grid_res=sim.res,
        max_coord=sim.max_coord,
        side_range=sim.side_range,
        fwd_range=sim.fwd_range,
        k_friction=properties["friction"],
        # Stiffness is not predicted: it is the robot's tyre constant.
        k_stiffness=properties.get("stiffness", sim.robot_model.contact_stiffness),
    )


def _apply_initial_state_override(state, initial_state, device):
    """Override the terrain-fitted initial pose/velocities from `initial_state`.

    The segmented trajectory loss (utils/trainer.py) seeds each segment at its
    GT start pose through this; without it every segment restarts at the origin.
    Unknown keys raise: wheel speeds are the controls, so an initial
    "wheel_omega" would be silently ignored.
    """
    if initial_state is not None:
        unused = set(initial_state) - {"x", "q", "xd", "omega"}
        if unused:
            raise ValueError(f"initial_state keys not used by the simulator: {unused}")
        if "x" in initial_state:
            state.x = initial_state["x"].to(device, torch.float32)
        if "q" in initial_state:
            state.q = initial_state["q"].to(device, torch.float32)
        if "xd" in initial_state:
            state.xd = initial_state["xd"].to(device, torch.float32)
        if "omega" in initial_state:
            state.omega = initial_state["omega"].to(device, torch.float32)
    return state


class Simulator(torch.nn.Module):
    def __init__(
        self,
        robot="helhest",
        device=None,
        T_horizon=1.0,
        dt=2e-3,  # see PhysicsEngineConfig.dt / scripts/analysis/sweep_dt.py
        com_transfer=True,
    ):
        """Differentiable torch simulator wrapper.

        T_horizon is the rollout length when predict_trajectory is given none (the
        eval paths build the wrapper with --t_horizon; the trainer always passes
        its segment horizon). The 1.0 s default matches the warp wrapper's.
        """
        super().__init__()

        robot_model = robot_model_config(robot)
        self.robot_model = robot_model

        engine_config = PhysicsEngineConfig(dt=dt, com_transfer=com_transfer)
        self.res = None
        self.max_coord = None
        self.side_range = None
        self.fwd_range = None

        self.T_horizon = T_horizon
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
        _update_world_config(
            self,
            res=res,
            max_coord=max_coord,
            side_range=side_range,
            fwd_range=fwd_range,
        )

    def _prepare_inputs(self, vel_ctrl, vel_ts, T_horizon=None):
        """Resample control inputs onto the physics engine's time step.

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
        dt = self.engine.config.dt
        device = vel_ctrl.device
        B, S = vel_ts.shape

        if T_horizon is None:
            T_horizon = self.T_horizon
        # Target timeline [t_start, t_start + dt, ...]. vel_ts must be float64:
        # absolute Unix stamps (~1.76e9 s) have ~128 s float32 spacing.
        vel_ts = vel_ts.to(torch.float64)
        t_start = vel_ts[:, 0].unsqueeze(1)
        num_steps = num_sim_steps(T_horizon, dt) + 1
        target_ts = t_start + (torch.arange(num_steps, device=device) * dt).unsqueeze(
            0
        ).expand(B, -1)

        target_ts_batch = target_ts.contiguous()  # (B, NumTargets)
        vel_ts = vel_ts.contiguous()

        idx_right = torch.searchsorted(vel_ts, target_ts_batch, right=True)
        idx_left = (idx_right - 1).clamp(0, S - 2)
        idx_right = idx_left + 1

        batch_indices = (
            torch.arange(B, device=device).unsqueeze(1).expand(-1, num_steps)
        )

        # (B, NumTargets, 3)
        v_left = vel_ctrl[batch_indices, idx_left]
        v_right = vel_ctrl[batch_indices, idx_right]

        t_left = vel_ts[batch_indices, idx_left]
        t_right = vel_ts[batch_indices, idx_right]

        denom = (t_right - t_left).clamp(min=1e-6)
        alpha = (target_ts_batch - t_left) / denom
        # Clamped: hold the end values instead of extrapolating.
        alpha = alpha.clamp(0.0, 1.0).unsqueeze(-1)

        new_ctrl = v_left + (v_right - v_left) * alpha

        return new_ctrl, target_ts_batch

    @torch.no_grad()
    def fit_plane_pose(
        self, world_config: WorldConfig, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initial pose from the 3-point (per-wheel) plane fit through the height field.

        Must be detached (hence @torch.no_grad()): one bad cell under a wheel
        swings the fit, and a differentiable fit lets training fix rotation error by
        saturating height_residual under a wheel instead of through contact dynamics.
        """
        return find_init_rob_position(
            world_config, self.robot_model, batch_size, self.device
        )

    def raise_to_clearance(
        self, x: torch.Tensor, q: torch.Tensor, world_config: WorldConfig
    ) -> torch.Tensor:
        """Raise the robot until all wheel circumference points clear the terrain.

        The plane fit only makes the wheel triangle tangent; on curved terrain the
        circumference points penetrate and the contact springs launch the robot at
        t=0. Returns a new position (`x` may be a view into GT data). Shared with
        the warp wrapper via simulator/utils.py.
        """
        return raise_to_clearance(self.engine, x, q, world_config)

    def predict_trajectory(
        self,
        properties: dict,
        vel_ctrl: torch.Tensor,
        vel_ts: torch.Tensor,
        initial_state: Optional[dict] = None,
        T_horizon: Optional[float] = None,
        include_start: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Roll out SE(3) poses for the given wheel velocities.

        Parameters
        ----------
        properties : dict
            - "support": (B, H, W) support surface (height_map - height_residual).
            - "friction": (B, H, W) or scalar.
            - "stiffness": optional; defaults to the fixed WorldConfig constant.
        vel_ctrl : torch.Tensor
            Wheel velocity controls, (B, Steps, 3).
        vel_ts : torch.Tensor
            Control timestamps, (B, Steps).
        initial_state : dict, optional
            May contain only "x" (B, 3), "q" (B, 4, wxyz), "xd", "omega";
            "x"/"q" override the terrain-fitted pose (e.g. a GT segment start).
        T_horizon : float, optional
            Simulation horizon; defaults to self.T_horizon.
        include_start : bool, optional
            Also return the start state (raised to clearance) first, stamped
            vel_ts[b, 0], so a loss can anchor on the state the rollout starts
            from. Outputs then have SimSteps + 1 entries.

        Returns:
        -------
        transformations : torch.Tensor
            Robot poses, (B, K, 4, 4), K = SimSteps + int(include_start).
        timestamps : torch.Tensor
            float64, (B, K). Pose i is the post-step state stamped at the
            step's end, so the first stamp is vel_ts[b, 0] + dt (same as the warp
            wrapper) unless include_start.
        xd_out : torch.Tensor
            Linear velocity, (B, K, 3).
        omega_out : torch.Tensor
            Angular velocity, (B, K, 3).
        """
        actual_bs = vel_ctrl.shape[0]

        world_config = _build_world_config(self, properties)
        state = PhysicsState.dummy(batch_size=actual_bs, device=self.device)
        # Skipped when the caller seeds both x and q (the segmented loss does).
        if not {"x", "q"} <= set(initial_state or ()):
            state.x, state.q = self.fit_plane_pose(world_config, actual_bs)

        state = _apply_initial_state_override(state, initial_state, self.device)

        state.x = self.raise_to_clearance(state.x, state.q, world_config)

        vel_ctrl, vel_ts = self._prepare_inputs(vel_ctrl, vel_ts, T_horizon=T_horizon)

        dt = self.engine.config.dt
        num_steps = vel_ts.shape[-1] - 1

        # Lists + stack keep the autograd graph (no in-place writes).
        R_list = []
        t_list = []
        xd_out_list = []
        omega_out_list = []
        timestamps_list = []

        ts = vel_ts[:, 0].clone()
        if include_start:
            R_list.append(quat_to_rotmat(state.q))
            t_list.append(state.x)
            xd_out_list.append(state.xd)
            omega_out_list.append(state.omega)
            timestamps_list.append(ts)

        for idx in range(num_steps):
            state.wheel_omega = vel_ctrl[:, idx]
            dstate = self.engine.forward(state, world_config)
            state = self.engine.update_state(state, dstate)

            # Poses are post-step states, stamped at the step's end.
            ts = ts + dt

            R_list.append(quat_to_rotmat(state.q))  # (B, 3, 3)
            t_list.append(state.x)  # (B, 3)
            xd_out_list.append(state.xd)
            omega_out_list.append(state.omega)
            timestamps_list.append(ts)

        R_all = torch.stack(R_list, dim=1)  # (B, SimSteps, 3, 3)
        t_all = torch.stack(t_list, dim=1)  # (B, SimSteps, 3)
        top_part = torch.cat(
            [R_all, t_all.unsqueeze(-1)], dim=-1
        )  # (B, SimSteps, 3, 4)
        bottom = torch.zeros(actual_bs, R_all.shape[1], 1, 4, device=self.device)
        bottom[:, :, 0, 3] = 1.0
        poses = torch.cat([top_part, bottom], dim=2)  # (B, SimSteps, 4, 4)

        xd_out = torch.stack(xd_out_list, dim=1)  # (B, SimSteps, 3)
        omega_out = torch.stack(omega_out_list, dim=1)  # (B, SimSteps, 3)
        timestamps = torch.stack(timestamps_list, dim=1)  # (B, SimSteps)

        return poses, timestamps, xd_out, omega_out
