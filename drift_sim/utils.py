from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from tensordict import TensorClass

from simulator.numerical import (
    interpolate_grid,
    quat_to_rotmat,
    unit_quaternion,
)


# ---------------------------------------------------------------------
# Configuration Classes
# ---------------------------------------------------------------------
class BaseConfig:
    """
    Base configuration class. This class is used to store the configuration of the simulation.
    """

    def to(self, device: Union[torch.device, str]):
        """
        Moves all tensors to the specified device.

        Args:
            device (torch.device or str): device to move the tensors to.
        """
        for attr, val in self.__dict__.items():
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))


@dataclass
class WorldConfig(BaseConfig):
    """
    World configuration. Contains the physical constants of the world, coordinates of the world frame etc.

    **Note**:
        1) The convention for grids is torch's "xy" indexing of the meshgrid. This means that the first
            dimension of the grid corresponds to the y-coordinate and the second dimension corresponds to the
            x-coordinate. The coordinate increases with increasing index, i.e. the Y down and X right. Generate
            them with torch.meshgrid(y, x, indexing="xy").
        2) origin [0,0] is at the center of the grid.

    z_grid (torch.Tensor): z-coordinates of the grid.
    normals (torch.Tensor): normals of the terrain. Shape (B, 3, grid_dim, grid_dim).
    grid_res (float): resolution of the grid in meters. Represents the metric distance between 2 centers of adjacent grid cells.
    max_coord (float): distance from the grid center to the outermost cell center
        (grid_sample align_corners=True convention). Every sampler normalises a
        metric query as ``u = (xy - grid_center) / max_coord`` and calls
        ``F.grid_sample(..., align_corners=True)``, where u = ±1 maps onto cell
        index 0 / n-1 -- the outermost cell *centers*, not the grid's outer
        edges. For an n-cell grid whose edges span ±E, that means
        ``max_coord = E - grid_res / 2``, half a cell inside the edge; passing E
        instead compresses all terrain queries by (n-1)/n. See
        ``utils/misc.py::world_cfg_from_config``, which derives it from the
        perception config. May be a scalar or a per-axis (x, y) pair.
    k_stiffness (float or torch.Tensor): stiffness of the terrain. Default is 40_000.
    k_friction (float or torch.Tensor): friction of the terrain. Default is 0.5.
    """

    z_grid: torch.Tensor
    grid_res: float
    max_coord: Union[float, Tuple[float, float]]
    side_range: Tuple[float, float] = (0.0, 0.0)
    fwd_range: Tuple[float, float] = (0.0, 0.0)
    k_stiffness: Union[float, torch.Tensor] = 400_000.0
    k_friction: Union[float, torch.Tensor] = 0.5

    def __post_init__(self):
        # Ensure all inputs are float32 to avoid dtype mismatches in torch operations
        self.z_grid = self.z_grid.to(torch.float32)

        # Convert max_coord into a 2D tensor for grid_sample broadcasting
        if isinstance(self.max_coord, (list, tuple)):
            self.max_coord = torch.tensor(
                self.max_coord, dtype=torch.float32, device=self.z_grid.device
            )
        elif isinstance(self.max_coord, (float, int)):
            self.max_coord = torch.tensor(
                [self.max_coord, self.max_coord],
                dtype=torch.float32,
                device=self.z_grid.device,
            )
        else:
            self.max_coord = self.max_coord.to(
                dtype=torch.float32, device=self.z_grid.device
            )

        z_grid_grad = self._compute_heightmap_gradients(self.z_grid)  # (B, 2, D, D)
        ones = torch.ones_like(z_grid_grad[:, 0]).unsqueeze(1)
        self.normals = torch.cat((-z_grid_grad, ones), dim=1)
        self.normals = self.normals / torch.linalg.norm(
            self.normals, dim=1, keepdim=True
        )

        if self.side_range == (0.0, 0.0):
            self.side_range = (-self.max_coord[0], self.max_coord[0])
        if self.fwd_range == (0.0, 0.0):
            self.fwd_range = (-self.max_coord[1], self.max_coord[1])

        if not isinstance(self.k_friction, torch.Tensor):
            self.k_friction = torch.full_like(
                self.z_grid, self.k_friction, dtype=torch.float32
            )
        else:
            self.k_friction = self.k_friction.to(torch.float32)

        if not isinstance(self.k_stiffness, torch.Tensor):
            self.k_stiffness = torch.full_like(
                self.z_grid, self.k_stiffness, dtype=torch.float32
            )
        else:
            self.k_stiffness = self.k_stiffness.to(torch.float32)

        # Precompute a single 6-channel grid
        # Channel layout: [z, nx, ny, nz, friction, stiffness]
        self._sample_grid = torch.cat(
            [
                self.z_grid.unsqueeze(1),  # (B, 1, H, W)
                self.normals,  # (B, 3, H, W)
                self.k_friction.unsqueeze(1),  # (B, 1, H, W)
                self.k_stiffness.unsqueeze(1),  # (B, 1, H, W)
            ],
            dim=1,
        )  # (B, 6, H, W)

    def _compute_heightmap_gradients(self, z_grid: torch.Tensor) -> torch.Tensor:
        """
        Computes the gradients of a heightmap along the x and y axes using torch.gradient.

        **Note**: The input grid is assumed to use torch's "xy" indexing convention, i.e., the first dimension is
                   the y-axis and the second dimension is the x-axis.

        Parameters:
        - z_grid: Heightmap tensor of shape (B, H, W).
        - grid_res: Resolution of the grid in meters.

        """
        return torch.stack(
            torch.gradient(z_grid, spacing=self.grid_res, dim=(2, 1), edge_order=2),
            dim=1,
        )

    @property
    def grid_center(self) -> torch.Tensor:
        """
        Returns the center of the grid based on the fwd and side bounds.

        Return:
            Tuple[float, float]: center of the grid
        """
        center_y = (self.fwd_range[0] + self.fwd_range[1]) / 2.0
        center_x = (self.side_range[0] + self.side_range[1]) / 2.0
        return torch.tensor(
            (center_x, center_y), dtype=torch.float32, device=self.z_grid.device
        )

    def ij_to_xyz(
        self, ij: Union[torch.IntTensor, torch.LongTensor]
    ) -> torch.FloatTensor:
        """
        ij is assumed to have the shape (N, B, 2) where N is the number of points and B is the batch size.
        """
        N, B = ij.shape[:-1]

        # Analytical calculation for x and y
        # ij[..., 1] is x-index, ij[..., 0] is y-index (due to "xy" indexing convention in torch)
        center = (self.grid_size - 1) / 2.0
        x = (ij[..., 1] - center) * self.grid_res
        y = (ij[..., 0] - center) * self.grid_res

        batch_idx = torch.arange(B, device=ij.device).expand(N, B)
        z = self.z_grid[batch_idx, ij[..., 0], ij[..., 1]]

        return torch.stack((x, y, z), dim=-1)


@dataclass
class RobotModelConfig(BaseConfig):
    mass: float = 0
    cog: torch.Tensor = torch.tensor([0, 0, 0])
    wheel_radii: torch.Tensor = torch.tensor([1, 1, 1])
    wheel_centers: torch.Tensor = torch.tensor(((1, -1, 0), (1, 1, 0), (-1, 0, 0)))
    wheel_inertias: torch.Tensor = torch.tensor(())
    inertia_tensor: torch.Tensor = torch.eye(3)


@dataclass
class PhysicsEngineConfig(BaseConfig):
    """
    Physics Engine configuration. Contains the physical constants of the simulation, such as gravity, time step, etc.

    Attributes:
        dt (float): time step for numerical integration. Default is 0.01. (100 Hz)
        gravity (float): acceleration due to gravity. Default is 9.81 m/s^2.
        torque_limit (float): torque limit that can be generated on CoG. The physics engine clips it to this value. Default is 200.0 Nm.
        damping_alpha (float): damping coefficient modifier, should be between larger than 1.0. Applies damping alpha times higher
        than critical damping.
        ang_damping (float): angular damping coefficient. Default is 0.1.
        soft_contact_sigma (float): soft contact sigma. Default is 0.5. This is the width of the soft contact force function.
        slip_ref_velocity (float): slip speed (m/s) at which the tangential force
            reaches ~76% of its Coulomb limit, via tanh(|v_t| / slip_ref).
            Default 1.0, which is the historical behaviour (a bare tanh(|v_t|)
            with |v_t| in m/s). NOTE that a wheel near rolling slips at ~0.01-0.3
            m/s, where tanh gives 0.01-0.29 -- so at 1.0 the contact is closer to
            VISCOUS than Coulomb and the effective traction is a small fraction of
            k_friction. That is the leading explanation for why the oracle sweep
            in probe_traction_degeneracy.py wants k_friction >= 3 (still falling
            at 14) when a physical coefficient belongs in (0, 1]. Lower this to
            saturate friction at realistic slip speeds. Changing it changes the
            dynamics, so leave it at 1.0 unless you mean to.
    """

    num_robots: int = 1
    # 2e-3 chosen by scripts/analysis/sweep_dt.py on real helhest terrain (see the
    # dt section in CLAUDE.md): it halves the step count versus the old 1e-3 while
    # GT trajectory error, final error and gradient norm are all indistinguishable.
    # 3e-3 is where gradient fidelity starts to degrade on both engines.
    dt: float = 2e-3
    g: float = 9.81
    torque_limit: float = 5000.0
    damping_alpha: float = 1.0
    ang_damping: float = 0.1
    soft_contact_sigma: Optional[float] = None  # None or 0.01
    slip_ref_velocity: float = 1.0


# ---------------------------------------------------------------------
# Physics Classes
# ---------------------------------------------------------------------
class PhysicsState(TensorClass):
    """Physics State

    Attributes:
        x (torch.Tensor): Position of the robot in the world frame. Shape (B, 3).
        xd (torch.Tensor): Velocity of the robot in the world frame. Shape (B, 3).
        q (torch.Tensor): Orientation quaternion of the robot in the world frame. Shape (B, 4).
        omega (torch.Tensor): Angular velocity of the robot in the world frame. Shape (B, 3).
        wheel_omega (torch.Tensor): Angular velocities of individual wheels. Shape (B, 3).
    """

    x: torch.Tensor
    xd: torch.Tensor
    q: torch.Tensor
    omega: torch.Tensor
    wheel_omega: torch.Tensor

    @staticmethod
    def dummy(**kwargs) -> "PhysicsState":
        """Create an empty dummy PhysicsState object with zero tensors.
        Some fields can be overridden by passing them as keyword arguments.
        """
        batch_size = kwargs.pop("batch_size", None)
        device = kwargs.pop("device", torch.get_default_device())
        if kwargs:
            t = next(iter(kwargs.values()))
            if batch_size is not None and batch_size != t.shape[0]:
                raise ValueError(
                    "Specified batch size does not match the shape of the tensors."
                )
            batch_size = t.shape[0]
        elif batch_size is None:
            batch_size = 0
        base = {
            "x": torch.zeros(batch_size, 3),
            "xd": torch.zeros(batch_size, 3),
            "q": unit_quaternion(batch_size, device=device),
            "omega": torch.zeros(batch_size, 3),
            "wheel_omega": torch.zeros(batch_size, 3, device=device),
            "batch_size": batch_size,
        }
        return PhysicsState(**{**base, **kwargs}, device=device)


class PhysicsStateDer(TensorClass):
    """Physics State Derivative

    Attributes:
        xdd (torch.tensor): Acceleration of the robot in the world frame. Shape (B, 3).
        omega_d (torch.tensor): Derivative of the angular velocity of the robot in the world frame. Shape (B, 3).
        wheel_omega_d (torch.Tensor): Angular acceleration of the 3 wheels. Shape (B, 3).
    """

    xdd: torch.Tensor
    omega_d: torch.Tensor
    wheel_omega_d: torch.Tensor


# ---------------------------------------------------------------------
# Other functions
# ---------------------------------------------------------------------
def find_init_rob_position(
    world_config: WorldConfig,
    robot_model: RobotModelConfig,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find the initial pose so that all 3 wheels rest on the terrain simultaneously.

    The 3 wheel contact points uniquely define a plane.  We iteratively:
      1. Compute world-frame wheel XY positions given current orientation.
      2. Sample terrain height at each wheel.
      3. Fit a plane through the 3 desired wheel-centre positions (terrain_z + R).
      4. Derive the body orientation that aligns the robot up-axis with that normal.
      5. Derive the body height so every wheel-centre z equals terrain_z + R.
      6. Repeat until convergence (typically 3-5 iterations).

    Returns:
        x_init: (B, 3)  world position of the robot CoM
        q_init: (B, 4)  unit quaternion [w, x, y, z]
    """
    grid_center = world_config.grid_center.to(device)
    wheel_centers = torch.as_tensor(
        robot_model.wheel_centers, device=device, dtype=torch.float32
    )  # (3, 3)
    wheel_radii = torch.as_tensor(
        robot_model.wheel_radii, device=device, dtype=torch.float32
    )
    R_wheel = wheel_radii.mean()

    q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(batch_size, 4)
    q_flip = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).expand(batch_size, 4)

    # Start upright at origin
    q = q_identity.clone()
    x = torch.zeros(batch_size, 3, device=device)

    for _ in range(10):
        # 1. World-frame wheel centre positions (B, 3, 3)
        R_mat = quat_to_rotmat(q)  # (B, 3, 3)
        wc_world = torch.einsum("bij,kj->bki", R_mat, wheel_centers) + x.unsqueeze(
            1
        )  # (B, 3, 3)
        wc_xy = wc_world[:, :, :2]  # (B, 3, 2)

        # 2. Terrain height under each wheel centre
        z_terrain = interpolate_grid(
            world_config.z_grid,
            wc_xy - grid_center,
            world_config.max_coord,
        ).reshape(batch_size, 3)  # (B, 3)

        # 3. Desired wheel-centre heights and plane through them
        wc_z_target = z_terrain + R_wheel  # (B, 3)
        P = torch.cat([wc_xy, wc_z_target.unsqueeze(-1)], dim=-1)  # (B, 3, 3)
        v1 = P[:, 1, :] - P[:, 0, :]
        v2 = P[:, 2, :] - P[:, 0, :]
        n = torch.linalg.cross(v1, v2, dim=1)  # (B, 3)
        n = torch.where(n[:, 2:3] < 0, -n, n)
        n = n / (n.norm(dim=1, keepdim=True) + 1e-6)

        # 4. Orientation: rotate body up-axis [0,0,1] to align with plane normal
        z_up = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(batch_size, 3)
        v_cross = torch.linalg.cross(z_up, n, dim=1)
        c = (z_up * n).sum(dim=1, keepdim=True)
        denom = (2 * (1 + c)).clamp_min(1e-6).sqrt()
        q_new = torch.cat([denom * 0.5, v_cross / denom], dim=1)
        q_new = q_new / (q_new.norm(dim=1, keepdim=True) + 1e-6)
        is_par = c.abs() > (1.0 - 1e-4)
        q_par = torch.where(c.expand_as(q_new) > 0, q_identity, q_flip)
        q_new = torch.where(is_par.expand_as(q_new), q_par, q_new)

        # 5. Body height: for each wheel, z_body = target_z - (R_new @ wc_body)_z
        #    Take the max so every wheel is at or above its terrain contact height.
        R_new = quat_to_rotmat(q_new)  # (B, 3, 3)
        wc_rot_z = torch.einsum("bi,ki->bk", R_new[:, 2, :], wheel_centers)  # (B, 3)
        z_new = (wc_z_target - wc_rot_z).max(dim=1).values  # (B,)

        x_new = torch.zeros(batch_size, 3, device=device)
        x_new[:, 2] = z_new

        # The iteration is idempotent once converged (a converged (q, x) maps to
        # itself), so the fixed 10 passes give the same answer as an early exit
        # without the two device->host syncs per iteration that testing for it
        # would cost.
        q, x = q_new, x_new

    return x, q


@torch.no_grad()
def raise_to_clearance(engine, x, q, world_config):
    """Raise the robot until all wheel circumference sample points clear the terrain.

    find_init_rob_position only makes the wheel-triangle plane tangent; on
    sloped/curved terrain the 16 circumference sample points per wheel can still
    penetrate, producing enormous initial spring forces that launch the robot
    upward at t=0. Returns the raised position (input `x` is not mutated — it may
    be a view into GT batch data).

    Each iteration applies -dh (vertical penetration depth), which removes fraction
    nz of the true geometric penetration per step, so convergence is geometric. The
    direct -dh/nz formula would be exact in one step but amplifies tiny dh at
    near-vertical faces (nz~0) into a huge over-lift. 5 iterations give <3% residual
    even on 60 degree slopes while never over-shooting.

    Free function rather than a method so both simulator wrappers seat the robot
    identically — the warp wrapper used to skip this step, which left it starting
    inside the terrain on any non-flat window. `engine` only supplies geometry
    (_wheel_centers, _wheel_pos_rel, _sample_terrain), so either engine's torch
    DPhysicsEngine works.
    """
    x = x.clone()
    batch_size = x.shape[0]
    R_body = quat_to_rotmat(q)  # (B, 3, 3)
    wp_rel = torch.einsum("bij,wpj->bwpi", R_body, engine._wheel_pos_rel)  # (B,W,P,3)

    # Fixed iteration count, no early exit: the `.all()` convergence test cost a
    # device->host sync per iteration, and max_pen is clamped at 0, so the extra
    # iterations add exactly nothing once converged.
    for _ in range(5):
        wc_world = (R_body @ engine._wheel_centers.T).transpose(-1, -2) + x.unsqueeze(1)
        wp_world = wp_rel + wc_world.unsqueeze(2)  # (B, W, P, 3)
        dh, _, _, _ = engine._sample_terrain(wp_world, world_config)  # (B, W, P, 1)
        max_pen = (-dh.reshape(batch_size, -1)).clamp(min=0).max(dim=1).values  # (B,)
        x[:, 2] += max_pen
    return x


@torch.no_grad()
def compute_wheel_contacts(engine, traj_se3, world_config, chunk_size=64):
    """Per-step in-contact mask for a single rollout's wheel-circumference points.

    `traj_se3` is a (T, 4, 4) world-frame SE3 pose stack (one batch element, e.g.
    one row of a `predict_trajectory` output). Returns a (T, W*P) bool array:
    which of `engine`'s fixed wheel-circumference points (`_wheel_centers`,
    `_wheel_pos_rel`, the same geometry used by `raise_to_clearance` above) are
    penetrating the terrain (`dh < 0`, see `engine.forward`'s docstring) at each
    step -- a debugging signal for spring-contact pathologies (e.g. the
    seating/launch bug this module already guards against), not something the
    engine keeps around after computing the resulting force.

    Chunked over T (rather than one `(T, H, W)` broadcast of `world_config`)
    since this only runs post-hoc over a handful of selected SimView-export
    trajectories, not inside the training loop.
    """
    T = traj_se3.shape[0]
    R = traj_se3[:, :3, :3]
    x = traj_se3[:, :3, 3]
    out = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        n = end - start
        Rc, xc = R[start:end], x[start:end]
        wp_rel = torch.einsum("tij,wpj->twpi", Rc, engine._wheel_pos_rel)
        wc_world = torch.einsum(
            "tij,wj->twi", Rc, engine._wheel_centers
        ) + xc.unsqueeze(1)
        wp_world = wp_rel + wc_world.unsqueeze(2)  # (n, W, P, 3)
        wc = WorldConfig(
            z_grid=world_config.z_grid.expand(n, -1, -1),
            grid_res=world_config.grid_res,
            max_coord=world_config.max_coord,
            side_range=world_config.side_range,
            fwd_range=world_config.fwd_range,
            k_friction=world_config.k_friction.expand(n, -1, -1),
            k_stiffness=world_config.k_stiffness.expand(n, -1, -1),
        )
        dh, _, _, _ = engine._sample_terrain(wp_world, wc)  # (n, W, P, 1)
        out.append((dh.squeeze(-1) < 0).reshape(n, -1))
    return torch.cat(out, dim=0)  # (T, W*P) bool


def generate_controls(
    duration,
    dt,
    num_robots,
    device,
    seed=42,
    wheel_radius=0.3685,
    current_vel_linear=1.0,
    clamp=None,
):
    """
    Generates random-walk angular wheel velocities.
    """
    torch.manual_seed(seed)
    steps = int(duration / dt)
    controls = []
    timestamps = []

    # Initial velocity (driving forward)
    current_omega = current_vel_linear / wheel_radius

    current_vel = torch.tensor(
        [current_omega, current_omega, current_omega], device=device
    )
    current_vel = current_vel.unsqueeze(0).expand(num_robots, -1).clone()

    for i in range(steps):
        ts = i * dt
        # Add small random noise to velocity
        noise = (
            torch.randn(num_robots, 3, device=device) * 0.05 * (1.0 / wheel_radius)
        )  # Scale noise to angular

        # Slowly vary the velocity (random walk)
        current_vel += noise * dt

        if clamp is not None:
            current_vel = torch.clamp(current_vel, -clamp, clamp)

        controls.append(current_vel.clone())
        timestamps.append(ts)

    return torch.stack(controls).permute(1, 0, 2), torch.tensor(
        timestamps, device=device
    ).expand(num_robots, -1)


def generate_dynamic_maneuvers(duration, dt, num_robots, device, seed=42):
    """
    Generates a sequence of dynamic maneuvers (Extended).
    """
    import numpy as np

    torch.manual_seed(seed)
    steps = int(duration / dt)
    controls = []
    timestamps = []

    # Parameters for Helhest
    wheel_radius = 0.3685
    max_speed_lin = 5.0  # m/s
    max_omega = max_speed_lin / wheel_radius

    for i in range(steps):
        ts = i * dt
        progress = ts / duration

        target_vel = torch.zeros(3, device=device, dtype=torch.float32)

        if progress < 0.05:
            # 0% - 5%: Warmup (Linear Accel)
            frac = progress / 0.05
            w = max_omega * frac
            target_vel = torch.tensor([w, w, w], device=device, dtype=torch.float32)

        elif progress < 0.15:
            # 5% - 15%: Cruise (Fast Forward)
            w = max_omega
            target_vel = torch.tensor([w, w, w], device=device, dtype=torch.float32)

        elif progress < 0.30:
            # 15% - 30%: Slalom (S-Turns)
            local_p = (progress - 0.15) / 0.15
            w_base = max_omega * 0.7
            w_diff = (max_omega * 0.6) * np.sin(local_p * 3 * 2 * np.pi)
            target_vel = torch.tensor(
                [w_base - w_diff, w_base + w_diff, w_base],
                device=device,
                dtype=torch.float32,
            )

        elif progress < 0.50:
            # 30% - 50%: Square Path (Sharp Turns)
            local_p = (progress - 0.30) / 0.20
            segment = (local_p * 4) % 1.0  # 0.0 to 1.0 within a segment

            if segment < 0.7:
                # Drive Straight
                w = max_omega * 0.8
                target_vel = torch.tensor([w, w, w], device=device, dtype=torch.float32)
            else:
                # Hard Left Turn
                w = max_omega
                target_vel = torch.tensor(
                    [w * 0.0, w, w * 0.5], device=device, dtype=torch.float32
                )

        elif progress < 0.60:
            # 50% - 60%: Stop & Go
            local_p = (progress - 0.50) / 0.10
            if local_p < 0.4:
                # Brake/Stop
                target_vel = torch.tensor(
                    [0.0, 0.0, 0.0], device=device, dtype=torch.float32
                )
            else:
                # Hard Accel
                w = max_omega
                target_vel = torch.tensor([w, w, w], device=device, dtype=torch.float32)

        elif progress < 0.75:
            # 60% - 75%: Reverse Maneuvering
            local_p = (progress - 0.60) / 0.15
            w_base = -max_omega * 0.6
            w_diff = (max_omega * 0.3) * np.sin(local_p * 2 * 2 * np.pi)
            target_vel = torch.tensor(
                [w_base + w_diff, w_base - w_diff, w_base],
                device=device,
                dtype=torch.float32,
            )

        elif progress < 0.85:
            # 75% - 85%: Spins
            local_p = (progress - 0.75) / 0.10
            w_spin = max_omega * 0.8
            if local_p < 0.5:
                # Clockwise
                target_vel = torch.tensor(
                    [-w_spin, w_spin, 0.0], device=device, dtype=torch.float32
                )
            else:
                # Counter-Clockwise
                target_vel = torch.tensor(
                    [w_spin, -w_spin, 0.0], device=device, dtype=torch.float32
                )

        elif progress < 0.95:
            # 85% - 95%: Agility (Rapid Random Changes)
            t_val = ts * 10.0
            n1 = np.sin(t_val)
            n2 = np.cos(t_val * 1.3)
            w_base = max_omega * 0.8
            w_l = w_base * (0.5 + 0.5 * n1)
            w_r = w_base * (0.5 + 0.5 * n2)
            w_rear = (w_l + w_r) / 2.0
            target_vel = torch.tensor(
                [w_l, w_r, w_rear], device=device, dtype=torch.float32
            )

        else:
            # 95% - 100%: Cooldown / Stop
            target_vel = torch.tensor(
                [0.0, 0.0, 0.0], device=device, dtype=torch.float32
            )

        # Expand to batch
        batch_vel = target_vel.unsqueeze(0).expand(num_robots, -1)
        controls.append(batch_vel)
        timestamps.append(ts)

    return torch.stack(controls).permute(1, 0, 2), torch.tensor(
        timestamps, device=device
    ).expand(num_robots, -1)


def add_helhest_visual_wheels(scene, parent: str = "box") -> None:
    """
    Adds rigidly attached visual wheels for the Helhest robot to the SimView scene.
    """
    from simview import BodyShapeType, SimViewBody

    wheels_config = {
        "left_wheel": {
            "pos": [0.4, 0.52, 0.0],
            "rot": [0.7071068, -0.7071068, 0.0, 0.0],
        },
        "right_wheel": {
            "pos": [0.4, -0.52, 0.0],
            "rot": [0.7071068, 0.7071068, 0.0, 0.0],
        },
        "rear_wheel": {
            "pos": [-0.5, 0.0, 0.0],
            "rot": [0.7071068, -0.7071068, 0.0, 0.0],
        },
    }
    for name, config in wheels_config.items():
        scene.add_body_object(
            SimViewBody.create(
                name=name,
                body_type=BodyShapeType.CYLINDER,
                radius=0.3685,
                height=0.142,
                parent=parent,
                local_transform=config["pos"] + config["rot"],
            )
        )
