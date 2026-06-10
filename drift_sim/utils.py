from dataclasses import dataclass
from typing import Optional, Union, Tuple

import torch
from tensordict import TensorClass

from simulator.numerical import (
    unit_quaternion,
    interpolate_grid,
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
    max_coord (float): maximum coordinate of the grid.
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
    """

    num_robots: int = 1
    dt: float = 1e-3
    g: float = 9.81
    torque_limit: float = 5000.0
    damping_alpha: float = 1.0
    ang_damping: float = 0.1
    soft_contact_sigma: Optional[float] = None  # None or 0.01


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
    Find the initial position (x) and orientation (q) of the robot so that it rests
    on the terrain at (0,0) by checking the height of the 3 wheel contact points.

    Returns:
        x_init: (B, 3)  world position
        q_init: (B, 4)  unit quaternion [w, x, y, z]
    """

    # --- 1. Get wheel XY positions assuming robot is at origin with 0 yaw ---
    # shape: (3, 3) -> (B, 3, 2)
    grid_center = world_config.grid_center.to(device)
    wheel_centers = torch.as_tensor(
        robot_model.wheel_centers, device=device, dtype=torch.float32
    )
    xy_wheels = wheel_centers[:, :2].unsqueeze(0).expand(batch_size, -1, -1)

    # --- 2. Sample terrain height at the 3 wheel positions ---
    # z_wheels shape: (B, 3, 1)
    z_wheels = (
        interpolate_grid(
            world_config.z_grid, xy_wheels - grid_center, world_config.max_coord
        )
        .reshape(batch_size, -1)
        .unsqueeze(-1)
    )

    # Construct 3D points on the terrain at the wheels: shape (B, 3, 3)
    P = torch.cat([xy_wheels, z_wheels], dim=-1)

    # --- 3. Compute normal of the plane passing through the 3 contact points ---
    v1 = P[:, 1, :] - P[:, 0, :]  # Vector from wheel 0 to wheel 1
    v2 = P[:, 2, :] - P[:, 0, :]  # Vector from wheel 0 to wheel 2
    n_ground = torch.linalg.cross(v1, v2, dim=1)

    # Ensure the normal vector points upwards
    n_ground = torch.where(n_ground[:, 2:3] < 0, -n_ground, n_ground)
    n_ground_norm = (n_ground.pow(2).sum(dim=1, keepdim=True) + 1e-6).sqrt()
    n_ground = n_ground / n_ground_norm

    # --- 4. Compute reference offset for wheels / CoM ---
    wheel_radii = torch.as_tensor(
        robot_model.wheel_radii, device=device, dtype=torch.float32
    )
    R = wheel_radii.mean()

    # --- 5. Compute correct Z origin height ---
    # Place the body so no wheel penetrates the terrain and the highest-terrain
    # wheel just touches.  The old plane formula lifted the origin (at XY=0) to
    # distance R from the contact plane, which overshoots on sloped terrain because
    # the wheel centres are at (wx, wy), not at (0, 0).
    # z_wheels shape: (B, 3, 1) → terrain heights at each of the 3 wheel positions.
    z_init = z_wheels[:, :, 0].max(dim=1, keepdim=True).values + R

    # --- 6. Compute orientation aligning robot up-axis with terrain normal ---
    z_body = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(batch_size, 3)

    # Compute axis–angle rotation: rotate z_body → n_ground
    v = torch.linalg.cross(z_body, n_ground, dim=1)

    # --- Rotation Calculation (Axis-Angle to Quaternion) ---
    c = (z_body * n_ground).sum(dim=1, keepdim=True)
    denom = (2 * (1 + c)).clamp_min(1e-6).sqrt()
    qxyz = v / denom
    qw = denom * 0.5

    # Stack to (B, 4): [w, x, y, z]
    q_nominal = torch.cat([qw, qxyz], dim=1)
    q_nom_norm = (q_nominal.pow(2).sum(dim=1, keepdim=True) + 1e-6).sqrt()
    q_nominal = q_nominal / q_nom_norm

    # --- Handle Singularities (Perfectly Upright or Upside Down) ---
    q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(batch_size, 4)
    q_flip = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).expand(batch_size, 4)

    is_parallel = c.abs() > (1.0 - 1e-4)
    q_parallel = torch.where(c.expand(batch_size, 4) > 0, q_identity, q_flip)
    q_final = torch.where(is_parallel.expand(batch_size, 4), q_parallel, q_nominal)

    # --- 7. Compose initial position ---
    x_init = torch.zeros((batch_size, 3), device=device)
    x_init[:, 2] = z_init.squeeze(-1)

    return x_init, q_final


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
