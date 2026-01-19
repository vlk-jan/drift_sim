from dataclasses import dataclass
from typing import Optional, Union

import torch
from tensordict import TensorClass

from simulator.numerical import (
    unit_quaternion,
    interpolate_grid,
    interpolate_surface_normals,
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
    max_coord: float
    k_stiffness: Union[float, torch.Tensor] = 40_000.0
    k_friction: Union[float, torch.Tensor] = 0.5

    def __post_init__(self):
        z_grid_grad = self._compute_heightmap_gradients(self.z_grid)  # (B, 2, D, D)
        ones = torch.ones_like(z_grid_grad[:, 0]).unsqueeze(1)
        self.normals = torch.cat((-z_grid_grad, ones), dim=1)
        self.normals = self.normals / torch.linalg.norm(
            self.normals, dim=1, keepdim=True
        )
        # if not isinstance(self.k_stiffness, torch.Tensor):
        #     self.k_stiffness = torch.full_like(self.z_grid, self.k_stiffness)
        if not isinstance(self.k_friction, torch.Tensor):
            self.k_friction = torch.full_like(self.z_grid, self.k_friction)

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
    def grid_size(self) -> int:
        """
        Returns the size of the grid.

        Returns:
            int: size of the grid.
        """
        return self.z_grid.shape[-1]

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
    dt: float = 1e-2
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
    on the terrain at (0,0,unknown height).

    Returns:
        x_init: (B, 3)  world position
        q_init: (B, 4)  unit quaternion [w, x, y, z]
    """

    # --- 1. Terrain height + normal at (0, 0) ---
    xy = torch.zeros((batch_size, 1, 2), device=device)  # shape (B, N=1, 2)
    z_ground = interpolate_grid(
        world_config.z_grid, xy, world_config.max_coord
    ).reshape(batch_size, 1)
    n_ground = interpolate_surface_normals(
        world_config.normals, xy, world_config.max_coord
    ).reshape(batch_size, 3)

    # Ensure the normal is normalized safely (larger eps for gradient stability)
    n_ground_norm = (n_ground.pow(2).sum(dim=1, keepdim=True) + 1e-6).sqrt()
    n_ground = n_ground / n_ground_norm

    # --- 2. Compute reference offset for wheels / CoM ---
    wheel_radii = torch.as_tensor(
        robot_model.wheel_radii, device=device, dtype=torch.float32
    )
    z_offset = wheel_radii.mean()
    z_init = z_ground + z_offset

    # --- 3. Compute orientation aligning robot up-axis with terrain normal ---
    z_body = torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(batch_size, 3)

    # Compute axis–angle rotation: rotate z_body → n_ground
    v = torch.linalg.cross(z_body, n_ground)

    # --- Rotation Calculation (Axis-Angle to Quaternion) ---
    c = (z_body * n_ground).sum(dim=1, keepdim=True)
    denom = (2 * (1 + c)).clamp_min(1e-6).sqrt()
    qxyz = v / denom
    qw = denom * 0.5

    # Stack to (B, 4): [w, x, y, z]
    q_nominal = torch.cat([qw, qxyz], dim=1)

    # Safe Normalize quaternion
    q_nom_norm = (q_nominal.pow(2).sum(dim=1, keepdim=True) + 1e-6).sqrt()
    q_nominal = q_nominal / q_nom_norm

    # --- Handle Singularities (Perfectly Upright or Upside Down) ---
    # Case A: Already aligned (c > 0) -> Identity Quaternion
    q_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(batch_size, 4)

    # Case B: Upside down (c < 0) -> 180 deg flip around X
    q_flip = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).expand(batch_size, 4)

    # If vectors are parallel, c is close to 1 or -1.
    # We use dot product for robust check since 's' is distorted by safe_norm.
    is_parallel = c.abs() > (1.0 - 1e-4)

    # Select correct quaternion based on alignment direction
    q_parallel = torch.where(c.expand(batch_size, 4) > 0, q_identity, q_flip)

    # Final selection
    q_final = torch.where(is_parallel.expand(batch_size, 4), q_parallel, q_nominal)

    # --- 4. Compose initial position ---
    x_init = torch.zeros((batch_size, 3), device=device)
    x_init[:, 2] = z_init.squeeze(-1)

    return x_init, q_final
