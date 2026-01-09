from dataclasses import dataclass
from typing import Union

import torch


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
    suitable_mask (torch.Tensor | None): mask of suitable terrain. Shape (grid_dim, grid_dim). 1 if suitable, 0 if not. Default is None.
    """

    z_grid: torch.Tensor
    grid_res: float
    max_coord: float
    k_stiffness: Union[float, torch.Tensor] = 40_000.0
    k_friction: Union[float, torch.Tensor] = 0.5
    suitable_mask: Union[torch.BoolTensor, None] = None

    def __post_init__(self):
        z_grid_grad = self._compute_heightmap_gradients(self.z_grid)  # (B, 2, D, D)
        ones = torch.ones_like(z_grid_grad[:, 0]).unsqueeze(1)
        self.normals = torch.cat((-z_grid_grad, ones), dim=1)
        self.normals /= torch.linalg.norm(self.normals, dim=1, keepdim=True)

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

    def ij_to_suited_mask(
        self, ij: Union[torch.IntTensor, torch.LongTensor]
    ) -> torch.BoolTensor:
        """
        ij is assumed to have the shape (N, B, 2) where N is the number of points and B is the batch size.
        """
        if self.suitable_mask is None:
            raise ValueError("Variable suitable_mask is not set")

        return self.suitable_mask[ij[..., 0], ij[..., 1]]
