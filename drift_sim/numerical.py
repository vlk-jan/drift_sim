from typing import Union

import torch
import torch.nn.functional as F

from utils.geometry import quat_to_rotmat

__all__ = [
    "quat_to_rotmat",
    "unit_quaternion",
    "integrate_quaternion",
    "interpolate_grid",
    "interpolate_surface_normals_multi",
    "interpolate_surface_normals",
]


def unit_quaternion(
    batch_size: int = 1, device: Union[str, torch.device] = "cpu"
) -> torch.Tensor:
    """
    Returns a unit quaternion tensor of shape (batch_size, 4).

    Parameters:
    - batch_size: Number of quaternions to generate.
    - device: Device on which to create the tensor.

    Returns:
    - Unit quaternion tensor.
    """
    q = torch.zeros(batch_size, 4, device=device)
    q[:, 0] = 1.0
    return q


def integrate_quaternion(
    q: torch.Tensor, omega: torch.Tensor, dt: float, eps: float = 1e-6
) -> torch.Tensor:
    """
    Integrate quaternion using the quaternion derivative.
    Optimized to use vectorized operations and avoid unnecessary intermediate tensor
        allocations.

    Parameters:
        - q: Quaternion. Shape [B, 4].
        - omega: Angular velocity. Shape [B, 3].
        - dt: Time step. Float value.
    Returns:
        - Updated quaternion. Shape [B, 4].
    """
    half_dt_omega = 0.5 * dt * omega  # shape: (B, 3)

    # Calculate theta (magnitude of half_dt_omega)
    theta = torch.linalg.norm(half_dt_omega, dim=-1, keepdim=True)  # shape: (B, 1)

    # Calculate scale = sin(theta)/theta using sinc(theta/pi)
    # sinc(x) = sin(pi*x)/(pi*x) -> sinc(theta/pi) = sin(theta)/theta
    # This handles theta close to 0 correctly.
    scale = torch.sinc(theta / torch.pi)

    # Components of delta quaternion (delta_q)
    d_w = torch.cos(theta)
    d_xyz = scale * half_dt_omega

    # Components of current quaternion (q)
    w = q[..., :1]
    xyz = q[..., 1:]

    # Quaternion multiplication: q_new = delta_q * q
    new_w = d_w * w - (d_xyz * xyz).sum(dim=-1, keepdim=True)
    new_xyz = d_w * xyz + w * d_xyz + torch.linalg.cross(d_xyz, xyz, dim=-1)

    q_new = torch.cat([new_w, new_xyz], dim=-1)

    # Normalize to prevent drift
    return F.normalize(q_new, dim=-1, eps=eps)


def interpolate_grid(grid, xy, max_coord):
    """
    grid: (B, 1, H, W) or (B, H, W)
    xy:   (B, Wh, P, 2) or (B, N, 2)
    """
    # Normalize xy to [-1, 1]
    xy_norm = xy / max_coord

    # Ensure grid is (B, 1, H, W)
    if grid.ndim == 3:
        grid = grid.unsqueeze(1)

    # Handle (B, N, 2) input by expanding to (B, 1, N, 2) to treat as Wh=1, P=N
    if xy_norm.ndim == 3:
        xy_norm = xy_norm.unsqueeze(1)

    # Check batch dimension match
    B = xy_norm.shape[0]
    if grid.shape[0] != B:
        grid = grid.expand(B, -1, -1, -1)

    # Apply sampling
    # grid: (B, 1, H_in, W_in)
    # xy_norm: (B, Wh, P, 2)
    sampled = F.grid_sample(grid, xy_norm, mode="bilinear", align_corners=True)

    return sampled.permute(0, 2, 3, 1)


def interpolate_surface_normals_multi(grid_normals, query, max_coord):
    """
    Compute surface normals for multiple wheels and multiple points.

    Args:
        grid_normals: (B, 3, H, W) pre-computed normals
        query:        (B, W, P, 2) world XY positions of wheel contact samples
        max_coord:    normalization coordinate

    Returns:
        normals: (B, W, P, 3)
    """
    assert query.shape[3] == 2  # plane
    B, W, P, _ = query.shape

    # --- Shape check ---
    assert grid_normals.shape[0] == query.shape[0], (
        f"Batch mismatch: grid_normals batch={grid_normals.shape[0]} query batch={query.shape[0]}"
    )

    # --- Flatten wheels & points, keep batch ---
    # (B, W*P, 2)
    query_flat = query.flatten(1, 2)

    # --- Compute normals using existing single-sample function ---
    # Output: (B, W*P, 3)
    normals_flat = interpolate_surface_normals(
        grid_normals,
        query_flat,
        max_coord,
    )

    # --- Restore wheel/point structure ---
    return normals_flat.reshape(B, W, P, 3)


def interpolate_surface_normals(
    grid_normals: torch.Tensor, query: torch.Tensor, max_coord: float
) -> torch.Tensor:
    """
    Computes the surface normals at the queried coordinates.

    Parameters:
    - grid_normals: torch.Tensor of normals (B, 3, H, W).
    - query: Tensor of desired point coordinates for interpolation (3D array), (B, N,
        2).

    Returns:
    - Surface normals at the queried coordinates (B, N, 3).
    """
    assert grid_normals.shape[0] == query.shape[0], (
        f"Batch mismatch: grid_normals batch={grid_normals.shape[0]} query batch={query.shape[0]}"
    )
    # Normalize to [-1, 1], clamp the query coordinates to the grid's valid range
    norm_query = torch.clamp(query / max_coord, -1, 1)

    # Query coordinates of shape (B, N, 1, 2)
    grid_coords = norm_query.unsqueeze(2)

    # Interpolate the normals: (B, 3, H, W) -> (B, 3, N, 1)
    interpolated = torch.nn.functional.grid_sample(
        grid_normals, grid_coords, align_corners=True, mode="bilinear"
    )

    # Reshape to (B, N, 3)
    # interpolated is (B, 3, N, 1) -> squeeze(-1) -> (B, 3, N) -> transpose -> (B, N, 3)
    normals = interpolated.squeeze(-1).transpose(1, 2)

    # Normalize (because interpolation doesn't preserve unit length)
    return F.normalize(normals, dim=-1, eps=1e-6)
