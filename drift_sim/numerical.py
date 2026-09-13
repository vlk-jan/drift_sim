import torch
import torch.nn.functional as F

__all__ = [
    "integrate_quaternion",
    "interpolate_grid",
]


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
