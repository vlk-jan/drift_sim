import torch
import torch.nn.functional as F

__all__ = [
    "integrate_quaternion",
    "interpolate_grid",
]


def integrate_quaternion(
    q: torch.Tensor, omega: torch.Tensor, dt: float, eps: float = 1e-6
) -> torch.Tensor:
    """Integrate a wxyz quaternion by a world-frame angular velocity over dt.

    Parameters:
        - q: Quaternion, [B, 4].
        - omega: Angular velocity, [B, 3].
        - dt: Time step.
    Returns:
        - Updated quaternion, [B, 4].
    """
    half_dt_omega = 0.5 * dt * omega  # (B, 3)
    theta = torch.linalg.norm(half_dt_omega, dim=-1, keepdim=True)  # (B, 1)

    # sinc(theta/pi) = sin(theta)/theta, stable at theta -> 0.
    scale = torch.sinc(theta / torch.pi)

    d_w = torch.cos(theta)
    d_xyz = scale * half_dt_omega

    w = q[..., :1]
    xyz = q[..., 1:]

    # q_new = delta_q * q
    new_w = d_w * w - (d_xyz * xyz).sum(dim=-1, keepdim=True)
    new_xyz = d_w * xyz + w * d_xyz + torch.linalg.cross(d_xyz, xyz, dim=-1)

    q_new = torch.cat([new_w, new_xyz], dim=-1)

    # Renormalize to prevent drift.
    return F.normalize(q_new, dim=-1, eps=eps)


def interpolate_grid(grid, xy, max_coord):
    """Bilinearly sample a grid at metric xy coordinates.

    grid: (B, 1, H, W) or (B, H, W)
    xy:   (B, Wh, P, 2) or (B, N, 2)
    """
    xy_norm = xy / max_coord  # [-1, 1]

    if grid.ndim == 3:
        grid = grid.unsqueeze(1)

    # (B, N, 2) -> (B, 1, N, 2)
    if xy_norm.ndim == 3:
        xy_norm = xy_norm.unsqueeze(1)

    B = xy_norm.shape[0]
    if grid.shape[0] != B:
        grid = grid.expand(B, -1, -1, -1)

    # Border padding, as the engines: zeros would put a cliff at the map edge.
    sampled = F.grid_sample(
        grid, xy_norm, mode="bilinear", align_corners=True, padding_mode="border"
    )

    return sampled.permute(0, 2, 3, 1)
