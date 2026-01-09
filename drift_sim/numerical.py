from typing import Union

import torch
import torch.nn.functional as F

__all__ = [
    "unit_quaternion",
    "integrate_quaternion",
    "quat_to_rotmat",
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


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a quaternion into a rotation matrix.

    Args:
        q: (..., 4) quaternion in (w, x, y, z) format

    Returns:
        R: (..., 3, 3) rotation matrix
    """
    # Normalize the quaternion
    q = F.normalize(q, dim=-1, eps=1e-12)

    w, x, y, z = q.unbind(dim=-1)

    # Precompute repeated terms
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    # Build rotation matrix
    R = torch.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - wz),
            2 * (xz + wy),
            2 * (xy + wz),
            1 - 2 * (xx + zz),
            2 * (yz - wx),
            2 * (xz - wy),
            2 * (yz + wx),
            1 - 2 * (xx + yy),
        ],
        dim=-1,
    )

    return R.reshape(q.shape[:-1] + (3, 3))


def integrate_quaternion(
    q: torch.Tensor, omega: torch.Tensor, dt: float, eps: float = 1e-8
) -> torch.Tensor:
    """
    Integrate quaternion using the quaternion derivative.
    Optimized to use vectorized operations and avoid unnecessary intermediate tensor allocations.

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
    return F.normalize(q_new, p=2, dim=-1, eps=eps)
