"""Quaternion <-> rotation-matrix conversions (wxyz)."""

import torch
import torch.nn.functional as F


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert (..., 4) wxyz quaternions to (..., 3, 3) rotation matrices.

    The quaternion is normalised first, so a slightly off-unit q still yields a
    rotation.
    """
    q = F.normalize(q, dim=-1, eps=1e-6)
    w, x, y, z = q.unbind(dim=-1)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

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


def rotmat_to_quat_batch(R: torch.Tensor) -> torch.Tensor:
    """Convert batched rotation matrices to quaternions (Shepperd's method).

    Branch-free: all four candidates are computed for every row and the right one
    is selected with torch.where, bit-identical to masked assignment but without
    the ~15 device->host syncs (`.any()`, boolean indexing) per call.

    Args:
        R: (B, 3, 3) rotation matrices
    Returns:
        q: (B, 4) quaternions [w, x, y, z]
    """
    r = lambda i, j: R[:, i, j]  # noqa: E731
    trace = r(0, 0) + r(1, 1) + r(2, 2)

    S = torch.sqrt((trace + 1.0).clamp_min(1e-10)) * 2
    q0 = torch.stack([0.25 * S, (r(2, 1) - r(1, 2)) / S, (r(0, 2) - r(2, 0)) / S, (r(1, 0) - r(0, 1)) / S], -1)
    S = torch.sqrt((1.0 + r(0, 0) - r(1, 1) - r(2, 2)).clamp_min(1e-10)) * 2
    q1 = torch.stack([(r(2, 1) - r(1, 2)) / S, 0.25 * S, (r(0, 1) + r(1, 0)) / S, (r(0, 2) + r(2, 0)) / S], -1)
    S = torch.sqrt((1.0 + r(1, 1) - r(0, 0) - r(2, 2)).clamp_min(1e-10)) * 2
    q2 = torch.stack([(r(0, 2) - r(2, 0)) / S, (r(0, 1) + r(1, 0)) / S, 0.25 * S, (r(1, 2) + r(2, 1)) / S], -1)
    S = torch.sqrt((1.0 + r(2, 2) - r(0, 0) - r(1, 1)).clamp_min(1e-10)) * 2
    q3 = torch.stack([(r(1, 0) - r(0, 1)) / S, (r(0, 2) + r(2, 0)) / S, (r(1, 2) + r(2, 1)) / S, 0.25 * S], -1)

    m0 = trace > 0
    m1 = ~m0 & (r(0, 0) > r(1, 1)) & (r(0, 0) > r(2, 2))
    m2 = ~m0 & ~m1 & (r(1, 1) > r(2, 2))
    q = torch.where(m0[:, None], q0, torch.where(m1[:, None], q1, torch.where(m2[:, None], q2, q3)))
    return F.normalize(q, dim=-1)
