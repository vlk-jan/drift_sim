"""Warp port of the spring-contact engine in simulator/engine.py.

Same physics (hard contact gate, Hunt-Crossley spring `relu(depth*k - c*vn)` with
the damping c capped per point, vn the body velocity of the point along the
normal), ~10x cheaper per rollout. Keep the engines formula-for-formula
identical: tests/test_engine_parity.py checks forward poses and gradients
between them.

The rollout is four kernel launches per timestep on a wp.Tape, captured once
per shape into a graph that each call replays (_Rollout), with state in
step-indexed arrays (`state_x[step]`, ...), and contact points are threads
(`dim=(B, NUM_WHEELS, NUM_WHEEL_POINTS)`). Warp only unrolls loops with
compile-time `range()` bounds; a dynamic loop that mutates state falls back to
`materialize_redefinitions`, whose adjoint uses end-of-loop values and silently
gives wrong gradients (forward still matches). Per-step launches keep every
primal input resident for an exact reverse pass.
`test_engine_parity.py::test_warp_kernels_have_no_dynamic_loops` forbids
reintroducing such loops.
"""

import numpy as np
import torch
import warp as wp

from simulator.engine import NUM_WHEEL_POINTS as _NUM_WHEEL_POINTS

wp.init()

# Compile-time constants so their loops unroll; don't make these kernel
# parameters (see module docstring).
NUM_WHEELS = wp.constant(3)
NUM_WHEEL_POINTS = wp.constant(_NUM_WHEEL_POINTS)


# --- Helper math (device functions) ---
@wp.func
def compute_uv_aligned(
    x: float,
    y: float,
    grid_center: wp.vec2,
    max_coord: wp.vec2,
    width: int,
    height: int,
):
    """Return grid indices (u, v) matching grid_sample(align_corners=True).

    Maps physical range [center - max_coord, center + max_coord] to [0, width-1].
    """
    u = ((x - grid_center[0]) / max_coord[0] + 1.0) * (float(width) - 1.0) * 0.5
    v = ((y - grid_center[1]) / max_coord[1] + 1.0) * (float(height) - 1.0) * 0.5
    return u, v


@wp.func
def sample_grid_bilinear(
    grid: wp.array3d(dtype=float), b: int, u: float, v: float, width: int, height: int
):
    """Bilinearly sample batch element `b`, replicating the border.

    Matches simulator/engine.py's padding_mode="border"; zero padding would put a
    phantom cliff at the map edge. A rollout leaving the map thus continues on the
    border height silently.
    """
    u0 = int(wp.floor(u))
    v0 = int(wp.floor(v))
    w_u = u - float(u0)
    w_v = v - float(v0)

    u0c = wp.clamp(u0, 0, width - 1)
    u1c = wp.clamp(u0 + 1, 0, width - 1)
    v0c = wp.clamp(v0, 0, height - 1)
    v1c = wp.clamp(v0 + 1, 0, height - 1)

    val0 = grid[b, v0c, u0c] * (1.0 - w_u) + grid[b, v0c, u1c] * w_u
    val1 = grid[b, v1c, u0c] * (1.0 - w_u) + grid[b, v1c, u1c] * w_u
    return val0 * (1.0 - w_v) + val1 * w_v


@wp.func
def sample_normal_bilinear(
    grid: wp.array3d(dtype=wp.vec3),
    b: int,
    u: float,
    v: float,
    width: int,
    height: int,
):
    """Bilinearly sample a normal field, renormalized after interpolation.

    Vertical off the map, where the border height extends flat (simulator/
    engine.py::_sample_terrain)."""
    if u < 0.0 or u > float(width - 1) or v < 0.0 or v > float(height - 1):
        return wp.vec3(0.0, 0.0, 1.0)
    u0 = int(wp.floor(u))
    v0 = int(wp.floor(v))
    w_u = u - float(u0)
    w_v = v - float(v0)

    u0c = wp.clamp(u0, 0, width - 1)
    u1c = wp.clamp(u0 + 1, 0, width - 1)
    v0c = wp.clamp(v0, 0, height - 1)
    v1c = wp.clamp(v0 + 1, 0, height - 1)

    n0 = grid[b, v0c, u0c] * (1.0 - w_u) + grid[b, v0c, u1c] * w_u
    n1 = grid[b, v1c, u0c] * (1.0 - w_u) + grid[b, v1c, u1c] * w_u
    n_res = n0 * (1.0 - w_v) + n1 * w_v

    # sqrt(|n|^2 + eps) rather than wp.normalize: the adjoint of the latter is
    # undefined at zero length, and interpolation can produce it at the border.
    return n_res / wp.sqrt(wp.dot(n_res, n_res) + 1.0e-12)


@wp.func
def get_wheel_point_local(i: int, total_points: int, radius: float):
    theta = (2.0 * wp.PI * float(i)) / float(total_points)
    return wp.vec3(radius * wp.cos(theta), 0.0, -radius * wp.sin(theta))


@wp.func
def sinc_stable(theta: float):
    """Return sin(theta)/theta, finite and differentiable at theta = 0.

    Mirrors torch.sinc(theta/pi) in simulator/numerical.py::integrate_quaternion.
    """
    if theta < 1.0e-4:
        t2 = theta * theta
        return 1.0 - t2 / 6.0 + t2 * t2 / 120.0
    return wp.sin(theta) / theta


@wp.func
def integrate_quat(q: wp.quat, omega: wp.vec3, dt: float):
    """Exponential-map quaternion update, matching the torch engine."""
    h = omega * (0.5 * dt)
    theta = wp.sqrt(wp.dot(h, h) + 1.0e-20)
    scale = sinc_stable(theta)

    d_xyz = h * scale
    dq = wp.quat(d_xyz[0], d_xyz[1], d_xyz[2], wp.cos(theta))
    q_new = dq * q
    return q_new / wp.sqrt(wp.dot(q_new, q_new) + 1.0e-12)


# --- Per-timestep kernels ---
@wp.kernel
def depth_sum_kernel(
    step: int,
    state_x: wp.array2d(dtype=wp.vec3),
    state_q: wp.array2d(dtype=wp.quat),
    wheel_centers: wp.array(dtype=wp.vec3),
    wheel_radii: wp.array(dtype=float),
    z_grid: wp.array3d(dtype=float),
    normals_grid: wp.array3d(dtype=wp.vec3),
    grid_center: wp.vec2,
    max_coord: wp.vec2,
    depth_sum: wp.array2d(dtype=float),
    wheel_depth_sum: wp.array3d(dtype=float),
    wheel_depth_max: wp.array3d(dtype=float),
):
    """Pass 1: sum the penetration depth over each robot's sample points, and
    sum and max it per wheel.

    The friction and damping impulse limits share over the contacts by depth,
    and a wheel's spring acts on its deepest point shared by depth (simulator/
    engine.py::_calculate_spring_force), so these need their own launch before
    force accumulation.
    """
    b, w, i = wp.tid()

    x = state_x[step, b]
    q = state_q[step, b]
    grid_h = z_grid.shape[1]
    grid_w = z_grid.shape[2]

    p_rel = get_wheel_point_local(i, NUM_WHEEL_POINTS, wheel_radii[w])
    p_world = x + wp.quat_rotate(q, wheel_centers[w] + p_rel)

    u, v = compute_uv_aligned(
        p_world[0], p_world[1], grid_center, max_coord, grid_w, grid_h
    )
    z_surf = sample_grid_bilinear(z_grid, b, u, v, grid_w, grid_h)
    n_surf = sample_normal_bilinear(normals_grid, b, u, v, grid_w, grid_h)

    dist = (p_world[2] - z_surf) * n_surf[2]
    if dist < 0.0:
        wp.atomic_add(depth_sum, step, b, -dist)
        wp.atomic_add(wheel_depth_sum, step, b, w, -dist)
        wp.atomic_max(wheel_depth_max, step, b, w, -dist)


@wp.func
def contact_point(
    step: int,
    b: int,
    w: int,
    i: int,
    state_x: wp.array2d(dtype=wp.vec3),
    state_q: wp.array2d(dtype=wp.quat),
    state_xd: wp.array2d(dtype=wp.vec3),
    state_omega: wp.array2d(dtype=wp.vec3),
    controls: wp.array2d(dtype=wp.vec3),
    wheel_centers: wp.array(dtype=wp.vec3),
    wheel_radii: wp.array(dtype=float),
    cog: wp.vec3,
    mass: float,
    inertia_body_inv: wp.mat33,
    gravity: float,
    z_grid: wp.array3d(dtype=float),
    normals_grid: wp.array3d(dtype=wp.vec3),
    friction_grid: wp.array3d(dtype=float),
    stiffness_grid: wp.array3d(dtype=float),
    grid_center: wp.vec2,
    max_coord: wp.vec2,
    dt: float,
    damping_alpha: float,
    slip_ref_ratio: float,
    slip_floor_velocity: float,
    depth_sum: wp.array2d(dtype=float),
    wheel_depth_sum: wp.array3d(dtype=float),
    wheel_depth_max: wp.array3d(dtype=float),
):
    """Spring force, friction law and impulse limit of one sample point.

    Returns (in_contact, F_spring, f_c_mag, v_dir, f_lim, r, v_tangential, v_rel),
    r = CoM -> point, v_rel = point velocity relative to the CoM.
    """
    x = state_x[step, b]
    q = state_q[step, b]
    xd = state_xd[step, b]
    omega = state_omega[step, b]

    grid_h = z_grid.shape[1]
    grid_w = z_grid.shape[2]

    w_center_world = x + wp.quat_rotate(q, wheel_centers[w])
    p_rel = get_wheel_point_local(i, NUM_WHEEL_POINTS, wheel_radii[w])
    p_world = w_center_world + wp.quat_rotate(q, p_rel)

    u, v = compute_uv_aligned(
        p_world[0], p_world[1], grid_center, max_coord, grid_w, grid_h
    )
    z_surf = sample_grid_bilinear(z_grid, b, u, v, grid_w, grid_h)
    n_surf = sample_normal_bilinear(normals_grid, b, u, v, grid_w, grid_h)

    dist = (p_world[2] - z_surf) * n_surf[2]
    zero = wp.vec3(0.0, 0.0, 0.0)
    if dist >= 0.0:
        return False, zero, 0.0, zero, 0.0, zero, zero, zero

    f_val = sample_grid_bilinear(friction_grid, b, u, v, grid_w, grid_h)
    k_s = sample_grid_bilinear(stiffness_grid, b, u, v, grid_w, grid_h)

    com_world = x + wp.quat_rotate(q, cog)

    # Contact-point velocity: body motion + wheel spin about the wheel's own axis.
    wheel_omega_world = wp.quat_rotate(q, wp.vec3(0.0, controls[step, b][w], 0.0))
    v_body = xd + wp.cross(omega, p_world - x)
    v_total = v_body + wp.cross(wheel_omega_world, p_world - w_center_world)
    # Rim points are body-fixed: only the body velocity compresses the contact;
    # wheel spin is tyre surface speed and enters friction only.
    vn = wp.dot(v_body, n_surf)

    b_damp = damping_alpha * 2.0 * wp.sqrt(mass) * wp.sqrt(k_s)
    # Hunt-Crossley damping, zero at zero depth (see simulator/engine.py).
    lam = b_damp * k_s / (mass * gravity)
    # Wheel spring k / n_wheels on its deepest point, shared by depth (see
    # simulator/engine.py).
    depth = -dist * wheel_depth_max[step, b, w] / wp.max(
        wheel_depth_sum[step, b, w], 1.0e-12
    )
    r = p_world - com_world
    r_x_n = wp.cross(r, n_surf)
    inv_m_eff_n = 1.0 / mass + wp.dot(
        r_x_n, wp.quat_rotate(q, inertia_body_inv * wp.quat_rotate_inv(q, r_x_n))
    )
    share = -dist / wp.max(depth_sum[step, b], 1.0e-12)
    c = wp.min(depth * lam / float(NUM_WHEELS), share / (dt * inv_m_eff_n))
    f_mag = wp.max(depth * k_s / float(NUM_WHEELS) - c * vn, 0.0)
    F_spring = n_surf * f_mag

    # Friction uses the tangential velocity only (see simulator/engine.py).
    v_tangential = v_total - wp.dot(v_total, n_surf) * n_surf
    # Slip at the end of the step under gravity (see simulator/engine.py).
    g_t = wp.vec3(0.0, 0.0, -gravity) + (gravity * n_surf[2]) * n_surf
    v_slip = v_tangential + g_t * dt
    v_mag = wp.sqrt(wp.dot(v_slip, v_slip) + 1.0e-12)
    v_dir = v_slip / (v_mag + 1.0e-6)

    F_N = wp.max(wp.dot(F_spring, n_surf), 0.0)
    # Slip ratio s = |v_t| / max(|omega_w| r_w, floor) (see simulator/engine.py).
    v_roll = wp.max(wp.abs(controls[step, b][w]) * wheel_radii[w], slip_floor_velocity)
    slip = v_mag / v_roll
    f_c_mag = f_val * F_N * wp.tanh(slip / slip_ref_ratio)
    if f_mag <= 0.0:
        f_c_mag = 0.0

    # Impulse limit: stop the slip in one step through the effective mass,
    # shared by depth (see simulator/engine.py::_friction).
    r_x_t = wp.cross(r, v_dir)
    inv_m_eff = 1.0 / mass + wp.dot(
        r_x_t, wp.quat_rotate(q, inertia_body_inv * wp.quat_rotate_inv(q, r_x_t))
    )
    share = -dist / wp.max(depth_sum[step, b], 1.0e-12)
    c_stop = v_tangential / inv_m_eff / dt + mass * g_t
    f_lim = share * wp.sqrt(wp.dot(c_stop, c_stop) + 1.0e-12)

    v_rel = v_total - (xd + wp.cross(omega, com_world - x))
    return True, F_spring, f_c_mag, v_dir, f_lim, r, v_tangential, v_rel


@wp.func
def body_accelerations(
    q: wp.quat,
    omega: wp.vec3,
    force_contacts: wp.vec3,
    torque_contacts: wp.vec3,
    inertia_body: wp.mat33,
    inertia_body_inv: wp.mat33,
    mass: float,
    gravity: float,
    ang_damping: float,
    torque_limit: float,
):
    """CoM linear and angular acceleration under the contact wrench + gravity."""
    rot_m = wp.quat_to_matrix(q)
    I_world = rot_m * inertia_body * wp.transpose(rot_m)
    # Closed form instead of wp.inverse(I_world): cheaper, and the adjoint never
    # differentiates a matrix inversion.
    I_inv_world = rot_m * inertia_body_inv * wp.transpose(rot_m)

    F_total = force_contacts + wp.vec3(0.0, 0.0, -mass * gravity)

    T_total = torque_contacts - ang_damping * omega
    t_mag = wp.sqrt(wp.dot(T_total, T_total) + 1.0e-12)
    T_total = T_total * wp.min(torque_limit / (t_mag + 1.0e-6), 1.0)

    omega_d = I_inv_world * (T_total - wp.cross(omega, I_world * omega))
    return F_total / mass, omega_d


@wp.kernel
def accumulate_forces_kernel(
    step: int,
    state_x: wp.array2d(dtype=wp.vec3),
    state_q: wp.array2d(dtype=wp.quat),
    state_xd: wp.array2d(dtype=wp.vec3),
    state_omega: wp.array2d(dtype=wp.vec3),
    controls: wp.array2d(dtype=wp.vec3),
    wheel_centers: wp.array(dtype=wp.vec3),
    wheel_radii: wp.array(dtype=float),
    cog: wp.vec3,
    mass: float,
    inertia_body_inv: wp.mat33,
    gravity: float,
    z_grid: wp.array3d(dtype=float),
    normals_grid: wp.array3d(dtype=wp.vec3),
    friction_grid: wp.array3d(dtype=float),
    stiffness_grid: wp.array3d(dtype=float),
    grid_center: wp.vec2,
    max_coord: wp.vec2,
    dt: float,
    damping_alpha: float,
    slip_ref_ratio: float,
    slip_floor_velocity: float,
    depth_sum: wp.array2d(dtype=float),
    wheel_depth_sum: wp.array3d(dtype=float),
    wheel_depth_max: wp.array3d(dtype=float),
    force_acc: wp.array2d(dtype=wp.vec3),
    torque_acc: wp.array2d(dtype=wp.vec3),
):
    """Pass 2: spring + uncut friction-law force and torque of one point."""
    b, w, i = wp.tid()
    hit, F_spring, f_c_mag, v_dir, f_lim, r, v_t, v_rel = contact_point(
        step, b, w, i, state_x, state_q, state_xd, state_omega, controls,
        wheel_centers, wheel_radii, cog, mass, inertia_body_inv, gravity, z_grid,
        normals_grid, friction_grid, stiffness_grid, grid_center, max_coord, dt,
        damping_alpha, slip_ref_ratio, slip_floor_velocity, depth_sum,
        wheel_depth_sum, wheel_depth_max,
    )  # fmt: skip
    if not hit:
        return
    F_point = F_spring - v_dir * f_c_mag
    wp.atomic_add(force_acc, step, b, F_point)
    wp.atomic_add(torque_acc, step, b, wp.cross(r, F_point))


@wp.kernel
def friction_cut_kernel(
    step: int,
    state_x: wp.array2d(dtype=wp.vec3),
    state_q: wp.array2d(dtype=wp.quat),
    state_xd: wp.array2d(dtype=wp.vec3),
    state_omega: wp.array2d(dtype=wp.vec3),
    controls: wp.array2d(dtype=wp.vec3),
    wheel_centers: wp.array(dtype=wp.vec3),
    wheel_radii: wp.array(dtype=float),
    cog: wp.vec3,
    mass: float,
    inertia_body: wp.mat33,
    inertia_body_inv: wp.mat33,
    gravity: float,
    z_grid: wp.array3d(dtype=float),
    normals_grid: wp.array3d(dtype=wp.vec3),
    friction_grid: wp.array3d(dtype=float),
    stiffness_grid: wp.array3d(dtype=float),
    grid_center: wp.vec2,
    max_coord: wp.vec2,
    dt: float,
    damping_alpha: float,
    slip_ref_ratio: float,
    slip_floor_velocity: float,
    ang_damping: float,
    torque_limit: float,
    depth_sum: wp.array2d(dtype=float),
    wheel_depth_sum: wp.array3d(dtype=float),
    wheel_depth_max: wp.array3d(dtype=float),
    force_acc: wp.array2d(dtype=wp.vec3),
    torque_acc: wp.array2d(dtype=wp.vec3),
    force_cut: wp.array2d(dtype=wp.vec3),
    torque_cut: wp.array2d(dtype=wp.vec3),
):
    """Pass 3: cap friction at the impulse limit where the uncut law would
    reverse the point's slip by the end of the step (simulator/engine.py
    ::_friction); the removed friction accumulates as a separate wrench."""
    b, w, i = wp.tid()
    hit, F_spring, f_c_mag, v_dir, f_lim, r, v_t, v_rel = contact_point(
        step, b, w, i, state_x, state_q, state_xd, state_omega, controls,
        wheel_centers, wheel_radii, cog, mass, inertia_body_inv, gravity, z_grid,
        normals_grid, friction_grid, stiffness_grid, grid_center, max_coord, dt,
        damping_alpha, slip_ref_ratio, slip_floor_velocity, depth_sum,
        wheel_depth_sum, wheel_depth_max,
    )  # fmt: skip
    if not hit:
        return
    if f_c_mag <= f_lim:
        return
    omega = state_omega[step, b]
    acc, omega_d = body_accelerations(
        state_q[step, b], omega, force_acc[step, b], torque_acc[step, b],
        inertia_body, inertia_body_inv, mass, gravity, ang_damping, torque_limit,
    )  # fmt: skip
    # Acceleration of the body-fixed point: a_com + alpha x r + omega x v_rel.
    a_pt = acc + wp.cross(omega_d, r) + wp.cross(omega, v_rel)
    if wp.dot(v_t + a_pt * dt, v_dir) >= 0.0:
        return
    F_cut = v_dir * (f_c_mag - f_lim)
    wp.atomic_add(force_cut, step, b, F_cut)
    wp.atomic_add(torque_cut, step, b, wp.cross(r, F_cut))


@wp.kernel
def integrate_kernel(
    step: int,
    state_x: wp.array2d(dtype=wp.vec3),
    state_q: wp.array2d(dtype=wp.quat),
    state_xd: wp.array2d(dtype=wp.vec3),
    state_omega: wp.array2d(dtype=wp.vec3),
    inertia_body: wp.mat33,
    inertia_body_inv: wp.mat33,
    mass: float,
    gravity: float,
    dt: float,
    ang_damping: float,
    torque_limit: float,
    cog: wp.vec3,
    com_transfer: float,
    force_acc: wp.array2d(dtype=wp.vec3),
    torque_acc: wp.array2d(dtype=wp.vec3),
    force_cut: wp.array2d(dtype=wp.vec3),
    torque_cut: wp.array2d(dtype=wp.vec3),
):
    """Semi-implicit Euler step: state[step] + accumulated wrench -> state[step+1]."""
    b = wp.tid()

    x = state_x[step, b]
    q = state_q[step, b]
    xd = state_xd[step, b]
    omega = state_omega[step, b]

    acc, omega_d = body_accelerations(
        q, omega, force_acc[step, b] + force_cut[step, b],
        torque_acc[step, b] + torque_cut[step, b], inertia_body, inertia_body_inv,
        mass, gravity, ang_damping, torque_limit,
    )  # fmt: skip

    # state.x is the body origin, not the CoM; com_transfer > 0 subtracts
    # alpha x r + omega x (omega x r), r = quat_rotate(q, cog). On by default,
    # as in simulator/engine.py.
    if com_transfer > 0.0:
        r_com = wp.quat_rotate(q, cog)
        acc = acc - wp.cross(omega_d, r_com) - wp.cross(omega, wp.cross(omega, r_com))

    next_xd = xd + acc * dt
    next_omega = omega + omega_d * dt
    # Same clamps as the torch engine; a runaway step would poison the rest of the
    # rollout and its adjoint.
    next_xd = wp.vec3(
        wp.clamp(next_xd[0], -50.0, 50.0),
        wp.clamp(next_xd[1], -50.0, 50.0),
        wp.clamp(next_xd[2], -50.0, 50.0),
    )
    next_omega = wp.vec3(
        wp.clamp(next_omega[0], -50.0, 50.0),
        wp.clamp(next_omega[1], -50.0, 50.0),
        wp.clamp(next_omega[2], -50.0, 50.0),
    )

    state_xd[step + 1, b] = next_xd
    state_omega[step + 1, b] = next_omega
    state_x[step + 1, b] = x + next_xd * dt
    state_q[step + 1, b] = integrate_quat(q, next_omega, dt)


# --- Autograd bridge ---
class _Rollout:
    """Persistent buffers and two captured graphs (forward, adjoint) for one shape.

    The rollout is ~1000 tiny launches per call (4 kernels x steps, and as many
    adjoints) and was CPU-launch-bound: ~95 ms wall for ~30 ms of GPU work at
    B*n = 128 x 125 steps. Captured once per shape and constants, a call is a
    copy into these buffers plus one graph launch each way. Same kernels in the
    same order, so the same numbers as launching them one by one.
    """

    def __init__(self, B, num_steps, H, W, device, consts):
        (wheel_centers_wp, wheel_radii_wp, inertia_wp, inertia_inv_wp, cog_wp, mass,
         gc, mc, dt, g, damping_alpha, slip_ref_ratio, slip_floor_velocity,
         ang_damping, torque_limit, com_transfer) = consts  # fmt: skip
        z = lambda shape, dtype, grad=True: wp.zeros(  # noqa: E731
            shape, dtype=dtype, device=device, requires_grad=grad
        )
        # Inputs, copied in per call; stiffness takes no gradient.
        self.ctrl = z((num_steps, B), wp.vec3)
        self.z, self.n = z((B, H, W), wp.float32), z((B, H, W), wp.vec3)
        self.kf, self.ks = z((B, H, W), wp.float32), z((B, H, W), wp.float32, False)
        self.init = [z(B, t, False) for t in (wp.vec3, wp.quat, wp.vec3, wp.vec3)]
        # One slot per step keeps every launch's primal inputs intact for the tape;
        # per-step accumulators too (a reused buffer needs an undifferentiable zeroing).
        self.state = [z((num_steps + 1, B), t) for t in (wp.vec3, wp.quat, wp.vec3, wp.vec3)]
        state_x, state_q, state_xd, state_omega = self.state
        depth_sum = z((num_steps, B), float)
        wheel_depth = [z((num_steps, B, int(NUM_WHEELS)), float) for _ in range(2)]
        self.acc = [depth_sum, *wheel_depth] + [
            z((num_steps, B), wp.vec3) for _ in range(4)
        ]
        (_, wheel_depth_sum, wheel_depth_max,
         force_acc, torque_acc, force_cut, torque_cut) = self.acc  # fmt: skip
        # Incoming output gradients, copied in before the adjoint graph.
        self.seed = [z((num_steps, B), t, False) for t in (wp.vec3, wp.quat, wp.vec3, wp.vec3)]
        self.generation = 0
        self.device = device

        ctrl, wz, wn, kf, ks = self.ctrl, self.z, self.n, self.kf, self.ks
        point_dim = (B, int(NUM_WHEELS), int(NUM_WHEEL_POINTS))
        self.tape = wp.Tape()
        with wp.ScopedCapture(device) as cap:
            for a in self.acc:
                a.zero_()
            for s, i in zip(self.state, self.init):
                wp.copy(s[0], i)
            with self.tape:
                for step in range(num_steps):
                    wp.launch(depth_sum_kernel, dim=point_dim, inputs=[
                        step, state_x, state_q, wheel_centers_wp, wheel_radii_wp, wz, wn,
                        gc, mc], outputs=[depth_sum, wheel_depth_sum, wheel_depth_max],
                        device=device)  # fmt: skip
                    wp.launch(accumulate_forces_kernel, dim=point_dim, inputs=[
                        step, state_x, state_q, state_xd, state_omega, ctrl,
                        wheel_centers_wp, wheel_radii_wp, cog_wp, mass, inertia_inv_wp, g,
                        wz, wn, kf, ks, gc, mc, dt, damping_alpha, slip_ref_ratio,
                        slip_floor_velocity, depth_sum, wheel_depth_sum, wheel_depth_max],
                        outputs=[force_acc, torque_acc],
                        device=device)  # fmt: skip
                    wp.launch(friction_cut_kernel, dim=point_dim, inputs=[
                        step, state_x, state_q, state_xd, state_omega, ctrl,
                        wheel_centers_wp, wheel_radii_wp, cog_wp, mass, inertia_wp,
                        inertia_inv_wp, g, wz, wn, kf, ks, gc, mc, dt, damping_alpha,
                        slip_ref_ratio, slip_floor_velocity, ang_damping, torque_limit,
                        depth_sum, wheel_depth_sum, wheel_depth_max, force_acc, torque_acc],
                        outputs=[force_cut, torque_cut],
                        device=device)  # fmt: skip
                    wp.launch(integrate_kernel, dim=B, inputs=[
                        step, state_x, state_q, state_xd, state_omega, inertia_wp,
                        inertia_inv_wp, mass, g, dt, ang_damping, torque_limit, cog_wp,
                        com_transfer, force_acc, torque_acc, force_cut, torque_cut],
                        device=device)  # fmt: skip
        self.forward_graph = cap.graph
        with wp.ScopedCapture(device) as cap:
            # Every adjoint accumulates, so all of them start from zero.
            for a in (*self.state, *self.acc, ctrl, wz, wn, kf):
                a.grad.zero_()
            # Step 0 is the seeded initial state, not an output.
            for s, sd in zip(self.state, self.seed):
                wp.copy(s.grad[1:], sd)
            self.tape.backward()
        self.adjoint_graph = cap.graph
        # The allocations above zero-filled on warp's stream; torch copies the
        # inputs in on its own (WarpTrajectoryFunc.forward).
        wp.synchronize_device(device)


def _launch(graph, device):
    """Replay a captured graph on warp's stream, ordered against torch's current
    stream both ways by events: after the torch copies into the buffers, before
    torch reads the outputs. Explicit, so it does not rest on warp's stream
    synchronising implicitly with torch's legacy default stream. (Replaying on
    torch's stream itself gave wrong adjoints, 2026-10-23.) Each DDP rank is its
    own process with its own cache."""
    if not device.is_cuda:
        wp.capture_launch(graph)
        return
    ws, ts = wp.get_stream(device), wp.stream_from_torch(torch.cuda.current_stream(device.ordinal))
    ws.wait_stream(ts)
    wp.capture_launch(graph, stream=ws)
    ts.wait_stream(ws)


# Rollouts by (shape, constants); a few shapes at most (train, val, a last batch).
# Each keeps its own grids and gradients: ~0.1 GB at B 128 on a 128 x 128 map.
_ROLLOUTS: dict = {}
_MAX_ROLLOUTS = 8


def _get_rollout(B, num_steps, H, W, device, consts, with_grad):
    # Everything a captured graph bakes in: shapes, device, and every scalar kernel
    # argument by value (floats, vec/mat entries); robot arrays by address (their
    # contents are read at replay, and the tape keeps them alive).
    key = (B, num_steps, H, W, str(device), bool(with_grad), tuple(
        c.ptr if isinstance(c, wp.array) else tuple(np.asarray(c, dtype=np.float64).ravel().tolist())
        for c in consts))
    ro = _ROLLOUTS.pop(key, None)
    if ro is None:
        if len(_ROLLOUTS) >= _MAX_ROLLOUTS:
            _ROLLOUTS.pop(next(iter(_ROLLOUTS)))  # least recently used
        ro = _Rollout(B, num_steps, H, W, device, consts)
    _ROLLOUTS[key] = ro
    return ro


class WarpTrajectoryFunc(torch.autograd.Function):
    """Roll the trajectory on a captured wp.Tape and hand the adjoint back to torch.

    Stiffness gets no gradient: it is neither predicted nor adapted. The buffers
    are shared per shape (_Rollout), so a second rollout of the same shape before
    the first one's backward invalidates the first; its backward then raises.
    """

    @staticmethod
    def forward(
        ctx,
        init_x,
        init_q,  # WXYZ
        init_xd,
        init_omega,
        control_velocities,  # (num_steps, B, 3)
        z_grid_t,  # (B, H, W)
        normals_grid_t,  # (B, H, W, 3)
        friction_grid_t,  # (B, H, W)
        stiffness_grid_t,  # (B, H, W), no gradient
        wheel_centers_wp,
        wheel_radii_wp,
        inertia_wp,
        inertia_inv_wp,
        cog_wp,
        mass,
        grid_center_t,
        max_coord_t,
        dt,
        num_steps,
        g,
        damping_alpha,
        slip_ref_ratio,
        slip_floor_velocity,
        ang_damping,
        torque_limit,
        com_transfer,
        with_grad,
        device_str,
    ):
        device = wp.get_device(device_str)
        B = init_x.shape[0]
        H, W = z_grid_t.shape[1:]
        consts = (
            wheel_centers_wp, wheel_radii_wp, inertia_wp, inertia_inv_wp, cog_wp, mass,
            wp.vec2(float(grid_center_t[0]), float(grid_center_t[1])),
            wp.vec2(float(max_coord_t[0]), float(max_coord_t[1])),
            dt, g, damping_alpha, slip_ref_ratio, slip_floor_velocity, ang_damping,
            torque_limit, com_transfer,
        )  # fmt: skip
        ro = _get_rollout(B, num_steps, H, W, device, consts, with_grad)

        def put(arr, t):
            wp.to_torch(arr).copy_(t.detach().reshape(wp.to_torch(arr).shape))

        put(ro.ctrl, control_velocities)
        put(ro.z, z_grid_t)
        put(ro.n, normals_grid_t)
        put(ro.kf, friction_grid_t)
        put(ro.ks, stiffness_grid_t)
        for arr, t in zip(ro.init, (init_x, init_q[:, [1, 2, 3, 0]], init_xd, init_omega)):
            put(arr, t)
        _launch(ro.forward_graph, device)

        ro.generation += 1
        ctx.rollout, ctx.generation = ro, ro.generation

        # Drop the seeded step 0 to match the torch engine, whose first pose is the
        # state after one step.
        state_x, state_q, state_xd, state_omega = ro.state
        out_q_xyzw = wp.to_torch(state_q)[1:]
        return (
            wp.to_torch(state_x)[1:].clone(),
            out_q_xyzw[..., [3, 0, 1, 2]].contiguous(),
            wp.to_torch(state_xd)[1:].clone(),
            wp.to_torch(state_omega)[1:].clone(),
        )

    @staticmethod
    def backward(ctx, grad_x, grad_q, grad_xd, grad_omega):
        ro = ctx.rollout
        if ro.generation != ctx.generation:
            raise RuntimeError(
                "warp engine: another rollout of the same shape ran between this "
                "rollout's forward and backward and overwrote its tape buffers; "
                "backpropagate each rollout before starting the next"
            )
        for seed, gr, quat in zip(ro.seed, (grad_x, grad_q, grad_xd, grad_omega),
                                  (False, True, False, False)):
            dst = wp.to_torch(seed)
            if gr is None:
                dst.zero_()
            else:
                dst.copy_(gr[..., [1, 2, 3, 0]] if quat else gr)  # WXYZ -> XYZW
        _launch(ro.adjoint_graph, ro.device)

        # Non-finite gradients propagate as in the torch engine, so the trainer
        # skips the step (utils/trainer.py) whichever engine blew up.
        def get_grad(arr, needed):
            if not needed:
                return None
            return wp.to_torch(arr.grad).clone().detach()

        need = ctx.needs_input_grad
        # The initial state's adjoint is what the tape left in state_*.grad[0]
        # (x0 takes one through the clearance raise, simulator/utils.py::
        # raise_to_clearance; the rest when a caller seeds them with grad).
        init = [
            wp.to_torch(s.grad)[0].clone() if need[i] else None
            for i, s in enumerate(ro.state)
        ]
        if init[1] is not None:
            init[1] = init[1][..., [3, 0, 1, 2]]  # XYZW -> WXYZ
        return (
            *init,
            get_grad(ro.ctrl, need[4]),
            get_grad(ro.z, need[5]),
            get_grad(ro.n, need[6]),
            get_grad(ro.kf, need[7]),
            # The input stiffness grid takes no gradient; the sampled stiffness
            # still varies with contact position.
            None,  # stiffness
            None,  # wheel_centers_wp
            None,  # wheel_radii_wp
            None,  # inertia_wp
            None,  # inertia_inv_wp
            None,  # cog_wp
            None,  # mass
            None,  # grid_center_t
            None,  # max_coord_t
            None,  # dt
            None,  # num_steps
            None,  # g
            None,  # damping_alpha
            None,  # slip_ref_ratio
            None,  # slip_floor_velocity
            None,  # ang_damping
            None,  # torque_limit
            None,  # com_transfer
            None,  # with_grad
            None,  # device_str
        )


class WarpStaticEngine:
    """Hold the robot constants; state and scratch buffers are cached per shape
    (_get_rollout)."""

    def __init__(
        self,
        robot_model_config,
        device="cuda",
    ):
        self.device = wp.get_device(device)

        r = robot_model_config
        self.mass = float(r.mass)
        self.wp_wheel_radii = wp.from_torch(
            r.wheel_radii.to(device).float().contiguous(), dtype=wp.float32
        )
        self.wp_wheel_centers = wp.from_torch(
            r.wheel_centers.to(device).float().contiguous(), dtype=wp.vec3
        )

        inertia = r.inertia_tensor.to(device).float()
        self.wp_inertia = wp.mat33(*inertia.flatten().tolist())
        self.wp_inertia_inv = wp.mat33(*torch.linalg.inv(inertia).flatten().tolist())
        self.wp_cog = wp.vec3(*r.cog.to(device).float().tolist())

    def run_trajectory(
        self, init_state, control_velocities, world_config, engine_config
    ):
        """Roll out (num_steps, B, 3) wheel speeds; outputs are (num_steps, B, ...)."""
        num_steps = control_velocities.shape[0]
        B = control_velocities.shape[1]
        stiffness = world_config.k_stiffness

        return WarpTrajectoryFunc.apply(
            init_state.x,
            init_state.q,
            init_state.xd,
            init_state.omega,
            control_velocities,
            world_config.z_grid,
            world_config.normals.permute(0, 2, 3, 1),  # (B,3,H,W) -> (B,H,W,3)
            world_config.k_friction,
            stiffness.expand(B, -1, -1) if stiffness.shape[0] == 1 else stiffness,
            self.wp_wheel_centers,
            self.wp_wheel_radii,
            self.wp_inertia,
            self.wp_inertia_inv,
            self.wp_cog,
            self.mass,
            world_config.grid_center_xy,  # floats, so no device read
            world_config.max_coord_xy,
            engine_config.dt,
            num_steps,
            engine_config.g,
            engine_config.damping_alpha,
            float(engine_config.slip_ref_ratio),
            float(engine_config.slip_floor_velocity),
            engine_config.ang_damping,
            engine_config.torque_limit,
            1.0 if engine_config.com_transfer else 0.0,
            # A rollout that will not be backpropagated (eval, no_grad) gets its own
            # cached buffers, so it can never overwrite a pending training tape.
            torch.is_grad_enabled(),
            str(self.device),
        )
