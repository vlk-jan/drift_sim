"""Warp port of the spring-contact engine in simulator/engine.py.

Same physics (hard ReLU contact, `max(-k*dh - b*vn, 0)`), ~10x cheaper per
rollout. Keep the engines formula-for-formula identical: tests/test_engine_parity.py
checks forward poses and gradients between them.

The rollout is one kernel launch per timestep on a wp.Tape, with state in
step-indexed arrays (`state_x[step]`, ...), and contact points are threads
(`dim=(B, NUM_WHEELS, NUM_WHEEL_POINTS)`). Warp only unrolls loops with
compile-time `range()` bounds; a dynamic loop that mutates state falls back to
`materialize_redefinitions`, whose adjoint uses end-of-loop values and silently
gives wrong gradients (forward still matches). Per-step launches keep every
primal input resident for an exact reverse pass.
`test_engine_parity.py::test_warp_kernels_have_no_dynamic_loops` forbids
reintroducing such loops.
"""

import warnings

import torch
import warp as wp

wp.init()

# The warp adjoint zeroes non-finite gradients while the torch engine propagates
# them, so the zeroing is announced once per process rather than done silently.
_WARNED_NONFINITE_GRAD = False

# Compile-time constants so their loops unroll; don't make these kernel
# parameters (see module docstring).
NUM_WHEELS = wp.constant(3)
NUM_WHEEL_POINTS = wp.constant(16)


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
    """Bilinearly sample a normal field, renormalized after interpolation."""
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
):
    """Pass 1: sum the penetration depth over each robot's sample points.

    The friction impulse limit shares itself over the contacts by depth, so the
    sum needs its own launch before force accumulation.
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
    v_total = (
        xd
        + wp.cross(omega, p_world - x)
        + wp.cross(wheel_omega_world, p_world - w_center_world)
    )
    vn = wp.dot(v_total, n_surf)

    b_damp = damping_alpha * 2.0 * wp.sqrt(mass) * wp.sqrt(k_s)
    # Per-point spring k / n_wheels, summed (see simulator/engine.py).
    f_mag = wp.max(-k_s * dist - b_damp * vn, 0.0) / float(NUM_WHEELS)
    F_spring = n_surf * f_mag

    # Friction uses the tangential velocity only (see simulator/engine.py).
    v_tangential = v_total - vn * n_surf
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
    if f_mag <= 1.0e-6:
        f_c_mag = 0.0

    # Impulse limit: stop the slip in one step through the effective mass,
    # shared by depth (see simulator/engine.py::_friction).
    r = p_world - com_world
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
class WarpTrajectoryFunc(torch.autograd.Function):
    """Roll the trajectory on a wp.Tape and hand the adjoint back to torch.

    Stiffness gets no gradient: it is neither predicted nor adapted.
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
        device_str,
    ):
        device = wp.get_device(device_str)
        B = init_x.shape[0]

        def to_wp(t, dtype, grad=None):
            requires_grad = t.requires_grad if grad is None else grad
            return wp.from_torch(
                t.detach().contiguous(), dtype=dtype, requires_grad=requires_grad
            )

        wp_ctrl = to_wp(control_velocities, wp.vec3)
        wp_z = to_wp(z_grid_t, wp.float32)
        wp_n = to_wp(normals_grid_t, wp.vec3)
        wp_kf = to_wp(friction_grid_t, wp.float32)
        wp_ks = to_wp(stiffness_grid_t, wp.float32, grad=False)

        # One slot per step keeps every launch's primal inputs intact for the tape.
        state_x = wp.zeros(
            (num_steps + 1, B), dtype=wp.vec3, device=device, requires_grad=True
        )
        state_q = wp.zeros(
            (num_steps + 1, B), dtype=wp.quat, device=device, requires_grad=True
        )
        state_xd = wp.zeros(
            (num_steps + 1, B), dtype=wp.vec3, device=device, requires_grad=True
        )
        state_omega = wp.zeros(
            (num_steps + 1, B), dtype=wp.vec3, device=device, requires_grad=True
        )

        wp.copy(state_x[0], to_wp(init_x, wp.vec3, grad=False))
        wp.copy(state_q[0], to_wp(init_q[:, [1, 2, 3, 0]], wp.quat, grad=False))
        wp.copy(state_xd[0], to_wp(init_xd, wp.vec3, grad=False))
        wp.copy(state_omega[0], to_wp(init_omega, wp.vec3, grad=False))

        # Per-step accumulators too: a reused buffer needs a zeroing pass the tape
        # cannot differentiate.
        depth_sum = wp.zeros(
            (num_steps, B), dtype=float, device=device, requires_grad=True
        )
        force_acc = wp.zeros(
            (num_steps, B), dtype=wp.vec3, device=device, requires_grad=True
        )
        torque_acc = wp.zeros(
            (num_steps, B), dtype=wp.vec3, device=device, requires_grad=True
        )
        force_cut = wp.zeros(
            (num_steps, B), dtype=wp.vec3, device=device, requires_grad=True
        )
        torque_cut = wp.zeros(
            (num_steps, B), dtype=wp.vec3, device=device, requires_grad=True
        )

        gc_wp = wp.vec2(float(grid_center_t[0]), float(grid_center_t[1]))
        mc_wp = wp.vec2(float(max_coord_t[0]), float(max_coord_t[1]))
        point_dim = (B, int(NUM_WHEELS), int(NUM_WHEEL_POINTS))

        ctx.tape = wp.Tape()
        with ctx.tape:
            for step in range(num_steps):
                wp.launch(
                    depth_sum_kernel,
                    dim=point_dim,
                    inputs=[
                        step,
                        state_x,
                        state_q,
                        wheel_centers_wp,
                        wheel_radii_wp,
                        wp_z,
                        wp_n,
                        gc_wp,
                        mc_wp,
                    ],
                    outputs=[depth_sum],
                    device=device,
                )
                wp.launch(
                    accumulate_forces_kernel,
                    dim=point_dim,
                    inputs=[
                        step,
                        state_x,
                        state_q,
                        state_xd,
                        state_omega,
                        wp_ctrl,
                        wheel_centers_wp,
                        wheel_radii_wp,
                        cog_wp,
                        mass,
                        inertia_inv_wp,
                        g,
                        wp_z,
                        wp_n,
                        wp_kf,
                        wp_ks,
                        gc_wp,
                        mc_wp,
                        dt,
                        damping_alpha,
                        slip_ref_ratio,
                        slip_floor_velocity,
                        depth_sum,
                    ],
                    outputs=[force_acc, torque_acc],
                    device=device,
                )
                wp.launch(
                    friction_cut_kernel,
                    dim=point_dim,
                    inputs=[
                        step,
                        state_x,
                        state_q,
                        state_xd,
                        state_omega,
                        wp_ctrl,
                        wheel_centers_wp,
                        wheel_radii_wp,
                        cog_wp,
                        mass,
                        inertia_wp,
                        inertia_inv_wp,
                        g,
                        wp_z,
                        wp_n,
                        wp_kf,
                        wp_ks,
                        gc_wp,
                        mc_wp,
                        dt,
                        damping_alpha,
                        slip_ref_ratio,
                        slip_floor_velocity,
                        ang_damping,
                        torque_limit,
                        depth_sum,
                        force_acc,
                        torque_acc,
                    ],
                    outputs=[force_cut, torque_cut],
                    device=device,
                )
                wp.launch(
                    integrate_kernel,
                    dim=B,
                    inputs=[
                        step,
                        state_x,
                        state_q,
                        state_xd,
                        state_omega,
                        inertia_wp,
                        inertia_inv_wp,
                        mass,
                        g,
                        dt,
                        ang_damping,
                        torque_limit,
                        cog_wp,
                        com_transfer,
                        force_acc,
                        torque_acc,
                        force_cut,
                        torque_cut,
                    ],
                    device=device,
                )

        ctx.wp_inputs = {"ctrl": wp_ctrl, "z": wp_z, "n": wp_n, "kf": wp_kf}
        ctx.state = (state_x, state_q, state_xd, state_omega)

        # Drop the seeded step 0 to match the torch engine, whose first pose is the
        # state after one step.
        out_q_xyzw = wp.to_torch(state_q)[1:]
        return (
            wp.to_torch(state_x)[1:].clone(),
            out_q_xyzw[..., [3, 0, 1, 2]].contiguous(),
            wp.to_torch(state_xd)[1:].clone(),
            wp.to_torch(state_omega)[1:].clone(),
        )

    @staticmethod
    def backward(ctx, grad_x, grad_q, grad_xd, grad_omega):
        state_x, state_q, state_xd, state_omega = ctx.state

        def seed(arr, g, quat=False):
            if g is None:
                return
            g = g.contiguous()
            if quat:  # WXYZ -> XYZW
                g = g[..., [1, 2, 3, 0]].contiguous()
            # Step 0 is the non-differentiable seeded initial state.
            wp.copy(arr.grad[1:], wp.from_torch(g, dtype=arr.dtype))

        seed(state_x, grad_x)
        seed(state_q, grad_q, quat=True)
        seed(state_xd, grad_xd)
        seed(state_omega, grad_omega)

        ctx.tape.backward()

        def get_grad(name):
            global _WARNED_NONFINITE_GRAD
            arr = ctx.wp_inputs[name]
            if not (arr.requires_grad and arr.grad is not None):
                return None
            g = wp.to_torch(arr.grad).clone().detach()
            nonfinite = int((~torch.isfinite(g)).sum())
            if nonfinite and not _WARNED_NONFINITE_GRAD:
                _WARNED_NONFINITE_GRAD = True
                warnings.warn(
                    f"warp engine: zeroing {nonfinite} non-finite gradient "
                    f"entries in '{name}'. The torch engine would propagate "
                    "them instead (utils/trainer.py zeroes them one level up), "
                    "so the two engines fail differently. Warned once per "
                    "process.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            None,  # init_x: detached plane fit
            None,  # init_q
            None,  # init_xd
            None,  # init_omega
            get_grad("ctrl"),
            get_grad("z"),
            get_grad("n"),
            get_grad("kf"),
            None,  # stiffness: constant, not a gradient path
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
            None,  # device_str
        )


class WarpStaticEngine:
    """Hold the robot constants; state and scratch are allocated per rollout."""

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
            world_config.grid_center,
            world_config.max_coord,
            engine_config.dt,
            num_steps,
            engine_config.g,
            engine_config.damping_alpha,
            float(engine_config.slip_ref_ratio),
            float(engine_config.slip_floor_velocity),
            engine_config.ang_damping,
            engine_config.torque_limit,
            1.0 if engine_config.com_transfer else 0.0,
            str(self.device),
        )
