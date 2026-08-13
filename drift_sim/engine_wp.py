"""Warp port of the spring-contact engine in simulator/engine.py.

Same physics, ~10x cheaper per rollout. The two engines are kept formula-for-formula
identical on purpose — tests/test_engine_parity.py checks forward poses *and*
gradients between them, and any deliberate divergence here shows up there.

Structure, and why it matters
-----------------------------
The rollout is driven by **one kernel launch per timestep, recorded on a wp.Tape**,
with the robot state living in step-indexed arrays (`state_x[step]`, ...) rather
than in kernel-local variables.

This is not a style choice. Warp only unrolls loops whose `range()` arguments are
compile-time constants (`warp.config.max_unroll`, default 16). A loop it cannot
unroll becomes a *dynamic* loop, and for a dynamic loop that mutates variables read
in later iterations Warp falls back to `materialize_redefinitions`, which — in its
own words — "violates SSA": the reverse pass then uses each variable's end-of-loop
value in place of its per-iteration value. An earlier version of this file ran the
entire 1000-step rollout inside a single kernel that way. Its forward matched the
torch engine to 3.3e-4 m while its gradients were cosine -0.01 against torch at 100
steps and NaN by 250 — silently, because nothing compared them.

Keeping each step in its own launch means every launch's primal inputs are still
resident in the state arrays when the tape replays the adjoint, so the reverse pass
is exact. The per-point loops are gone entirely: contact points are threads
(`dim=(B, NUM_WHEELS, NUM_WHEEL_POINTS)`), which removes the second dynamic loop and
also fixes occupancy — the old kernel launched with `dim=B`, so at the batch size
used for online adaptation (B=1) the whole rollout ran on a single CUDA thread.

`tests/test_engine_parity.py::test_warp_kernels_have_no_dynamic_loops` statically
forbids reintroducing either loop.
"""

import torch
import warp as wp

wp.init()

# Compile-time constants, so the loops that use them unroll. Do NOT turn these
# back into kernel parameters (see module docstring).
NUM_WHEELS = wp.constant(3)
NUM_WHEEL_POINTS = wp.constant(16)


# -----------------------------------------------------------------------------
# 1. Helper math (device functions)
# -----------------------------------------------------------------------------
@wp.func
def compute_uv_aligned(
    x: float,
    y: float,
    grid_center: wp.vec2,
    max_coord: wp.vec2,
    width: int,
    height: int,
):
    """Grid indices (u, v) matching PyTorch's grid_sample(align_corners=True).

    Maps physical range [center - max_coord, center + max_coord] to [0, width-1].
    """
    u = ((x - grid_center[0]) / max_coord[0] + 1.0) * (float(width) - 1.0) * 0.5
    v = ((y - grid_center[1]) / max_coord[1] + 1.0) * (float(height) - 1.0) * 0.5
    return u, v


@wp.func
def sample_grid_bilinear(
    grid: wp.array3d(dtype=float), b: int, u: float, v: float, width: int, height: int
):
    """Bilinear sample of batch element `b`, replicating the border.

    padding_mode="border" matches simulator/engine.py's grid_sample call; zero
    padding would put a phantom cliff at the edge of the property map.
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
    """Bilinear sample of a normal field, renormalized after interpolation."""
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
def softplus_stable(x: float, sigma: float):
    """Soft contact force. sigma <= 0 falls back to hard contact (ReLU)."""
    if sigma <= 0.0:
        return wp.max(x, 0.0)
    beta = 1.0 / sigma
    xb = x * beta
    if xb > 20.0:  # exp() would overflow; softplus is the identity out here
        return x
    return wp.log(1.0 + wp.exp(xb)) / beta


@wp.func
def sinc_stable(theta: float):
    """sin(theta)/theta, finite and differentiable at theta = 0.

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


# -----------------------------------------------------------------------------
# 2. Per-timestep kernels
# -----------------------------------------------------------------------------
@wp.kernel
def count_contacts_kernel(
    step: int,
    state_x: wp.array2d(dtype=wp.vec3),
    state_q: wp.array2d(dtype=wp.quat),
    wheel_centers: wp.array(dtype=wp.vec3),
    wheel_radii: wp.array(dtype=float),
    z_grid: wp.array3d(dtype=float),
    normals_grid: wp.array3d(dtype=wp.vec3),
    grid_center: wp.vec2,
    max_coord: wp.vec2,
    contact_counts: wp.array2d(dtype=float),
):
    """Pass 1: how many sample points are penetrating, per robot.

    The spring force is divided by this count, so it must be known before any
    force is accumulated — hence a separate launch rather than a fused kernel.
    The count is a step function of the state and carries no gradient, exactly
    as in the torch engine (`(dh < 0).float()`).
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

    if (p_world[2] - z_surf) * n_surf[2] < 0.0:
        wp.atomic_add(contact_counts, step, b, 1.0)


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
    z_grid: wp.array3d(dtype=float),
    normals_grid: wp.array3d(dtype=wp.vec3),
    friction_grid: wp.array3d(dtype=float),
    stiffness_grid: wp.array3d(dtype=float),
    grid_center: wp.vec2,
    max_coord: wp.vec2,
    dt: float,
    damping_alpha: float,
    soft_contact_sigma: float,
    contact_counts: wp.array2d(dtype=float),
    force_acc: wp.array2d(dtype=wp.vec3),
    torque_acc: wp.array2d(dtype=wp.vec3),
):
    """Pass 2: spring + friction force and torque of one contact point."""
    b, w, i = wp.tid()

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
    if dist >= 0.0:
        return

    f_val = sample_grid_bilinear(friction_grid, b, u, v, grid_w, grid_h)
    k_s = sample_grid_bilinear(stiffness_grid, b, u, v, grid_w, grid_h)

    count = wp.max(contact_counts[step, b], 1.0)
    com_world = x + wp.quat_rotate(q, cog)

    # Contact-point velocity: body motion + wheel spin about the wheel's own axis
    wheel_omega_world = wp.quat_rotate(q, wp.vec3(0.0, controls[step, b][w], 0.0))
    v_total = (
        xd
        + wp.cross(omega, p_world - x)
        + wp.cross(wheel_omega_world, p_world - w_center_world)
    )
    vn = wp.dot(v_total, n_surf)

    b_damp = damping_alpha * 2.0 * wp.sqrt(mass) * wp.sqrt(k_s)
    f_mag = softplus_stable(-k_s * dist - b_damp * vn, soft_contact_sigma) / count
    F_spring = n_surf * f_mag

    # Friction is tangential: strip the normal component of the contact velocity
    # first. Using the full 3D velocity injects a spurious normal term that can
    # launch the robot off the terrain (same reasoning as simulator/engine.py).
    v_tangential = v_total - vn * n_surf
    v_mag = wp.sqrt(wp.dot(v_tangential, v_tangential) + 1.0e-12)
    v_dir = v_tangential / (v_mag + 1.0e-6)

    F_N = wp.max(wp.dot(F_spring, n_surf), 0.0)
    f_c_mag = wp.min(f_val * F_N * wp.tanh(v_mag), mass * v_mag / 3.0 / dt)

    F_fric = v_dir * f_c_mag
    if f_mag <= 1.0e-6:
        F_fric = wp.vec3(0.0, 0.0, 0.0)

    F_point = F_spring - F_fric
    wp.atomic_add(force_acc, step, b, F_point)
    wp.atomic_add(torque_acc, step, b, wp.cross(p_world - com_world, F_point))


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
    force_acc: wp.array2d(dtype=wp.vec3),
    torque_acc: wp.array2d(dtype=wp.vec3),
):
    """Semi-implicit Euler step: state[step] + accumulated wrench -> state[step+1]."""
    b = wp.tid()

    x = state_x[step, b]
    q = state_q[step, b]
    xd = state_xd[step, b]
    omega = state_omega[step, b]

    rot_m = wp.quat_to_matrix(q)
    I_world = rot_m * inertia_body * wp.transpose(rot_m)
    # Closed form beats wp.inverse(I_world): cheaper, and better conditioned in
    # the adjoint since it never differentiates a matrix inversion.
    I_inv_world = rot_m * inertia_body_inv * wp.transpose(rot_m)

    F_total = force_acc[step, b] + wp.vec3(0.0, 0.0, -mass * gravity)

    T_total = torque_acc[step, b] - ang_damping * omega
    t_mag = wp.sqrt(wp.dot(T_total, T_total) + 1.0e-12)
    T_total = T_total * wp.min(torque_limit / (t_mag + 1.0e-6), 1.0)

    omega_d = I_inv_world * (T_total - wp.cross(omega, I_world * omega))

    next_xd = xd + (F_total / mass) * dt
    next_omega = omega + omega_d * dt
    # Matches the torch engine's clamps; a runaway step would otherwise poison
    # every later step of the rollout and its adjoint.
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


# -----------------------------------------------------------------------------
# 3. Autograd bridge
# -----------------------------------------------------------------------------
class WarpTrajectoryFunc(torch.autograd.Function):
    """Rolls the trajectory on a wp.Tape and hands the adjoint back to torch.

    Stiffness is passed through as a plain input with no gradient: it is not a
    predicted or adaptable property (see ADAPT_HEADS in utils/experiment.py), so
    taping it would only add work.
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
        soft_contact_sigma,
        ang_damping,
        torque_limit,
        device_str,
        clip_grad_norm,
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

        # State trajectories: one slot per step, so the tape's reverse pass finds
        # every launch's primal inputs still intact.
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

        # Per-step accumulators, likewise one slot per step: reusing a single
        # buffer would need a zeroing pass that the tape cannot differentiate.
        counts = wp.zeros(
            (num_steps, B), dtype=float, device=device, requires_grad=True
        )
        force_acc = wp.zeros(
            (num_steps, B), dtype=wp.vec3, device=device, requires_grad=True
        )
        torque_acc = wp.zeros(
            (num_steps, B), dtype=wp.vec3, device=device, requires_grad=True
        )

        gc_wp = wp.vec2(float(grid_center_t[0]), float(grid_center_t[1]))
        mc_wp = wp.vec2(float(max_coord_t[0]), float(max_coord_t[1]))
        point_dim = (B, int(NUM_WHEELS), int(NUM_WHEEL_POINTS))

        ctx.tape = wp.Tape()
        with ctx.tape:
            for step in range(num_steps):
                wp.launch(
                    count_contacts_kernel,
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
                    outputs=[counts],
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
                        wp_z,
                        wp_n,
                        wp_kf,
                        wp_ks,
                        gc_wp,
                        mc_wp,
                        dt,
                        damping_alpha,
                        soft_contact_sigma,
                        counts,
                    ],
                    outputs=[force_acc, torque_acc],
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
                        force_acc,
                        torque_acc,
                    ],
                    device=device,
                )

        ctx.wp_inputs = {"ctrl": wp_ctrl, "z": wp_z, "n": wp_n, "kf": wp_kf}
        ctx.state = (state_x, state_q, state_xd, state_omega)
        ctx.clip_grad_norm = clip_grad_norm

        # Drop the seeded step 0 so the outputs line up with the torch engine,
        # whose first returned pose is the state *after* one step.
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
            # step 0 is the (non-differentiable) seeded initial state
            wp.copy(arr.grad[1:], wp.from_torch(g, dtype=arr.dtype))

        seed(state_x, grad_x)
        seed(state_q, grad_q, quat=True)
        seed(state_xd, grad_xd)
        seed(state_omega, grad_omega)

        ctx.tape.backward()

        def get_grad(name):
            arr = ctx.wp_inputs[name]
            if not (arr.requires_grad and arr.grad is not None):
                return None
            g = wp.to_torch(arr.grad).clone().detach()
            g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            if ctx.clip_grad_norm is not None:
                norm = g.norm()
                if norm > ctx.clip_grad_norm:
                    g = g * (ctx.clip_grad_norm / norm)
            return g

        return (
            None,  # init_x   — seeded by a detached plane fit, see models/simulator_wp.py
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
            None,  # soft_contact_sigma
            None,  # ang_damping
            None,  # torque_limit
            None,  # device_str
            None,  # clip_grad_norm
        )


class WarpStaticEngine:
    """Holds the robot constants; state and scratch are allocated per rollout."""

    def __init__(
        self,
        robot_model_config,
        batch_size,
        max_steps,
        device="cuda",
        clip_grad_norm=None,
    ):
        self.device = wp.get_device(device)
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.clip_grad_norm = clip_grad_norm

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
        """control_velocities: (num_steps, B, 3) wheel speeds. Returns (T,B,...)."""
        num_steps = control_velocities.shape[0]
        if num_steps > self.max_steps:
            raise ValueError(f"Steps {num_steps} > Max Allocated {self.max_steps}")

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
            engine_config.soft_contact_sigma
            if engine_config.soft_contact_sigma is not None
            else 0.0,
            engine_config.ang_damping,
            engine_config.torque_limit,
            str(self.device),
            self.clip_grad_norm,
        )
