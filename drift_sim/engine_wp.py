import torch
import warp as wp

# Initialize Warp
wp.init()


# -----------------------------------------------------------------------------
# 1. Helper Math Functions (Device Functions)
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
    """
    Computes grid indices (u, v) matching PyTorch's grid_sample(align_corners=True).
    Maps physical range [center - max_coord, center + max_coord] to [0, width-1].
    """
    # Normalize x to [-1, 1], then map to [0, width-1]
    u = ((x - grid_center[0]) / max_coord[0] + 1.0) * (float(width) - 1.0) * 0.5
    v = ((y - grid_center[1]) / max_coord[1] + 1.0) * (float(height) - 1.0) * 0.5
    return u, v


@wp.func
def length_safe(v: wp.vec3):
    return wp.sqrt(wp.dot(v, v) + 1.0e-6)


@wp.func
def sample_grid_bilinear(
    grid: wp.array2d(dtype=float), u: float, v: float, width: int, height: int
):
    """
    Bilinear sampling with padding_mode='zeros' (PyTorch default).
    Returns 0.0 if sampling outside the grid.
    """
    u_f = wp.floor(u)
    v_f = wp.floor(v)
    u0 = int(u_f)
    v0 = int(v_f)
    u1 = u0 + 1
    v1 = v0 + 1

    w_u = u - float(u0)
    w_v = v - float(v0)

    # Sampling with Zero Padding checks
    val00 = 0.0
    if u0 >= 0 and u0 < width and v0 >= 0 and v0 < height:
        val00 = grid[v0, u0]

    val01 = 0.0
    if u1 >= 0 and u1 < width and v0 >= 0 and v0 < height:
        val01 = grid[v0, u1]

    val10 = 0.0
    if u0 >= 0 and u0 < width and v1 >= 0 and v1 < height:
        val10 = grid[v1, u0]

    val11 = 0.0
    if u1 >= 0 and u1 < width and v1 >= 0 and v1 < height:
        val11 = grid[v1, u1]

    val0 = val00 * (1.0 - w_u) + val01 * w_u
    val1 = val10 * (1.0 - w_u) + val11 * w_u
    return val0 * (1.0 - w_v) + val1 * w_v


@wp.func
def sample_normal_bilinear(
    grid: wp.array2d(dtype=wp.vec3), u: float, v: float, width: int, height: int
):
    """
    Bilinear sampling for normals with padding_mode='zeros'.
    Returns (0,0,0) if out of bounds, preventing phantom forces.
    """
    u_f = wp.floor(u)
    v_f = wp.floor(v)
    u0 = int(u_f)
    v0 = int(v_f)
    u1 = u0 + 1
    v1 = v0 + 1

    w_u = u - float(u0)
    w_v = v - float(v0)

    n00 = wp.vec3(0.0)
    if u0 >= 0 and u0 < width and v0 >= 0 and v0 < height:
        n00 = grid[v0, u0]

    n01 = wp.vec3(0.0)
    if u1 >= 0 and u1 < width and v0 >= 0 and v0 < height:
        n01 = grid[v0, u1]

    n10 = wp.vec3(0.0)
    if u0 >= 0 and u0 < width and v1 >= 0 and v1 < height:
        n10 = grid[v1, u0]

    n11 = wp.vec3(0.0)
    if u1 >= 0 and u1 < width and v1 >= 0 and v1 < height:
        n11 = grid[v1, u1]

    n0 = n00 * (1.0 - w_u) + n01 * w_u
    n1 = n10 * (1.0 - w_u) + n11 * w_u
    n_res = n0 * (1.0 - w_v) + n1 * w_v

    # Robust normalization using length_safe
    return n_res / length_safe(n_res)


@wp.func
def get_wheel_point_local(i: int, total_points: int, radius: float):
    theta = (2.0 * wp.PI * float(i)) / float(total_points)
    c = wp.cos(theta)
    s = wp.sin(theta)
    return wp.vec3(radius * c, 0.0, -radius * s)


@wp.func
def softplus_stable(x: float, beta: float):
    threshold = 20.0
    xb = x * beta
    if xb > threshold:
        return x
    return wp.log(1.0 + wp.exp(xb)) / beta


@wp.func
def integrate_quat(q: wp.quat, w: wp.vec3, dt: float):
    w_len = length_safe(w)
    theta = w_len * dt * 0.5

    # Numerically stable sinc approximation for small angles to avoid 1/w_len explosion
    # With length_safe, w_len >= 1e-3, so division is safe.
    scale = wp.sin(theta) / w_len

    dq_xyz = w * scale
    dq_w = wp.cos(theta)

    # Construct quaternion directly (x, y, z, w)
    dq = wp.quat(dq_xyz[0], dq_xyz[1], dq_xyz[2], dq_w)

    return wp.normalize(dq * q)


# -----------------------------------------------------------------------------
# 2. The Super Kernel (Fused Loop)
# -----------------------------------------------------------------------------
@wp.kernel
def simulate_trajectory_kernel(
    init_x: wp.array(dtype=wp.vec3),
    init_q: wp.array(dtype=wp.quat),
    init_xd: wp.array(dtype=wp.vec3),
    init_omega: wp.array(dtype=wp.vec3),
    control_velocities: wp.array(dtype=wp.vec3, ndim=2),
    wheel_centers: wp.array(dtype=wp.vec3),
    wheel_radii: wp.array(dtype=float),
    inertia_body: wp.array(dtype=wp.mat33),
    cog_body: wp.array(dtype=wp.vec3),
    mass: float,
    z_grid: wp.array2d(dtype=float),
    normals_grid: wp.array2d(dtype=wp.vec3),
    friction_grid: wp.array2d(dtype=float),
    stiffness_grid: wp.array2d(dtype=float),
    grid_center: wp.vec2,
    max_coord: wp.vec2,
    dt: float,
    num_steps: int,
    num_wheel_points: int,
    gravity: float,
    damping_alpha: float,
    soft_contact_sigma: float,
    torque_limit: float,
    out_x: wp.array(dtype=wp.vec3, ndim=2),
    out_q: wp.array(dtype=wp.quat, ndim=2),
):
    tid = wp.tid()

    # Load Initial State
    x = init_x[tid]
    q = init_q[tid]
    xd = init_xd[tid]
    omega = init_omega[tid]

    I_body_val = inertia_body[0]
    cog_val = cog_body[0]

    grid_h = z_grid.shape[0]
    grid_w = z_grid.shape[1]

    m_eff = mass / 3.0

    for step in range(num_steps):
        wheel_omega = control_velocities[step, tid]

        # B. Dynamics
        rot_m = wp.quat_to_matrix(q)
        I_world = rot_m * I_body_val * wp.transpose(rot_m)
        I_inv_world = wp.inverse(I_world)

        F_total = wp.vec3(0.0, 0.0, -mass * gravity)
        T_total = wp.vec3(0.0, 0.0, 0.0)

        # ---------------------------------------------------------------------
        # PASS 1: Contact Counting (Global)
        # ---------------------------------------------------------------------
        total_active_contacts = float(0.0)

        for w in range(3):
            w_r = wheel_radii[w]
            w_c_local = wheel_centers[w]
            w_c_world = x + wp.quat_rotate(q, w_c_local)

            for i in range(num_wheel_points):
                p_rel = get_wheel_point_local(i, num_wheel_points, w_r)
                p_world = w_c_world + wp.quat_rotate(q, p_rel)

                u, v = compute_uv_aligned(
                    p_world[0], p_world[1], grid_center, max_coord, grid_w, grid_h
                )

                z_surf = sample_grid_bilinear(z_grid, u, v, grid_w, grid_h)
                n_surf = sample_normal_bilinear(normals_grid, u, v, grid_w, grid_h)

                dist = (p_world[2] - z_surf) * n_surf[2]
                if dist < 0.0:
                    total_active_contacts += 1.0

        # ---------------------------------------------------------------------
        # Damping Parameter Calculation
        # ---------------------------------------------------------------------
        contact_count_safe = total_active_contacts
        if contact_count_safe < 1.0:
            contact_count_safe = 1.0

        # Pre-calculate damping constant
        damp_const = damping_alpha * 2.0 * wp.sqrt(mass / contact_count_safe)
        com_world = x + wp.quat_rotate(q, cog_val)

        # ---------------------------------------------------------------------
        # PASS 2: Force Accumulation (All Points)
        # ---------------------------------------------------------------------
        for w in range(3):
            w_r = wheel_radii[w]
            w_c_local = wheel_centers[w]

            w_speed = wheel_omega[w]
            w_vec_local = wp.vec3(0.0, w_speed, 0.0)
            w_vec_world = wp.quat_rotate(q, w_vec_local)
            w_center_world = x + wp.quat_rotate(q, w_c_local)

            for i in range(num_wheel_points):
                p_rel = get_wheel_point_local(i, num_wheel_points, w_r)
                p_world = w_center_world + wp.quat_rotate(q, p_rel)

                # Re-sample grid (necessary to avoid large local memory usage)
                u, v = compute_uv_aligned(
                    p_world[0], p_world[1], grid_center, max_coord, grid_w, grid_h
                )
                z_surf = sample_grid_bilinear(z_grid, u, v, grid_w, grid_h)
                n_surf = sample_normal_bilinear(normals_grid, u, v, grid_w, grid_h)
                f_val = sample_grid_bilinear(friction_grid, u, v, grid_w, grid_h)
                k_s = sample_grid_bilinear(stiffness_grid, u, v, grid_w, grid_h)

                dist = (p_world[2] - z_surf) * n_surf[2]

                if dist < 0.0:
                    # Physics
                    r_w_c = p_world - w_center_world
                    r_com_p = p_world - com_world

                    v_rigid = xd + wp.cross(omega, r_com_p)
                    v_spin = wp.cross(w_vec_world, r_w_c)
                    v_total = v_rigid + v_spin
                    vn = wp.dot(v_total, n_surf)

                    b_damp = damp_const * wp.sqrt(k_s)
                    f_raw = -k_s * dist - b_damp * vn
                    beta = 1.0 / soft_contact_sigma
                    f_mag = softplus_stable(f_raw, beta)

                    f_mag = f_mag / contact_count_safe
                    F_spring = n_surf * f_mag

                    v_mag = length_safe(v_total)
                    v_dir = v_total / (v_mag + 1.0e-6)
                    F_N = wp.dot(F_spring, n_surf)
                    if F_N < 0.0:
                        F_N = 0.0

                    f_c_mag = f_val * F_N * wp.tanh(v_mag)
                    f_imp_lim = m_eff * v_mag / dt
                    if f_c_mag > f_imp_lim:
                        f_c_mag = f_imp_lim
                    F_fric = v_dir * f_c_mag
                    if f_mag <= 1e-6:
                        F_fric = wp.vec3(0.0)

                    F_point = F_spring - F_fric
                    F_total = F_total + F_point

                    tau_p = wp.cross(r_com_p, F_point)
                    T_total = T_total + tau_p

        # Integration
        ang_damping = 0.1
        T_total = T_total - ang_damping * omega
        t_mag = length_safe(T_total)
        if t_mag > torque_limit:
            T_total = T_total * (torque_limit / t_mag)
        xdd = F_total / mass
        Iw = I_world * omega
        w_x_Iw = wp.cross(omega, Iw)
        omega_d = I_inv_world * (T_total - w_x_Iw)

        xd = xd + xdd * dt
        omega = omega + omega_d * dt
        x = x + xd * dt
        q = integrate_quat(q, omega, dt)

        out_x[step, tid] = x
        out_q[step, tid] = q


# -----------------------------------------------------------------------------
# 2.5 Autograd Function
# -----------------------------------------------------------------------------
class WarpTrajectoryFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        init_x,
        init_q,  # WXYZ
        init_xd,
        init_omega,
        control_velocities,
        z_grid_t,
        normals_grid_t,
        friction_grid_t,
        stiffness_grid_t,
        wheel_centers_wp,
        wheel_radii_wp,
        inertia_wp,
        cog_wp,
        mass,
        grid_center_t,
        max_coord_t,
        dt,
        num_steps,
        num_wheel_points,
        g,
        damping_alpha,
        soft_contact_sigma,
        torque_limit,
        device_str,
    ):
        device = wp.get_device(device_str)
        B = init_x.shape[0]

        # Convert Torch -> Warp (zero-copy view)
        # Use .detach() to isolate gradient buffers
        wp_x = wp.from_torch(
            init_x.detach(), dtype=wp.vec3, requires_grad=init_x.requires_grad
        )

        # WXYZ -> XYZW
        init_q_xyzw = init_q.detach()[:, [1, 2, 3, 0]].contiguous()
        wp_q = wp.from_torch(
            init_q_xyzw, dtype=wp.quat, requires_grad=init_q.requires_grad
        )

        wp_xd = wp.from_torch(
            init_xd.detach(), dtype=wp.vec3, requires_grad=init_xd.requires_grad
        )
        wp_w = wp.from_torch(
            init_omega.detach(), dtype=wp.vec3, requires_grad=init_omega.requires_grad
        )

        wp_ctrl = wp.from_torch(
            control_velocities.detach(),
            dtype=wp.vec3,
            requires_grad=control_velocities.requires_grad,
        )

        wp_z = wp.from_torch(
            z_grid_t.detach(), dtype=wp.float32, requires_grad=z_grid_t.requires_grad
        )
        wp_n = wp.from_torch(
            normals_grid_t.detach(),
            dtype=wp.vec3,
            requires_grad=normals_grid_t.requires_grad,
        )
        wp_kf = wp.from_torch(
            friction_grid_t.detach(),
            dtype=wp.float32,
            requires_grad=friction_grid_t.requires_grad,
        )
        wp_ks = wp.from_torch(
            stiffness_grid_t.detach(),
            dtype=wp.float32,
            requires_grad=stiffness_grid_t.requires_grad,
        )

        # grid_center and max_coord are small tensors (2,), convert to wp.vec2
        gc_wp = wp.vec2(float(grid_center_t[0]), float(grid_center_t[1]))
        mc_wp = wp.vec2(float(max_coord_t[0]), float(max_coord_t[1]))

        wp_out_x = wp.zeros(
            (num_steps, B), dtype=wp.vec3, device=device, requires_grad=True
        )
        wp_out_q = wp.zeros(
            (num_steps, B), dtype=wp.quat, device=device, requires_grad=True
        )

        ctx.tape = wp.Tape()
        with ctx.tape:
            wp.launch(
                kernel=simulate_trajectory_kernel,
                dim=B,
                inputs=[
                    wp_x,
                    wp_q,
                    wp_xd,
                    wp_w,
                    wp_ctrl,
                    wheel_centers_wp,
                    wheel_radii_wp,
                    inertia_wp,
                    cog_wp,
                    mass,
                    wp_z,
                    wp_n,
                    wp_kf,
                    wp_ks,
                    gc_wp,
                    mc_wp,
                    dt,
                    num_steps,
                    num_wheel_points,
                    g,
                    damping_alpha,
                    soft_contact_sigma,
                    torque_limit,
                ],
                outputs=[wp_out_x, wp_out_q],
                device=device,
            )

        ctx.wp_inputs = {
            "x": wp_x,
            "q": wp_q,
            "xd": wp_xd,
            "w": wp_w,
            "ctrl": wp_ctrl,
            "z": wp_z,
            "n": wp_n,
            "kf": wp_kf,
            "ks": wp_ks,
        }
        ctx.wp_outputs = [wp_out_x, wp_out_q]

        # Permute Output Quaternion XYZW -> WXYZ
        out_q_xyzw = wp.to_torch(wp_out_q)
        out_q_wxyz = out_q_xyzw[:, :, [3, 0, 1, 2]]

        return wp.to_torch(wp_out_x), out_q_wxyz

    @staticmethod
    def backward(ctx, grad_out_x, grad_out_q):
        grads = {}
        if grad_out_x is not None:
            grads[ctx.wp_outputs[0]] = wp.from_torch(
                grad_out_x.contiguous(), dtype=wp.vec3
            )
        if grad_out_q is not None:
            # WXYZ -> XYZW
            grad_q_xyzw = grad_out_q[..., [1, 2, 3, 0]].contiguous()
            grads[ctx.wp_outputs[1]] = wp.from_torch(grad_q_xyzw, dtype=wp.quat)

        ctx.tape.backward(grads=grads)

        def get_grad(name):
            wpa = ctx.wp_inputs[name]
            if wpa.requires_grad and wpa.grad is not None:
                g = wp.to_torch(wpa.grad)
                return g.clone().detach()
            return None

        g_q = get_grad("q")
        if g_q is not None:
            # XYZW -> WXYZ
            g_q = g_q[..., [3, 0, 1, 2]]

        return (
            get_grad("x"),
            g_q,
            get_grad("xd"),
            get_grad("w"),
            get_grad("ctrl"),
            get_grad("z"),
            get_grad("n"),
            get_grad("kf"),
            get_grad("ks"),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class WarpStaticEngine:
    def __init__(self, robot_model_config, batch_size, max_steps, device="cuda"):
        self.device = wp.get_device(device)
        self.batch_size = batch_size
        self.max_steps = max_steps

        # A. Static Constants
        self.mass = float(robot_model_config.mass)
        self.wp_wheel_radii = wp.from_torch(
            robot_model_config.wheel_radii.to(device).float(), dtype=wp.float32
        )
        self.wp_wheel_centers = wp.from_torch(
            robot_model_config.wheel_centers.to(device).float(), dtype=wp.vec3
        )
        self.wp_inertia = wp.from_torch(
            robot_model_config.inertia_tensor.unsqueeze(0).to(device).float(),
            dtype=wp.mat33,
        )
        self.wp_cog = wp.from_torch(
            robot_model_config.cog.unsqueeze(0).to(device).float(), dtype=wp.vec3
        )

        # B. Static Buffers
        self.static_x = wp.zeros(batch_size, dtype=wp.vec3, device=self.device)
        self.static_q = wp.zeros(batch_size, dtype=wp.quat, device=self.device)
        self.static_xd = wp.zeros(batch_size, dtype=wp.vec3, device=self.device)
        self.static_w = wp.zeros(batch_size, dtype=wp.vec3, device=self.device)

        self.static_controls = wp.zeros(
            (max_steps, batch_size), dtype=wp.vec3, device=self.device
        )
        self.static_out_x = wp.zeros(
            (max_steps, batch_size), dtype=wp.vec3, device=self.device
        )
        self.static_out_q = wp.zeros(
            (max_steps, batch_size), dtype=wp.quat, device=self.device
        )

        self.cached_z = None
        self.cached_n = None
        self.cached_kf = None
        self.cached_ks = None

    def _update_world_cache(self, world_config):
        if self.cached_z is None:
            z_t = world_config.z_grid[0]
            self.cached_z = wp.from_torch(
                z_t, dtype=wp.float32, requires_grad=z_t.requires_grad
            )

            n_t = world_config.normals[0].permute(1, 2, 0).contiguous()
            self.cached_n = wp.from_torch(
                n_t, dtype=wp.vec3, requires_grad=n_t.requires_grad
            )

            if isinstance(world_config.k_friction, torch.Tensor):
                kf_t = world_config.k_friction[0]
                self.cached_kf = wp.from_torch(
                    kf_t, dtype=wp.float32, requires_grad=kf_t.requires_grad
                )
            else:
                fric_t = torch.full_like(
                    world_config.z_grid[0], world_config.k_friction
                )
                self.cached_kf = wp.from_torch(
                    fric_t, dtype=wp.float32, requires_grad=False
                )

            if isinstance(world_config.k_stiffness, torch.Tensor):
                ks_t = world_config.k_stiffness[0]
                self.cached_ks = wp.from_torch(
                    ks_t, dtype=wp.float32, requires_grad=ks_t.requires_grad
                )
            else:
                stiff_t = torch.full_like(
                    world_config.z_grid[0], world_config.k_stiffness
                )
                self.cached_ks = wp.from_torch(
                    stiff_t, dtype=wp.float32, requires_grad=False
                )

    def run_trajectory(
        self, init_state, control_velocities, world_config, engine_config
    ):
        num_steps = control_velocities.shape[0]

        requires_grad = (
            control_velocities.requires_grad
            or world_config.z_grid.requires_grad
            or (
                isinstance(world_config.k_friction, torch.Tensor)
                and world_config.k_friction.requires_grad
            )
            or (
                isinstance(world_config.k_stiffness, torch.Tensor)
                and world_config.k_stiffness.requires_grad
            )
        )

        if requires_grad:
            z_t = world_config.z_grid[0]
            n_t = world_config.normals[0].permute(1, 2, 0).contiguous()
            kf_t = world_config.k_friction[0]
            ks_t = world_config.k_stiffness[0]

            return WarpTrajectoryFunc.apply(
                init_state.x,
                init_state.q,
                init_state.xd,
                init_state.omega,
                control_velocities,
                z_t,
                n_t,
                kf_t,
                ks_t,
                self.wp_wheel_centers,
                self.wp_wheel_radii,
                self.wp_inertia,
                self.wp_cog,
                self.mass,
                world_config.grid_center,
                world_config.max_coord,
                engine_config.dt,
                num_steps,
                16,
                engine_config.g,
                engine_config.damping_alpha,
                engine_config.soft_contact_sigma
                if engine_config.soft_contact_sigma
                else 0.01,
                engine_config.torque_limit,
                str(self.device),
            )

        if num_steps > self.max_steps:
            raise ValueError(f"Steps {num_steps} > Max Allocated {self.max_steps}")

        self._update_world_cache(world_config)

        # Copy State
        wp.copy(self.static_x, wp.from_torch(init_state.x, dtype=wp.vec3))
        q_xyzw = init_state.q[:, [1, 2, 3, 0]].contiguous()
        wp.copy(self.static_q, wp.from_torch(q_xyzw, dtype=wp.quat))
        wp.copy(self.static_xd, wp.from_torch(init_state.xd, dtype=wp.vec3))
        wp.copy(self.static_w, wp.from_torch(init_state.omega, dtype=wp.vec3))

        wp_ctrl_view = wp.from_torch(control_velocities.contiguous(), dtype=wp.vec3)
        wp.copy(self.static_controls[:num_steps], wp_ctrl_view)

        # grid_center and max_coord are small tensors (2,), convert to wp.vec2
        gc_wp = wp.vec2(
            float(world_config.grid_center[0]), float(world_config.grid_center[1])
        )
        mc_wp = wp.vec2(
            float(world_config.max_coord[0]), float(world_config.max_coord[1])
        )

        wp.launch(
            kernel=simulate_trajectory_kernel,
            dim=self.batch_size,
            inputs=[
                self.static_x,
                self.static_q,
                self.static_xd,
                self.static_w,
                self.static_controls,
                self.wp_wheel_centers,
                self.wp_wheel_radii,
                self.wp_inertia,
                self.wp_cog,
                self.mass,
                self.cached_z,
                self.cached_n,
                self.cached_kf,
                self.cached_ks,
                gc_wp,
                mc_wp,
                engine_config.dt,
                num_steps,
                16,
                engine_config.g,
                engine_config.damping_alpha,
                engine_config.soft_contact_sigma
                if engine_config.soft_contact_sigma
                else 0.01,
                engine_config.torque_limit,
            ],
            outputs=[self.static_out_x, self.static_out_q],
            device=self.device,
        )

        out_x = wp.to_torch(self.static_out_x)[:num_steps]
        out_q_xyzw = wp.to_torch(self.static_out_q)[:num_steps]
        out_q_wxyz = out_q_xyzw[..., [3, 0, 1, 2]]

        return out_x, out_q_wxyz
