import torch
import torch.nn.functional as F

from simulator.numerical import integrate_quaternion
from simulator.utils import (
    PhysicsEngineConfig,
    PhysicsState,
    PhysicsStateDer,
    RobotModelConfig,
    WorldConfig,
)
from utils.geometry import quat_to_rotmat

# Rim sample points per wheel, shared with simulator/engine_wp.py. The polygon
# chord costs r (1 - cos(pi / N)) of radius where the local ground tilts against
# the body: 6.4 mm at 16 (~ the 7.3 mm tyre sag), 1.6 mm at 32.
NUM_WHEEL_POINTS = 32
# Version of the contact and friction law in both engines. Bump it with any change
# to how contact forces are computed: run records store it (utils.misc.
# robot_constants), and checkpoints from another law are refused for scoring.
CONTACT_LAW = "wheel-deepest-point-v2"

# --- Rigid-body model ---
# COG and wheel triangle are parameters; contact forces do not spin the wheels
# and are transferred straight to the axle.


class DPhysicsEngine(torch.nn.Module):
    def __init__(
        self,
        config: PhysicsEngineConfig,
        robot_model: RobotModelConfig,
    ):
        super().__init__()
        self.config = config
        self.robot_model = robot_model
        self.register_buffer(
            "_wheel_pos_rel", self._precompute_wheel_points(NUM_WHEEL_POINTS)
        )
        self.register_buffer("_I_body", robot_model.inertia_tensor.float())
        self.register_buffer(
            "_I_body_inv", torch.linalg.inv(robot_model.inertia_tensor.float())
        )
        self.register_buffer("_wheel_centers", robot_model.wheel_centers.float())
        self.register_buffer("_wheel_radii", robot_model.wheel_radii.float())
        self.register_buffer("_cog", robot_model.cog.float())
        self._mass = robot_model.mass
        F_g = torch.tensor(
            [0.0, 0.0, -self._mass * self.config.g],
        ).unsqueeze(0)
        self.register_buffer("_F_g", F_g)

    def _precompute_wheel_points(self, n_points: int) -> torch.Tensor:
        """Precompute wheel points relative to wheel centers."""
        # Drop the endpoint, which duplicates theta=0.
        theta = torch.linspace(0, 2 * torch.pi, n_points + 1)[:-1]
        n_wheels = 3

        wheel_radii = self.robot_model.wheel_radii  # (3,)

        y = torch.zeros((n_wheels, n_points))
        x = wheel_radii[:, None] * torch.cos(theta)[None, :]
        z = -wheel_radii[:, None] * torch.sin(theta)[None, :]

        # (n_wheels, n_points, 3)
        return torch.stack([x, y, z], dim=-1)

    def _sample_terrain(
        self,
        wheel_points: torch.Tensor,
        world_config: WorldConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample all terrain properties at the wheel points in one grid_sample.

        Uses the precomputed 6-channel ``world_config._sample_grid``
        (z, nx, ny, nz, friction, stiffness). Points off the map read the border
        cell, so a rollout leaving the grid continues on the edge height silently.

        Args:
            wheel_points:  (B, W, P, 3) world-frame positions of wheel sample points
            world_config:  WorldConfig with ``_sample_grid`` of shape (B, 6, H, W)

        Returns:
            dh:           (B, W, P, 1) penetration depth (negative = in contact)
            normals:      (B, W, P, 3) unit surface normals
            stif_coeffs:  (B, W, P)   stiffness values per sample point
            fric_coeffs:  (B, W, P)   friction values per sample point
        """
        B, W, P, _ = wheel_points.shape

        xy = wheel_points[..., :2] - world_config.grid_center
        xy_norm = (xy / world_config.max_coord).clamp(-1.0, 1.0)  # (B, W, P, 2)
        query = xy_norm.reshape(B, 1, W * P, 2)  # one-row output grid

        sampled = F.grid_sample(
            world_config._sample_grid,
            query,
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )  # (B, 6, 1, W*P)

        sampled = sampled.squeeze(2).permute(0, 2, 1).reshape(B, W, P, 6)

        z_terrain = sampled[..., :1]  # (B, W, P, 1)
        normals = sampled[..., 1:4]  # (B, W, P, 3)
        fric_coeffs = sampled[..., 4]  # (B, W, P)
        stif_coeffs = sampled[..., 5]  # (B, W, P)

        # Bilinear interpolation denormalizes the normals.
        normals = F.normalize(normals, dim=-1, eps=1e-6)

        # Penetration depth: vertical gap scaled by the normal's z component.
        dh = (wheel_points[..., 2:3] - z_terrain) * normals[..., 2:3]  # (B, W, P, 1)

        return dh, normals, stif_coeffs, fric_coeffs

    def _calculate_spring_force(
        self,
        dh_points: torch.Tensor,
        xd_points: torch.Tensor,
        normals: torch.Tensor,
        stiffness: torch.Tensor,
        in_contact: torch.Tensor,
        r: torch.Tensor,
        I_world_inv: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the damped contact spring force on the wheel points.

        Each wheel is a spring of k / n_wheels on its DEEPEST point's depth D,
        shared over its penetrating points by depth (point i carries
        d_i D / sum d): flat ground gives the whole-robot k (sag m g / k) at any
        point count. Summing a k / n_wheels spring per point instead stiffened the
        wheel with every extra rim point under the sag (at 32 points the
        neighbours sit 6.4 mm up, inside v2's 7.3 mm sag). A point starts
        touching at zero share, so the force stays continuous.

        A point pushes relu(depth k / n_wheels - c v_n) along the normal, with
        Hunt-Crossley damping proportional to depth, c = depth lam / n_wheels,
        lam = b k / (m g), b = alpha 2 sqrt(m k): uncapped, the robot's total is
        b, alpha times critical heave damping, at the rest sag depth m g / k
        only, and zero at zero depth, as for a real tyre with no deflection. A
        depth-independent b v_n made the force jump by b |v_n| / n_wheels at
        every touchdown, which the analytic gradient cannot see: support
        gradients came out 5-35 % low and, from a seated start on rough ground,
        pointed the wrong way (cos -0.28 at 0.25 s).

        c is capped per point at share m_eff_n / dt, so the damping impulse can
        at most stop the point's normal velocity through its normal effective
        mass, 1 / m_eff_n = 1 / m + (r x n) . I^-1 (r x n), in one step; share
        is the point's depth over the robot's total, as in the friction impulse
        limit (_friction). Uncapped, c grows with depth while the velocity
        update is explicit: past ~56 mm / alpha of sinkage (helhest_v2; ~18 mm
        / alpha on helhest) the step is unstable, and while the forward stays
        finite under the +-50 clamps the adjoint explodes (gradients 1e13-1e17,
        NaN; tests/test_contact_damping_regression.py). At alpha 2.0 the cap
        already binds at rest on helhest and on helhest_v2's front wheels
        (m_eff_n ~35 kg). Local, no solve, and still zero at zero depth: the
        force is continuous in the terrain, so `in_contact` only masks.
        """
        # The robot's tyre constant, not a gradient path (warp has none either).
        k = stiffness.detach().unsqueeze(-1)  # (B, W, P, 1)
        b = self.config.damping_alpha * 2 * (self._mass * k) ** 0.5
        lam = b * k / (self._mass * self.config.g)
        n_wheels = self._wheel_radii.shape[0]

        xd_points_n = (xd_points * normals).sum(dim=-1, keepdim=True)

        # relu makes it unilateral (the ground only pushes).
        depth_raw = (-dh_points).clamp_min(0)
        D = depth_raw.amax(dim=2, keepdim=True)  # (B, W, 1, 1)
        depth = depth_raw * D / depth_raw.sum(dim=2, keepdim=True).clamp_min(1e-12)

        r_x_n = torch.linalg.cross(r, normals)
        inv_m_eff_n = 1.0 / self._mass + (
            r_x_n * torch.einsum("bij,bwpj->bwpi", I_world_inv, r_x_n)
        ).sum(dim=-1, keepdim=True)
        m_eff_n = 1.0 / inv_m_eff_n
        share = depth_raw / depth_raw.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        damping_unlimited = depth * lam * xd_points_n.abs() / n_wheels
        damping_limit = share * m_eff_n * xd_points_n.abs() / self.config.dt
        damping = torch.minimum(damping_unlimited, damping_limit)
        F_spring_mag = torch.relu(
            depth * k / n_wheels - damping * torch.sign(xd_points_n)
        )

        F_spring = F_spring_mag * normals

        return F_spring * in_contact

    def _contact_terms(self, state: PhysicsState, world_config: WorldConfig) -> dict:
        """Spring and friction terms of every wheel sample point.

        Returns a dict with the spring force, the friction law ``F_c`` (tanh,
        uncapped), the impulse limit ``F_lim``, the mask ``capped`` where the cap
        applies, the applied magnitude ``F_fric``, and the geometry forward needs.
        Shared by forward and tests/test_engine_invariants.py.
        """
        R = quat_to_rotmat(state.q)
        R_T = R.transpose(-1, -2)

        wheel_centers = self._wheel_centers
        wheel_centers_world = (R @ wheel_centers.T).transpose(
            -1, -2
        ) + state.x.unsqueeze(1)

        # (B, 3, 3) x (W, P, 3) -> (B, W, P, 3)
        rotated_wheel_pos_rel = torch.einsum("bij,wpj->bwpi", R, self._wheel_pos_rel)
        wheel_points = rotated_wheel_pos_rel + wheel_centers_world.unsqueeze(2)

        dh_points, normals, stif_coeffs, fric_coeffs = self._sample_terrain(
            wheel_points, world_config
        )

        # wheel_omega are spin rates about the body Y axis: (B, 3) -> (B, 3, 3).
        zeros = torch.zeros_like(state.wheel_omega)
        # .float() is required: wheel_omega can arrive as float64.
        omega_local = torch.stack([zeros, state.wheel_omega, zeros], dim=-1).float()
        omega_world = (R.unsqueeze(1) @ omega_local.unsqueeze(-1)).squeeze(-1)

        omega_world_expanded = omega_world.unsqueeze(2)  # (B, W, 1, 3)
        r_contact_rel = wheel_points - wheel_centers_world.unsqueeze(2)

        # v_P = v_body + omega_body x (P - x) + omega_wheel x (P - wheel_center)
        r_body_P = wheel_points - state.x.unsqueeze(1).unsqueeze(1)

        v_rigid = state.xd.unsqueeze(1).unsqueeze(1) + torch.linalg.cross(
            state.omega.unsqueeze(1).unsqueeze(1), r_body_P
        )
        v_spin = torch.linalg.cross(omega_world_expanded, r_contact_rel)

        vels_wheel_contact = v_rigid + v_spin

        # Hard contact gate, shared by the spring force and the impulse limit.
        in_contact = (dh_points < 0).float()  # (B, W, P, 1)

        r_com = R @ self._cog  # body origin -> CoM, world frame (B, 3)
        com_world = state.x + r_com
        r = wheel_points - com_world.unsqueeze(1).unsqueeze(1)  # CoM -> contact
        I_world_inv = R @ self._I_body_inv @ R_T

        # (B, W, P, 3)
        # Rim points are body-fixed: only the body velocity compresses the
        # contact; wheel spin is tyre surface speed and enters friction only.
        F_springs = self._calculate_spring_force(
            dh_points,
            v_rigid,
            normals,
            stif_coeffs,
            in_contact,
            r,
            I_world_inv,
        )

        # Normal force magnitude.
        norms_spring = (F_springs * normals).sum(dim=-1, keepdim=True)  # (B, W, P, 1)
        F_N = norms_spring.clamp_min(1e-9)

        # Friction uses the tangential velocity only: a normal component would add
        # an unphysical upward force when sinking and launch the robot.
        v_normal_comp = (vels_wheel_contact * normals).sum(
            dim=-1, keepdim=True
        ) * normals
        v_tangential = vels_wheel_contact - v_normal_comp  # (B, W, P, 3)
        # Friction acts on the slip at the end of the step under gravity, so a
        # robot parked on a slope is held instead of creeping by g sin(theta) dt.
        g_t = self._F_g / self._mass
        g_t = g_t - (g_t * normals).sum(dim=-1, keepdim=True) * normals
        v_slip = v_tangential + g_t * self.config.dt
        v_mag = torch.norm(v_slip, dim=-1, keepdim=True)
        v_dir = v_slip / (v_mag + 1e-6)

        # F = mu * F_N * tanh(s / slip_ref_ratio) with the slip ratio
        # s = |v_t| / max(|omega_w| r_w, slip_floor_velocity): mu is the peak
        # traction coefficient; a stopped wheel falls back to the velocity form.
        v_roll = (state.wheel_omega.abs().float() * self._wheel_radii).clamp_min(
            self.config.slip_floor_velocity
        )[:, :, None, None]  # (B, W, 1, 1)
        slip = v_mag / v_roll
        F_c = fric_coeffs.unsqueeze(-1) * F_N * torch.tanh(
            slip / self.config.slip_ref_ratio
        )

        # 1 / m_eff = 1 / m + (r x t) . I^-1 (r x t): the body's effective mass
        # at the point along the slip.
        u = torch.linalg.cross(r, v_dir)
        inv_m_eff = 1.0 / self._mass + (
            u * torch.einsum("bij,bwpj->bwpi", I_world_inv, u)
        ).sum(dim=-1, keepdim=True)
        # Penetration-depth shares (sum to 1; zero for a point at zero depth).
        depth = (-dh_points).clamp_min(0)
        share = depth / depth.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        F_lim = share * torch.norm(
            v_tangential / inv_m_eff / self.config.dt + self._mass * g_t,
            dim=-1,
            keepdim=True,
        )

        return {
            "R": R,
            "I_world_inv": I_world_inv,
            "r_com": r_com,
            "r": r,
            "F_springs": F_springs,
            "v_tangential": v_tangential,
            # Point velocity relative to the CoM, omega x r + wheel spin.
            "v_rel": vels_wheel_contact
            - (state.xd + torch.linalg.cross(state.omega, r_com))[:, None, None],
            "v_dir": v_dir,
            "slip": slip,
            # Friction only where the spring pushes.
            "F_c": F_c * (norms_spring > 0).float(),
            "F_lim": F_lim,
        }

    def _body_accelerations(
        self, state: PhysicsState, c: dict, F_fric: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CoM linear and angular acceleration under springs, gravity and the
        friction magnitudes `F_fric` (B, W, P, 1) along -v_dir."""
        R, r = c["R"], c["r"]
        F_net_wheels = c["F_springs"] - F_fric * c["v_dir"]  # (B, W, P, 3)

        # Expand gravity to the actual batch size.
        F_grav_batch = self._F_g.expand(R.shape[0], -1)
        F_net_world = F_grav_batch + F_net_wheels.sum(dim=(1, 2))

        xdd_com = F_net_world / self._mass

        M_net = torch.linalg.cross(r, F_net_wheels).sum(dim=(1, 2))  # (B, 3)
        M_net = M_net - self.config.ang_damping * state.omega

        # Clamp the torque magnitude.
        M_mag = torch.norm(M_net, dim=-1, keepdim=True)
        scale = torch.clamp(self.config.torque_limit / (M_mag + 1e-6), max=1.0)
        M_net = M_net * scale

        # dot(omega) = inv(I) @ (M - omega x (I @ omega)), with
        # inv(I_world) = R @ inv(I_body) @ R.T (R orthonormal)
        I_world = R @ self._I_body @ R.transpose(-1, -2)
        Iw = (I_world @ state.omega.unsqueeze(-1)).squeeze(-1)
        w_x_Iw = torch.linalg.cross(state.omega, Iw)

        omega_d = (c["I_world_inv"] @ (M_net - w_x_Iw).unsqueeze(-1)).squeeze(-1)
        return xdd_com, omega_d

    def _friction(self, state: PhysicsState, c: dict) -> torch.Tensor:
        """Applied friction magnitude (B, W, P, 1): the tanh law, cut where it
        would reverse the slip within one step.

        Coulomb friction can stop a slip but never reverse it. Near rest the law's
        slope mu F_N / (slip_ref_ratio slip_floor) is ~3x what explicit Euler
        integrates at dt 2e-3, so uncut it chatters (a stopped robot drifted
        ~1.8 mm / 0.5 s forever). Predict each point's tangential velocity at the
        end of the step under ALL forces (springs, gravity, every contact's
        friction); where it points against the slip, cap the point at the
        impulse limit F_lim: the force that stops its own slip within the step
        through its effective mass, 1 / m_eff = 1 / m + (r x t) . I^-1 (r x t),
        plus gravity's tangential pull through the whole mass, shared over the
        contacts by penetration depth (shares sum to 1, so the joint impulse
        cannot overshoot). Sustained sliding under drive torque (the dragged
        wheel of a pivot turn) does not reverse and follows the law; gating the
        cap on the slip ratio instead clipped it (always on) or let stopping
        wheels chatter (slip < slip_ref_ratio).
        Pinned by tests/test_engine_invariants.py.
        """
        F_c = c["F_c"]
        dt = self.config.dt
        xdd_com, omega_d = self._body_accelerations(state, c, F_c)
        # Acceleration of the body-fixed sample point:
        # a_com + alpha x r + omega x (omega x r + v_spin).
        r = c["r"]
        a_pt = (
            xdd_com[:, None, None]
            + torch.linalg.cross(omega_d[:, None, None].expand_as(r), r)
            + torch.linalg.cross(state.omega[:, None, None].expand_as(r), c["v_rel"])
        )
        v_end = c["v_tangential"] + a_pt * dt
        reverses = (v_end * c["v_dir"]).sum(dim=-1, keepdim=True) < 0
        return torch.where(reverses, torch.min(F_c, c["F_lim"]), F_c)

    def forward(
        self,
        state: PhysicsState,
        world_config: WorldConfig,
    ) -> PhysicsStateDer:
        """Compute state time derivatives given the current state."""
        c = self._contact_terms(state, world_config)
        r_com = c["r_com"]
        xdd_com, omega_d = self._body_accelerations(state, c, self._friction(state, c))

        # state.x is the body origin (front axle), not the CoM, so
        # a_origin = a_com - alpha x r - omega x (omega x r), r = R @ cog.
        # On by default; checkpoints before 2026-10-22 trained without it.
        if self.config.com_transfer:
            xdd = (
                xdd_com
                - torch.linalg.cross(omega_d, r_com)
                - torch.linalg.cross(
                    state.omega, torch.linalg.cross(state.omega, r_com)
                )
            )
        else:
            xdd = xdd_com

        return PhysicsStateDer(xdd=xdd, omega_d=omega_d)

    def update_state(
        self, state: PhysicsState, dstate: PhysicsStateDer
    ) -> PhysicsState:
        """Integrate state for one timestep."""
        actual_bs = state.x.shape[0]

        # Semi-implicit Euler: positions use the updated velocities.
        next_xd = (state.xd + dstate.xdd * self.config.dt).clamp(-50.0, 50.0)
        next_x = state.x + next_xd * self.config.dt

        next_omega = (state.omega + dstate.omega_d * self.config.dt).clamp(-50.0, 50.0)
        next_q = integrate_quaternion(state.q, next_omega, self.config.dt)

        return PhysicsState(
            x=next_x,
            xd=next_xd,
            q=next_q,
            omega=next_omega,
            # Wheel speeds are commanded, not integrated (set by the caller).
            wheel_omega=state.wheel_omega,
            batch_size=actual_bs,
            device=state.device,
        )

