import torch
import torch.nn.functional as F

from simulator.numerical import (
    integrate_quaternion,
    quat_to_rotmat,
)
from simulator.utils import (
    PhysicsEngineConfig,
    PhysicsState,
    PhysicsStateDer,
    RobotModelConfig,
    WorldConfig,
)

###############################################################################
# Less Simple Model
# - rigid body only
# - COG, triangle as parameters
# - **assumes contact forces do not spin wheels and are *just* transferred to the axis**
###############################################################################


class DPhysicsEngine(torch.nn.Module):
    def __init__(
        self,
        config: PhysicsEngineConfig,
        robot_model: RobotModelConfig,
    ):
        super().__init__()
        self.config = config
        self.robot_model = robot_model
        # Precompute relative wheel points
        self.register_buffer(
            "_wheel_pos_rel", self._precompute_wheel_points(n_points=16)
        )
        self.register_buffer("_I_body", robot_model.inertia_tensor.float())
        self.register_buffer(
            "_I_body_inv", torch.linalg.inv(robot_model.inertia_tensor.float())
        )
        self.register_buffer("_wheel_centers", robot_model.wheel_centers.float())
        self.register_buffer("_cog", robot_model.cog.float())
        self._mass = robot_model.mass
        F_g = torch.tensor(
            [0.0, 0.0, -self._mass * self.config.g],
        ).unsqueeze(0)
        self.register_buffer("_F_g", F_g)

    def _precompute_wheel_points(self, n_points: int = 16) -> torch.Tensor:
        """Precompute wheel points relative to wheel centers."""
        # [:-1] to remove endpoint to avoid duplicate points
        theta = torch.linspace(0, 2 * torch.pi, n_points + 1)[:-1]
        n_wheels = 3

        # radii: (3,)
        wheel_radii = self.robot_model.wheel_radii

        # Shape (n_wheels, n_pts)
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
        """
        Single batched grid_sample for all terrain properties at wheel contact points.

        Uses the precomputed 6-channel ``world_config._sample_grid``
        (z, nx, ny, nz, friction, stiffness) to replace the three separate
        grid_sample calls previously needed for contact detection and coefficient
            lookup.

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

        # Normalise XY to [-1, 1] for grid_sample
        xy = wheel_points[..., :2] - world_config.grid_center
        xy_norm = (xy / world_config.max_coord).clamp(-1.0, 1.0)  # (B, W, P, 2)

        # Flatten to (B, 1, W*P, 2) — one-row output grid
        query = xy_norm.reshape(B, 1, W * P, 2)

        # Single grid_sample over the precomputed (B, 6, H, W) grid
        sampled = F.grid_sample(
            world_config._sample_grid,
            query,
            mode="bilinear",
            align_corners=True,
            padding_mode="border",
        )  # (B, 6, 1, W*P)

        # (B, 6, W*P) -> (B, W*P, 6) -> (B, W, P, 6)
        sampled = sampled.squeeze(2).permute(0, 2, 1).reshape(B, W, P, 6)

        z_terrain = sampled[..., :1]  # (B, W, P, 1)
        normals = sampled[..., 1:4]  # (B, W, P, 3)
        fric_coeffs = sampled[..., 4]  # (B, W, P)
        stif_coeffs = sampled[..., 5]  # (B, W, P)

        # Re-normalise after bilinear interpolation
        normals = F.normalize(normals, dim=-1, eps=1e-6)

        # Penetration depth along the surface normal z component
        dh = (wheel_points[..., 2:3] - z_terrain) * normals[..., 2:3]  # (B, W, P, 1)

        return dh, normals, stif_coeffs, fric_coeffs

    def _calculate_spring_force(
        self,
        dh_points: torch.Tensor,
        xd_points: torch.Tensor,
        normals: torch.Tensor,
        stiffness: torch.Tensor,
        in_contact: torch.Tensor,
        num_contacts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the spring force acting on the robot points.

        `in_contact` is a HARD (dh < 0) step, so the contact BOUNDARY is not
        differentiable.
        """
        k = stiffness.unsqueeze(-1)  # shape (B, W, 1, 1)
        b = self.config.damping_alpha * 2 * (self._mass * k) ** 0.5

        # normal component of the velocity
        xd_points_n = (xd_points * normals).sum(dim=-1, keepdim=True)

        # F_spring = -k * dh - b * v_n, multiply by -n to get the force vector
        # add unilateral constraint - to ensure the ground can only push, never pull.
        force_magnitude = -(k * dh_points + b * xd_points_n)

        F_spring_mag = torch.relu(force_magnitude)

        F_spring = F_spring_mag * normals

        # shape (B, 3, 3), the spring force acting on the robot points
        return F_spring * in_contact / num_contacts

    def forward(
        self,
        state: PhysicsState,
        whl_acc_ctrl: torch.Tensor,
        world_config: WorldConfig,
    ) -> PhysicsStateDer:
        """
        Computes time derivatives of the robot state given current state and controls.
        """
        R = quat_to_rotmat(state.q)

        # Transform Inertia to World Frame: I_world = R @ I_body @ R.T
        R_T = R.transpose(-1, -2)
        I_world = R @ self._I_body @ R_T  # (B, 3, 3)

        wheel_centers = self._wheel_centers
        wheel_centers_world = (R @ wheel_centers.T).transpose(
            -1, -2
        ) + state.x.unsqueeze(1)

        # Rotate precomputed wheel points with the robot body
        # R is (B, 3, 3), _wheel_pos_rel is (W, P, 3) -> (B, W, P, 3)
        rotated_wheel_pos_rel = torch.einsum("bij,wpj->bwpi", R, self._wheel_pos_rel)

        # Add to world-frame wheel centers
        wheel_points = rotated_wheel_pos_rel + wheel_centers_world.unsqueeze(2)

        dh_points, normals, stif_coeffs, fric_coeffs = self._sample_terrain(
            wheel_points, world_config
        )

        # Interpret wheel_omega as scalars around Y-axis (Robot Frame)
        # (B, 3) -> (B, 3, 3) vectors [0, w, 0]
        zeros = torch.zeros_like(state.wheel_omega)
        # .float() is critical, wheel points are passed as float64
        omega_local = torch.stack([zeros, state.wheel_omega, zeros], dim=-1).float()
        # Rotate to World Frame
        omega_world = (R.unsqueeze(1) @ omega_local.unsqueeze(-1)).squeeze(-1)

        # v = v_center + ω x (r_contact - r_center)
        # contact_points is (B, W, P, 3)
        # wheel_centers_world is (B, W, 3) -> expand to (B, W, 1, 3)
        # omega_world is (B, W, 3) -> expand to (B, W, 1, 3)
        omega_world_expanded = omega_world.unsqueeze(2)
        r_contact_rel = wheel_points - wheel_centers_world.unsqueeze(2)

        # Velocity of contact point P = v_body + omega_body x r_body_P + v_spin
        # r_body_P = contact_points - state.x
        r_body_P = wheel_points - state.x.unsqueeze(1).unsqueeze(1)

        v_rigid = state.xd.unsqueeze(1).unsqueeze(1) + torch.linalg.cross(
            state.omega.unsqueeze(1).unsqueeze(1), r_body_P
        )
        v_spin = torch.linalg.cross(omega_world_expanded, r_contact_rel)

        vels_wheel_contact = v_rigid + v_spin

        # Hard contact gate, shared by the spring force and the impulse limit
        in_contact = (dh_points < 0).float()  # (B, W, P, 1)
        num_contacts = in_contact.sum(dim=(1, 2), keepdim=True).clamp_min(1)

        # F_springs will be (B, W, P, 3)
        F_springs = self._calculate_spring_force(
            dh_points,
            vels_wheel_contact,
            normals,
            stif_coeffs,
            in_contact,
            num_contacts,
        )

        # Normal Force Magnitude: Project Spring Force onto Surface Normal
        # F_springs: (B, W, P, 3). normals: (B, W, P, 3).
        norms_spring = (F_springs * normals).sum(dim=-1, keepdim=True)  # (B, W, P, 1)
        F_N = norms_spring.clamp_min(1e-9)

        # F_fric = mu * F_N * tanh(v_rel)
        # Friction is a tangential force — project contact velocity onto the terrain
        # surface
        # plane before computing direction and magnitude.  Using the full 3D velocity
        # would
        # inject a spurious normal component: when the body has a downward z-velocity,
        # v_dir[z] < 0, so Fs_fric[z] < 0, and F_net[z] = F_spring[z] - Fs_fric[z] gains
        # an unphysical extra-upward term that can launch the robot off the terrain.
        v_normal_comp = (vels_wheel_contact * normals).sum(
            dim=-1, keepdim=True
        ) * normals
        v_tangential = vels_wheel_contact - v_normal_comp  # (B, W, P, 3)
        v_mag = torch.norm(v_tangential, dim=-1, keepdim=True)
        v_dir = v_tangential / (v_mag + 1e-6)

        # F_c_raw_mag = mu * F_N * tanh(v)
        # tanh(|v_t| / v_ref): v_ref is the slip speed at which traction
        # saturates. At the default 1.0 this is the historical bare tanh(|v_t|).
        v_ref = getattr(self.config, "slip_ref_velocity", 1.0) or 1.0
        F_c_mag = fric_coeffs.unsqueeze(-1) * F_N * torch.tanh(v_mag / v_ref)

        # Impulse limit: friction may not do more than kill the tangential
        # velocity of this point within one step. The momentum share of one
        # contact point is m / num_contacts, so the divisor is the CONTACT
        # COUNT, not the wheel count (it used to be a hard 3). Measured inert at
        # dt 2e-3 either way -- F_N per point is ~mg/48, so the friction force is
        # ~2 orders below the cap; it starts to bind at much larger dt. Pinned by
        # tests/test_engine_invariants.py::test_impulse_limit_is_inert_at_training_dt.
        F_imp_lim = self._mass * v_mag / num_contacts / self.config.dt

        # Clamp magnitude
        F_c_mag = torch.min(F_c_mag, F_imp_lim)

        # Apply direction
        Fs_fric = F_c_mag * v_dir

        # Only apply friction if in contact (F_spring > 0)
        wheels_in_contact = (norms_spring > 0).float()
        Fs_fric = Fs_fric * wheels_in_contact

        # Net Force per Wheel Point: (B, W, P, 3)
        F_net_wheels = F_springs - Fs_fric

        # Total Force (World Frame)
        # F_grav: (B, 3) — expand to actual batch size, not the static config value
        F_grav_batch = self._F_g.expand(R.shape[0], -1)
        # Sum over wheels (dim 1) and points (dim 2)
        F_net_world = F_grav_batch + F_net_wheels.sum(dim=(1, 2))

        # Acceleration of the CENTRE OF MASS
        xdd_com = F_net_world / self._mass

        # Angular Dynamics
        # CoM in world: state.x + R @ cog
        r_com = R @ self._cog  # body origin -> CoM, world frame (B, 3)
        com_world = state.x + r_com

        # r: Vector from CoM to Contact Points
        r = wheel_points - com_world.unsqueeze(1).unsqueeze(1)

        # Torque: r x F
        torque_per_wheel = torch.linalg.cross(r, F_net_wheels)

        # Compute net torque and apply angular damping and torque limits
        M_net = torque_per_wheel.sum(dim=(1, 2))  # (B, 3)
        M_net = M_net - self.config.ang_damping * state.omega

        # Apply torque limit (Magnitude clamping)
        M_mag = torch.norm(M_net, dim=-1, keepdim=True)
        scale = torch.clamp(self.config.torque_limit / (M_mag + 1e-6), max=1.0)
        M_net = M_net * scale

        # Solve for angular acceleration
        # dot(omega) = inv(I) @ (M - omega x (I @ omega)), with
        # inv(I_world) = R @ inv(I_body) @ R.T (R orthonormal)
        Iw = (I_world @ state.omega.unsqueeze(-1)).squeeze(-1)
        w_x_Iw = torch.linalg.cross(state.omega, Iw)

        I_world_inv = R @ self._I_body_inv @ R_T
        omega_d = (I_world_inv @ (M_net - w_x_Iw).unsqueeze(-1)).squeeze(-1)

        # state.x is the BODY ORIGIN (base_link = the front axle), not the CoM,
        # so a_origin = a_com - alpha x r - omega x (omega x r) with
        # r = R @ cog. The transfer terms were dropped historically; every
        # production checkpoint was trained without them, so this is OFF by
        # default and turning it on changes the dynamics.
        if getattr(self.config, "com_transfer", False):
            xdd = (
                xdd_com
                - torch.linalg.cross(omega_d, r_com)
                - torch.linalg.cross(
                    state.omega, torch.linalg.cross(state.omega, r_com)
                )
            )
        else:
            xdd = xdd_com

        # next state derivative
        next_state_der = PhysicsStateDer(
            xdd=xdd,
            omega_d=omega_d,
            wheel_omega_d=whl_acc_ctrl,
        )

        return next_state_der

    def update_state(
        self, state: PhysicsState, dstate: PhysicsStateDer
    ) -> PhysicsState:
        """Integrate state for one timestep."""
        actual_bs = state.x.shape[0]

        # translational kinematics
        next_xd = (state.xd + dstate.xdd * self.config.dt).clamp(-50.0, 50.0)
        next_x = state.x + next_xd * self.config.dt

        # rotation kinematics
        next_omega = (state.omega + dstate.omega_d * self.config.dt).clamp(-50.0, 50.0)
        next_q = integrate_quaternion(state.q, next_omega, self.config.dt)
        next_wheel_omega = state.wheel_omega + dstate.wheel_omega_d * self.config.dt

        return PhysicsState(
            x=next_x,
            xd=next_xd,
            q=next_q,
            omega=next_omega,
            wheel_omega=next_wheel_omega,
            batch_size=actual_bs,
            device=state.device,
        )

    def get_dt(self) -> float:
        return self.config.dt
