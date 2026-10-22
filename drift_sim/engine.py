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
        (z, nx, ny, nz, friction, stiffness).

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
        num_contacts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the damped contact spring force on the wheel points.

        `in_contact` is a hard (dh < 0) step, so the contact boundary is not
        differentiable.
        """
        k = stiffness.unsqueeze(-1)  # (B, W, 1, 1)
        b = self.config.damping_alpha * 2 * (self._mass * k) ** 0.5

        xd_points_n = (xd_points * normals).sum(dim=-1, keepdim=True)

        # F = -k * dh - b * v_n along the normal; relu makes it unilateral (the
        # ground only pushes).
        force_magnitude = -(k * dh_points + b * xd_points_n)

        F_spring_mag = torch.relu(force_magnitude)

        F_spring = F_spring_mag * normals

        return F_spring * in_contact / num_contacts

    def forward(
        self,
        state: PhysicsState,
        whl_acc_ctrl: torch.Tensor,
        world_config: WorldConfig,
    ) -> PhysicsStateDer:
        """Compute state time derivatives given the current state and controls."""
        R = quat_to_rotmat(state.q)

        # I_world = R @ I_body @ R.T
        R_T = R.transpose(-1, -2)
        I_world = R @ self._I_body @ R_T  # (B, 3, 3)

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
        num_contacts = in_contact.sum(dim=(1, 2), keepdim=True).clamp_min(1)

        # (B, W, P, 3)
        F_springs = self._calculate_spring_force(
            dh_points,
            vels_wheel_contact,
            normals,
            stif_coeffs,
            in_contact,
            num_contacts,
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
        v_mag = torch.norm(v_tangential, dim=-1, keepdim=True)
        v_dir = v_tangential / (v_mag + 1e-6)

        # F = mu * F_N * tanh(|v_t| / v_ref); v_ref is the slip speed where
        # traction saturates.
        v_ref = self.config.slip_ref_velocity
        F_c_mag = fric_coeffs.unsqueeze(-1) * F_N * torch.tanh(v_mag / v_ref)

        # Impulse limit: friction may at most stop this point's tangential velocity
        # within one step, with a per-contact momentum share of m / num_contacts.
        # Inert at dt 2e-3 (friction is ~2 orders below the cap); pinned by
        # tests/test_engine_invariants.py::test_impulse_limit_is_inert_at_training_dt.
        F_imp_lim = self._mass * v_mag / num_contacts / self.config.dt

        F_c_mag = torch.min(F_c_mag, F_imp_lim)

        Fs_fric = F_c_mag * v_dir

        wheels_in_contact = (norms_spring > 0).float()
        Fs_fric = Fs_fric * wheels_in_contact

        F_net_wheels = F_springs - Fs_fric  # (B, W, P, 3)

        # Expand gravity to the actual batch size.
        F_grav_batch = self._F_g.expand(R.shape[0], -1)
        F_net_world = F_grav_batch + F_net_wheels.sum(dim=(1, 2))

        xdd_com = F_net_world / self._mass

        r_com = R @ self._cog  # body origin -> CoM, world frame (B, 3)
        com_world = state.x + r_com

        r = wheel_points - com_world.unsqueeze(1).unsqueeze(1)  # CoM -> contact

        torque_per_wheel = torch.linalg.cross(r, F_net_wheels)

        M_net = torque_per_wheel.sum(dim=(1, 2))  # (B, 3)
        M_net = M_net - self.config.ang_damping * state.omega

        # Clamp the torque magnitude.
        M_mag = torch.norm(M_net, dim=-1, keepdim=True)
        scale = torch.clamp(self.config.torque_limit / (M_mag + 1e-6), max=1.0)
        M_net = M_net * scale

        # dot(omega) = inv(I) @ (M - omega x (I @ omega)), with
        # inv(I_world) = R @ inv(I_body) @ R.T (R orthonormal)
        Iw = (I_world @ state.omega.unsqueeze(-1)).squeeze(-1)
        w_x_Iw = torch.linalg.cross(state.omega, Iw)

        I_world_inv = R @ self._I_body_inv @ R_T
        omega_d = (I_world_inv @ (M_net - w_x_Iw).unsqueeze(-1)).squeeze(-1)

        # state.x is the body origin (front axle), not the CoM, so
        # a_origin = a_com - alpha x r - omega x (omega x r), r = R @ cog.
        # Off by default: all production checkpoints were trained without it.
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

        # Semi-implicit Euler: positions use the updated velocities.
        next_xd = (state.xd + dstate.xdd * self.config.dt).clamp(-50.0, 50.0)
        next_x = state.x + next_xd * self.config.dt

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

