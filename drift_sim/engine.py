from typing import Union

import torch

from simulator.numerical import (
    integrate_quaternion,
    interpolate_grid,
    interpolate_surface_normals_multi,
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
        device: Union[torch.device, str],
    ):
        super().__init__()
        self.config = config
        self.robot_model = robot_model
        self.device = device
        self._F_g = torch.tensor(
            [0.0, 0.0, -self.robot_model.mass * self.config.g],
            device=self.device,
            requires_grad=False,
        ).unsqueeze(0)
        # Precompute relative wheel points
        self.register_buffer(
            "_wheel_pos_rel", self._precompute_wheel_points(n_points=16)
        )

    def _precompute_wheel_points(self, n_points: int = 16) -> torch.Tensor:
        """Precompute wheel points relative to wheel centers."""
        # [:-1] to remove endpoint to avoid duplicate points
        theta = torch.linspace(0, 2 * torch.pi, n_points + 1, device=self.device)[:-1]
        n_wheels = 3

        # radii: (3,)
        wheel_radii = self.robot_model.wheel_radii.to(self.device)

        # Shape (n_wheels, n_pts)
        x = torch.zeros((n_wheels, n_points), device=self.device)
        y = wheel_radii[:, None] * torch.cos(theta)[None, :]
        z = -wheel_radii[:, None] * torch.sin(theta)[None, :]

        # (n_wheels, n_points, 3)
        return torch.stack([x, y, z], dim=-1)

    def _T_onto_normal(
        self, vec: torch.Tensor, normal_vec: torch.Tensor
    ) -> torch.Tensor:
        """Transform vector onto a vector (i.e. via projection)."""
        n = normal_vec / torch.norm(normal_vec, dim=-1, keepdim=True).clamp_min(1e-9)
        dot = (vec * n).sum(dim=-1, keepdim=True)
        return dot

    def _T_onto_plane(
        self, vec: torch.Tensor, normal_vec: torch.Tensor
    ) -> torch.Tensor:
        """Transform vector onto a plane given by normal vector (i.e. via rejection on it)."""
        return vec - self._T_onto_normal(vec, normal_vec)

    def _find_contact_points(
        self,
        robot_points: torch.Tensor,
        world_config: WorldConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the points of each wheel and the associated normals and penetration depth.

        Args:
            robot_points: (B, W, P, 3) positions of wheel sample points
            world_config: World configuration containing terrain grids

        Returns:
            points:    (B, W, 1, 3) deepest contact point per wheel
            dh_points: (B, W, 1, 1) penetration depth at that point
            normals:   (B, W, 1, 3) surface normal at that point
        """
        # --- Interpolate terrain heights for all wheel points ---
        z_points = interpolate_grid(
            world_config.z_grid, robot_points[..., :2], world_config.max_coord
        )  # (B, W, P, 1)

        # --- Compute normals per wheel sample point ---
        normals = interpolate_surface_normals_multi(
            world_config.normals, robot_points[..., :2], world_config.max_coord
        )  # (B, W, P, 3)

        # --- Compute penetration depth along normal z ---
        dh = (robot_points[..., 2:3] - z_points) * normals[..., 2:3]  # (B, W, P, 1)

        # Multi-point contact: Return all points, depths, and normals
        # No reduction (argmin) here.
        return robot_points, dh, normals

    def _calculate_spring_force(
        self,
        dh_points: torch.Tensor,
        xd_points: torch.Tensor,
        normals: torch.Tensor,
        world_config: WorldConfig,
    ) -> torch.Tensor:
        """
        Calculate the spring force acting on the robot points.
        """
        in_contact = (dh_points < 0).float()  # shape (B, W, P, 1)

        # Count total contacts across all wheels and points to distribute stiffness
        num_contacts = in_contact.sum(dim=(1, 2), keepdim=True).clamp_min(1)

        k = world_config.k_stiffness
        b = (
            self.config.damping_alpha
            * 2
            * (self.robot_model.mass * k / num_contacts) ** 0.5
        )

        # normal component of the velocity
        xd_points_n = (xd_points * normals).sum(dim=-1, keepdim=True)

        # F_spring = -k * dh - b * v_n, multiply by -n to get the force vector
        # add unilateral constraint - to ensure the ground can only push, never pull.
        force_magnitude = -(k * dh_points + b * xd_points_n)

        if self.config.soft_contact_sigma is not None:
            # Soft contact: Smooth approximation of ReLU using Softplus
            F_spring_mag = torch.nn.functional.softplus(
                force_magnitude, beta=1.0 / self.config.soft_contact_sigma
            )
        else:
            # Hard contact: ReLU
            F_spring_mag = torch.relu(force_magnitude)

        F_spring = F_spring_mag * normals

        # shape (B, 3, 3), the spring force acting on the robot points
        return F_spring * in_contact / num_contacts

    def _find_frict_coeffs_at_contact(
        self, contact_points: torch.Tensor, world_config: WorldConfig
    ) -> torch.Tensor:
        if isinstance(world_config.k_friction, torch.Tensor):
            fric_coeffs = interpolate_grid(
                world_config.k_friction,
                contact_points[..., :2],
                world_config.max_coord,
            ).squeeze(-1)
        else:
            k = world_config.k_friction
            B, W = contact_points.shape[:2]
            fric_coeffs = torch.full(
                (B, W, 1), k, device=self.device, dtype=contact_points.dtype
            )

        return fric_coeffs

    def forward(
        self,
        state: PhysicsState,
        whl_acc_ctrl: torch.Tensor,
        world_config: WorldConfig,
    ) -> PhysicsStateDer:
        """
        Computes time derivatives of the robot state given current state and controls.
        """
        # Inertia Tensor
        I_body = self.robot_model.inertia_tensor.to(self.device)  # (3, 3)

        R = quat_to_rotmat(state.q)

        # Transform Inertia to World Frame: I_world = R @ I_body @ R.T
        I_world = R @ I_body.unsqueeze(0) @ R.transpose(-1, -2)  # (B, 3, 3)
        # Compute Inverse Inertia
        I_inv_world = torch.linalg.inv(I_world)

        wheel_centers = self.robot_model.wheel_centers.to(self.device)
        wheel_centers_world = (R @ wheel_centers.T).transpose(
            -1, -2
        ) + state.x.unsqueeze(1)

        # Rotate precomputed wheel points with the robot body
        # R is (B, 3, 3), _wheel_pos_rel is (W, P, 3) -> (B, W, P, 3)
        rotated_wheel_pos_rel = torch.einsum("bij,wpj->bwpi", R, self._wheel_pos_rel)

        # Add to world-frame wheel centers
        wheel_points = rotated_wheel_pos_rel + wheel_centers_world.unsqueeze(2)

        # INFO: if contact points do not touch, they return the closest one
        contact_points, dh_points, normals = self._find_contact_points(
            wheel_points, world_config
        )

        # Interpret wheel_omega as scalars around Y-axis (Robot Frame)
        # (B, 3) -> (B, 3, 3) vectors [0, w, 0]
        zeros = torch.zeros_like(state.wheel_omega)
        omega_local = torch.stack([zeros, state.wheel_omega, zeros], dim=-1)
        # Rotate to World Frame
        omega_world = (R.unsqueeze(1) @ omega_local.unsqueeze(-1)).squeeze(-1)

        # v = v_center + ω x (r_contact - r_center)
        # contact_points is (B, W, P, 3)
        # wheel_centers_world is (B, W, 3) -> expand to (B, W, 1, 3)
        # omega_world is (B, W, 3) -> expand to (B, W, 1, 3)
        omega_world_expanded = omega_world.unsqueeze(2)
        r_contact_rel = contact_points - wheel_centers_world.unsqueeze(2)

        vels_wheel_contact = state.xd.unsqueeze(1).unsqueeze(1) + torch.linalg.cross(
            omega_world_expanded, r_contact_rel
        )

        # F_springs will be (B, W, P, 3)
        F_springs = self._calculate_spring_force(
            dh_points, vels_wheel_contact, normals, world_config
        )

        # Friction
        fric_coeffs = self._find_frict_coeffs_at_contact(contact_points, world_config)

        # Normal Force Magnitude: Project Spring Force onto Surface Normal
        # F_springs: (B, W, P, 3). normals: (B, W, P, 3).
        norms_spring = (F_springs * normals).sum(dim=-1, keepdim=True)  # (B, W, P, 1)
        F_N = norms_spring.clamp_min(1e-9)

        # F_fric = mu * F_N * tanh(v_rel)
        # vels_wheel_contact is absolute velocity of contact point (ground is static)
        # Refined friction model: direction based on velocity vector, magnitude based on speed
        v_mag = torch.norm(vels_wheel_contact, dim=-1, keepdim=True)
        v_dir = vels_wheel_contact / (v_mag + 1e-6)
        Fs_fric = fric_coeffs.unsqueeze(-1) * F_N * torch.tanh(v_mag) * v_dir

        # Only apply friction if in contact (F_spring > 0)
        wheels_in_contact = (norms_spring > 0).float()
        Fs_fric = Fs_fric * wheels_in_contact

        # Net Force per Wheel Point: (B, W, P, 3)
        F_net_wheels = F_springs - Fs_fric

        # Total Force (World Frame)
        # F_grav: (B, 3)
        F_grav_batch = self._F_g.expand(self.config.num_robots, -1)
        # Sum over wheels (dim 1) and points (dim 2)
        F_net_world = F_grav_batch + F_net_wheels.sum(dim=(1, 2))

        # Acceleration
        xdd = F_net_world / self.robot_model.mass

        # Angular Dynamics
        # CoM in world: state.x + R @ cog
        com_world = state.x + (R @ self.robot_model.cog.to(self.device).float())

        # r: Vector from CoM to Contact Points
        r = contact_points - com_world.unsqueeze(1).unsqueeze(1)

        # Torque: r x F
        torque_per_wheel = torch.linalg.cross(r, F_net_wheels)

        # Compute net torque and apply angular damping and torque limits
        M_net = torque_per_wheel.sum(dim=(1, 2))  # (B, 3)
        M_net = M_net - self.config.ang_damping * state.omega
        M_net = M_net.clamp(
            -self.config.torque_limit, self.config.torque_limit
        )  # apply torque limit

        # Solve for angular acceleration
        # dot(omega) = inv(I) @ (M - omega x (I @ omega))
        Iw = (I_world @ state.omega.unsqueeze(-1)).squeeze(-1)
        w_x_Iw = torch.linalg.cross(state.omega, Iw)

        omega_d = (I_inv_world @ (M_net - w_x_Iw).unsqueeze(-1)).squeeze(-1)

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
        # basic kinematics
        next_xd = state.xd + dstate.xdd * self.config.dt
        next_x = state.x + next_xd * self.config.dt

        # rotation kinematics
        next_omega = state.omega + dstate.omega_d * self.config.dt
        next_q = integrate_quaternion(state.q, next_omega, self.config.dt)
        next_wheel_omega = state.wheel_omega + dstate.wheel_omega_d * self.config.dt

        return PhysicsState(
            x=next_x,
            xd=next_xd,
            q=next_q,
            omega=next_omega,
            wheel_omega=next_wheel_omega,
            batch_size=self.config.num_robots,
            device=state.device,
        )

    def get_dt(self) -> float:
        return self.config.dt
