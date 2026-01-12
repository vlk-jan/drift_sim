from typing import Union

import torch

from simulator.numerical import (
    integrate_quaternion,
    quat_to_rotmat,
    interpolate_grid,
    interpolate_surface_normals,
)
from simulator.utils import (
    PhysicsEngineConfig,
    PhysicsState,
    PhysicsStateDer,
    RobotModelConfig,
)
from simulator.world_config import WorldConfig


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

        # Use precomputed relative wheel points
        wheel_points = (
            self._wheel_pos_rel[None, :, :, :] + wheel_centers_world[:, :, None, :]
        )

        # INFO: if contact points do not touch, they return the closest one
        contact_points, dh_points, normals = self.find_contact_points(
            wheel_points, world_config
        )

        # Interpret wheel_omega as scalars around Y-axis (Robot Frame)
        # (B, 3) -> (B, 3, 3) vectors [0, w, 0]
        zeros = torch.zeros_like(state.wheel_omega)
        omega_local = torch.stack([zeros, state.wheel_omega, zeros], dim=-1)
        # Rotate to World Frame
        omega_world = (R.unsqueeze(1) @ omega_local.unsqueeze(-1)).squeeze(-1)

        # v = v_center + ω x (r_contact - r_center)
        # Squeeze contact_points from (B, 3, 1, 3) to (B, 3, 3)
        vels_wheel_contact = state.xd.unsqueeze(1) + torch.linalg.cross(
            omega_world,
            contact_points.squeeze(2) - wheel_centers_world,
        ).unsqueeze(2)

        # F_springs will be (B, 3, 1, 3)
        F_springs = self.calculate_spring_force(
            dh_points, vels_wheel_contact, normals, world_config
        )

        # Friction
        fric_coeffs = self.find_frict_coeffs_at_contact(contact_points, world_config)

        # Normal Force Magnitude: Project Spring Force onto Surface Normal
        # F_springs: (B, W, 1, 3). normals: (B, W, 1, 3).
        norms_spring = (F_springs * normals).sum(dim=-1, keepdim=True)  # (B, W, 1, 1)
        F_N = norms_spring.clamp_min(1e-9)

        # F_fric = mu * F_N * tanh(v_rel)
        # vels_wheel_contact is absolute velocity of contact point (ground is static)
        Fs_fric = fric_coeffs.unsqueeze(-1) * F_N * torch.tanh(vels_wheel_contact)

        # Only apply friction if in contact (F_spring > 0)
        wheels_in_contact = (norms_spring > 0).float()
        Fs_fric = Fs_fric * wheels_in_contact

        # Net Force per Wheel: (B, W, 1, 3)
        F_net_wheels = F_springs - Fs_fric

        # Total Force (World Frame)
        # F_grav: (B, 3)
        F_grav_batch = self._F_g.expand(self.config.num_robots, -1)
        F_net_world = F_grav_batch + F_net_wheels.sum(dim=1).squeeze(1)

        # Acceleration
        xdd = F_net_world / self.robot_model.mass

        # Angular Dynamics
        # CoM in world: state.x + R @ cog
        com_world = state.x + (R @ self.robot_model.cog.to(self.device).float())

        # r: Vector from CoM to Contact Points
        r = contact_points - com_world.unsqueeze(1).unsqueeze(1)

        # Torque: r x F
        torque_per_wheel = torch.linalg.cross(r, F_net_wheels)
        torque_per_wheel = torch.clamp(
            torque_per_wheel, -self.config.torque_limit, self.config.torque_limit
        )
        M_net = torque_per_wheel.sum(dim=1).squeeze(1)  # (B, 3)
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

    def find_contact_points(
        self,
        robot_points: torch.Tensor,
        world_config: WorldConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns the deepest contact points for each wheel and the associated normals.

        Args:
            robot_points: (B, W, P, 3) positions of wheel sample points
            world_config: World configuration containing terrain grids

        Returns:
            points:    (B, W, 1, 3) deepest contact point per wheel
            dh_points: (B, W, 1, 1) penetration depth at that point
            normals:   (B, W, 1, 3) surface normal at that point
        """
        B, W, _, _ = robot_points.shape

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

        # --- Find deepest point per wheel ---
        deepest_idx = torch.argmin(dh, dim=2, keepdim=True)  # (B, W, 1)

        # --- Gather deepest point, depth, and normal ---
        idx_points = deepest_idx.expand(B, W, 1, 3)
        points = robot_points.gather(dim=2, index=idx_points)  # (B, W, 1, 3)
        normals = normals.gather(dim=2, index=idx_points)  # (B, W, 1, 3)

        idx_dh = deepest_idx.expand(B, W, 1, 1)
        dh_points = dh.gather(dim=2, index=idx_dh)  # (B, W, 1, 1)

        return points, dh_points, normals

    def calculate_spring_force(
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

        num_contacts = in_contact.sum(dim=1, keepdim=True).clamp_min(1)

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
        F_spring = torch.relu(-(k * dh_points + b * xd_points_n)) * normals

        # shape (B, 3, 3), the spring force acting on the robot points
        return F_spring * in_contact / num_contacts

    def find_frict_coeffs_at_contact(
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


# TODO: check thoroughly
def find_init_rob_position(
    world_config: WorldConfig, robot_model: RobotModelConfig, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Find the initial position (x) and orientation (q) of the robot so that it rests
    on the terrain at (0,0,unknown height).

    Returns:
        x_init: (3,)  world position
        q_init: (4,)  unit quaternion [w, x, y, z]
    """

    # --- 1. Terrain height + normal at (0, 0) ---
    xy = torch.tensor([[[0.0, 0.0]]], device=device)  # shape (B=1, N=1, 2)
    z_ground = interpolate_grid(world_config.z_grid, xy, world_config.max_coord)[
        0, 0, 0, 0
    ]
    n_ground = interpolate_surface_normals(
        world_config.normals, xy, world_config.max_coord
    )[0, 0]

    # Ensure the normal is normalized
    n_ground = n_ground / torch.norm(n_ground)

    # --- 2. Compute reference offset for wheels / CoM ---
    # Assume wheel_radii is iterable or tensor-like
    wheel_radii = torch.as_tensor(
        robot_model.wheel_radii, device=device, dtype=torch.float32
    )
    z_offset = wheel_radii.mean()

    z_init = z_ground + z_offset

    # --- 3. Compute orientation aligning robot up-axis with terrain normal ---
    z_body = torch.tensor([0.0, 0.0, 1.0], device=device)

    # Compute axis–angle rotation: rotate z_body → n_ground
    v = torch.linalg.cross(z_body, n_ground)
    s = torch.norm(v)
    c = torch.dot(z_body, n_ground)

    if s < 1e-8:
        # The vectors are parallel — robot is either upright or upside-down
        if c > 0:  # already aligned
            q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        else:  # 180-degree flip around any horizontal axis
            q = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)
    else:
        axis = v / s
        angle = torch.atan2(s, c)

        # Convert axis–angle to quaternion
        half = angle * 0.5
        qw = torch.cos(half)
        qx, qy, qz = axis * torch.sin(half)

        q = torch.stack([qw, qx, qy, qz])
        q = q / torch.norm(q)

    # --- 4. Compose initial position ---
    x_init = torch.tensor([0.0, 0.0, z_init], device=device).unsqueeze(0)
    q = q.unsqueeze(0).to(device)

    return x_init, q


def get_transformations_from_height_map(
    world_config: WorldConfig,
    phys_engine_cfg: PhysicsEngineConfig,
    robot_model: RobotModelConfig,
    vel_ctrl: list[VelocitiesStamped],
    device,
) -> list[TransformStamped]:
    """Compute a list of SO3 pairs for the given wheel velocities

    Parameters
    ----------
        TODO:

    Returns:
    -------
    list[TransformStamped]
        List of transformations between steps with timestamp.
    """

    engine = DPhysicsEngine(phys_engine_cfg, robot_model, device)
    state = PhysicsState.dummy(batch_size=1, device=device)
    state.x, state.q = find_init_rob_position(world_config, robot_model, device)

    start_time = vel_ctrl[0].timestamp
    stop_time = vel_ctrl[-1].timestamp
    t = start_time

    transformations = []
    prev_x = state.x.clone()
    prev_q = state.q.clone()

    curr_idx = 0

    while t <= stop_time:
        while curr_idx + 1 < len(vel_ctrl) and t >= vel_ctrl[curr_idx + 1].timestamp:
            curr_idx += 1

        # Linear Interpolation of Velocity
        if curr_idx + 1 < len(vel_ctrl):
            t0 = vel_ctrl[curr_idx].timestamp
            t1 = vel_ctrl[curr_idx + 1].timestamp
            v0 = vel_ctrl[curr_idx].velocities
            v1 = vel_ctrl[curr_idx + 1].velocities

            # Avoid division by zero
            dt = max(t1 - t0, 1e-6)
            ratio = (t - t0) / dt

            curr_velocities = v0 + (v1 - v0) * ratio
            curr_acceleration = (v1 - v0) / dt
        else:
            # End of trajectory, hold last velocity, zero acceleration
            curr_velocities = vel_ctrl[curr_idx].velocities
            curr_acceleration = torch.zeros_like(curr_velocities)

        # Update state wheel velocities to match interpolated input
        state.wheel_omega = curr_velocities.unsqueeze(0)

        # Forward with acceleration control
        dstate = engine.forward(state, curr_acceleration, world_config)
        state = engine.update_state(state, dstate)

        rel_trans = state.x - prev_x

        R_prev = quat_to_rotmat(prev_q)
        R_curr = quat_to_rotmat(state.q)
        R_rel = R_prev.transpose(-1, -2) @ R_curr

        transformations.append(
            TransformStamped(
                timestamp=t,
                transl_vec=rel_trans,
                rot_mat=R_rel,
            )
        )

        prev_x = state.x.clone()
        prev_q = state.q.clone()
        t += phys_engine_cfg.dt

    return transformations
