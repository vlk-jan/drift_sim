import math
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Tuple, Union

import torch
from tensordict import TensorClass

from simulator.numerical import interpolate_grid
from utils.geometry import quat_to_rotmat


# --- Configuration classes ---
@dataclass
class WorldConfig:
    """Physical constants of the world and the coordinates of the world frame.

    Grids use torch's "xy" meshgrid indexing (first dim y, second x, both
    increasing with index); build them with ``torch.meshgrid(y, x, indexing="xy")``.
    Origin [0, 0] is the grid center.

    z_grid (torch.Tensor): z-coordinates of the grid.
    normals (torch.Tensor): terrain normals, (B, 3, grid_dim, grid_dim).
    grid_res (float): metric distance between adjacent cell centers.
    max_coord (float): distance from the grid center to the outermost cell
        center (grid_sample align_corners=True), i.e. ``E - grid_res / 2`` for a
        grid whose edges span +-E; passing E compresses queries by (n-1)/n. See
        ``utils/misc.py::world_cfg_from_config``. Scalar or per-axis (x, y).
    side_range / fwd_range (Tuple[float, float]): metric grid extents. The names
        are swapped: ``side_range`` is the first (x / forward) axis and
        ``fwd_range`` the second (y / lateral), consistently with
        ``utils/pcd.py::get_height_map``. Harmless while extents are square and
        zero-centred; pinned by ``tests/test_world_config.py``.
    k_stiffness (float or torch.Tensor): contact stiffness in N/m. A rollout takes
        the robot's ``contact_stiffness`` (models/simulator.py::_build_world_config);
        the 400_000 default only serves geometry-only uses (seating, plots).
    k_friction (float or torch.Tensor): terrain friction, default 0.5.
    """

    z_grid: torch.Tensor
    grid_res: float
    max_coord: Union[float, Tuple[float, float]]
    side_range: Tuple[float, float] = (0.0, 0.0)
    fwd_range: Tuple[float, float] = (0.0, 0.0)
    k_stiffness: Union[float, torch.Tensor] = 400_000.0
    k_friction: Union[float, torch.Tensor] = 0.5

    def __post_init__(self):
        self.z_grid = self.z_grid.to(torch.float32)

        # max_coord as a 2-vector for grid_sample broadcasting.
        if isinstance(self.max_coord, (list, tuple)):
            self.max_coord = torch.tensor(
                self.max_coord, dtype=torch.float32, device=self.z_grid.device
            )
        elif isinstance(self.max_coord, (float, int)):
            self.max_coord = torch.tensor(
                [self.max_coord, self.max_coord],
                dtype=torch.float32,
                device=self.z_grid.device,
            )
        else:
            self.max_coord = self.max_coord.to(
                dtype=torch.float32, device=self.z_grid.device
            )

        # Heightmap gradients along x/y ("xy" indexing: dim 1 is y, dim 2 is x).
        z_grid_grad = torch.stack(
            torch.gradient(
                self.z_grid, spacing=self.grid_res, dim=(2, 1), edge_order=2
            ),
            dim=1,
        )  # (B, 2, D, D)
        ones = torch.ones_like(z_grid_grad[:, 0]).unsqueeze(1)
        self.normals = torch.cat((-z_grid_grad, ones), dim=1)
        self.normals = self.normals / torch.linalg.norm(
            self.normals, dim=1, keepdim=True
        )

        if self.side_range == (0.0, 0.0):
            self.side_range = (-self.max_coord[0], self.max_coord[0])
        if self.fwd_range == (0.0, 0.0):
            self.fwd_range = (-self.max_coord[1], self.max_coord[1])

        if not isinstance(self.k_friction, torch.Tensor):
            self.k_friction = torch.full_like(
                self.z_grid, self.k_friction, dtype=torch.float32
            )
        else:
            self.k_friction = self.k_friction.to(torch.float32)

        if not isinstance(self.k_stiffness, torch.Tensor):
            self.k_stiffness = torch.full_like(
                self.z_grid, self.k_stiffness, dtype=torch.float32
            )
        else:
            # The robot's tyre constant: no gradient into the grid (both engines).
            self.k_stiffness = self.k_stiffness.to(torch.float32).detach()

        # One 6-channel grid [z, nx, ny, nz, friction, stiffness] for a single
        # grid_sample per step.
        self._sample_grid = torch.cat(
            [
                self.z_grid.unsqueeze(1),  # (B, 1, H, W)
                self.normals,  # (B, 3, H, W)
                self.k_friction.unsqueeze(1),  # (B, 1, H, W)
                self.k_stiffness.unsqueeze(1),  # (B, 1, H, W)
            ],
            dim=1,
        )  # (B, 6, H, W)

    @cached_property
    def grid_center(self) -> torch.Tensor:
        """Return the (x, y) grid center from the side and fwd bounds.

        Cached, as the torch engine reads it every step, and built without a host
        sync whether the bounds are floats or device scalars.
        """
        center_y = (self.fwd_range[0] + self.fwd_range[1]) / 2.0
        center_x = (self.side_range[0] + self.side_range[1]) / 2.0
        return torch.stack(
            [
                torch.as_tensor(c, dtype=torch.float32).to(
                    self.z_grid.device, non_blocking=True
                )
                for c in (center_x, center_y)
            ]
        )


@dataclass
class RobotModelConfig:
    """Mass properties and wheel geometry of one robot.

    Every field is required: only robot_model_config() may state robot geometry.
    """

    mass: float
    cog: torch.Tensor
    wheel_radii: torch.Tensor
    wheel_centers: torch.Tensor
    inertia_tensor: torch.Tensor
    # Whole-robot tyre stiffness (N/m): static sag on flat ground is m*g/k, since
    # each wheel is a spring of k / n_wheels on its deepest rim point.
    contact_stiffness: float
    # Height of the rig's lidar height map above the wheel-contact model on rigid
    # ground (m): proprio labels add it so rigid ground is hr = 0, as hr_rigid says.
    heightmap_datum_offset: float


def num_sim_steps(T_horizon, dt) -> int:
    """Physics steps for a horizon, shared by both wrappers: ceil, so the rollout
    covers the whole horizon, after rounding off float noise (1.0 / 2e-3 is
    500.0000001 or 499.99997 depending on precision; 3 decimals, as a float32
    horizon like tensor(0.3) is off by ~1e-5 steps)."""
    n = math.ceil(round(float(T_horizon) / dt, 3))
    if n <= 0:
        raise ValueError(f"horizon {float(T_horizon)} s gives {n} steps at dt {dt}")
    return n


# --- Robot models: the single source of truth for wheel geometry ---
# Anything needing wheel positions/radii must call robot_model_config() rather
# than restate numbers; stale copies once put contacts 0.2-0.4 m off the wheels.
def helhest_robot_model() -> "RobotModelConfig":
    """Return HELHEST's wheel geometry and mass properties.

    WHEEL GEOMETRY IS THE URDF'S (`/robot_description`, latched in every bag).
    base_link is the front axle: front wheels (0, +-0.36), rear (-0.697, 0),
    radius 0.36. Verified 2026-09-30 by reading the published URDF out of both
    the 2025-09-12 v1 bag and the 2026-09-29 v2 bag: identical in both.

    MASS, COG AND INERTIA ARE NOT THE URDF'S, despite what this docstring
    claimed until 2026-09-30. They arrived in commit "update robot config"
    (2026-01-13) with no source recorded. The URDF's own links sum to 67.0 kg
    with the centre of mass 0.171 m behind the front axle and inertia
    diag ~(3.38, 4.04, 7.17); this model carries 93.8 kg, 0.255 m and
    ~(8.65, 10.64, 19.29) -- 1.4x the mass and 2.6x the inertia.

    THE MASS HERE IS THE RIGHT ONE. The user puts the robot at about 96 kg
    (stated 2026-09-30), so 93.8 is within 2-3% and the URDF is 30% low --
    unsurprising, since its inertias are exact uniform boxes and its battery is
    2 kg. The COG and INERTIA remain unverified (issue 36 in docs/METHOD.md): a
    weight does not fix either, and inertia enters the transient response the
    trajectory loss scores.

    THE URDF IS STALE IN OTHER RESPECTS, which is the reason not to adopt it
    wholesale: the copy latched in the 2026-09-29 bag still declares an Ouster
    `os_sensor` and four e-CAM cameras, a sensor suite this robot does not
    carry. Its wheel radius is stale for the same reason -- see
    helhest_v2_robot_model.
    """
    return RobotModelConfig(
        mass=93.8,
        cog=torch.tensor([-0.255, 0.009, 0.0]),
        wheel_radii=torch.tensor([0.36, 0.36, 0.36]),
        wheel_centers=torch.tensor(
            [[0.0, 0.36, 0.0], [0.0, -0.36, 0.0], [-0.697, 0.0, 0.0]]
        ),
        inertia_tensor=torch.tensor(
            [[8.65, -0.18, 0.0], [-0.18, 10.64, 0.0], [0.0, 0.0, 19.29]]
        ),
        contact_stiffness=400_000.0,  # unmeasured guess; v1 tyres no longer exist
        heightmap_datum_offset=0.0,  # unmeasured; v1 maps are legacy
    )


CONTACT_STIFFNESS_V2 = 210_000.0  # N/m, top of the measured range; see helhest_v2_robot_model
# Median (height map - wheel-contact z) over driven cells of the paved Stromovka
# training scenes 0000/0002 (4.3/3.8 cm; held-out 0001: 4.2), measured 2026-10-22
# after the window-start fix: the lidar ground sits above axle - r on rigid ground.
# This is the datum of the p90 maps of HELHEST v2.0-v2.4.
HEIGHTMAP_DATUM_OFFSET_V2 = 0.040  # m
# The same rule on HELHEST v2.5's median maps (2026-10-23): paved 0000-0002 incl.
# held-out 0001, measured driven cells at 1-2 m, 4 s label horizon, contacts from
# the scan pose: 3.24 cm (20 s block bootstrap 95 % CI 3.17-3.32, n 288,113 cells,
# 175 blocks; per scene 3.01/3.42/3.23). The same rule gives 3.90 on v2.4.
HEIGHTMAP_DATUM_OFFSET_V2_MEDIAN = 0.032  # m


def helhest_v2_robot_model() -> "RobotModelConfig":
    """Return HELHEST on the v2 rig (Odin1 + Fixposition), measured 2026-09-29.

    Differs from helhest_robot_model() in the wheel RADIUS, which is 0.334 and
    not the URDF's 0.36, and in the MASS: the v2 rig weighs 106 kg (user,
    2026-10-22; the Odin1 + Fixposition fit is heavier than the Livox rig).
    Inertia is scaled with the mass (same geometry assumed; where the extra
    13 kg sits is not measured), cog is kept. Until 2026-10-22 only the radius
    moved, so arms before that date simulated the v2 rig at 93.8 kg.

    WHEEL CENTRES are the URDF's and are trusted: `/robot_description` is a
    latched topic in every v2 bag, and the 26-09-14, 26-09-22 and 26-09-28 bags
    all give left (0, +0.36, 0), right (0, -0.36, 0), rear (-0.697, 0, 0).

    THE RADIUS IS NOT. The URDF says 0.36, but git history (helhest-common,
    inside subtdata:/data/helhest/helhest-jr-workspaces.tar.gz) shows
    `wheel_radius` was set to 0.36 on 2025-04-02 in commit 04daac0 and never
    edited again -- 18 months before this rig existed. It is inherited, not
    measured, and the user confirmed 2026-09-25 that the current wheels differ.

    0.334 is metres of ground per radian of `/joint_states` wheel rate, from two
    independent measurements on HELHEST_v2_0 agreeing to 0.5%: GT speed over
    wheel rate (0.336, dataset-tools-63) and total path over integrated rotation
    (0.334 on the made-surface scenes, 0.3268 pooled over all 12). Slip only
    inflates integrated rotation, so the made-surface scenes are the estimate and
    every figure is a lower bound. The URDF's 0.36 over-predicts the driven path
    by 10% -- 5396 m against 4899 m actual.

    Caveat kept in view: this is the constant that maps a REPORTED joint rate to
    ground travel, which is exactly what the rollout needs, since the trajectory
    loss drives the simulator from `joint_states` and scores it against GT. It is
    the PHYSICAL radius only if `/joint_states` is wheel-side; the rate reaches it
    from the ODrive controllers via the ESP and no one has ruled out a gear or
    encoder scale. The physical radius is separately indicated near 0.317 by the
    lidar ground plane on the flat plaza.

    STIFFNESS is measured (user, 2026-10-22, robot standing on flat floor vs
    each wheel just lifted; axle height, cm): left 33.1/33.6, right 32.5/33.5,
    rear 33.2/33.9 -> mean sag 7.3 mm (5-10 mm, one-hand lift), k = 106 * 9.81
    / 0.0073 = 1.42e5 N/m (1.0e5-2.1e5). Same probe: xy identical across
    1.0e5-4e5; |z| 2.93 cm vs 2.70 at the old 4e5 guess.
    SET TO 2.1e5 (2026-10-24), the top of that range (sag 4.9 mm): with 32 rim
    points the 6.4 mm neighbours touch inside a 7.3 mm sag, the wider patch adds
    scrub and the sim under-turns more. Re-tuned with damping_alpha 2.0 on the
    v2.5 fidelity windows (rigid 0000/0002-0006, rough 0021-0023/0028-0033,
    held-out 0001; mu <= 1.0) against the 16-point engine at 1.42e5 / 1.0: xy
    ties on rigid and rough, held-out xy -0.2 mm, |z| -1.7 to -2.7 mm, held-out
    yaw +0.03 deg. The constants were picked on those same windows.
    The same numbers confirm the RADIUS as physical: mean free radius 0.337,
    loaded 0.329, and the rolling radius of a pneumatic tyre r_free - sag/3 =
    0.334 -- so /joint_states is wheel-side, no gear or encoder scale.

    SENSITIVITY (2026-10-22, scripts/analysis/probe_robot_constants.py: rigid GT
    support, constant mu, 512 windows of the made-surface TRAINING scenes
    0000/0002-0006 of HELHEST_v2_4, mean xy error over 0.5-2 s; baseline 3.58 cm):
    radius is sharp and 0.334 is the optimum (0.325 +0.5 cm, 0.345 +1.3 cm, 0.36
    +4.3 cm); cog x -0.255 is the optimum (URDF -0.171 +0.1 cm, +-0.1 m +0.5 to
    +0.9 cm); inertia x0.75-x2 within 0.1 cm (x0.5 +0.4 cm), so the mass-scaled
    tensor is not the error's limit; k_stiffness 1e5-4e5 N/m within 0.1 cm (worse
    from 8e5); slip_ref_ratio 0.1-0.2 tie (0.02 +0.2 cm, 0.4 +0.2 cm);
    slip_floor_velocity 0.01-0.2 m/s has no effect. mu 0.8-1.5 within 0.15 cm.
    """
    base = helhest_robot_model()
    mass = 106.0
    return RobotModelConfig(
        mass=mass,
        cog=base.cog,
        wheel_radii=torch.tensor([0.334, 0.334, 0.334]),
        wheel_centers=base.wheel_centers,
        inertia_tensor=base.inertia_tensor * (mass / base.mass),
        contact_stiffness=CONTACT_STIFFNESS_V2,
        heightmap_datum_offset=HEIGHTMAP_DATUM_OFFSET_V2,
    )


def helhest_v2_median_robot_model() -> "RobotModelConfig":
    """helhest_v2 on median-built height maps (HELHEST v2.5 on).

    The robot is the same; only the height-map datum differs, because the lidar
    ground a median map reads on rigid ground sits lower than a p90 map's. The
    datum lives in the robot model so that the run record (robot +
    robot_constants) pins it; datasets.helhest.robot_model_name_for_root picks
    this model for a v2 root whose maps are median-built.
    """
    return replace(
        helhest_v2_robot_model(),
        heightmap_datum_offset=HEIGHTMAP_DATUM_OFFSET_V2_MEDIAN,
    )


ROBOT_MODELS = {
    "helhest": helhest_robot_model,
    "helhest_v2": helhest_v2_robot_model,
    "helhest_v2_median": helhest_v2_median_robot_model,
}


def robot_model_config(robot: str = "helhest") -> "RobotModelConfig":
    """Return wheel geometry, mass and inertia of a supported robot."""
    try:
        return ROBOT_MODELS[robot]()
    except KeyError:
        raise ValueError(
            f"Unsupported robot: {robot!r} (have {sorted(ROBOT_MODELS)})"
        ) from None


@dataclass
class PhysicsEngineConfig:
    """Physical constants of the simulation: gravity, time step, limits.

    Attributes:
        dt (float): integration time step. Default 2e-3 (500 Hz).
        g (float): gravity. Default 9.81 m/s^2.
        torque_limit (float): clip on the net torque about the CoG. Default 5000 Nm.
        damping_alpha (float): contact damping in units of critical heave
            damping, at the rest sag only: it scales with depth and is capped
            per point (simulator/engine.py::_calculate_spring_force). Default 2.0.
        ang_damping (float): angular damping coefficient. Default 0.1.
        com_transfer (bool): integrate acceleration onto the body origin rather
            than treating the origin as the CoM. See the field comment below.
        slip_ref_ratio (float): slip ratio s = |v_t| / max(|omega_w| r_w,
            slip_floor_velocity) at which the tangential force reaches ~76% of
            mu * F_N, via tanh(s / slip_ref_ratio). Default 0.1.
        slip_floor_velocity (float): floor (m/s) on the rolling speed in the
            slip-ratio denominator, so a stopped wheel reduces to the old
            velocity law s = |v_t| / slip_floor_velocity. Default 0.05.
        Changing either changes the dynamics; trajectory metrics are not
        comparable across values.
    """

    # 2e-3 from scripts/analysis/sweep_dt.py: half the steps of 1e-3 with
    # indistinguishable error and gradients; fidelity degrades from 3e-3.
    dt: float = 2e-3
    g: float = 9.81
    torque_limit: float = 5000.0
    # 2.0 since 2026-10-24, re-tuned with the 32-point engine and the v2 stiffness
    # 2.1e5 (helhest_v2_robot_model); 1.0 before.
    damping_alpha: float = 2.0
    ang_damping: float = 0.1
    # Slip RATIO law (since 2026-10-22): |F_t| = mu F_N tanh(s / slip_ref_ratio),
    # s = |v_t| / max(|omega_w| r_w, slip_floor_velocity). With the old absolute
    # slip speed (tanh(|v_t| / 0.1)) only mu / v_ref was identifiable below
    # saturation and the fitted global optimum sat at ~1.6 against a head ceiling
    # of 1.0. In slip ratio, mu is the peak traction coefficient: tyres peak at
    # 10-20 % slip (Pacejka), so 0.1 puts ~76 % of mu at s = 0.1 and saturation
    # by s ~ 0.3.
    slip_ref_ratio: float = 0.1
    # Rolling-speed floor (m/s); a stationary or barely turning wheel uses the
    # velocity form s = |v_t| / 0.05, which keeps the force finite and smooth.
    slip_floor_velocity: float = 0.05
    # state.x is the body origin (front axle), not the CoM (|r| = 0.255 m on
    # helhest). On = the correct rigid-body dynamics (alpha x r and
    # omega x (omega x r) terms); checkpoints before 2026-10-22 trained with it off.
    com_transfer: bool = True


# --- Physics classes ---
class PhysicsState(TensorClass):
    """Robot state; all vectors in the world frame.

    Attributes:
        x (torch.Tensor): body origin position, (B, 3).
        xd (torch.Tensor): linear velocity, (B, 3).
        q (torch.Tensor): orientation quaternion wxyz, (B, 4).
        omega (torch.Tensor): angular velocity, (B, 3).
        wheel_omega (torch.Tensor): wheel spin rates, (B, 3).
    """

    x: torch.Tensor
    xd: torch.Tensor
    q: torch.Tensor
    omega: torch.Tensor
    wheel_omega: torch.Tensor

    @staticmethod
    def dummy(**kwargs) -> "PhysicsState":
        """Create a zero PhysicsState; keyword arguments override fields."""
        batch_size = kwargs.pop("batch_size", None)
        device = kwargs.pop("device", torch.get_default_device())
        if kwargs:
            t = next(iter(kwargs.values()))
            if batch_size is not None and batch_size != t.shape[0]:
                raise ValueError(
                    "Specified batch size does not match the shape of the tensors."
                )
            batch_size = t.shape[0]
        elif batch_size is None:
            batch_size = 0
        base = {
            "x": torch.zeros(batch_size, 3),
            "xd": torch.zeros(batch_size, 3),
            "q": torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(
                batch_size, 1
            ),
            "omega": torch.zeros(batch_size, 3),
            "wheel_omega": torch.zeros(batch_size, 3, device=device),
            "batch_size": batch_size,
        }
        return PhysicsState(**{**base, **kwargs}, device=device)


class PhysicsStateDer(TensorClass):
    """Time derivative of PhysicsState; world frame.

    Attributes:
        xdd (torch.Tensor): linear acceleration, (B, 3).
        omega_d (torch.Tensor): angular acceleration, (B, 3).
    """

    xdd: torch.Tensor
    omega_d: torch.Tensor


# --- Other functions ---
def find_init_rob_position(
    world_config: WorldConfig,
    robot_model: RobotModelConfig,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the initial pose with all 3 wheels resting on the terrain.

    Iterates: sample terrain under the wheel centres, fit the plane through
    terrain_z + R, align the body up-axis with its normal, and set the height so
    no wheel is below its target (converges in ~3-5 iterations).

    Returns:
        x_init: (B, 3)  world position of the body origin
        q_init: (B, 4)  unit quaternion [w, x, y, z]
    """
    # Host constants go over asynchronously: no host sync in the fit.
    def to_device(x):
        return torch.as_tensor(x, dtype=torch.float32).to(device, non_blocking=True)

    grid_center = world_config.grid_center.to(device)
    wheel_centers = to_device(robot_model.wheel_centers)  # (3, 3)
    wheel_radii = to_device(robot_model.wheel_radii)
    R_wheel = wheel_radii.mean()

    q_identity = to_device([1.0, 0.0, 0.0, 0.0]).expand(batch_size, 4)
    q_flip = to_device([0.0, 1.0, 0.0, 0.0]).expand(batch_size, 4)
    z_up = to_device([[0.0, 0.0, 1.0]]).expand(batch_size, 3)

    q = q_identity.clone()
    x = torch.zeros(batch_size, 3, device=device)

    for _ in range(10):
        R_mat = quat_to_rotmat(q)  # (B, 3, 3)
        wc_world = torch.einsum("bij,kj->bki", R_mat, wheel_centers) + x.unsqueeze(
            1
        )  # (B, 3, 3)
        wc_xy = wc_world[:, :, :2]  # (B, 3, 2)

        z_terrain = interpolate_grid(
            world_config.z_grid,
            wc_xy - grid_center,
            world_config.max_coord,
        ).reshape(batch_size, 3)  # (B, 3)

        # Plane through the desired wheel-centre heights.
        wc_z_target = z_terrain + R_wheel  # (B, 3)
        P = torch.cat([wc_xy, wc_z_target.unsqueeze(-1)], dim=-1)  # (B, 3, 3)
        v1 = P[:, 1, :] - P[:, 0, :]
        v2 = P[:, 2, :] - P[:, 0, :]
        n = torch.linalg.cross(v1, v2, dim=1)  # (B, 3)
        n = torch.where(n[:, 2:3] < 0, -n, n)
        n = n / (n.norm(dim=1, keepdim=True) + 1e-6)

        # Shortest-arc rotation of the body up-axis onto the plane normal.
        v_cross = torch.linalg.cross(z_up, n, dim=1)
        c = (z_up * n).sum(dim=1, keepdim=True)
        denom = (2 * (1 + c)).clamp_min(1e-6).sqrt()
        q_new = torch.cat([denom * 0.5, v_cross / denom], dim=1)
        q_new = q_new / (q_new.norm(dim=1, keepdim=True) + 1e-6)
        # Only the antiparallel case is singular; at c -> 1 the formula above
        # tends to identity on its own (guarding |c| snapped slopes < 0.81 deg flat).
        q_new = torch.where((c < -1.0 + 1e-4).expand_as(q_new), q_flip, q_new)

        # z_body = target_z - (R_new @ wc_body)_z per wheel; the max keeps every
        # wheel at or above its contact height.
        R_new = quat_to_rotmat(q_new)  # (B, 3, 3)
        wc_rot_z = torch.einsum("bi,ki->bk", R_new[:, 2, :], wheel_centers)  # (B, 3)
        z_new = (wc_z_target - wc_rot_z).max(dim=1).values  # (B,)

        x_new = torch.zeros(batch_size, 3, device=device)
        x_new[:, 2] = z_new

        # Fixed passes, no early exit: converged iterates are fixed points, and a
        # convergence test would cost device->host syncs.
        q, x = q_new, x_new

    return x, q


def raise_to_clearance(engine, x, q, world_config):
    """Raise the robot until all wheel circumference sample points clear the terrain.

    The plane fit only makes the wheel triangle tangent; circumference points can
    still penetrate and the contact springs launch the robot at t=0. Returns a new
    position (`x` may be a view into GT data).

    A vertical lift leaves every point's xy, so the exact lift is the largest
    vertical gap terrain_z - point_z (dh / nz), in one step. Differentiable in
    the terrain: the start height is a function of the support under the
    deepest point, and a detached lift made the support gradient disagree with
    finite differences (cos 0.98 -> 1.000 at 2 s, rough ground, GT-seeded start).
    The attitude is not differentiated (see Simulator.fit_plane_pose). `engine`
    only supplies geometry, so both simulator wrappers share this.
    """
    R_body = quat_to_rotmat(q.detach())  # (B, 3, 3)
    wp_rel = torch.einsum("bij,wpj->bwpi", R_body, engine._wheel_pos_rel)  # (B,W,P,3)
    wc_world = (R_body @ engine._wheel_centers.T).transpose(-1, -2) + x.unsqueeze(1)
    dh, normals, _, _ = engine._sample_terrain(wp_rel + wc_world.unsqueeze(2), world_config)
    gap = -dh / normals[..., 2:3].clamp_min(1e-3)  # vertical gap, (B, W, P, 1)
    lift = gap.flatten(1).max(dim=1).values.clamp_min(0)  # (B,)
    return x + torch.stack([torch.zeros_like(lift), torch.zeros_like(lift), lift], -1)


@torch.no_grad()
def compute_wheel_contacts(engine, traj_se3, world_config, chunk_size=64):
    """Return the per-step in-contact mask of one rollout's wheel points.

    `traj_se3` is a (T, 4, 4) world-frame pose stack for one batch element.
    Returns a (T, W*P) bool array of penetrating (`dh < 0`) wheel-circumference
    points, a post-hoc debugging signal for SimView exports. Chunked over T.
    """
    T = traj_se3.shape[0]
    R = traj_se3[:, :3, :3]
    x = traj_se3[:, :3, 3]
    out = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        n = end - start
        Rc, xc = R[start:end], x[start:end]
        wp_rel = torch.einsum("tij,wpj->twpi", Rc, engine._wheel_pos_rel)
        wc_world = torch.einsum(
            "tij,wj->twi", Rc, engine._wheel_centers
        ) + xc.unsqueeze(1)
        wp_world = wp_rel + wc_world.unsqueeze(2)  # (n, W, P, 3)
        wc = WorldConfig(
            z_grid=world_config.z_grid.expand(n, -1, -1),
            grid_res=world_config.grid_res,
            max_coord=world_config.max_coord,
            side_range=world_config.side_range,
            fwd_range=world_config.fwd_range,
            k_friction=world_config.k_friction.expand(n, -1, -1),
            k_stiffness=world_config.k_stiffness.expand(n, -1, -1),
        )
        dh, _, _, _ = engine._sample_terrain(wp_world, wc)  # (n, W, P, 1)
        out.append((dh.squeeze(-1) < 0).reshape(n, -1))
    return torch.cat(out, dim=0)  # (T, W*P) bool


def generate_controls(
    duration,
    dt,
    num_robots,
    device,
    seed=42,
    wheel_radius=None,
    current_vel_linear=1.0,
    clamp=None,
):
    """Generate random-walk angular wheel velocities.

    wheel_radius (default: helhest URDF) converts linear speed to rad/s.
    """
    if wheel_radius is None:
        wheel_radius = float(robot_model_config("helhest").wheel_radii[0])
    torch.manual_seed(seed)
    steps = int(duration / dt)
    controls = []
    timestamps = []

    current_omega = current_vel_linear / wheel_radius

    current_vel = torch.tensor(
        [current_omega, current_omega, current_omega], device=device
    )
    current_vel = current_vel.unsqueeze(0).expand(num_robots, -1).clone()

    for i in range(steps):
        ts = i * dt
        noise = (
            torch.randn(num_robots, 3, device=device) * 0.05 * (1.0 / wheel_radius)
        )  # linear noise scaled to angular
        current_vel += noise * dt

        if clamp is not None:
            current_vel = torch.clamp(current_vel, -clamp, clamp)

        controls.append(current_vel.clone())
        timestamps.append(ts)

    return torch.stack(controls).permute(1, 0, 2), torch.tensor(
        timestamps, device=device
    ).expand(num_robots, -1)


def generate_dynamic_maneuvers(duration, dt, num_robots, device, seed=42):
    """Generate a scripted sequence of dynamic maneuvers."""
    import numpy as np

    torch.manual_seed(seed)
    steps = int(duration / dt)
    controls = []
    timestamps = []

    wheel_radius = float(robot_model_config("helhest").wheel_radii[0])
    max_speed_lin = 5.0  # m/s
    max_omega = max_speed_lin / wheel_radius

    for i in range(steps):
        ts = i * dt
        progress = ts / duration

        target_vel = torch.zeros(3, device=device, dtype=torch.float32)

        if progress < 0.05:
            # 0-5%: warmup (linear acceleration).
            frac = progress / 0.05
            w = max_omega * frac
            target_vel = torch.tensor([w, w, w], device=device, dtype=torch.float32)

        elif progress < 0.15:
            # 5-15%: cruise.
            w = max_omega
            target_vel = torch.tensor([w, w, w], device=device, dtype=torch.float32)

        elif progress < 0.30:
            # 15-30%: slalom.
            local_p = (progress - 0.15) / 0.15
            w_base = max_omega * 0.7
            w_diff = (max_omega * 0.6) * np.sin(local_p * 3 * 2 * np.pi)
            target_vel = torch.tensor(
                [w_base - w_diff, w_base + w_diff, w_base],
                device=device,
                dtype=torch.float32,
            )

        elif progress < 0.50:
            # 30-50%: square path, straight then hard left turn.
            local_p = (progress - 0.30) / 0.20
            segment = (local_p * 4) % 1.0  # 0.0 to 1.0 within a segment

            if segment < 0.7:
                w = max_omega * 0.8
                target_vel = torch.tensor([w, w, w], device=device, dtype=torch.float32)
            else:
                w = max_omega
                target_vel = torch.tensor(
                    [w * 0.0, w, w * 0.5], device=device, dtype=torch.float32
                )

        elif progress < 0.60:
            # 50-60%: stop, then hard acceleration.
            local_p = (progress - 0.50) / 0.10
            if local_p < 0.4:
                target_vel = torch.tensor(
                    [0.0, 0.0, 0.0], device=device, dtype=torch.float32
                )
            else:
                w = max_omega
                target_vel = torch.tensor([w, w, w], device=device, dtype=torch.float32)

        elif progress < 0.75:
            # 60-75%: reverse maneuvering.
            local_p = (progress - 0.60) / 0.15
            w_base = -max_omega * 0.6
            w_diff = (max_omega * 0.3) * np.sin(local_p * 2 * 2 * np.pi)
            target_vel = torch.tensor(
                [w_base + w_diff, w_base - w_diff, w_base],
                device=device,
                dtype=torch.float32,
            )

        elif progress < 0.85:
            # 75-85%: spins, clockwise then counter-clockwise.
            local_p = (progress - 0.75) / 0.10
            w_spin = max_omega * 0.8
            if local_p < 0.5:
                target_vel = torch.tensor(
                    [-w_spin, w_spin, 0.0], device=device, dtype=torch.float32
                )
            else:
                target_vel = torch.tensor(
                    [w_spin, -w_spin, 0.0], device=device, dtype=torch.float32
                )

        elif progress < 0.95:
            # 85-95%: agility (rapid changes).
            t_val = ts * 10.0
            n1 = np.sin(t_val)
            n2 = np.cos(t_val * 1.3)
            w_base = max_omega * 0.8
            w_l = w_base * (0.5 + 0.5 * n1)
            w_r = w_base * (0.5 + 0.5 * n2)
            w_rear = (w_l + w_r) / 2.0
            target_vel = torch.tensor(
                [w_l, w_r, w_rear], device=device, dtype=torch.float32
            )

        else:
            # 95-100%: stop.
            target_vel = torch.tensor(
                [0.0, 0.0, 0.0], device=device, dtype=torch.float32
            )

        batch_vel = target_vel.unsqueeze(0).expand(num_robots, -1)
        controls.append(batch_vel)
        timestamps.append(ts)

    return torch.stack(controls).permute(1, 0, 2), torch.tensor(
        timestamps, device=device
    ).expand(num_robots, -1)


WHEEL_VISUAL_WIDTH = 0.142  # tyre width, visual only -- no dynamics reads it

# THE DRAWN CHASSIS. Visual only: no dynamics reads it, the engine knows the
# robot as three wheel contacts plus mass/inertia (simulator/engine*.py).
#
# These are the machine's envelope, matching dataset_tools' viewer
# (dataset_tools/viewer/build.py CHASSIS / CHASSIS_CENTER) so the same robot is
# drawn in both tools. base_link IS the front axle, so a box centred on the body
# origin is wrong in an obvious way: it hangs half its length out in front of
# the front wheels. The box is therefore pushed back by CHASSIS_CENTER, which
# puts it at x in [-0.50, +0.20] -- behind the wheel holders (tf_static
# -0.477) and up to the front sensor face (+0.10..+0.17).
#
# NOT the URDF's base_link box, which is `0.26 0.6 0.18` at xyz `-0.047 0 0`
# (/robot_description, identical in the 2025-09-12 v1 bag and the 2026-09-29 v2
# bag). That box is the frame member alone; it leaves out the motors, the
# battery and the wheel forks, so it draws a machine a quarter the length of
# the real one. Half-extents here, as SimView wants; the URDF quotes full
# extents.
#
# Until 2026-09-30 this was hx=0.5, hy=0.3, hz=0.15 centred on base_link: a
# 1.00 x 0.60 x 0.30 m slab reaching 0.5 m ahead of the front axle and stopping
# 0.2 m short of the rear wheel, which is what made the robot look wrong.
HELHEST_CHASSIS = {"hx": 0.35, "hy": 0.25, "hz": 0.15}
HELHEST_CHASSIS_CENTER = (-0.15, 0.0, 0.0)


def chassis_positions(positions, quats):
    """Offset a base_link path to the drawn chassis box's centre.

    The box is drawn about the body origin it is given, so the body that
    carries the trajectory rides HELHEST_CHASSIS_CENTER behind base_link,
    rotated by each pose. add_helhest_visual_wheels undoes the same offset, so
    the wheels stay on the axles.

    Args:
        positions: (..., 3) base_link positions, array or tensor. Leading
                   dimensions are free, so a (T, B, 3) batch works.
        quats:     (..., 4) orientations as [w, x, y, z], same leading shape.
    Returns:
        positions of the box centre, same shape and type as `positions`.
    """
    import numpy as np

    was_tensor = isinstance(positions, torch.Tensor)
    p = positions.detach().cpu().numpy() if was_tensor else np.asarray(positions)
    q = (
        quats.detach().cpu().numpy()
        if isinstance(quats, torch.Tensor)
        else np.asarray(quats)
    )
    shape = p.shape
    p = p.reshape(-1, 3)
    q = q.reshape(-1, 4)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Rotate the constant body-frame offset by each pose (quaternion sandwich,
    # written out to avoid a scipy dependency in the simulator package).
    ox, oy, oz = HELHEST_CHASSIS_CENTER
    tx = 2.0 * (y * oz - z * oy)
    ty = 2.0 * (z * ox - x * oz)
    tz = 2.0 * (x * oy - y * ox)
    rot = np.stack(
        [
            ox + w * tx + (y * tz - z * ty),
            oy + w * ty + (z * tx - x * tz),
            oz + w * tz + (x * ty - y * tx),
        ],
        axis=1,
    )
    out = (p + rot).reshape(shape)
    return torch.as_tensor(out, dtype=positions.dtype) if was_tensor else out


def chassis_local(points):
    """Re-express body-frame points for the offset chassis body.

    Contact points handed to SimView are local to the body that carries the
    trajectory, and that body is the box, not base_link.
    """
    import numpy as np

    off = np.asarray(HELHEST_CHASSIS_CENTER)
    if isinstance(points, torch.Tensor):
        return points - torch.as_tensor(off, dtype=points.dtype, device=points.device)
    return np.asarray(points) - off


def add_helhest_visual_body(scene, name: str = "box", **kwargs) -> None:
    """Create the chassis box SimView body at the shared envelope."""
    from simview import BodyShapeType

    scene.create_body(
        body_name=name, shape_type=BodyShapeType.BOX, **HELHEST_CHASSIS, **kwargs
    )


def add_helhest_visual_wheels(
    scene, parent: str = "box", robot: str = "helhest"
) -> None:
    """Add rigidly attached visual wheels for Helhest to a SimView scene.

    Geometry comes from robot_model_config(robot) so drawn wheels match the
    simulated contacts; pass the robot the rollout used, since the v1 and v2
    models differ by 26 mm of radius. Hubs are expressed relative to the drawn
    chassis body, which sits HELHEST_CHASSIS_CENTER behind base_link.
    """
    from simview import BodyShapeType, SimViewBody

    model = robot_model_config(robot)
    centers = model.wheel_centers.tolist()
    radii = model.wheel_radii.tolist()
    # Rows are [+y (left), -y (right), rear]; the quaternion lays the cylinder
    # axis along y, signed by the wheel's side.
    names = ("left_wheel", "right_wheel", "rear_wheel")
    for name_i, pos, radius in zip(names, centers, radii):
        sign = 1.0 if pos[1] < 0 else -1.0
        rot = [0.7071068, sign * 0.7071068, 0.0, 0.0]
        hub = [p - o for p, o in zip(pos, HELHEST_CHASSIS_CENTER)]
        scene.add_body_object(
            SimViewBody.create(
                name=name_i,
                body_type=BodyShapeType.CYLINDER,
                radius=float(radius),
                height=WHEEL_VISUAL_WIDTH,
                parent=parent,
                local_transform=hub + rot,
            )
        )
