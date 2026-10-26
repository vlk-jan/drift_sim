import math
from dataclasses import dataclass, replace
from typing import Tuple, Union

import torch

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
    k_friction (float or torch.Tensor): terrain friction, default 0.5.

    Contact stiffness is not a terrain property: both engines read the robot's
    ``contact_stiffness`` (contact_spring_constants).
    """

    z_grid: torch.Tensor
    grid_res: float
    max_coord: Union[float, Tuple[float, float]]
    side_range: Tuple[float, float] = (0.0, 0.0)
    fwd_range: Tuple[float, float] = (0.0, 0.0)
    k_friction: Union[float, torch.Tensor] = 0.5

    def __post_init__(self):
        # A wrapper's unset bounds (None) take the default, +-max_coord (below).
        self.side_range = (0.0, 0.0) if self.side_range is None else self.side_range
        self.fwd_range = (0.0, 0.0) if self.fwd_range is None else self.fwd_range
        self.z_grid = self.z_grid.to(torch.float32)

        # max_coord as a 2-vector for grid_sample broadcasting; max_coord_xy keeps
        # it as given, the warp kernel arguments (floats: no device read).
        if isinstance(self.max_coord, (float, int)):
            self.max_coord = (self.max_coord, self.max_coord)
        self.max_coord_xy = self.max_coord
        if isinstance(self.max_coord, (list, tuple)):
            self.max_coord = torch.tensor(self.max_coord, dtype=torch.float32).to(
                self.z_grid.device, non_blocking=True
            )
        else:
            self.max_coord = self.max_coord.to(
                dtype=torch.float32, device=self.z_grid.device
            )

        # Heightmap gradients along x/y ("xy" indexing: dim 1 is y, dim 2 is x);
        # the normal is (-gx, -gy, 1) normalised, in closed form: linalg.norm over
        # the channel dim is ~20x slower on CPU.
        gx, gy = torch.gradient(
            self.z_grid, spacing=self.grid_res, dim=(2, 1), edge_order=2
        )
        nz = (gx.square() + gy.square() + 1).rsqrt()
        self.normals = torch.stack((-gx * nz, -gy * nz, nz), dim=1)  # (B, 3, D, D)

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

        # One 5-channel grid [z, nx, ny, nz, friction] for a single grid_sample
        # per step.
        self._sample_grid = torch.cat(
            [
                self.z_grid.unsqueeze(1),  # (B, 1, H, W)
                self.normals,  # (B, 3, H, W)
                self.k_friction.unsqueeze(1),  # (B, 1, H, W)
            ],
            dim=1,
        )  # (B, 5, H, W)

        # (x, y) grid center from the side and fwd bounds: grid_center_xy in the
        # bounds' own type (floats for the warp kernels), grid_center a tensor
        # for the torch engine, which reads it every step; built once, without a
        # host sync whether the bounds are floats or device scalars.
        self.grid_center_xy = tuple(
            (r[0] + r[1]) / 2.0 for r in (self.side_range, self.fwd_range)
        )
        self.grid_center = torch.stack(
            [
                torch.as_tensor(c, dtype=torch.float32).to(
                    self.z_grid.device, non_blocking=True
                )
                for c in self.grid_center_xy
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
# than restate numbers: a stale copy puts contacts 0.2-0.4 m off the wheels.
CONTACT_STIFFNESS_V2 = 210_000.0  # N/m, top of the measured range; see helhest_v2_robot_model
# Median (height map - wheel-contact z) over driven cells of the paved Stromovka
# training scenes 0000/0002 (4.3/3.8 cm; held-out 0001: 4.2): the lidar ground
# sits above axle - r on rigid ground. This is the datum of the p90 maps of
# HELHEST v2.0-v2.4.
HEIGHTMAP_DATUM_OFFSET_V2 = 0.040  # m
# The same rule on HELHEST v2.5's median maps: paved 0000-0002 incl. held-out
# 0001, measured driven cells at 1-2 m, 4 s label horizon, contacts from the
# scan pose: 3.24 cm (20 s block bootstrap 95 % CI 3.17-3.32, n 288,113 cells,
# 175 blocks; per scene 3.01/3.42/3.23). The same rule gives 3.90 on v2.4.
HEIGHTMAP_DATUM_OFFSET_V2_MEDIAN = 0.032  # m


def helhest_v2_robot_model() -> "RobotModelConfig":
    """Return HELHEST on the v2 rig (Odin1 + Fixposition).

    WHEEL CENTRES are the URDF's and are trusted: `/robot_description` is a
    latched topic in every v2 bag, and bags from three different days all give
    left (0, +0.36, 0), right (0, -0.36, 0), rear (-0.697, 0, 0); base_link is
    the front axle.

    MASS 106 kg. COG AND INERTIA ARE NOT THE URDF'S, and their source is not
    recorded: the URDF's links sum to 67.0 kg (exact uniform boxes, a 2 kg
    battery) with the centre of mass 0.171 m behind the front axle and inertia
    diag ~(3.38, 4.04, 7.17). The cog and inertia here are those of a 93.8 kg
    model of the lighter Livox rig, the inertia scaled to 106 kg (where the
    extra mass sits is not measured). Both remain unverified (issue 36 in
    docs/METHOD.md); inertia enters the transient response the trajectory loss
    scores. The URDF is stale in other respects too: it declares an Ouster
    `os_sensor` and four e-CAM cameras this robot does not carry.

    THE RADIUS IS NOT. The URDF's 0.36 is helhest-common's `wheel_radius`
    (subtdata:/data/helhest/helhest-jr-workspaces.tar.gz), which predates this
    rig and was never measured; the current wheels differ.

    0.334 is metres of ground per radian of `/joint_states` wheel rate, from two
    independent measurements on HELHEST_v2_0 agreeing to 0.5%: GT speed over
    wheel rate (0.336, dataset-tools-63) and total path over integrated rotation
    (0.334 on the made-surface scenes, 0.3268 pooled over all 12). Slip only
    inflates integrated rotation, so the made-surface scenes are the estimate and
    every figure is a lower bound. The URDF's 0.36 over-predicts the driven path
    by 10% -- 5396 m against 4899 m actual.

    This is the constant that maps a REPORTED joint rate to ground travel,
    which is exactly what the rollout needs, since the trajectory loss drives
    the simulator from `joint_states` and scores it against GT. It is the
    PHYSICAL radius only if `/joint_states` is wheel-side (the rate reaches it
    from the ODrive controllers via the ESP); the tyre measurement under
    STIFFNESS shows it is. The lidar ground plane on the flat plaza separately
    indicates a radius near 0.317.

    STIFFNESS is measured (robot standing on flat floor vs each wheel just
    lifted; axle height, cm): left 33.1/33.6, right 32.5/33.5, rear 33.2/33.9
    -> mean sag 7.3 mm (5-10 mm, one-hand lift), k = 106 * 9.81 / 0.0073 =
    1.42e5 N/m (1.0e5-2.1e5). Same probe: xy identical across 1.0e5-4e5; |z|
    2.93 cm vs 2.70 at 4e5.
    SET TO 2.1e5, the top of that range (sag 4.9 mm): with 32 rim points the
    6.4 mm neighbours touch inside a 7.3 mm sag, the wider patch adds scrub and
    the sim under-turns more. Tuned with damping_alpha 2.0 on the v2.5 fidelity
    windows (rigid 0000/0002-0006, rough 0021-0023/0028-0033, held-out 0001;
    mu <= 1.0) against the 16-point engine at 1.42e5 / 1.0: xy ties on rigid
    and rough, held-out xy -0.2 mm, |z| -1.7 to -2.7 mm, held-out yaw
    +0.03 deg. The constants were picked on those same windows. These figures
    are from contact law v1 (uncapped damping); the fidelity windows are to be
    re-run under the capped law.
    The same numbers confirm the RADIUS as physical: mean free radius 0.337,
    loaded 0.329, and the rolling radius of a pneumatic tyre r_free - sag/3 =
    0.334 -- so /joint_states is wheel-side, no gear or encoder scale.

    SENSITIVITY (scripts/analysis/probe_robot_constants.py: rigid GT support,
    constant mu, 512 windows of the made-surface TRAINING scenes 0000/0002-0006
    of HELHEST_v2_4, mean xy error over 0.5-2 s; baseline 3.58 cm):
    radius is sharp and 0.334 is the optimum (0.325 +0.5 cm, 0.345 +1.3 cm, 0.36
    +4.3 cm); cog x -0.255 is the optimum (URDF -0.171 +0.1 cm, +-0.1 m +0.5 to
    +0.9 cm); inertia x0.75-x2 within 0.1 cm (x0.5 +0.4 cm), so the mass-scaled
    tensor is not the error's limit; k_stiffness 1e5-4e5 N/m within 0.1 cm (worse
    from 8e5); slip_ref_ratio 0.1-0.2 tie (0.02 +0.2 cm, 0.4 +0.2 cm);
    slip_floor_velocity 0.01-0.2 m/s has no effect. mu 0.8-1.5 within 0.15 cm.
    """
    mass = 106.0
    return RobotModelConfig(
        mass=mass,
        cog=torch.tensor([-0.255, 0.009, 0.0]),
        wheel_radii=torch.tensor([0.334, 0.334, 0.334]),
        wheel_centers=torch.tensor(
            [[0.0, 0.36, 0.0], [0.0, -0.36, 0.0], [-0.697, 0.0, 0.0]]
        ),
        inertia_tensor=torch.tensor(
            [[8.65, -0.18, 0.0], [-0.18, 10.64, 0.0], [0.0, 0.0, 19.29]]
        )
        * (mass / 93.8),
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
    "helhest_v2": helhest_v2_robot_model,
    "helhest_v2_median": helhest_v2_median_robot_model,
}


def robot_model_config(robot: str) -> "RobotModelConfig":
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
            slip-ratio denominator, so a stopped wheel reduces to the velocity
            form s = |v_t| / slip_floor_velocity. Default 0.05.
        Changing either changes the dynamics; trajectory metrics are not
        comparable across values.
    """

    # 2e-3 from scripts/analysis/sweep_dt.py: half the steps of 1e-3 with
    # indistinguishable error and gradients; fidelity degrades from 3e-3.
    dt: float = 2e-3
    g: float = 9.81
    torque_limit: float = 5000.0
    # Tuned jointly with the 32-point engine and the v2 stiffness 2.1e5
    # (helhest_v2_robot_model).
    damping_alpha: float = 2.0
    ang_damping: float = 0.1
    # Slip RATIO law: |F_t| = mu F_N tanh(s / slip_ref_ratio),
    # s = |v_t| / max(|omega_w| r_w, slip_floor_velocity). With an absolute slip
    # speed (tanh(|v_t| / 0.1)) only mu / v_ref is identifiable below saturation
    # and the fitted global optimum sits at ~1.6 against a head ceiling of 1.0.
    # In slip ratio, mu is the peak traction coefficient: tyres peak at
    # 10-20 % slip (Pacejka), so 0.1 puts ~76 % of mu at s = 0.1 and saturation
    # by s ~ 0.3.
    slip_ref_ratio: float = 0.1
    # Rolling-speed floor (m/s); a stationary or barely turning wheel uses the
    # velocity form s = |v_t| / 0.05, which keeps the force finite and smooth.
    slip_floor_velocity: float = 0.05
    # state.x is the body origin (front axle), not the CoM (|r| = 0.255 m on
    # helhest). On = the correct rigid-body dynamics (alpha x r and
    # omega x (omega x r) terms).
    com_transfer: bool = True


def contact_spring_constants(robot_model, config):
    """(k, lam): the contact law's spring and damping constants, as floats.

    k is the robot's whole-robot tyre stiffness; lam = b k / (m g) with
    b = damping_alpha 2 sqrt(m k) (simulator/engine.py::_calculate_spring_force).
    Both engines take them from here, so they run on the same numbers.
    """
    m, k = float(robot_model.mass), float(robot_model.contact_stiffness)
    b = config.damping_alpha * 2.0 * math.sqrt(m * k)
    return k, b * k / (m * config.g)


# --- Physics classes ---
@dataclass(eq=False)
class PhysicsState:
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

    def __post_init__(self):
        sizes = {t.shape[0] for t in self.__dict__.values()}
        if len(sizes) != 1:
            raise ValueError(f"PhysicsState fields disagree on batch size: {sizes}")

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
            "x": torch.zeros(batch_size, 3, device=device),
            "xd": torch.zeros(batch_size, 3, device=device),
            "q": torch.eye(1, 4, device=device).repeat(batch_size, 1),  # identity
            "omega": torch.zeros(batch_size, 3, device=device),
            "wheel_omega": torch.zeros(batch_size, 3, device=device),
        }
        # Overrides are moved here: a host tensor on a device state would
        # otherwise be copied implicitly (blocking) at its first use.
        base.update({k: v.to(device) for k, v in kwargs.items()})
        return PhysicsState(**base)


@dataclass(eq=False)
class PhysicsStateDer:
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
    R_mat = quat_to_rotmat(q)  # (B, 3, 3)

    for _ in range(10):
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
        # tends to identity on its own (a guard on |c| would snap slopes
        # < 0.81 deg flat).
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
        # R_new is the next pass's R_mat: one quat_to_rotmat per pass (~35 small
        # ops, launch-bound on GPU).
        q, x, R_mat = q_new, x_new, R_new

    return x, q


def seat_on_terrain(engine, x, q, world_config):
    """Shift the robot vertically so it starts at rest on its static sag.

    A vertical shift leaves every rim point's xy, so after lifting by s each
    wheel's deepest penetration is G_w - s, with G_w its largest vertical gap
    terrain_z - point_z now. Each wheel is a spring of k / n_wheels on that depth
    (simulator/engine.py::_calculate_spring_force), so the springs carry the
    robot's weight when sum_w relu(G_w - s) = m g n_wheels / k. Over the wheels
    sorted deepest first, every s_j = (sum_{i<=j} G_i - m g n_wheels / k) / j
    bounds the solution from below and the actual contact set attains it, so
    s = max_j s_j: closed form, one step. The vertical gap is the normal depth
    over n_z and the normal forces sum to m g n_z, so the same sum holds on a
    uniform slope whose friction carries the downhill load.

    Flat ground at a consistent attitude seats every wheel at m g / k (4.95 mm
    at 2.1e5 N/m). Seated at zero depth instead, every GT-seeded segment sank
    ~4.7 mm onto its sag within 0.1 s, a transient the GT does not have. A
    start above the terrain is lowered onto it too. The attitude is kept (GT or
    the plane fit), so a tilt the static loads disagree with still settles.

    Differentiable in the terrain: the start height is a function of the support
    under each wheel's deepest point; a detached lift makes the support gradient
    disagree with finite differences (cos 0.98 instead of 1.000 at 2 s, rough
    ground, GT-seeded start). The attitude is not differentiated (see
    Simulator.fit_plane_pose). `engine` only supplies geometry and the robot's
    constants, so both simulator wrappers share this. Returns a new position
    (`x` may be a view into GT data).
    """
    R_body = quat_to_rotmat(q.detach())  # (B, 3, 3)
    wp_rel = torch.einsum("bij,wpj->bwpi", R_body, engine._wheel_pos_rel)  # (B,W,P,3)
    wc_world = (R_body @ engine._wheel_centers.T).transpose(-1, -2) + x.unsqueeze(1)
    dh, normals, _ = engine._sample_terrain(wp_rel + wc_world.unsqueeze(2), world_config)
    gap = -dh / normals[..., 2:3].clamp_min(1e-3)  # vertical gap, (B, W, P, 1)
    G = gap.squeeze(-1).amax(dim=2).sort(dim=1, descending=True).values  # (B, W)
    n = G.shape[1]
    sag_sum = engine._mass * engine.config.g * n / engine.robot_model.contact_stiffness
    j = torch.arange(1, n + 1, device=G.device, dtype=G.dtype)
    lift = ((G.cumsum(dim=1) - sag_sum) / j).amax(dim=1)  # (B,)
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
        )
        dh, _, _ = engine._sample_terrain(wp_world, wc)  # (n, W, P, 1)
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

    wheel_radius (default: the v2 rig's) converts linear speed to rad/s.
    """
    if wheel_radius is None:
        wheel_radius = float(robot_model_config("helhest_v2").wheel_radii[0])
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

    wheel_radius = float(robot_model_config("helhest_v2").wheel_radii[0])
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
# (/robot_description, identical in the v1 and v2 bags). That box is the frame
# member alone; it leaves out the motors, the battery and the wheel forks, so it
# draws a machine a quarter the length of the real one. Half-extents here, as
# SimView wants; the URDF quotes full extents.
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


def add_helhest_visual_wheels(scene, parent: str, robot: str) -> None:
    """Add rigidly attached visual wheels for Helhest to a SimView scene.

    Geometry comes from robot_model_config(robot) so drawn wheels match the
    simulated contacts; pass the robot the rollout used. Hubs are expressed
    relative to the drawn chassis body, which sits HELHEST_CHASSIS_CENTER
    behind base_link.
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
