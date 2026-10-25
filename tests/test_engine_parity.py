"""Torch-engine vs. Warp-engine parity, forward and backward.

A dynamic loop in a Warp kernel gets an adjoint built on end-of-loop values:
the forward still agrees while support gradients go to cosine ~0 at 100 steps
and NaN at 1000. So gradients are checked at full rollout length, both engines
are cross-checked against finite differences, and the loop construct is banned
statically. Stiffness is the robot's constant in both engines, not an input.
"""

import pytest
import torch
from _helpers import make_sim as _make_sim_factory

# Warp has a CPU device too: the suite runs on CUDA when present, else on CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROBOTS = ("helhest_v2",)
GRID_N = 128
RES = 0.1
MAX_COORD = GRID_N * RES / 2 - RES / 2  # outermost cell centre, as world_cfg_from_config: 6.35
DT = 1e-3
# Training runs at 2e-3 (PhysicsEngineConfig.dt), so parity is checked there too.
DT_TRAIN = 2e-3
DTS = (DT, DT_TRAIN)

# Measured worst case (V100, fp32) over dt in {1e-3, 2e-3} x steps {50, 250, 1000}:
#
#   final pose        1.6e-4 m          (dt 2e-3, 1000 steps)
#   support grad      cosine 0.999384   norm ratio 1.0255
#   friction grad     cosine 0.999949   norm ratio 1.0119
#   warp self-vs-self 7.5e-9 m          (nondeterminism floor)
#   TF32 effect       cosine 0.99892    norm ratio 1.018
#
# GRAD_COS_MIN is tighter than the TF32 effect; the ratio cannot be, since the
# engines' own 1.0255 already exceeds it, so 1.05 is the tightest honest bound.
POS_ATOL = 5e-4
GRAD_COS_MIN = 0.999
GRAD_RATIO_RANGE = (0.95, 1.05)
# Unmeasured B=4 distinct-terrain rollouts stack more contact jitter into one norm.
GRAD_RATIO_RANGE_BATCHED = (0.9, 1.1)
# Same-run floor: torch self-vs-self is bit-identical, warp 7.5e-9 m.
SAME_RUN_ATOL = 1e-4
# Support gradient vs central finite differences (eps 5e-4 m, 250 x 1e-3 s),
# worst cell (both engines, CPU): flat 0.0063, rough 0.0065. eps is 5e-4 because
# 32 rim points put neighbours within mm of touching, where a 2 mm step is
# truncation error (flat 0.024 -> 0.017 -> 0.006 at 2e-3/1e-3/5e-4). Rough fails
# a detached raise (0.071) and depth-independent touchdown damping (0.33), both
# measured at eps 2e-3.
FD_REL_TOL = {"flat": 0.02, "rough": 0.06}


def _make_sim(engine, t_horizon, batch_size=1, dt=DT, robot="helhest_v2"):
    return _make_sim_factory(
        t_horizon, dt, engine=engine, device=DEVICE, max_coord=MAX_COORD,
        eval_mode=False, robot=robot,
    )


def _make_inputs(t_horizon, batch_size=1, n_ctrl=50, distinct_batch=False, dt=DT):
    """Build flat terrain and a forward drive with a slight turn.

    The turn keeps the friction gradient non-zero. `distinct_batch` gives each
    element its own slope, friction and wheel speeds, so sampling terrain[0]
    for every robot shows up.
    """
    support = torch.zeros(batch_size, GRID_N, GRID_N, device=DEVICE, requires_grad=True)
    friction = torch.full(
        (batch_size, GRID_N, GRID_N), 0.6, device=DEVICE, requires_grad=True
    )
    ctrl = torch.ones(batch_size, n_ctrl, 3, device=DEVICE)
    ctrl[..., 1] *= 0.7

    if distinct_batch:
        ramp = torch.linspace(-1.0, 1.0, GRID_N, device=DEVICE)
        slopes = torch.linspace(0.02, 0.10, batch_size, device=DEVICE)
        # A distinct tilt per element, so mixed-up batch indices move the trajectory.
        terrain = slopes.view(-1, 1, 1) * ramp.view(1, 1, -1).expand(
            batch_size, GRID_N, GRID_N
        )
        support = terrain.detach().clone().requires_grad_(True)
        fr = torch.linspace(0.4, 0.9, batch_size, device=DEVICE).view(-1, 1, 1)
        friction = fr.expand(batch_size, GRID_N, GRID_N).clone().requires_grad_(True)
        ctrl = ctrl * torch.linspace(0.8, 1.2, batch_size, device=DEVICE).view(-1, 1, 1)

    ctrl_ts = torch.linspace(0.0, max(t_horizon, dt), n_ctrl, device=DEVICE)
    ctrl_ts = ctrl_ts.unsqueeze(0).expand(batch_size, -1)
    return support, friction, ctrl, ctrl_ts


def _rollout(
    engine,
    n_steps,
    batch_size=1,
    distinct_batch=False,
    dt=DT,
    flat=False,
    robot="helhest_v2",
):
    """Run one rollout and backprop a trajectory loss.

    Returns (final_pos_per_batch, grad_support, grad_friction).
    """
    t_horizon = n_steps * dt
    sim = _make_sim(engine, t_horizon, batch_size, dt, robot)
    support, friction, ctrl, ctrl_ts = _make_inputs(
        t_horizon, batch_size, distinct_batch=distinct_batch, dt=dt
    )
    if flat:
        support = torch.zeros_like(support, requires_grad=True)

    traj, _, _, _ = sim.predict_trajectory(
        {"support": support, "friction": friction}, ctrl, ctrl_ts
    )
    traj[:, :, :3, 3].pow(2).sum().backward()

    return (
        traj[:, -1, :3, 3].detach().clone(),
        support.grad.clone(),
        friction.grad.clone(),
    )


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def _ratio(a, b):
    return (a.norm() / b.norm().clamp_min(1e-30)).item()


# --- Forward parity ---


@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("dt", DTS)
@pytest.mark.parametrize("n_steps", [50, 250, 1000])
def test_forward_pose_matches_torch(n_steps, dt, robot):
    pos_t, _, _ = _rollout("torch", n_steps, dt=dt, robot=robot)
    pos_w, _, _ = _rollout("warp", n_steps, dt=dt, robot=robot)
    err = (pos_w - pos_t).norm().item()
    assert err < POS_ATOL, (
        f"warp/torch final pose diverged by {err:.3e} m over {n_steps} steps "
        f"at dt={dt:g} (torch={pos_t.tolist()}, warp={pos_w.tolist()})"
    )


# --- Gradient parity ---


# A dynamic-loop adjoint breaks at 100 steps and goes NaN at 1000, the training
# length; both must stay covered.
@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("dt", DTS)
@pytest.mark.parametrize("n_steps", [50, 100, 250, 1000])
def test_gradients_match_torch(n_steps, dt, robot):
    _, gs_t, gf_t = _rollout("torch", n_steps, dt=dt, robot=robot)
    _, gs_w, gf_w = _rollout("warp", n_steps, dt=dt, robot=robot)

    assert torch.isfinite(gs_w).all(), (
        f"warp support grad is non-finite at {n_steps} steps, dt={dt:g}"
    )
    assert torch.isfinite(gf_w).all(), (
        f"warp friction grad is non-finite at {n_steps} steps, dt={dt:g}"
    )

    for name, gw, gt in (("support", gs_w, gs_t), ("friction", gf_w, gf_t)):
        cos, ratio = _cos(gw, gt), _ratio(gw, gt)
        assert cos > GRAD_COS_MIN, (
            f"{name} gradient direction diverged at {n_steps} steps, dt={dt:g}: "
            f"cosine={cos:.4f} (need > {GRAD_COS_MIN})"
        )
        lo, hi = GRAD_RATIO_RANGE
        assert lo < ratio < hi, (
            f"{name} gradient magnitude diverged at {n_steps} steps, dt={dt:g}: "
            f"‖warp‖/‖torch‖={ratio:.3e} (need {lo}..{hi})"
        )


def _brake_rollout(engine, robot):
    """Drive 0.5 s at 3 rad/s with a slight turn, brake to 0 by 0.55 s, hold."""
    t_horizon = 1.5
    sim = _make_sim(engine, t_horizon, dt=DT_TRAIN, robot=robot)
    support = torch.zeros(1, GRID_N, GRID_N, device=DEVICE, requires_grad=True)
    friction = torch.full((1, GRID_N, GRID_N), 0.8, device=DEVICE, requires_grad=True)
    ctrl = torch.tensor([[[3.5, 2.5, 3.0]] * 2 + [[0.0] * 3] * 2], device=DEVICE)
    ctrl_ts = torch.tensor([[0.0, 0.5, 0.55, t_horizon]], device=DEVICE)
    traj, _, xd, om = sim.predict_trajectory(
        {"support": support, "friction": friction}, ctrl, ctrl_ts
    )
    traj[:, :, :3, 3].pow(2).sum().backward()
    late = max(
        xd[0, -100:, :2].std(0).max().item(), om[0, -100:, 2].std().item()
    )
    return traj[0, -1, :3, 3].detach(), late, support.grad, friction.grad


@pytest.mark.parametrize("robot", ROBOTS)
def test_brake_to_stop_settles_and_matches_torch(robot):
    """A braked robot must come to rest in both engines (the friction cut in
    simulator/engine.py::_friction), with matching pose and gradients."""
    pos_t, late_t, gs_t, gf_t = _brake_rollout("torch", robot)
    pos_w, late_w, gs_w, gf_w = _brake_rollout("warp", robot)
    assert late_t < 1e-4 and late_w < 1e-4, (late_t, late_w)
    assert (pos_w - pos_t).norm().item() < POS_ATOL, (pos_t.tolist(), pos_w.tolist())
    for name, gw, gt in (("support", gs_w, gs_t), ("friction", gf_w, gf_t)):
        assert _cos(gw, gt) > GRAD_COS_MIN, (name, _cos(gw, gt))
        lo, hi = GRAD_RATIO_RANGE
        assert lo < _ratio(gw, gt) < hi, (name, _ratio(gw, gt))


@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_batch_elements_are_independent(engine):
    """Element b of a batched rollout must equal that window run on its own.

    Catches sampling element 0's terrain for every robot (`z_grid[0]`), which is
    invisible at B=1 or with identical batch elements.
    """
    n_steps, B = 100, 4
    t_horizon = n_steps * DT
    support, friction, ctrl, ctrl_ts = _make_inputs(t_horizon, B, distinct_batch=True)

    sim = _make_sim(engine, t_horizon, B)
    traj, _, _, _ = sim.predict_trajectory(
        {"support": support, "friction": friction}, ctrl, ctrl_ts
    )
    batched = traj[:, -1, :3, 3].detach().clone()

    sim1 = _make_sim(engine, t_horizon, 1)
    for b in range(B):
        traj_b, _, _, _ = sim1.predict_trajectory(
            {
                "support": support[b : b + 1].detach(),
                "friction": friction[b : b + 1].detach(),
            },
            ctrl[b : b + 1],
            ctrl_ts[b : b + 1],
        )
        alone = traj_b[0, -1, :3, 3].detach()
        err = (batched[b] - alone).norm().item()
        assert err < 1e-4, (
            f"{engine}: batch element {b} moved {err:.3e} m when run alongside "
            f"others — batch elements are not independent (terrain indexing?)"
        )

    # The elements must actually differ, or the check above is vacuous.
    spread = (batched - batched.mean(dim=0)).norm(dim=-1).max().item()
    assert spread > 1e-3, "batch elements are identical; the test proves nothing"


@pytest.mark.parametrize("n_steps", [100, 500])
def test_gradients_match_torch_batched(n_steps):
    """Gradient parity at B>1 with a different terrain per element.

    Catches a wrapper that mixes up batch indices, which B=1 parity cannot see.
    """
    B = 4
    _, gs_t, gf_t = _rollout("torch", n_steps, B, distinct_batch=True)
    _, gs_w, gf_w = _rollout("warp", n_steps, B, distinct_batch=True)

    assert torch.isfinite(gs_w).all() and torch.isfinite(gf_w).all()
    for name, gw, gt in (("support", gs_w, gs_t), ("friction", gf_w, gf_t)):
        for b in range(B):
            cos = _cos(gw[b], gt[b])
            assert cos > GRAD_COS_MIN, (
                f"{name} gradient of batch element {b} diverged at {n_steps} "
                f"steps: cosine={cos:.4f}"
            )
        ratio = _ratio(gw, gt)
        lo, hi = GRAD_RATIO_RANGE_BATCHED
        assert lo < ratio < hi, f"{name} gradient magnitude ratio {ratio:.3e}"


@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_initial_pose_override_is_honoured(engine):
    """initial_state x/q must override the terrain-fitted start pose.

    The segmented loss seeds each segment from GT this way; catches a wrapper
    silently re-seeding from the plane fit.
    """
    n_steps = 50
    t_horizon = n_steps * DT
    sim = _make_sim(engine, t_horizon)
    support, friction, ctrl, ctrl_ts = _make_inputs(t_horizon)
    props = {"support": support.detach(), "friction": friction.detach()}

    base, _, _, _ = sim.predict_trajectory(props, ctrl, ctrl_ts)
    shifted, _, _, _ = sim.predict_trajectory(
        props,
        ctrl,
        ctrl_ts,
        initial_state={"x": torch.tensor([[3.0, -2.0, 0.0]], device=DEVICE)},
    )
    dxy = (shifted[0, 0, :2, 3] - base[0, 0, :2, 3]).detach()
    assert torch.allclose(dxy, torch.tensor([3.0, -2.0], device=DEVICE), atol=1e-3), (
        f"{engine}: initial_state['x'] was ignored — start offset {dxy.tolist()} "
        "instead of [3.0, -2.0]"
    )


@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_include_start_prepends_the_seated_start(engine):
    """include_start returns the state the rollout steps from first, at vel_ts[:, 0]:
    the seed seated on the terrain (the segmented loss anchors on it), its seeded
    velocities; the steps after it are the default call's."""
    from models.simulator import _build_world_config

    t_horizon = 20 * DT
    sim = _make_sim(engine, t_horizon)
    support, friction, ctrl, ctrl_ts = _make_inputs(t_horizon)
    props = {"support": support.detach() + 0.05, "friction": friction.detach()}
    init = {
        "x": torch.tensor([[0.3, -0.2, 0.0]], device=DEVICE),  # wheels below ground
        "q": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE),
        "xd": torch.tensor([[0.5, 0.0, 0.0]], device=DEVICE),
        "omega": torch.tensor([[0.0, 0.0, 0.1]], device=DEVICE),
    }
    base = sim.predict_trajectory(props, ctrl, ctrl_ts, initial_state=init)
    full = sim.predict_trajectory(
        props, ctrl, ctrl_ts, initial_state=init, include_start=True
    )
    for b, f in zip(base, full):
        assert torch.allclose(f[:, 1:], b, atol=SAME_RUN_ATOL), engine

    pose, ts, xd, omega = (o[:, 0] for o in full)
    seated = sim.seat_on_terrain(
        init["x"], init["q"], _build_world_config(sim, props)
    )
    assert seated[0, 2] > 0.01, "the seed must start below ground to test the seat"
    assert torch.allclose(pose[:, :3, 3], seated, atol=1e-6), engine
    assert torch.allclose(pose[:, :3, :3], torch.eye(3, device=DEVICE)), engine
    assert torch.equal(ts, ctrl_ts[:, 0].double())
    assert torch.equal(xd, init["xd"]) and torch.equal(omega, init["omega"])


def test_initial_state_gradients_match_torch():
    """Gradients into a seeded initial q / xd / omega must flow in both engines.

    Returning None for them would silently cut a graph that seeds a segment
    from a differentiable state. q is compared in the tangent space:
    torch normalises q, warp's rotation does not, so they differ along q itself.
    """
    n_steps = 250
    t_horizon = n_steps * DT
    support, friction, ctrl, ctrl_ts = _make_inputs(t_horizon)
    props = {"support": support.detach(), "friction": friction.detach()}
    grads = {}
    for engine in ("torch", "warp"):
        init = {
            "q": torch.tensor([[0.999, 0.0, 0.03, 0.02]], device=DEVICE),
            "xd": torch.tensor([[0.5, 0.1, 0.0]], device=DEVICE),
            "omega": torch.tensor([[0.0, 0.0, 0.2]], device=DEVICE),
        }
        init["q"] = init["q"] / init["q"].norm()
        for t in init.values():
            t.requires_grad_(True)
        traj, _, _, _ = _make_sim(engine, t_horizon).predict_trajectory(
            props, ctrl, ctrl_ts, initial_state=init
        )
        traj[:, :, :3, 3].pow(2).sum().backward()
        q = init["q"].detach()
        gq = init["q"].grad
        grads[engine] = {
            "q": gq - (gq * q).sum(-1, keepdim=True) * q,
            "xd": init["xd"].grad,
            "omega": init["omega"].grad,
        }
    for name in ("q", "xd", "omega"):
        gt, gw = grads["torch"][name], grads["warp"][name]
        assert gw is not None and gw.norm() > 0, f"warp: no gradient into init {name}"
        assert _cos(gw, gt) > GRAD_COS_MIN, (name, _cos(gw, gt))
        lo, hi = GRAD_RATIO_RANGE
        assert lo < _ratio(gw, gt) < hi, (name, _ratio(gw, gt))


def test_rest_start_gradients_match_torch():
    """Seeded at rest at its rest sag, the robot's v_n is exactly 0 on step 0.
    The damping is c * v_n in both engines (c = min(depth lam / n_w,
    share m_eff_n / dt)), so dF/dv_n = -c there and matches a central finite
    difference; the |v_n| sign(v_n) form gives 0 in torch and -share m_eff_n / dt
    in warp (init xd gradient x0.45). The wheels spin, so the slip is not 0
    (where torch's |v| and warp's sqrt(|v|^2 + 1e-12) still differ).
    """
    t_horizon = 5 * DT_TRAIN
    support, friction, ctrl, ctrl_ts = _make_inputs(t_horizon, dt=DT_TRAIN)
    props = {"support": support.detach(), "friction": friction.detach()}
    grads = {}
    for engine in ("torch", "warp"):
        sim = _make_sim(engine, t_horizon, dt=DT_TRAIN, robot="helhest_v2")
        sim.seat_on_terrain = lambda x, q, wc: x  # keep the sag
        rm = sim.robot_model
        z_rest = float(rm.wheel_radii[0]) - rm.mass * 9.81 / rm.contact_stiffness
        init = {
            "x": torch.tensor([[0.0, 0.0, z_rest]], device=DEVICE),
            "q": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE),
            "xd": torch.zeros(1, 3, device=DEVICE, requires_grad=True),
            "omega": torch.zeros(1, 3, device=DEVICE, requires_grad=True),
        }
        traj, _, _, _ = sim.predict_trajectory(props, ctrl, ctrl_ts, initial_state=init)
        traj[:, :, :3, 3].pow(2).sum().backward()
        grads[engine] = (init["xd"].grad, init["omega"].grad)
    for name, gt, gw in zip(("xd", "omega"), grads["torch"], grads["warp"]):
        assert _cos(gw, gt) > GRAD_COS_MIN, (name, _cos(gw, gt))
        lo, hi = GRAD_RATIO_RANGE
        assert lo < _ratio(gw, gt) < hi, (name, _ratio(gw, gt))


@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_returned_timestamps_survive_epoch_scale_input(engine):
    """Predicted timestamps must resolve a 1 s window sitting at Unix-epoch scale.

    float32 spacing at ~1.76e9 is ~128 s; catches a float32 query grid that
    collapses every timestamp-matched GT metric onto predicted index 0.
    """
    n_steps = 100
    t_horizon = n_steps * DT
    sim = _make_sim(engine, t_horizon)
    support, friction, ctrl, _ = _make_inputs(t_horizon)

    t0 = 1_757_690_067.516  # a real timestamp from the helhest dataset
    ctrl_ts = torch.linspace(
        t0, t0 + t_horizon, ctrl.shape[1], device=DEVICE, dtype=torch.float64
    ).unsqueeze(0)

    _, pred_ts, _, _ = sim.predict_trajectory(
        {"support": support, "friction": friction}, ctrl, ctrl_ts
    )
    ts = pred_ts[0].double()

    span = (ts[-1] - ts[0]).item()
    assert span > 0.5 * t_horizon, (
        f"{engine}: predicted timestamps span {span:.3e} s for a {t_horizon} s "
        "horizon — the window collapsed (float32 at epoch scale?)"
    )
    assert (ts.diff() > 0).all(), f"{engine}: predicted timestamps are not increasing"
    assert abs(ts[0].item() - t0) < 1.0, (
        f"{engine}: predicted start {ts[0].item():.3f} is {abs(ts[0].item() - t0):.1f} s "
        f"off the control start {t0:.3f}"
    )


def _rough_support():
    """+-3 cm smooth random terrain (~0.3 m wavelength), fixed seed."""
    g = torch.Generator().manual_seed(0)
    k = torch.exp(-torch.arange(-12, 13.0) ** 2 / 18.0)
    k = k / k.sum()
    z = torch.randn(1, 1, GRID_N, GRID_N, generator=g)
    z = torch.nn.functional.conv2d(z, k.view(1, 1, 1, -1), padding=(0, 12))
    z = torch.nn.functional.conv2d(z, k.view(1, 1, -1, 1), padding=(12, 0))[:, 0]
    return (0.03 * z / z.abs().max()).to(DEVICE)


def _fd_rel_errors(engine, robot, terrain, eps=5e-4):
    """Relative error |fd - analytic| / max(|fd|, |analytic|) of the support
    gradient at the 4 cells of largest |gradient| (wheel contacts sit off the
    grid centre).

    The start attitude is pinned (the plane fit is detached by design, see
    Simulator.fit_plane_pose). The seed is 5 mm above ("flat") or into
    ("rough") the terrain, as a GT-seeded training segment often is; the
    differentiable seat sets the start height either way.
    """
    from models.simulator import _build_world_config

    n_steps = 250
    t_horizon = n_steps * DT
    sim = _make_sim(engine, t_horizon, robot=robot)
    support, friction, ctrl, ctrl_ts = _make_inputs(t_horizon)
    if terrain == "rough":
        support = _rough_support().requires_grad_(True)
    with torch.no_grad():
        wc = _build_world_config(sim, {"support": support, "friction": friction})
        x0, q0 = sim.fit_plane_pose(wc, 1)
        x0 = sim.seat_on_terrain(x0, q0, wc)
        x0[:, 2] += 5e-3 if terrain == "flat" else -5e-3
    init = {"x": x0, "q": q0}

    traj, _, _, _ = sim.predict_trajectory(
        {"support": support, "friction": friction}, ctrl, ctrl_ts, initial_state=init
    )
    traj[:, :, :3, 3].pow(2).sum().backward()
    grad = support.grad.clone()
    assert torch.isfinite(grad).all(), f"{engine} support grad is non-finite"

    def loss_at(z):
        props = {"support": z, "friction": friction.detach()}
        tr, _, _, _ = sim.predict_trajectory(props, ctrl, ctrl_ts, initial_state=init)
        return tr[:, :, :3, 3].pow(2).sum().item()

    rels = []
    for k in torch.topk(grad[0].abs().flatten(), 4).indices:
        i, j = int(k) // GRID_N, int(k) % GRID_N
        analytic = grad[0, i, j].item()
        if abs(analytic) < 1e-6:
            continue  # cell the trajectory never touched; nothing to verify
        zp = support.detach().clone()
        zp[0, i, j] += eps
        zm = support.detach().clone()
        zm[0, i, j] -= eps
        numeric = (loss_at(zp) - loss_at(zm)) / (2 * eps)
        rels.append(abs(numeric - analytic) / max(abs(numeric), abs(analytic), 1e-12))
    return rels


@pytest.mark.parametrize("terrain", ["flat", "rough"])
@pytest.mark.parametrize("robot", ROBOTS)
@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_gradient_matches_finite_differences(engine, robot, terrain):
    """Each engine's support gradient must match central finite differences.

    Catches both engines agreeing on the same mistake; torch is the parity
    reference and nothing else checks it. A contact force discontinuous at
    touchdown makes the analytic gradient run 5-35 % low (rel errors up to 0.43
    here); see FD_REL_TOL for the measured errors.
    """
    rels = _fd_rel_errors(engine, robot, terrain)
    assert rels, "no cell carried gradient — the test verified nothing"
    assert max(rels) < FD_REL_TOL[terrain], f"{engine} {robot} {terrain}: rel errors {rels}"


def _off_map_rollout(engine):
    """3 s from the centre of a 3.2 m map of 3 cm noise, so every edge cell is
    tilted: the robot drives off the +x edge (the audit's H2 probe)."""
    n, t_horizon = 32, 3.0
    mc = (n - 1) * RES / 2
    sim = _make_sim_factory(t_horizon, DT_TRAIN, engine=engine, device=DEVICE,
                            max_coord=mc, eval_mode=False, robot="helhest_v2")
    g = torch.Generator().manual_seed(3)
    support = (0.03 * torch.randn(1, n, n, generator=g)).to(DEVICE).requires_grad_(True)
    friction = torch.full((1, n, n), 0.7, device=DEVICE)
    ctrl = torch.tensor([[[3.0, 2.4, 3.0]] * 2], device=DEVICE)
    ctrl_ts = torch.tensor([[0.0, t_horizon]], device=DEVICE, dtype=torch.float64)
    traj, _, _, _ = sim.predict_trajectory(
        {"support": support, "friction": friction}, ctrl, ctrl_ts
    )
    traj[:, :, :3, 3].pow(2).sum().backward()
    return traj[0, :, :3, 3].detach(), support.grad, mc


def test_off_map_rollout_matches_torch_and_stays_bounded():
    """Off the map both engines extend the edge height flat under a vertical
    normal. The edge cell's tilted normal there would push sideways forever: a
    non-conservative force, support gradient 1e28 here without the damping cap."""
    pos_t, g_t, mc = _off_map_rollout("torch")
    pos_w, g_w, _ = _off_map_rollout("warp")
    off = (pos_t[:, 0] > mc).float().mean().item()
    assert off > 0.3, f"origin off the map on only {off:.0%} of steps"
    err = (pos_w - pos_t).norm(dim=-1).max().item()
    assert err < POS_ATOL, f"off-map poses diverged by {err:.3e} m"
    for name, g in (("torch", g_t), ("warp", g_w)):
        assert torch.isfinite(g).all() and g.norm() < 2e4, (name, g.norm().item())


@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_a_stiffness_property_is_refused(engine):
    """Stiffness is the robot's contact_stiffness in both engines; a per-call
    "stiffness" would be ignored, so it raises."""
    t_horizon = 10 * DT
    support, friction, ctrl, ctrl_ts = _make_inputs(t_horizon)
    with pytest.raises(ValueError, match="contact_stiffness"):
        _make_sim(engine, t_horizon).predict_trajectory(
            {"support": support, "friction": friction, "stiffness": 2.1e5}, ctrl, ctrl_ts
        )


def test_gradients_match_torch_on_a_friction_ramp():
    """A coefficient sampled under the wheels that varies with contact position:
    both engines differentiate it the same way."""
    t_horizon = 100 * DT_TRAIN
    xs = torch.linspace(-MAX_COORD, MAX_COORD, GRID_N, device=DEVICE)
    mu = (0.6 + 0.3 * torch.tanh(2.0 * xs)).expand(1, GRID_N, GRID_N)
    grads = []
    for engine in ("torch", "warp"):
        support, _, ctrl, ctrl_ts = _make_inputs(t_horizon, dt=DT_TRAIN)
        sim = _make_sim(engine, t_horizon, dt=DT_TRAIN, robot="helhest_v2")
        traj, _, _, _ = sim.predict_trajectory(
            {"support": support, "friction": mu.clone()}, ctrl, ctrl_ts
        )
        traj[:, :, :3, 3].pow(2).sum().backward()
        grads.append(support.grad)
    gt, gw = grads
    rel = ((gw - gt).norm() / gt.norm()).item()
    assert _cos(gw, gt) > 0.9999 and rel < 1e-3, (_cos(gw, gt), rel)


def test_both_wrappers_share_one_default_horizon():
    """Both wrappers must share one T_horizon default.

    The single-segment path passes no horizon; catches warp rolling out 5x further.
    """
    import inspect

    from models.simulator import Simulator as TorchSim
    from models.simulator_wp import Simulator as WarpSim

    defaults = {
        cls.__module__: inspect.signature(cls.__init__).parameters["T_horizon"].default
        for cls in (TorchSim, WarpSim)
    }
    assert len(set(defaults.values())) == 1, f"T_horizon defaults differ: {defaults}"


def test_wheel_geometry_has_one_source():
    """Both wrappers read the URDF geometry from simulator.utils, not a copy."""
    from models.simulator import Simulator as TorchSim
    from models.simulator_wp import Simulator as WarpSim
    from simulator.utils import robot_model_config

    ref = robot_model_config("helhest_v2")
    assert torch.allclose(
        ref.wheel_centers,
        torch.tensor([[0.0, 0.36, 0.0], [0.0, -0.36, 0.0], [-0.697, 0.0, 0.0]]),
    ), "the source of truth no longer matches the URDF tuple"
    for cls in (TorchSim, WarpSim):
        sim = cls(robot="helhest_v2", device="cpu")
        assert torch.equal(sim.robot_model.wheel_centers, ref.wheel_centers)
        assert torch.equal(sim.robot_model.wheel_radii, ref.wheel_radii)
        assert torch.equal(sim.robot_model.cog, ref.cog)


# --- Non-finite gradients: both engines propagate them ---


def _rollout_with_poisoned_cotangent(engine, n_steps=50):
    """Backprop a loss whose cotangent contains an inf; return d(support)."""
    dt = DT
    t_horizon = n_steps * dt
    sim = _make_sim(engine, t_horizon, dt=dt)
    support, friction, ctrl, ctrl_ts = _make_inputs(t_horizon, dt=dt)
    traj, _, _, _ = sim.predict_trajectory(
        {"support": support, "friction": friction}, ctrl, ctrl_ts
    )
    w = torch.ones_like(traj[:, :, :3, 3])
    w[0, n_steps // 2, 0] = float("inf")
    (traj[:, :, :3, 3] * w).sum().backward()
    return support.grad


@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_non_finite_gradients_propagate(engine):
    """Both engines must hand a non-finite gradient back as it is.

    The trainer skips the step on one (utils/trainer.py). Zeroing them would let
    a simulator blow-up train on as a silent zero gradient, indistinguishable
    from a converged rollout.
    """
    g = _rollout_with_poisoned_cotangent(engine)
    assert not torch.isfinite(g).all(), f"{engine} sanitised a non-finite gradient"


# --- Compiler-level guard ---


def test_warp_kernels_have_no_dynamic_loops():
    """No `@wp.kernel` may loop over a range() bound by a runtime value.

    Warp only unrolls compile-time-constant ranges; a dynamic loop gets a wrong
    adjoint. Source is parsed because Warp's codegen warning is skipped when a
    kernel is served from its on-disk cache.
    """
    import ast
    import pathlib

    src_path = (
        pathlib.Path(__file__).resolve().parents[1] / "simulator" / "engine_wp.py"
    )
    tree = ast.parse(src_path.read_text())

    def is_wp_kernel(node):
        for dec in node.decorator_list:
            attr = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(attr, ast.Attribute) and attr.attr == "kernel":
                return True
        return False

    # Module-level names bound to wp.constant(...) or a plain int are unrollable.
    constants = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if not isinstance(tgt, ast.Name):
                continue
            val = node.value
            if isinstance(val, ast.Constant) and isinstance(val.value, int):
                constants.add(tgt.id)
            elif (
                isinstance(val, ast.Call)
                and isinstance(val.func, ast.Attribute)
                and val.func.attr == "constant"
            ):
                constants.add(tgt.id)

    offending = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or not is_wp_kernel(fn):
            continue
        params = {a.arg for a in fn.args.args}
        for node in ast.walk(fn):
            if not isinstance(node, ast.For):
                continue
            it = node.iter
            if not (
                isinstance(it, ast.Call) and getattr(it.func, "id", None) == "range"
            ):
                continue
            for arg in it.args:
                if isinstance(arg, ast.Constant):
                    continue
                if isinstance(arg, ast.Name) and arg.id in constants:
                    continue
                bound = getattr(arg, "id", ast.dump(arg))
                kind = "kernel parameter" if bound in params else "runtime value"
                offending.append(
                    f"{fn.name} (line {node.lineno}): range() bound by {kind} "
                    f"{bound!r} — Warp cannot unroll this"
                )

    assert not offending, (
        "dynamic loop(s) in a Warp kernel; any state mutated inside gets a wrong "
        "adjoint (see this file's docstring):\n  " + "\n  ".join(offending)
    )
