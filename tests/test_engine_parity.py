"""Torch-engine vs. Warp-engine parity, forward *and* backward.

The Warp engine exists to make the rollout ~10x cheaper, but it is only useful if
its gradients match the torch engine's. They did not: the whole rollout used to
live inside one kernel as `for step in range(num_steps)` with `num_steps` a kernel
*parameter*, so Warp could not unroll it, and a dynamic loop that mutates state
read in later iterations violates Warp's SSA assumption (it emits
`assign(var1, var2)` in `materialize_redefinitions`). The adjoint then substitutes
end-of-loop values for per-iteration values. Measured symptom, gradient wrt the
support surface vs. the torch engine:

    steps    50 -> cosine  0.9998, norm ratio 1.13
    steps   100 -> cosine -0.0122, norm ratio 5.5e6
    steps   250 -> NaN
    steps  1000 -> NaN                (the length actually used in training)

So the forward agreed to 3.3e-4 m while the backward was garbage, and short
rollouts looked fine. Hence this file checks gradients at the *full* rollout
length rather than a convenient short one, cross-checks both engines against
finite differences, and statically forbids the loop construct that caused it.

Most tests need a GPU and so only run on the cluster; the static guard runs
everywhere.

Stiffness is deliberately not covered: it is no longer a predicted or adaptable
property (see ADAPT_HEADS in utils/experiment.py), only a simulator input.
"""

import warnings

import pytest
import torch

# Only the tests that actually roll out the warp engine need a GPU; the static
# dynamic-loop guard below is pure source parsing and must run in the local suite,
# where it is the only thing standing between a refactor and a silently wrong
# adjoint that nobody sees until a cluster run.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the warp engine is CUDA-only"
)

DEVICE = "cuda"
GRID_N = 128
RES = 0.1
MAX_COORD = 6.4
DT = 1e-3
# Training runs 2e-3 (PhysicsEngineConfig.dt, chosen by sweep_dt.py). Parity was
# only ever checked at 1e-3, so a divergence that only shows up at the step size
# production uses would have gone unseen. Both are covered now.
DT_TRAIN = 2e-3
DTS = (DT, DT_TRAIN)

# MEASURED, not aspirational: scripts/analysis/measure_engine_parity.py, job
# 11564503 (gpufast, V100, IEEE fp32, slurm/out/audit_engine_parity_11564503.out).
# Worst case over dt in {1e-3, 2e-3} x steps in {50, 250, 1000}:
#
#   final pose        1.6e-4 m          (at dt 2e-3, 1000 steps)
#   support grad      cosine 0.999384   norm ratio 1.0255
#   friction grad     cosine 0.999949   norm ratio 1.0119
#   warp self-vs-self 7.5e-9 m          (the nondeterminism floor)
#
# The old bounds (2e-3 m, cosine 0.99, ratio 0.5..2.0) were looser than the TF32
# effect the configs were changed to avoid (cosine 0.99892, ratio 1.018), so a
# TF32-sized divergence would have passed. GRAD_COS_MIN is now tighter than that
# effect. GRAD_RATIO_RANGE cannot be: the engines' own worst norm ratio (1.0255)
# is already larger than TF32's, so 1.05 is the tightest honest bound.
POS_ATOL = 5e-4
GRAD_COS_MIN = 0.999
GRAD_RATIO_RANGE = (0.95, 1.05)
# The batched (B=4, distinct terrain per element) rollouts were not part of the
# measurement above and stack four problems' worth of contact-boundary jitter
# into one norm, so they keep a looser magnitude bound.
GRAD_RATIO_RANGE_BATCHED = (0.9, 1.1)
# Measured floor for "the two rollouts are the same run": torch self-vs-self is
# bit-identical, warp self-vs-self 7.5e-9 m (job 11564503, section 7).
SAME_RUN_ATOL = 1e-4


def _make_sim(engine, t_horizon, batch_size=1, dt=DT):
    if engine == "torch":
        from models.simulator import Simulator
    else:
        from models.simulator_wp import Simulator

    sim = Simulator(
        robot="helhest",
        batch_size=batch_size,
        device=DEVICE,
        T_horizon=t_horizon,
        dt=dt,
    )
    sim.update_world_config(
        res=RES,
        max_coord=MAX_COORD,
        side_range=(-MAX_COORD, MAX_COORD),
        fwd_range=(-MAX_COORD, MAX_COORD),
    )
    return sim


def _make_inputs(t_horizon, batch_size=1, n_ctrl=50, distinct_batch=False, dt=DT):
    """Flat terrain, forward drive with a slight turn.

    The turn matters: on a perfectly straight drive the friction gradient is
    legitimately near zero, so a friction check would pass vacuously.

    `distinct_batch` gives every batch element its own sloped terrain, its own
    friction and its own wheel speeds. Without that, a batch is B copies of one
    problem and a wrapper that samples terrain[0] for every robot looks perfectly
    correct — which is exactly the bug the warp engine shipped with.
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
        # b-th element gets a distinct tilt, so mixing up batch indices moves the
        # trajectory rather than cancelling out
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
):
    """Run one rollout and backprop a trajectory loss.

    Returns (final_pos_per_batch, grad_support, grad_friction).
    """
    t_horizon = n_steps * dt
    sim = _make_sim(engine, t_horizon, batch_size, dt)
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


# ---------------------------------------------------------------------------
# Forward parity
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("dt", DTS)
@pytest.mark.parametrize("n_steps", [50, 250, 1000])
def test_forward_pose_matches_torch(n_steps, dt):
    pos_t, _, _ = _rollout("torch", n_steps, dt=dt)
    pos_w, _, _ = _rollout("warp", n_steps, dt=dt)
    err = (pos_w - pos_t).norm().item()
    assert err < POS_ATOL, (
        f"warp/torch final pose diverged by {err:.3e} m over {n_steps} steps "
        f"at dt={dt:g} (torch={pos_t.tolist()}, warp={pos_w.tolist()})"
    )


# ---------------------------------------------------------------------------
# Gradient parity — the actual gate
# ---------------------------------------------------------------------------


# 100 steps is where the old fused-kernel adjoint first detonated (cosine went to
# ~0 and the norm ratio to 5.5e6); 1000 is the length training and adaptation
# really use, and is where it returned NaN. Both must be covered or the
# regression walks straight back in.
@requires_cuda
@pytest.mark.parametrize("dt", DTS)
@pytest.mark.parametrize("n_steps", [50, 100, 250, 1000])
def test_gradients_match_torch(n_steps, dt):
    _, gs_t, gf_t = _rollout("torch", n_steps, dt=dt)
    _, gs_w, gf_w = _rollout("warp", n_steps, dt=dt)

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


@requires_cuda
@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_batch_elements_are_independent(engine):
    """Element b of a batched rollout must equal that window run on its own.

    This is the check that the warp engine's original `world_config.z_grid[0]`
    bug needed and never got: it sampled batch element 0's terrain for every
    robot, which is invisible whenever a test batch is B copies of one problem,
    and invisible at B=1. Training runs at B=48 with 48 different terrains.
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

    # and the elements must actually differ, or the check above is vacuous
    spread = (batched - batched.mean(dim=0)).norm(dim=-1).max().item()
    assert spread > 1e-3, "batch elements are identical; the test proves nothing"


@requires_cuda
@pytest.mark.parametrize("n_steps", [100, 500])
def test_gradients_match_torch_batched(n_steps):
    """Gradient parity at B>1 with a different terrain per element.

    B=1 parity cannot see a wrapper that mixes up batch indices, and training runs
    batched. Each element's terrain gradient must land on that element's slice.
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


@requires_cuda
@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_initial_pose_override_is_honoured(engine):
    """initial_state x/q must override the terrain-fitted start pose.

    The segmented trajectory loss (utils/trainer.py) seeds each segment at the
    GT pose closest to its own start timestamp through exactly this path. The
    warp wrapper used to ignore x/q and silently re-seed from the plane fit,
    which produces no error — just wrong segments.
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


@requires_cuda
@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_returned_timestamps_survive_epoch_scale_input(engine):
    """Predicted timestamps must resolve a 1 s window sitting at Unix-epoch scale.

    The dataset supplies absolute timestamps (~1.76e9). float32 spacing up there is
    ~128 s, so building the query grid at the default dtype collapses every sample
    in the window onto one value. Nothing crashes — but every GT metric is
    timestamp-matched, so all of them silently collapse onto predicted index 0.
    That is exactly what `--engine warp` did before this test existed.
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


@requires_cuda
@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_gradient_matches_finite_differences(engine):
    """Independent reference: perturb single terrain cells and difference the loss.

    The two engines share a contact model, so agreeing with each other could in
    principle mean agreeing on the same mistake — finite differences do not care.
    The torch case matters as much as the warp one: it is the reference the parity
    test above trusts, and nothing else checks it.

    Cells are chosen by largest |analytic gradient| rather than hardcoded. The
    robot's wheel contacts sit off the grid centre (URDF: front wheels at
    (0, ±0.36), rear at (−0.697, 0)), so the centre cell carries little or no
    gradient — a fixed cell list silently verifies nothing.
    """
    n_steps = 250
    t_horizon = n_steps * DT
    eps = 2e-3

    sim = _make_sim(engine, t_horizon)
    support, friction, ctrl, ctrl_ts = _make_inputs(t_horizon)

    traj, _, _, _ = sim.predict_trajectory(
        {"support": support, "friction": friction}, ctrl, ctrl_ts
    )
    traj[:, :, :3, 3].pow(2).sum().backward()
    grad = support.grad.clone()
    assert torch.isfinite(grad).all(), f"{engine} support grad is non-finite"

    def loss_at(z):
        props = {"support": z, "friction": friction.detach()}
        tr, _, _, _ = sim.predict_trajectory(props, ctrl, ctrl_ts)
        return tr[:, :, :3, 3].pow(2).sum().item()

    flat = grad[0].abs().flatten()
    top = torch.topk(flat, 4).indices
    cells = [(int(k) // GRID_N, int(k) % GRID_N) for k in top]

    checked = 0
    for i, j in cells:
        analytic = grad[0, i, j].item()
        if abs(analytic) < 1e-6:
            continue  # cell the trajectory never touched; nothing to verify
        zp = support.detach().clone()
        zp[0, i, j] += eps
        zm = support.detach().clone()
        zm[0, i, j] -= eps
        numeric = (loss_at(zp) - loss_at(zm)) / (2 * eps)
        rel = abs(numeric - analytic) / max(abs(numeric), abs(analytic), 1e-12)
        # Loose: the contact gate is a hard `dist < 0` switch, so a finite
        # perturbation can add or drop sample points that the derivative at the
        # unperturbed point knows nothing about.
        assert rel < 0.35, (
            f"{engine} cell ({i},{j}): analytic={analytic:.6e} "
            f"finite-diff={numeric:.6e} (rel err {rel:.2%})"
        )
        checked += 1

    assert checked > 0, "no cell carried gradient — the test verified nothing"


def test_both_wrappers_share_one_default_horizon():
    """T_horizon was 1.0 in the torch wrapper and 5.0 in the warp one.

    The trainer's single-segment path passes no horizon, so `--engine warp`
    silently rolled out 5x further than `--engine torch` on the same config.
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

    ref = robot_model_config("helhest")
    assert torch.allclose(
        ref.wheel_centers,
        torch.tensor([[0.0, 0.36, 0.0], [0.0, -0.36, 0.0], [-0.697, 0.0, 0.0]]),
    ), "the source of truth no longer matches the URDF tuple"
    for cls in (TorchSim, WarpSim):
        sim = cls(robot="helhest", batch_size=1, device="cpu")
        assert torch.equal(sim.robot_model.wheel_centers, ref.wheel_centers)
        assert torch.equal(sim.robot_model.wheel_radii, ref.wheel_radii)
        assert torch.equal(sim.robot_model.cog, ref.cog)


# ---------------------------------------------------------------------------
# Non-finite gradients: the two engines fail differently
# ---------------------------------------------------------------------------


def _rollout_with_poisoned_cotangent(engine, n_steps=50):
    """Backprop a loss whose cotangent contains an inf, and return d(support)."""
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


@requires_cuda
def test_warp_zeroes_non_finite_gradients_but_says_so():
    """The warp adjoint silently zeroed non-finite gradients; now it warns once.

    The torch engine propagates them instead and lets utils/trainer.py's
    `isfinite` check zero them one level up (utils/trainer.py:462-465), so the
    two engines fail differently on the same input. Zeroing is kept -- the
    trainer does the same thing -- but it is no longer silent, because a
    NaN-producing rollout that quietly contributes a zero gradient is
    indistinguishable from a converged one.
    """
    import simulator.engine_wp as ewp

    ewp._WARNED_NONFINITE_GRAD = False
    with pytest.warns(RuntimeWarning, match="non-finite gradient"):
        g_warp = _rollout_with_poisoned_cotangent("warp")
    assert torch.isfinite(g_warp).all(), "warp still returned a non-finite gradient"

    # and the warning fires only once per process, so a training loop is not
    # flooded by it
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _rollout_with_poisoned_cotangent("warp")
    assert not [w for w in caught if "non-finite gradient" in str(w.message)]


@requires_cuda
def test_torch_propagates_non_finite_gradients():
    """The other half of the asymmetry, pinned so it is a decision, not a bug."""
    g_torch = _rollout_with_poisoned_cotangent("torch")
    assert not torch.isfinite(g_torch).all(), (
        "the torch engine now sanitises its gradients too -- update the "
        "asymmetry documented in simulator/engine_wp.py and utils/trainer.py"
    )


# ---------------------------------------------------------------------------
# Compiler-level guard
# ---------------------------------------------------------------------------


def test_warp_kernels_have_no_dynamic_loops():
    """Static guard: no `@wp.kernel` may loop over a range() bound by a runtime value.

    Warp only unrolls loops whose range() arguments are compile-time constants
    (warp.config.max_unroll = 16). Anything else becomes a dynamic loop, and a
    dynamic loop that mutates state read in a later iteration gets an adjoint
    built on end-of-loop values — the bug this whole file exists to prevent.

    Checked by parsing the source rather than by watching for Warp's build-time
    warning: that warning is emitted during codegen, which Warp skips entirely
    when it serves the kernel from its on-disk cache, so a warning-based check
    passes vacuously on every run after the first. This one is deterministic,
    needs no GPU, and so runs in the normal local suite.
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

    # module-level names bound to wp.constant(...) or a plain int are unrollable
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
