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

# The forward passes already agree to ~3e-4 m over a 1 s rollout; anything much
# looser would stop catching real physics divergence.
POS_ATOL = 2e-3
# Gradient direction is what the optimizer actually follows, so it is the tight
# bound; magnitude is allowed a factor of 2 to absorb contact-boundary jitter
# (the hard `dist < 0` gate can flip a sample point between the two engines).
GRAD_COS_MIN = 0.99
GRAD_RATIO_RANGE = (0.5, 2.0)


def _make_sim(engine, t_horizon, batch_size=1):
    if engine == "torch":
        from models.simulator import Simulator
    else:
        from models.simulator_wp import Simulator

    sim = Simulator(
        robot="helhest",
        batch_size=batch_size,
        device=DEVICE,
        T_horizon=t_horizon,
        dt=DT,
    )
    sim.update_world_config(
        res=RES,
        max_coord=MAX_COORD,
        side_range=(-MAX_COORD, MAX_COORD),
        fwd_range=(-MAX_COORD, MAX_COORD),
    )
    return sim


def _make_inputs(t_horizon, batch_size=1, n_ctrl=50, distinct_batch=False):
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

    ctrl_ts = torch.linspace(0.0, max(t_horizon, DT), n_ctrl, device=DEVICE)
    ctrl_ts = ctrl_ts.unsqueeze(0).expand(batch_size, -1)
    return support, friction, ctrl, ctrl_ts


def _rollout(engine, n_steps, batch_size=1, distinct_batch=False):
    """Run one rollout and backprop a trajectory loss.

    Returns (final_pos_per_batch, grad_support, grad_friction).
    """
    t_horizon = n_steps * DT
    sim = _make_sim(engine, t_horizon, batch_size)
    support, friction, ctrl, ctrl_ts = _make_inputs(
        t_horizon, batch_size, distinct_batch=distinct_batch
    )

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
@pytest.mark.parametrize("n_steps", [50, 250, 1000])
def test_forward_pose_matches_torch(n_steps):
    pos_t, _, _ = _rollout("torch", n_steps)
    pos_w, _, _ = _rollout("warp", n_steps)
    err = (pos_w - pos_t).norm().item()
    assert err < POS_ATOL, (
        f"warp/torch final pose diverged by {err:.3e} m over {n_steps} steps "
        f"(torch={pos_t.tolist()}, warp={pos_w.tolist()})"
    )


# ---------------------------------------------------------------------------
# Gradient parity — the actual gate
# ---------------------------------------------------------------------------


# 100 steps is where the old fused-kernel adjoint first detonated (cosine went to
# ~0 and the norm ratio to 5.5e6); 1000 is the length training and adaptation
# really use, and is where it returned NaN. Both must be covered or the
# regression walks straight back in.
@requires_cuda
@pytest.mark.parametrize("n_steps", [50, 100, 250, 1000])
def test_gradients_match_torch(n_steps):
    _, gs_t, gf_t = _rollout("torch", n_steps)
    _, gs_w, gf_w = _rollout("warp", n_steps)

    assert torch.isfinite(gs_w).all(), (
        f"warp support grad is non-finite at {n_steps} steps"
    )
    assert torch.isfinite(gf_w).all(), (
        f"warp friction grad is non-finite at {n_steps} steps"
    )

    for name, gw, gt in (("support", gs_w, gs_t), ("friction", gf_w, gf_t)):
        cos, ratio = _cos(gw, gt), _ratio(gw, gt)
        assert cos > GRAD_COS_MIN, (
            f"{name} gradient direction diverged at {n_steps} steps: "
            f"cosine={cos:.4f} (need > {GRAD_COS_MIN})"
        )
        lo, hi = GRAD_RATIO_RANGE
        assert lo < ratio < hi, (
            f"{name} gradient magnitude diverged at {n_steps} steps: "
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
        lo, hi = GRAD_RATIO_RANGE
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
    robot's wheel contacts sit at x=+0.4/-0.5, y=±0.52, so the grid centre lies
    *between* the wheels and carries no gradient at all — a fixed cell list
    silently verifies nothing.
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
