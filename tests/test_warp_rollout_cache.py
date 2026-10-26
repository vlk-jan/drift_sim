"""The warp engine's captured, shape-cached rollout (simulator/engine_wp.py::_Rollout)."""

import pytest
import torch
from _helpers import make_sim


def _rollout(sim, sup, B=2):
    ctrl = torch.full((B, 20, 3), 1.0)
    ts = (
        (1.7e9 + torch.arange(20, dtype=torch.float64) * 0.005)
        .expand(B, -1)
        .contiguous()
    )
    fr = torch.full_like(sup, 0.6)
    T, _, xd, _ = sim.predict_trajectory({"support": sup, "friction": fr}, ctrl, ts)
    return T[..., :3, 3].sum() + xd.sum()


def test_repeat_call_reuses_buffers_and_matches():
    sim = make_sim(t_horizon=0.05, dt=0.005, engine="warp")
    sup = (
        0.02 * torch.randn(2, 32, 32, generator=torch.Generator().manual_seed(0))
    ).requires_grad_()
    grads = []
    for _ in range(2):
        sup.grad = None
        _rollout(sim, sup).backward()
        grads.append(sup.grad.clone())
    assert torch.isfinite(grads[0]).all() and grads[0].abs().sum() > 0
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-5, atol=1e-6)


def test_a_warm_call_on_new_terrain_matches_a_fresh_capture():
    """The per-point contact flags (simulator/engine_wp.py) must not leak from one
    call into the next: a stale "out of contact" would skip a point in contact."""
    from simulator import engine_wp

    g = torch.Generator().manual_seed(0)
    ground = [
        (0.04 * torch.randn(2, 32, 32, generator=g)).requires_grad_() for _ in range(2)
    ]

    def pose_and_grad(sim, sup):
        sup.grad = None
        loss = _rollout(sim, sup)
        loss.backward()
        return loss.detach(), sup.grad.clone()

    sim = make_sim(t_horizon=0.05, dt=0.005, engine="warp")
    pose_and_grad(sim, ground[0])
    warm = pose_and_grad(sim, ground[1])
    engine_wp._ROLLOUTS.clear()
    fresh = pose_and_grad(sim, ground[1])
    for a, b in zip(warm, fresh):
        assert torch.equal(a, b)


def test_backward_after_a_newer_rollout_of_the_same_shape_raises():
    sim = make_sim(t_horizon=0.05, dt=0.005, engine="warp")
    sup = torch.zeros(2, 32, 32, requires_grad=True)
    first = _rollout(sim, sup)
    _rollout(sim, sup)
    with pytest.raises(RuntimeError, match="another rollout"):
        first.backward()


def _run(sim, B=2, grid=32, n_ctrl=20, sup_scale=0.02):
    """Final pose and support gradient of one rollout (CPU warp: deterministic)."""
    g = torch.Generator().manual_seed(0)
    sup = (sup_scale * torch.randn(B, grid, grid, generator=g)).requires_grad_()
    ctrl = 1.0 + 0.5 * torch.randn(B, n_ctrl, 3, generator=g)
    ts = (
        (1.7e9 + torch.arange(n_ctrl, dtype=torch.float64) * 0.005)
        .expand(B, -1)
        .contiguous()
    )
    fr = torch.full_like(sup, 0.6)
    T, _, xd, _ = sim.predict_trajectory({"support": sup, "friction": fr}, ctrl, ts)
    (T[..., :3, 3].sum() + xd.sum()).backward()
    return T[:, -1].detach(), sup.grad


@pytest.mark.parametrize(
    "change",
    [
        "batch",
        "horizon",
        "grid",
        "dt",
        "slip_ref_ratio",
        "slip_floor_velocity",
        "damping_alpha",
        "ang_damping",
        "torque_limit",
        "com_transfer",
        "robot",
    ],
)
def test_changed_shape_or_scalar_never_replays_a_stale_graph(change, monkeypatch):
    """After a warm call, a call that differs in one key component equals the same
    call on an empty cache (a fresh capture), and differs from the warm result."""
    from dataclasses import replace

    from simulator import engine_wp, utils

    # Another robot: the v2 rig on 0.36 m wheels.
    monkeypatch.setitem(
        utils.ROBOT_MODELS,
        "other",
        lambda: replace(
            utils.helhest_v2_robot_model(), wheel_radii=torch.tensor([0.36] * 3)
        ),
    )
    kw = {}

    def setup():
        sim = make_sim(
            t_horizon=0.05,
            dt=0.005,
            engine="warp",
            max_coord=1.6,
            robot="other" if change == "robot" else "helhest_v2",
        )
        if change in (
            "dt",
            "slip_ref_ratio",
            "slip_floor_velocity",
            "damping_alpha",
            "ang_damping",
            "torque_limit",
        ):
            # Factors large enough that the term is active in this short rollout;
            # the damping cap binds at rest at alpha 2.0, so alpha goes below it.
            f = {
                "slip_floor_velocity": 20.0,
                "torque_limit": 1e-4,
                "damping_alpha": 0.25,
            }.get(change, 1.5)
            setattr(sim.engine_config, change, getattr(sim.engine_config, change) * f)
        if change == "com_transfer":
            sim.engine_config.com_transfer = not sim.engine_config.com_transfer
        if change == "horizon":
            sim.T_horizon = 0.08
        if change == "grid":
            sim.update_world_config(
                res=0.08,
                max_coord=1.28,
                side_range=(-1.28, 1.28),
                fwd_range=(-1.28, 1.28),
            )
        return sim

    if change == "batch":
        kw["B"] = 3
    if change == "grid":
        kw["grid"] = 33
    base = _run(make_sim(t_horizon=0.05, dt=0.005, engine="warp"))
    warm = _run(setup(), **kw)
    engine_wp._ROLLOUTS.clear()
    fresh = _run(setup(), **kw)
    for a, b in zip(warm, fresh):
        assert torch.equal(a, b)
    if warm[0].shape == base[0].shape:
        assert not torch.equal(warm[0], base[0])


def test_no_grad_rollout_between_forward_and_backward_is_harmless():
    sim = make_sim(t_horizon=0.05, dt=0.005, engine="warp")
    sup = (
        0.02 * torch.randn(2, 32, 32, generator=torch.Generator().manual_seed(0))
    ).requires_grad_()
    _rollout(sim, sup).backward()
    ref = sup.grad.clone()
    sup.grad = None
    loss = _rollout(sim, sup)
    with torch.no_grad():
        other = _rollout(sim, sup * 2)  # same shape: must use its own buffers
    loss.backward()
    assert torch.equal(sup.grad, ref)
    assert torch.isfinite(other)
