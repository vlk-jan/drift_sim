"""Physical sanity invariants of the differentiable spring-contact engine."""

import torch
from _helpers import make_sim as _make_sim_factory

GRID_N = 64
RES = 0.1
MAX_COORD = GRID_N * RES / 2  # +-3.2 m


def _flat_props(mu, k=4e5):
    return {
        "support": torch.zeros(1, GRID_N, GRID_N),
        "friction": torch.full((1, GRID_N, GRID_N), mu),
        "stiffness": torch.full((1, GRID_N, GRID_N), k),
    }


def _make_sim(t_horizon, dt=0.002, com_transfer=False):
    return _make_sim_factory(
        t_horizon, dt, com_transfer=com_transfer, max_coord=MAX_COORD
    )


def _rollout(sim, props, wheel_omega, t_horizon):
    ctrl = torch.full((1, 2, 3), wheel_omega)
    ctrl_ts = torch.tensor([[100.0, 100.0 + t_horizon]])
    with torch.no_grad():
        poses, _, _, _ = sim.predict_trajectory(props, ctrl, ctrl_ts)
    return poses[0, :, :3, 3]


def test_static_equilibrium_on_flat_ground():
    """Zero wheel speed on flat ground: the robot must stay put."""
    t_horizon = 0.5
    sim = _make_sim(t_horizon)
    pos = _rollout(sim, _flat_props(mu=0.6), wheel_omega=0.0, t_horizon=t_horizon)

    horiz_drift = (pos[-1, :2] - pos[0, :2]).norm().item()
    z_drift = (pos[-1, 2] - pos[0, 2]).abs().item()
    assert horiz_drift < 0.05, f"robot drifted {horiz_drift:.3f} m at rest"
    assert z_drift < 0.05, f"robot sank/bounced {z_drift:.3f} m at rest"


def test_friction_monotonic_travel():
    """Same wheel commands from rest: higher friction must travel farther."""
    t_horizon = 1.0
    sim = _make_sim(t_horizon)

    def travel(mu):
        pos = _rollout(sim, _flat_props(mu=mu), wheel_omega=3.0, t_horizon=t_horizon)
        return (pos[-1, :2] - pos[0, :2]).norm().item()

    d_low, d_high = travel(0.1), travel(0.8)
    assert d_high > d_low + 0.05, (
        f"friction 0.8 travelled {d_high:.3f} m vs {d_low:.3f} m at friction 0.1"
    )


def test_impulse_limit_is_inert_at_the_training_dt():
    """The friction impulse cap must not bind at dt 2e-3, and must be reachable.

    Cap `F_fric <= m*|v_t| / N / dt`; measured headroom F_imp_lim / F_c_raw over
    a 0.5 s sloped rollout:

        dt 2e-3   min 18.4x   median 79x    0 points binding
        dt 5e-3   min  6.5x   median 29x    0 points binding
        dt 1e-2   min  1.4x   median 2.3x   0 points binding
        dt 2e-2   min 0.64x   median 0.83x  4 points binding
    """
    from simulator.utils import PhysicsState, WorldConfig
    from utils.geometry import quat_to_rotmat

    def min_headroom(dt):
        sim = _make_sim(0.2, dt=dt)
        engine = sim.engine
        props = _flat_props(mu=0.6)
        world = WorldConfig(
            z_grid=props["support"],
            grid_res=RES,
            max_coord=MAX_COORD,
            side_range=(-MAX_COORD, MAX_COORD),
            fwd_range=(-MAX_COORD, MAX_COORD),
            k_friction=props["friction"],
            k_stiffness=props["stiffness"],
        )
        state = PhysicsState.dummy(batch_size=1, device="cpu")
        state.x, state.q = sim.fit_plane_pose(world, 1)
        state.x = sim.raise_to_clearance(state.x, state.q, world)
        worst = float("inf")
        with torch.no_grad():
            for _ in range(int(0.2 / dt)):
                state.wheel_omega = torch.full((1, 3), 3.0)
                R = quat_to_rotmat(state.q)
                wc = (R @ engine._wheel_centers.T).transpose(
                    -1, -2
                ) + state.x.unsqueeze(1)
                pts = torch.einsum(
                    "bij,wpj->bwpi", R, engine._wheel_pos_rel
                ) + wc.unsqueeze(2)
                dh, n, stif, fric = engine._sample_terrain(pts, world)
                om = torch.stack(
                    [
                        torch.zeros_like(state.wheel_omega),
                        state.wheel_omega,
                        torch.zeros_like(state.wheel_omega),
                    ],
                    dim=-1,
                ).float()
                om_w = (R.unsqueeze(1) @ om.unsqueeze(-1)).squeeze(-1)
                v = (
                    state.xd.unsqueeze(1).unsqueeze(1)
                    + torch.linalg.cross(
                        state.omega.unsqueeze(1).unsqueeze(1),
                        pts - state.x.unsqueeze(1).unsqueeze(1),
                    )
                    + torch.linalg.cross(om_w.unsqueeze(2), pts - wc.unsqueeze(2))
                )
                in_c = (dh < 0).float()
                n_c = in_c.sum(dim=(1, 2), keepdim=True).clamp_min(1)
                F_s = engine._calculate_spring_force(dh, v, n, stif, in_c, n_c)
                F_N = (F_s * n).sum(dim=-1, keepdim=True).clamp_min(1e-9)
                v_t = v - (v * n).sum(dim=-1, keepdim=True) * n
                v_mag = v_t.norm(dim=-1, keepdim=True)
                F_c = fric.unsqueeze(-1) * F_N * torch.tanh(v_mag)
                F_lim = engine._mass * v_mag / n_c / dt
                live = (in_c > 0) & (F_c > 1e-12)
                if live.any():
                    worst = min(
                        worst, (F_lim / F_c.clamp_min(1e-12))[live].min().item()
                    )
                der = engine.forward(state, torch.zeros(1, 3), world)
                state = engine.update_state(state, der)
        return worst

    assert min_headroom(2e-3) > 2.0, "the impulse cap binds at the training dt"
    assert min_headroom(5e-2) < 1.0, (
        "the impulse cap never binds even at dt 5e-2 -- it is dead code and the "
        "measurement in this docstring is stale"
    )


def test_com_transfer_defaults_off_and_is_live_when_enabled():
    """com_transfer must default off and change the rollout when enabled.

    Without it xdd lands on the body origin, not the CoM; the dropped term is
    oscillatory (~5.8e-3 m pose error at 2 s) and production checkpoints were
    trained without it. Catches the flag not reaching the engine.
    """
    props = _flat_props(mu=0.6)
    ramp = torch.linspace(-1.0, 1.0, GRID_N)
    props["support"] = 0.06 * ramp.view(1, 1, -1).expand(1, GRID_N, GRID_N).clone()

    from simulator.utils import PhysicsEngineConfig

    assert PhysicsEngineConfig().com_transfer is False
    assert _make_sim(0.2).engine.config.com_transfer is False

    def final(com_transfer):
        sim = _make_sim(0.2, dt=0.002, com_transfer=com_transfer)
        return _rollout(sim, props, wheel_omega=3.0, t_horizon=0.2)[-1]

    delta = (final(True) - final(False)).norm().item()
    assert delta > 1e-5, (
        f"com_transfer changed the 0.2 s rollout by only {delta:.3e} m -- the "
        "flag is not reaching the engine"
    )
