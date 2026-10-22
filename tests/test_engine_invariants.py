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


def _make_sim(t_horizon, dt=0.002, com_transfer=True):
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
    """The friction impulse cap must not bind while driving at dt 2e-3, and must
    be reachable (it binds at rest).

    Cap `F_fric <= share * |v_t m_eff / dt + m g_t|` below slip_ref_ratio
    (simulator/engine.py); it binds at rest by design, so this drives from the
    drop: straight at 3 rad/s on both rigs, and a helhest_v2 pivot turn whose
    dragged rear wheel slides past slip_ref_ratio, where the cap must not apply.
    """
    from simulator.utils import PhysicsState, WorldConfig
    from utils.geometry import quat_to_rotmat

    def min_headroom(dt, robot="helhest", omega=(3.0, 3.0, 3.0), settle=0.0):
        sim = _make_sim_factory(
            0.2, dt, com_transfer=True, max_coord=MAX_COORD, robot=robot
        )
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
            for i in range(int(0.2 / dt)):
                state.wheel_omega = torch.tensor([omega])
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
                F_s = engine._calculate_spring_force(dh, v, n, stif, in_c)
                F_N = (F_s * n).sum(dim=-1, keepdim=True).clamp_min(1e-9)
                v_t = v - (v * n).sum(dim=-1, keepdim=True) * n
                g_t = engine._F_g / engine._mass
                g_t = g_t - (g_t * n).sum(dim=-1, keepdim=True) * n
                v_s = v_t + g_t * dt
                v_mag = v_s.norm(dim=-1, keepdim=True)
                v_roll = (
                    state.wheel_omega.abs().float() * engine._wheel_radii
                ).clamp_min(engine.config.slip_floor_velocity)[:, :, None, None]
                slip = v_mag / v_roll
                F_c = (
                    fric.unsqueeze(-1)
                    * F_N
                    * torch.tanh(slip / engine.config.slip_ref_ratio)
                )
                r = pts - (state.x + R @ engine._cog)[:, None, None]
                u = torch.linalg.cross(r, v_s / (v_mag + 1e-6))
                I_inv = R @ engine._I_body_inv @ R.transpose(-1, -2)
                inv_m = 1 / engine._mass + (
                    u * torch.einsum("bij,bwpj->bwpi", I_inv, u)
                ).sum(-1, keepdim=True)
                depth = (-dh).clamp_min(0)
                share = depth / depth.sum(dim=(1, 2), keepdim=True)
                F_lim = share * (v_t / inv_m / dt + engine._mass * g_t).norm(
                    dim=-1, keepdim=True
                )
                live = (
                    (in_c > 0) & (F_c > 1e-12) & (slip < engine.config.slip_ref_ratio)
                )
                if live.any() and i * dt >= settle:
                    worst = min(
                        worst, (F_lim / F_c.clamp_min(1e-12))[live].min().item()
                    )
                der = engine.forward(state, torch.zeros(1, 3), world)
                state = engine.update_state(state, der)
        return worst

    assert min_headroom(2e-3) > 2.0, "the impulse cap binds at the training dt"
    assert min_headroom(2e-3, "helhest_v2") > 2.0, "cap binds driving helhest_v2"
    pivot = (0.5 / 0.334, -0.5 / 0.334, 0.0)
    # The dragged rear wheel binds for ~10 steps passing through the stiff
    # regime out of rest (the cap's job), then must slide uncapped.
    assert min_headroom(2e-3, "helhest_v2", pivot, settle=0.05) > 2.0, (
        "the cap binds in a sustained pivot turn"
    )
    assert min_headroom(2e-3, omega=(0.0, 0.0, 0.0)) < 1.0, (
        "the impulse cap never binds even at rest -- it is dead code"
    )


def test_com_transfer_defaults_on_and_is_live():
    """com_transfer must default on (the correct rigid-body dynamics) and
    change the rollout when toggled.

    Without it xdd lands on the body origin, not the CoM; the dropped term is
    oscillatory (~5.8e-3 m pose error at 2 s). Catches the flag not reaching
    the engine.
    """
    props = _flat_props(mu=0.6)
    ramp = torch.linspace(-1.0, 1.0, GRID_N)
    props["support"] = 0.06 * ramp.view(1, 1, -1).expand(1, GRID_N, GRID_N).clone()

    from simulator.utils import PhysicsEngineConfig

    assert PhysicsEngineConfig().com_transfer is True
    assert _make_sim(0.2).engine.config.com_transfer is True

    def final(com_transfer):
        sim = _make_sim(0.2, dt=0.002, com_transfer=com_transfer)
        return _rollout(sim, props, wheel_omega=3.0, t_horizon=0.2)[-1]

    delta = (final(True) - final(False)).norm().item()
    assert delta > 1e-5, (
        f"com_transfer changed the 0.2 s rollout by only {delta:.3e} m -- the "
        "flag is not reaching the engine"
    )
