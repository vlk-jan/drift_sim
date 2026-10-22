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


def test_impulse_cut_is_inert_while_driving():
    """The friction cut (DPhysicsEngine._friction) must not bind while driving
    at dt 2e-3, and must bind at rest, where it is the stabiliser.

    Drives from the drop: straight at 3 rad/s on both rigs, and a helhest_v2
    pivot turn whose dragged rear wheel slides under drive torque (a gravity-only
    cap once clipped it on 998/1000 steps). Checks the engine's own rule.
    """
    from simulator.utils import PhysicsState, WorldConfig

    def cut_steps(robot="helhest", omega=(3.0, 3.0, 3.0), t=0.2, dt=2e-3):
        sim = _make_sim_factory(
            t, dt, com_transfer=True, max_coord=MAX_COORD, robot=robot
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
        n = 0
        with torch.no_grad():
            for _ in range(int(t / dt)):
                state.wheel_omega = torch.tensor([omega])
                c = engine._contact_terms(state, world)
                n += bool((engine._friction(state, c) < c["F_c"]).any())
                state = engine.update_state(state, engine.forward(state, world))
        return n

    assert cut_steps() == 0, "the friction cut binds driving helhest"
    assert cut_steps("helhest_v2") == 0, "the friction cut binds driving helhest_v2"
    pivot = (0.5 / 0.334, -0.5 / 0.334, 0.0)
    assert cut_steps("helhest_v2", pivot, t=1.0) == 0, "the cut binds in a pivot"
    assert cut_steps(omega=(0.0, 0.0, 0.0)) > 0, (
        "the friction cut never binds even at rest -- it is dead code"
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
