"""Physical sanity invariants of the differentiable spring-contact engine."""

import pytest
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
    cap would clip it on 998/1000 steps). Checks the engine's own rule.
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


@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_step_normal_force_ignores_wheel_spin(engine):
    """A static robot pressed into a 4 cm step: the contact force must not depend
    on wheel spin. Rim points are body-fixed, so only the body velocity
    compresses the contact; with the spin in v_n the force would read 2384 N
    still, 5336 N at +4.5 rad/s and 1482 N at -4.5 rad/s (friction 0 isolates it).
    """
    device = "cuda" if engine == "warp" and torch.cuda.is_available() else "cpu"
    dt = 2e-3
    sim = _make_sim_factory(dt, dt, engine=engine, device=device,
                            max_coord=MAX_COORD - RES / 2, robot="helhest_v2")
    sim.raise_to_clearance = lambda x, q, wc: x  # keep the pressed-in start
    xs = torch.linspace(-sim.max_coord, sim.max_coord, GRID_N)
    S = torch.zeros(1, GRID_N, GRID_N)
    S[..., xs > 0.2] = 0.04  # step ahead of the front axle (last grid axis is x)
    r = float(sim.robot_model.wheel_radii[0])
    init = {
        # 5 mm into flat ground; the front rims dip ~9 mm into the step top
        "x": torch.tensor([[0.1, 0.0, r - 0.005]], device=device),
        "q": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        "xd": torch.zeros(1, 3, device=device),
        "omega": torch.zeros(1, 3, device=device),
    }
    props = {"support": S.to(device), "friction": torch.zeros_like(S).to(device)}
    ts = torch.tensor([[0.0, dt]], device=device)

    def contact_force(omega):
        ctrl = torch.full((1, 2, 3), omega, device=device)
        with torch.no_grad():
            _, _, xd, _ = sim.predict_trajectory(props, ctrl, ts, initial_state=init)
        g = torch.tensor([0.0, 0.0, 9.81], device=device)
        return (sim.robot_model.mass * (xd[0, 0] / dt + g)).cpu()

    F0 = contact_force(0.0)
    assert F0[0] < -50 and F0[2] > 1000, f"robot not pressed into the step: {F0}"
    for omega in (4.5, -4.5):
        F = contact_force(omega)
        assert torch.allclose(F, F0, rtol=1e-4, atol=1e-2), (
            f"{engine}: wheel spin {omega} rad/s changed the contact force {F0} -> {F}"
        )
