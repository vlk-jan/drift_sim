"""A training rollout whose support gradient spiked: batch 41, row 83 (segment 5,
sample 3), the robot standing still on rough ground under zero wheel commands,
with row 51 (segment 3, same sample) as a normal row. data/spike_rollout_b41.pt
holds both rows' simulator inputs and the final pose each reaches.

Row 83 is the field case of tests/test_low_slip_gradient.py. With the exact
slip-direction derivative its gradient here is 7e8 (torch) and 7e7 (warp)
instead of 22, and a 1e-8 m support field scales it by 1.5 (torch) and 15
(warp). The final poses are the seated start's (simulator/utils.py::
seat_on_terrain), torch's.
"""

from pathlib import Path

import pytest
import torch
from _helpers import make_sim

FIXTURE = Path(__file__).parent / "data" / "spike_rollout_b41.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Support gradient norms of the summed squared positions (test_engine_parity's
# loss): 22 (row 83) and 22 (row 51) in both engines.
GRAD_NORM_MAX = 1e3
# The stick-slip at rest is rounding-sensitive: ten 1e-8 m support fields move
# the final pose by up to 9e-6 m, and warp ends 7e-6 m off the torch reference.
POSE_ATOL = 2e-4


@pytest.fixture(scope="module")
def rollouts():
    """Reference final poses (rows 83, 51) and, per engine, the final poses and
    support gradients of row 83, row 51 and row 83 plus a N(0, 1e-8 m) field."""
    d = torch.load(FIXTURE, weights_only=True)
    rows = [0, 1, 0]
    field = 1e-8 * torch.randn(
        d["support"].shape[1:], generator=torch.Generator().manual_seed(0)
    )
    out = {}
    for engine in ("torch", "warp"):
        sim = make_sim(
            d["T_horizon"], d["dt"], engine=engine, device=DEVICE, robot=d["robot"],
            **d["world"],
        )  # fmt: skip
        support = d["support"][rows].clone()
        support[2] += field
        support = support.to(DEVICE).requires_grad_(True)
        traj, _, _, _ = sim.predict_trajectory(
            {"support": support, "friction": d["friction"][rows].to(DEVICE)},
            d["ctrl"][rows].to(DEVICE),
            d["ctrl_ts"][rows].to(DEVICE),
            initial_state={
                k: v[rows].to(DEVICE) for k, v in d["initial_state"].items()
            },
            T_horizon=d["T_horizon"],
            include_start=d["include_start"],
        )
        traj[:, :, :3, 3].pow(2).sum().backward()
        out[engine] = traj[:, -1].detach(), support.grad.flatten(1)
    return d["final_pose"].to(DEVICE), out


@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_support_gradients_are_bounded(engine, rollouts):
    _, grad = rollouts[1][engine]
    norms = grad[:2].norm(dim=1)
    assert torch.isfinite(grad).all(), f"{engine}: non-finite support gradient"
    assert (norms < GRAD_NORM_MAX).all(), f"{engine}: rows 83, 51 {norms.tolist()}"


@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_rounding_noise_keeps_the_gradient(engine, rollouts):
    _, grad = rollouts[1][engine]
    change = (grad[2].norm() / grad[0].norm() - 1).abs().item()
    assert change < 0.05, f"{engine}: a 1e-8 m field moved row 83 by {change:.1%}"


@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_final_poses_match_the_reference(engine, rollouts):
    ref, out = rollouts
    err = (out[engine][0][:2] - ref).abs().max().item()
    assert err < POSE_ATOL, f"{engine}: final pose off the reference by {err:.3e}"


def test_engines_agree_on_the_spike_row(rollouts):
    out = rollouts[1]
    cos = torch.nn.functional.cosine_similarity(
        out["torch"][1][0], out["warp"][1][0], dim=0
    ).item()
    assert cos > 0.999, f"row 83 torch/warp support gradient cosine {cos:.6f}"


# Batch 1019, row 14, segment 2 (v26 recipe, trained heads): zero commands, the
# robot slides ~17 cm off its seat. With the exact slip-ratio derivative the
# stopped wheels' tanh law outruns the explicit step and the adjoint grows from
# step 20: |dL/dsupport| 2604 (torch), 1242 (warp) where finite differences give
# < 0.02. SLIP_GRAD_FLOOR brings both to ~1e-4. L = sum(upstream . outputs), the
# training loss's own upstream gradient on this row.
FIXTURE_B1019 = Path(__file__).parent / "data" / "spike_rollout_b1019.pt"


@pytest.mark.parametrize("engine", ["torch", "warp"])
def test_sliding_off_the_seat_keeps_the_gradient_bounded(engine):
    d = torch.load(FIXTURE_B1019, weights_only=True)
    sim = make_sim(
        d["T_horizon"], 2e-3, engine=engine, device=DEVICE, robot=d["robot"],
        res=d["world"]["res"], max_coord=d["world"]["max_coord"],
    )  # fmt: skip
    sim.update_world_config(**d["world"])
    support = d["support"].to(DEVICE).requires_grad_(True)
    traj, _, xd, omega = sim.predict_trajectory(
        {"support": support, "friction": d["friction"].to(DEVICE)},
        d["ctrl"].to(DEVICE),
        d["ctrl_ts"].to(DEVICE),
        initial_state={k: v.to(DEVICE) for k, v in d["initial_state"].items()},
        T_horizon=d["T_horizon"],
        include_start=True,
    )
    up = {k: v.to(DEVICE) for k, v in d["upstream"].items()}
    loss = sum((o * up[k]).sum() for o, k in ((traj, "traj"), (xd, "xd"), (omega, "omega")))
    loss.backward()
    norm = support.grad.norm().item()
    assert torch.isfinite(support.grad).all(), f"{engine}: non-finite support gradient"
    assert norm < 1e-2, f"{engine}: support gradient {norm:.3g}"
