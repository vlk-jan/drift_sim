"""The drawn robot must be the robot.

Nothing in the physics reads the chassis box -- the engine knows the machine as
three wheel contacts plus mass and inertia -- so a wrong box costs nothing and
is never caught by a metric. It is still worth being right: every judgement made
by looking at a SimView scene is made against this outline. A 1.00 x 0.60 x
0.30 m box centred on base_link, which is the FRONT AXLE, hangs 0.5 m out in
front of the front wheels and stops 0.2 m short of the rear one.

The envelope asserted here is dataset_tools' viewer's (CHASSIS / CHASSIS_CENTER
in dataset_tools/viewer/build.py), so the same robot is drawn in both tools.
It is deliberately NOT the URDF's base_link box (`0.26 0.6 0.18` at
`-0.047 0 0`): that is the frame member alone, without motors, battery or
wheel forks.
"""

import numpy as np
import pytest
import torch

from drift_sim.utils import (
    HELHEST_CHASSIS,
    HELHEST_CHASSIS_CENTER,
    WHEEL_VISUAL_WIDTH,
    add_helhest_visual_body,
    add_helhest_visual_wheels,
    chassis_local,
    chassis_positions,
    robot_model_config,
)

FRONT_SENSOR_X = 0.17  # tf_static: front sensor face, the most forward fixture
WHEEL_HOLDER_X = -0.477  # tf_static: the rear-wheel fork mount


def box_span():
    cx = HELHEST_CHASSIS_CENTER[0]
    return cx - HELHEST_CHASSIS["hx"], cx + HELHEST_CHASSIS["hx"]


def wheel_bodies(robot):
    """The wheels add_helhest_visual_wheels puts into a real SimView scene."""
    from simview import SimulationScene

    scene = SimulationScene(batch_size=1, scalar_names=[], dt=0.01)
    add_helhest_visual_body(scene, "box")
    add_helhest_visual_wheels(scene, "box", robot=robot)
    return [b for name, b in scene.model.bodies.items() if name != "box"]


# --- the box covers the machine and nothing more ---


def test_the_box_reaches_the_front_sensor_but_not_past_the_front_wheels():
    back, front = box_span()
    assert front >= FRONT_SENSOR_X, "the box stops short of the front fixtures"
    r = robot_model_config("helhest_v2").wheel_radii[0].item()
    assert front < r, "the box juts out past the front wheels' own outline"


def test_the_box_covers_the_rear_wheel_fork():
    back, _ = box_span()
    assert back <= WHEEL_HOLDER_X, "the box ends before the wheel holders"


def test_the_box_does_not_reach_the_rear_wheel_hub():
    """It is a chassis, not a bounding box: the rear wheel hangs off a fork."""
    back, _ = box_span()
    rear_hub = robot_model_config("helhest_v2").wheel_centers[2, 0].item()
    assert back > rear_hub


def test_the_box_is_narrow_enough_to_clear_the_tyres():
    track = robot_model_config("helhest_v2").wheel_centers[0, 1].item()
    assert HELHEST_CHASSIS["hy"] < track - WHEEL_VISUAL_WIDTH / 2.0


def test_the_box_sits_within_the_wheels_vertically():
    assert HELHEST_CHASSIS["hz"] < robot_model_config("helhest_v2").wheel_radii[0].item()


# --- the wheels land on the axles despite the offset ---


def test_drawn_hubs_undo_the_chassis_offset():
    bodies = wheel_bodies("helhest_v2")
    model = robot_model_config("helhest_v2")
    assert len(bodies) == 3
    for body, want in zip(bodies, model.wheel_centers.tolist()):
        hub = np.array(body.local_transform[:3]) + np.array(HELHEST_CHASSIS_CENTER)
        np.testing.assert_allclose(hub, want, atol=1e-9)


def test_drawn_wheels_use_the_radius_of_the_robot_asked_for():
    bodies = wheel_bodies("helhest_v2")
    want = robot_model_config("helhest_v2").wheel_radii[0].item()
    for body in bodies:
        assert body.shape["radius"] == pytest.approx(want)


# --- the body that carries the trajectory is the box, not base_link ---


def test_identity_pose_shifts_by_the_body_frame_offset():
    pos = np.zeros((3, 3))
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (3, 1))
    np.testing.assert_allclose(
        chassis_positions(pos, quat), np.tile(HELHEST_CHASSIS_CENTER, (3, 1)), atol=1e-9
    )


def test_the_offset_rotates_with_the_pose():
    """Half a turn about z must put the box ahead of base_link in world x."""
    pos = np.zeros((1, 3))
    quat = np.array([[0.0, 0.0, 0.0, 1.0]])  # 180 deg about z, [w, x, y, z]
    got = chassis_positions(pos, quat)
    assert got[0, 0] == pytest.approx(-HELHEST_CHASSIS_CENTER[0], abs=1e-9)


def test_a_batch_of_trajectories_keeps_its_shape():
    pos = np.zeros((5, 4, 3))
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (5, 4, 1))
    assert chassis_positions(pos, quat).shape == (5, 4, 3)


def test_tensors_stay_tensors():
    pos = torch.zeros(2, 3)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
    out = chassis_positions(pos, quat)
    assert isinstance(out, torch.Tensor) and out.shape == (2, 3)


def test_contact_points_move_into_the_box_frame():
    pts = np.zeros((4, 3))
    np.testing.assert_allclose(
        chassis_local(pts), -np.tile(HELHEST_CHASSIS_CENTER, (4, 1)), atol=1e-9
    )


def test_a_wheel_contact_point_lands_under_its_hub():
    """The end-to-end check: hub offset + contact offset = the real contact."""
    model = robot_model_config("helhest_v2")
    hub = model.wheel_centers[0].numpy()
    contact_base = hub - np.array([0.0, 0.0, model.wheel_radii[0].item()])
    in_box = chassis_local(contact_base[None])[0]
    np.testing.assert_allclose(in_box + np.array(HELHEST_CHASSIS_CENTER), contact_base)
