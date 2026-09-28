"""Tests for Tier B demo pipeline."""

import numpy as np
import pytest

from human2robot.data.allegro_demos import generate_synthetic_demos, load_demo_npz
from human2robot.data.limits import BASE_DOF, FINGER_LOWER, FINGER_UPPER
from human2robot.data.processing import demo_actions, differentiate, smooth
from human2robot.data.schema import DemoTrajectory


def test_smooth_and_differentiate() -> None:
    x = np.linspace(0, 1, 20)[:, None].repeat(2, axis=1)
    s = smooth(x, window=5)
    assert s.shape == x.shape
    v, a = differentiate(x, dt=0.02)
    assert v.shape == x.shape and a.shape == x.shape


def test_demo_actions_shape() -> None:
    q = np.zeros((10, 22))
    q[1:] = 0.1
    a = demo_actions(q, np.zeros(22), np.ones(22) * 2)
    assert a.shape == (10, 22)
    assert np.all(np.abs(a) <= 1.0)


def test_generate_and_validate_tier_b_demos(tmp_path) -> None:
    paths = generate_synthetic_demos(tmp_path / "demos", count=5, seed=1)
    assert len(paths) == 5
    traj = load_demo_npz(paths[0])
    assert traj.q.shape[1] == 22
    # Generated demos must respect the real actuator range, not a stub one.
    # Checking against `zeros`/`ones` here is what let a retargeting bug put the
    # thumb's motion in the middle finger's slot unnoticed.
    fingers = traj.q[:, BASE_DOF:]
    assert fingers.min() >= FINGER_LOWER.min() - 1e-6
    assert fingers.max() <= FINGER_UPPER.max() + 1e-6


def test_all_generated_demos_round_trip(tmp_path) -> None:
    paths = generate_synthetic_demos(tmp_path / "demos", count=4, seed=3)
    for path in paths:
        traj = load_demo_npz(path)
        assert traj.q.shape == traj.a_demo.shape
        assert traj.trajectory_id == path.stem


def test_schema_validation_rejects_bad_shape() -> None:
    traj = DemoTrajectory(
        q=np.zeros((10, 5)),
        qdot=np.zeros((10, 22)),
        a_demo=np.zeros((10, 22)),
        object_pose=np.zeros((10, 7)),
        object_vel=np.zeros((10, 6)),
        contact=np.zeros((10, 5)),
        trajectory_id="x",
        task_id="y",
        source="synthetic",
    )
    with pytest.raises(ValueError):
        traj.validate()
