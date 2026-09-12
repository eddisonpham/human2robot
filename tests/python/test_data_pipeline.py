"""Tests for Tier B demo pipeline."""

import numpy as np
import pytest

from dynhand.data.allegro_demos import (
    generate_synthetic_demos,
    load_demo_npz,
    validate_open_loop_replay,
)
from dynhand.data.processing import demo_actions, differentiate, smooth
from dynhand.data.schema import DemoTrajectory


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
    report = validate_open_loop_replay(paths, tolerance=0.5, sample=3)
    assert "mean_error" in report


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
