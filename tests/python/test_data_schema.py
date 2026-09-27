"""Demo schema validation branches, which guard every pipeline artifact."""

import numpy as np
import pytest

from human2robot.data.schema import DEMO_SCHEMA_VERSION, DemoTrajectory

HORIZON = 5


def make_traj(**overrides) -> DemoTrajectory:
    """Build a schema-valid trajectory, then apply any field overrides."""
    base = {
        "q": np.zeros((HORIZON, 22), dtype=np.float32),
        "qdot": np.zeros((HORIZON, 22), dtype=np.float32),
        "a_demo": np.zeros((HORIZON, 22), dtype=np.float32),
        "object_pose": np.zeros((HORIZON, 7), dtype=np.float32),
        "object_vel": np.zeros((HORIZON, 6), dtype=np.float32),
        "contact": np.zeros((HORIZON, 5), dtype=np.float32),
        "trajectory_id": "t0",
        "task_id": "allegro_pickup",
        "source": "unit-test",
    }
    base.update(overrides)
    return DemoTrajectory(**base)


def test_schema_version_is_declared():
    assert isinstance(DEMO_SCHEMA_VERSION, str) and DEMO_SCHEMA_VERSION


def test_valid_trajectory_passes():
    make_traj().validate()


def test_empty_trajectory_rejected():
    with pytest.raises(ValueError, match="empty trajectory"):
        make_traj(
            q=np.zeros((0, 22)),
            qdot=np.zeros((0, 22)),
            a_demo=np.zeros((0, 22)),
            object_pose=np.zeros((0, 7)),
            object_vel=np.zeros((0, 6)),
            contact=np.zeros((0, 5)),
        ).validate()


@pytest.mark.parametrize("field", ["q", "qdot", "a_demo"])
def test_wrong_dof_rejected(field):
    bad = np.zeros((HORIZON, 16), dtype=np.float32)
    with pytest.raises(ValueError, match=field):
        make_traj(**{field: bad}).validate()


def test_wrong_horizon_rejected_for_dof_field():
    with pytest.raises(ValueError, match="qdot shape"):
        make_traj(qdot=np.zeros((HORIZON + 1, 22), dtype=np.float32)).validate()


def test_object_pose_shape_rejected():
    with pytest.raises(ValueError, match="object_pose shape"):
        make_traj(object_pose=np.zeros((HORIZON, 3), dtype=np.float32)).validate()


def test_object_vel_shape_rejected():
    with pytest.raises(ValueError, match="object_vel shape"):
        make_traj(object_vel=np.zeros((HORIZON, 2), dtype=np.float32)).validate()


def test_contact_horizon_mismatch_rejected():
    with pytest.raises(ValueError, match="contact horizon mismatch"):
        make_traj(contact=np.zeros((HORIZON + 1, 5), dtype=np.float32)).validate()


def test_contact_too_few_columns_rejected():
    with pytest.raises(ValueError, match="2-7 columns"):
        make_traj(contact=np.zeros((HORIZON, 1), dtype=np.float32)).validate()


def test_contact_too_many_columns_rejected():
    with pytest.raises(ValueError, match="2-7 columns"):
        make_traj(contact=np.zeros((HORIZON, 8), dtype=np.float32)).validate()


@pytest.mark.parametrize("width", [2, 5, 7])
def test_contact_width_bounds_accepted(width):
    make_traj(contact=np.zeros((HORIZON, width), dtype=np.float32)).validate()
