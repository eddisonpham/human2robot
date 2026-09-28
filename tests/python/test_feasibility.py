"""Feasibility scoring: can the simulated Allegro hand execute a trajectory?

The metric is calibrated against three cases whose difficulty is known
independently of the code: a constant in-bounds target, a slow in-bounds ramp,
and a jump between opposite extremes. If the ordering of those three ever
changes, the metric has stopped measuring executability.
"""

from __future__ import annotations

import numpy as np
import pytest

from human2robot.data.limits import (
    ACTUATOR_LOWER,
    ACTUATOR_UPPER,
    BASE_DOF,
    DOF,
    FINGER_LOWER,
    FINGER_UPPER,
)
from human2robot.evaluation import feasibility as F


@pytest.fixture(scope="module", name="env")
def env_fixture():
    from human2robot.envs.allegro import AllegroPickupEnv

    env = AllegroPickupEnv()
    yield env
    env.close()


def _constant(horizon: int = 60) -> np.ndarray:
    mid = 0.5 * (FINGER_LOWER + FINGER_UPPER)
    return np.tile(np.concatenate([np.zeros(BASE_DOF), mid]), (horizon, 1))


def _ramp(horizon: int = 60) -> np.ndarray:
    span = FINGER_UPPER - FINGER_LOWER
    low = np.concatenate([np.zeros(BASE_DOF), FINGER_LOWER])
    step = np.concatenate([np.zeros(BASE_DOF), span])
    return low + np.linspace(0.0, 1.0, horizon)[:, None] * step


def _alternating_extremes(horizon: int = 60) -> np.ndarray:
    q = np.zeros((horizon, DOF))
    q[::2, BASE_DOF:] = FINGER_LOWER
    q[1::2, BASE_DOF:] = FINGER_UPPER
    return q


def test_constant_in_bounds_target_is_feasible(env):
    report = F.score_trajectory(_constant(), env, trajectory_id="constant")
    assert not report.diverged
    assert report.is_feasible
    assert report.out_of_bounds_frames == 0
    assert report.sustained_limit_violation == 0.0
    # Well below the 0.35 threshold: this is the hand's steady-state floor.
    assert report.tracking_drift_max < 0.10


def test_extremes_are_harder_than_a_constant(env):
    """The three calibration cases must order by difficulty."""
    constant = F.score_trajectory(_constant(), env, trajectory_id="c")
    ramp = F.score_trajectory(_ramp(), env, trajectory_id="r")
    extreme = F.score_trajectory(_alternating_extremes(), env, trajectory_id="e")
    assert constant.tracking_drift_max < ramp.tracking_drift_max
    assert ramp.tracking_drift_max < extreme.tracking_drift_max
    assert constant.is_feasible
    assert not extreme.is_feasible


def test_settling_removes_the_initial_transient(env):
    """Without a prelude the hand reports its start displacement, not tracking.

    Measured on a constant target this is the difference between ~0.53 rad of
    initial displacement and ~0.03 rad of genuine tracking error, so the metric
    would reject perfectly feasible motion.
    """
    settled = F.score_trajectory(_constant(), env, trajectory_id="c", settle_steps=25)
    unsettled = F.score_trajectory(_constant(), env, trajectory_id="c", settle_steps=0)
    assert settled.tracking_drift_max < 0.1
    assert unsettled.tracking_drift_max > 0.3
    assert settled.tracking_drift_max < unsettled.tracking_drift_max


def test_out_of_bounds_demand_is_measured_even_though_the_action_clamps(env):
    """A target outside the range must still be reported as a violation.

    The normalized action clips, so the simulator never sees the excess. If the
    metric measured the replayed action instead of the requested configuration
    it would report zero violation for an impossible demand.
    """
    q = _constant()
    q[:, BASE_DOF + 1] = FINGER_UPPER[1] + 0.5
    report = F.score_trajectory(q, env, trajectory_id="oob")
    assert report.sustained_limit_violation > 0.0
    assert report.out_of_bounds_frames == len(q)
    assert not report.is_feasible


def test_limit_tolerance_ignores_rounding_noise(env):
    """Resampling and smoothing put values within a milliradian of a bound."""
    q = _constant()
    q[:, BASE_DOF] = FINGER_UPPER[0] + 0.005
    strict = F.score_trajectory(q, env, trajectory_id="n", limit_tolerance=0.001)
    lenient = F.score_trajectory(q, env, trajectory_id="n", limit_tolerance=0.01)
    assert strict.out_of_bounds_frames == len(q)
    assert lenient.out_of_bounds_frames == 0


def test_score_trajectory_rejects_wrong_shape(env):
    with pytest.raises(ValueError, match="q shape"):
        F.score_trajectory(np.zeros((5, 21)), env, trajectory_id="bad")


def test_non_finite_trajectory_is_reported_diverged_not_scored(env):
    """A corrupt configuration must not be silently scored as in-bounds.

    The bound arithmetic on NaN comparisons all come out False, so an
    unguarded implementation reports zero violation and, with a short enough
    horizon, a perfectly feasible trajectory. It is reported diverged with
    unbounded violation instead of raising, so one bad file cannot abort a
    directory.
    """
    q = _constant()
    q[5, BASE_DOF + 3] = np.nan
    report = F.score_trajectory(q, env, trajectory_id="div")
    assert report.diverged
    assert not report.is_feasible
    assert report.tracking_drift_max == float("inf")
    assert report.sustained_limit_violation == float("inf")
    assert report.out_of_bounds_frames == 1


def test_empty_trajectory_is_rejected(env):
    with pytest.raises(ValueError, match="empty trajectory"):
        F.score_trajectory(np.zeros((0, DOF)), env, trajectory_id="none")


def test_summarize_counts_feasible_and_worst_cases(env):
    reports = [
        F.score_trajectory(_constant(), env, trajectory_id="ok"),
        F.score_trajectory(_alternating_extremes(), env, trajectory_id="bad"),
    ]
    summary = F.summarize(reports)
    assert summary["count"] == 2
    assert summary["feasible"] == 1
    assert summary["feasible_fraction"] == 0.5
    assert summary["tracking_drift_max_worst"] == pytest.approx(
        reports[1].tracking_drift_max
    )
    assert summary["diverged"] == 0


def test_summarize_rejects_empty():
    with pytest.raises(ValueError, match="no reports"):
        F.summarize([])


def test_score_demo_directory_selects_a_suffix(tmp_path, env):
    from human2robot.data.schema import DEMO_SCHEMA_VERSION, DemoTrajectory

    def write(stem: str) -> None:
        traj = DemoTrajectory(
            q=_constant(10).astype(np.float32),
            qdot=np.zeros((10, DOF), dtype=np.float32),
            a_demo=np.zeros((10, DOF), dtype=np.float32),
            object_pose=np.tile(np.array([0, 0, 0.1, 0, 0, 0, 1], np.float32), (10, 1)),
            object_vel=np.zeros((10, 6), dtype=np.float32),
            contact=np.zeros((10, 5), dtype=np.float32),
            trajectory_id=stem,
            task_id="allegro_pickup",
            source="unit-test",
        )
        traj.validate()
        import numpy as _np

        _np.savez_compressed(
            tmp_path / f"{stem}.npz",
            q=traj.q,
            qdot=traj.qdot,
            a_demo=traj.a_demo,
            object_pose=traj.object_pose,
            object_vel=traj.object_vel,
            contact=traj.contact,
            trajectory_id=traj.trajectory_id,
            task_id=traj.task_id,
            source=traj.source,
            schema_version=_np.array(DEMO_SCHEMA_VERSION),
        )

    write("seq_opt")
    write("seq_resampled")
    only_opt = F.score_demo_directory(tmp_path, env, suffix="_opt")
    assert len(only_opt) == 1
    assert only_opt[0].trajectory_id == "seq_opt"
    everything = F.score_demo_directory(tmp_path, env)
    assert len(everything) == 2


def test_score_demo_directory_rejects_empty(tmp_path, env):
    with pytest.raises(FileNotFoundError, match="no demos"):
        F.score_demo_directory(tmp_path, env)


def test_defaults_are_ordered_by_strictness():
    assert F.DEFAULT_MAX_DRIFT > F.DEFAULT_MAX_RMS_DRIFT
    assert F.DEFAULT_LIMIT_TOLERANCE < F.DEFAULT_MAX_RMS_DRIFT
    assert F.DEFAULT_SETTLE_STEPS > 0
    assert F.DEFAULT_MAX_LIMIT_VIOLATION == 0.0
    assert ACTUATOR_LOWER.shape == (DOF,)
    assert ACTUATOR_UPPER.shape == (DOF,)
