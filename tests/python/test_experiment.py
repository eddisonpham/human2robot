"""The shared experiment machinery.

These are the definitions every experiment script imports, so a mistake here
propagates into every published table. The properties tested are the ones a
reader assumes silently: splits are disjoint and complete, reductions are
signed correctly, and the two resampling rates are consistent with each other.
"""

from __future__ import annotations

import numpy as np
import pytest

from human2robot.data.limits import (
    ACTUATOR_LOWER,
    ACTUATOR_UPPER,
    BASE_DOF,
    CAPTURE_DT,
    CONTROL_DT,
    DOF,
)
from human2robot.optimization import experiment as X


def _trajectories(n: int = 3, length: int = 20, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(length, DOF)) for _ in range(n)]


def test_resampling_rates_match_the_hardware():
    capture_dt, control_dt = CAPTURE_DT, CONTROL_DT
    assert capture_dt == pytest.approx(1.0 / 30.0)
    assert control_dt == pytest.approx(0.02)
    assert control_dt < capture_dt


def test_optimizer_config_uses_the_canonical_bounds():
    cfg = X.make_optimizer_config().raw
    assert cfg.step_size == 0.35
    assert cfg.noise_scale == 0.0
    assert cfg.step_size == X.DEFAULT_STEP_SIZE
    assert cfg.noise_scale == X.NOISE_SCALE
    assert np.allclose(cfg.limits.lower, ACTUATOR_LOWER)
    assert np.allclose(cfg.limits.upper, ACTUATOR_UPPER)
    assert np.allclose(cfg.limits.lower[:BASE_DOF], 0.0)
    fingers_lower = np.asarray(cfg.limits.lower[BASE_DOF:])
    fingers_upper = np.asarray(cfg.limits.upper[BASE_DOF:])
    assert np.all(fingers_lower < fingers_upper)


def test_optimizer_config_accepts_an_override():
    cfg = X.make_optimizer_config(step_size=0.25, max_iterations=7, seed=3).raw
    assert cfg.step_size == 0.25
    assert cfg.max_iterations == 7
    assert cfg.seed == 3


def test_optimizer_config_weights_are_not_smoothness_dominated():
    """Tracking must outweigh smoothness or the optimizer erases the motion."""
    cfg = X.make_optimizer_config().raw
    assert cfg.weights.tracking > cfg.weights.velocity
    assert cfg.weights.tracking > cfg.weights.acceleration
    assert cfg.weights.tracking > cfg.weights.jerk
    assert cfg.weights.limits > cfg.weights.tracking


def test_to_transitions_pairs_adjacent_steps():
    traj = np.arange(12.0).reshape(4, 3)
    pairs = X.to_transitions([traj])
    assert len(pairs) == 1
    obs, act = pairs[0]
    assert obs.shape == (3, 3)
    assert act.shape == (3, 3)
    assert np.allclose(act, np.array([3.0, 3.0, 3.0]))


def test_to_transitions_handles_single_step_trajectories():
    assert X.to_transitions([np.zeros((1, DOF))]) == []


def test_aggregate_reports_mean_std_and_max():
    rows = [{key: v for key in X.METRIC_KEYS} for v in (1.0, 2.0, 3.0)]
    agg = X.aggregate_metrics(rows)
    for key in X.METRIC_KEYS:
        assert agg[key]["mean"] == pytest.approx(2.0)
        assert agg[key]["std"] == pytest.approx(np.std([1.0, 2.0, 3.0]))
        assert agg[key]["max"] == pytest.approx(3.0)


def test_pairwise_reductions_is_positive_when_the_metric_falls():
    before = {key: {"mean": 2.0} for key in X.METRIC_KEYS}
    after = {key: {"mean": 1.0} for key in X.METRIC_KEYS}
    out = X.pairwise_reductions(before, after)
    assert set(out) == {f"{k}_reduction_pct" for k in X.METRIC_KEYS}
    assert all(v == pytest.approx(50.0) for v in out.values())
    assert X.pairwise_reductions(after, before)[
        f"{X.METRIC_KEYS[0]}_reduction_pct"
    ] == (pytest.approx(-100.0))


def test_pairwise_reductions_rejects_an_empty_baseline():
    before = {key: {"mean": 0.0} for key in X.METRIC_KEYS}
    after = {key: {"mean": 1.0} for key in X.METRIC_KEYS}
    with pytest.raises(ValueError, match="non-positive baseline"):
        X.pairwise_reductions(before, after)


def _assert_disjoint_and_complete(split_a, split_b, total_rows):
    (train_obs, train_act), (test_obs, test_act) = split_a
    assert train_obs.shape == train_act.shape
    assert test_obs.shape == test_act.shape
    assert len(train_obs) + len(test_obs) == total_rows
    assert len(train_obs) > 0
    assert len(test_obs) > 0
    joined_train = {tuple(r) for r in np.round(train_obs, 6)}
    joined_test = {tuple(r) for r in np.round(test_obs, 6)}
    assert not (joined_train & joined_test)


def test_split_transition_partitions_all_transitions():
    trajs = _trajectories(n=4, length=10)
    total = sum(len(t) - 1 for t in trajs)
    rng = np.random.default_rng(0)
    _assert_disjoint_and_complete(
        X.split_transition(trajs, rng, fraction=0.1),
        X.split_transition(trajs, np.random.default_rng(0), fraction=0.1),
        total,
    )


def test_split_trajectory_holds_out_whole_trajectories():
    trajs = _trajectories(n=5, length=8)
    rng = np.random.default_rng(1)
    (train_obs, _), (test_obs, _) = X.split_trajectory(trajs, rng, fraction=0.2)
    assert len(test_obs) == 7
    assert len(train_obs) == 4 * 7


def test_split_prefix_extrapolates_in_time():
    """Every held-out transition is strictly later than every training one."""
    traj = np.stack([np.arange(12.0), np.zeros(12)], axis=1)
    (train_obs, _), (test_obs, _) = X.split_prefix(
        [traj], np.random.default_rng(0), 0.25
    )
    assert len(train_obs) == 8
    assert len(test_obs) == 2
    assert train_obs[:, 0].max() < test_obs[:, 0].min()


def test_split_prefix_partitions_every_trajectory():
    trajs = _trajectories(n=3, length=12)
    (train_obs, _), (test_obs, _) = X.split_prefix(
        trajs, np.random.default_rng(0), 0.25
    )
    assert len(train_obs) == 3 * 8
    assert len(test_obs) == 3 * 2


def test_split_prefix_rejects_a_tail_that_yields_no_transition():
    with pytest.raises(ValueError, match="too short"):
        X.split_prefix([np.zeros((2, DOF))], np.random.default_rng(0), 0.1)
    with pytest.raises(ValueError, match="too short"):
        X.split_prefix([np.zeros((5, DOF))], np.random.default_rng(0), 0.9)


def test_split_prefix_rejects_a_fraction_outside_zero_one():
    with pytest.raises(ValueError, match="must be in"):
        X.split_prefix(_trajectories(n=1, length=6), np.random.default_rng(0), 1.0)


def test_splits_reject_empty_input():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="no trajectories"):
        X.split_transition([], rng)
    with pytest.raises(ValueError, match="no trajectories"):
        X.split_trajectory([], rng, fraction=0.5)


def test_split_transition_is_reproducible_from_the_seed():
    trajs = _trajectories()
    first = X.split_transition(trajs, np.random.default_rng(7))
    second = X.split_transition(trajs, np.random.default_rng(7))
    assert np.array_equal(first[0][0], second[0][0])
    assert np.array_equal(first[1][1], second[1][1])
