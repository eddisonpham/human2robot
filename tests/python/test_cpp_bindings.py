"""Integration tests for the h2r_cpp pybind11 bindings."""

import numpy as np
import pytest

from human2robot.cpp_bindings import (
    OptimizerConfig,
    compute_metrics,
    cubic_resample,
    linear_resample,
    moving_average_smooth,
    optimize_trajectory,
    project_all,
)


@pytest.fixture(name="limits")
def limits_fixture():
    lower = np.full(2, -1.0)
    upper = np.full(2, 1.0)
    max_vel = np.full(2, 2.0)
    max_acc = np.full(2, 20.0)
    return lower, upper, max_vel, max_acc


def make_demo(T=60, dof=2, noise=0.15, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(T) * 0.01
    clean = np.stack([np.sin(0.5 * t + j) for j in range(dof)], axis=1)
    noisy = clean + rng.normal(0, noise, clean.shape)
    return noisy, clean


def test_numpy_roundtrip_matches():
    from human2robot import cpp_bindings

    arr, _ = make_demo()
    traj = cpp_bindings._mod.trajectory_from_numpy(0.01, arr)
    back = cpp_bindings.trajectory_to_numpy(traj)
    np.testing.assert_allclose(back, arr, atol=1e-15)


def test_resample_matches_python_reference():
    positions = np.linspace(0.0, 1.0, 21)[:, None]
    out = linear_resample(positions, 0.01, 0.02)
    expected = positions[::2]
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_cubic_resample_passes_through_knots():
    positions, _ = make_demo(T=40, noise=0.0)
    out = cubic_resample(positions, 0.01, 0.01)
    np.testing.assert_allclose(out, positions, atol=1e-8)


def test_smoothing_reduces_jerk():
    positions, _ = make_demo(T=80, noise=0.1)
    before = compute_metrics(positions, 0.01)["max_jerk"]
    smoothed = moving_average_smooth(positions, 0.01, 7)
    after = compute_metrics(smoothed, 0.01)["max_jerk"]
    assert after < before


def test_projection_enforces_limits(limits):
    lower, upper, max_vel, max_acc = limits
    positions, _ = make_demo(T=50, noise=1.5)
    out = project_all(positions, 0.01, lower, upper, max_vel, max_acc)
    assert (out <= upper + 1e-9).all()
    assert (out >= lower - 1e-9).all()


def test_optimizer_reduces_cost_and_is_deterministic(limits):
    lower, upper, max_vel, max_acc = limits
    initial, reference = make_demo()
    config = OptimizerConfig(
        dof=2,
        lower=lower,
        upper=upper,
        max_velocity=max_vel,
        max_acceleration=max_acc,
        seed=0,
    )
    first = optimize_trajectory(initial, reference, 0.01, config)
    second = optimize_trajectory(initial, reference, 0.01, config)
    assert first.final_cost < first.initial_cost
    assert first.final_cost == second.final_cost
    np.testing.assert_array_equal(
        np.asarray(first.trajectory), np.asarray(second.trajectory)
    )


def test_optimizer_seed_change_still_valid_and_comparable(limits):
    lower, upper, max_vel, max_acc = limits
    initial, reference = make_demo()
    config_a = OptimizerConfig(
        dof=2,
        lower=lower,
        upper=upper,
        max_velocity=max_vel,
        max_acceleration=max_acc,
        seed=0,
    )
    config_b = OptimizerConfig(
        dof=2,
        lower=lower,
        upper=upper,
        max_velocity=max_vel,
        max_acceleration=max_acc,
        seed=1,
    )
    result_a = optimize_trajectory(initial, reference, 0.01, config_a)
    result_b = optimize_trajectory(initial, reference, 0.01, config_b)
    assert result_a.final_cost < result_a.initial_cost
    assert result_b.final_cost < result_b.initial_cost
    rel = abs(result_a.final_cost - result_b.final_cost) / max(
        result_a.final_cost, 1e-12
    )
    assert rel < 0.5


def test_metrics_consistent_with_numpy():
    positions, _ = make_demo(T=100, noise=0.0)
    metrics = compute_metrics(positions, 0.01)
    vel = np.diff(positions, axis=0) / 0.01
    assert metrics["max_velocity"] == pytest.approx(np.abs(vel).max(), rel=0.2)
    assert metrics["max_jerk"] >= 0.0
