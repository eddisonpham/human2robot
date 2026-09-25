"""Numerical equivalence and throughput benchmarks, C++ versus Python.

Spec section 19: no speedup claims without benchmarking; these tests print
measured timings and assert numerical consistency.
"""

import time

import numpy as np
import pytest

from human2robot.cpp_bindings import (
    OptimizerConfig,
    compute_metrics,
    moving_average_smooth,
    optimize_trajectory,
)


def py_moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    half = max(1, window // 2)
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        out[i] = arr[lo:hi].mean(axis=0)
    return out


def py_metrics(positions: np.ndarray, dt: float) -> dict:
    n = len(positions)
    vel = np.zeros_like(positions)
    vel[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    vel[0] = (positions[1] - positions[0]) / dt
    vel[-1] = (positions[-1] - positions[-2]) / dt
    acc = np.zeros_like(positions)
    if n >= 3:
        acc[1:-1] = (positions[2:] - 2 * positions[1:-1] + positions[:-2]) / dt**2
    jerk = np.zeros_like(positions)
    if n >= 4:
        jerk[1:-1] = (acc[2:] - acc[:-2]) / (2 * dt)
    return {
        "max_velocity": float(np.abs(vel).max()),
        "max_acceleration": float(np.abs(acc).max()),
        "max_jerk": float(np.abs(jerk).max()),
    }


@pytest.fixture(name="trajectory")
def trajectory_fixture():
    rng = np.random.default_rng(0)
    t = np.arange(200) * 0.01
    clean = np.stack([np.sin(0.5 * t + j) for j in range(22)], axis=1)
    return clean + rng.normal(0, 0.05, clean.shape)


def test_metrics_equivalence_within_tolerance(trajectory):
    cpp = compute_metrics(trajectory, 0.01)
    py = py_metrics(trajectory, 0.01)
    assert cpp["max_velocity"] == pytest.approx(py["max_velocity"], rel=0.05)
    assert cpp["max_acceleration"] == pytest.approx(py["max_acceleration"], rel=0.05)
    assert cpp["max_jerk"] == pytest.approx(py["max_jerk"], rel=0.05)


def test_smoothing_equivalence(trajectory):
    cpp = moving_average_smooth(trajectory, 0.01, 7)
    py = py_moving_average(trajectory, 7)
    np.testing.assert_allclose(cpp, py, atol=1e-12)


def test_benchmark_optimizer_throughput(trajectory):
    lower = np.full(22, -1.5)
    upper = np.full(22, 1.5)
    config = OptimizerConfig(
        dof=22,
        lower=lower,
        upper=upper,
        max_velocity=np.full(22, 2.0),
        max_acceleration=np.full(22, 20.0),
        max_iterations=300,
        seed=0,
    )
    started = time.perf_counter()
    result = optimize_trajectory(trajectory, trajectory, 0.01, config)
    elapsed = time.perf_counter() - started
    assert result.iterations > 0
    print(
        f"\n[benchmark] optimizer: {elapsed * 1000:.1f} ms for "
        f"{result.iterations} iterations on T=200 dof=22 "
        f"({elapsed / max(result.iterations, 1) * 1000:.2f} ms/iter)"
    )
    assert elapsed < 30.0


def test_benchmark_smoothing_speedup(trajectory):
    config = {"window": 7}
    started = time.perf_counter()
    for _ in range(20):
        moving_average_smooth(trajectory, 0.01, config["window"])
    cpp_elapsed = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(20):
        py_moving_average(trajectory, config["window"])
    py_elapsed = time.perf_counter() - started
    print(
        f"\n[benchmark] smoothing: C++ {cpp_elapsed * 1000:.1f} ms vs "
        f"Python {py_elapsed * 1000:.1f} ms for 20 runs "
        f"(speedup {py_elapsed / max(cpp_elapsed, 1e-9):.1f}x)"
    )
