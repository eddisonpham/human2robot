"""Python wrapper around the h2r_cpp pybind11 module.

Keeps the C++ build artifact discovery in one place so the rest of the
package never touches the raw module directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_BUILD_DIRS = (
    Path(__file__).resolve().parents[3] / "cpp" / "build",
    Path(__file__).resolve().parents[2] / "cpp" / "build",
)


def _load_module():
    for candidate in _BUILD_DIRS:
        if candidate.is_dir():
            for pyd in sorted(candidate.glob("h2r_cpp*.pyd")):
                import importlib.util

                spec = importlib.util.spec_from_file_location("h2r_cpp", pyd)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules["h2r_cpp"] = module
                spec.loader.exec_module(module)
                return module
    raise ImportError(
        "h2r_cpp bindings not found. Build them with: "
        "cmake -S cpp -B cpp/build && cmake --build cpp/build -j 16"
    )


_mod = _load_module()


class OptimizerConfig:
    """Typed facade over the C++ optimizer config.

    The bound arrays are what the optimizer actually enforces, so a `dof` that
    disagrees with their length would be recorded here and silently ignored
    there. Lengths are checked at construction.
    """

    def __init__(
        self,
        dof: int,
        lower: np.ndarray,
        upper: np.ndarray,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        tracking: float = 1.0,
        velocity: float = 0.1,
        acceleration: float = 0.1,
        jerk: float = 0.05,
        collision: float = 0.0,
        limits_weight: float = 10.0,
        max_iterations: int = 300,
        convergence_tolerance: float = 1e-4,
        step_size: float = 0.05,
        noise_scale: float = 0.1,
        seed: int = 0,
    ) -> None:
        arrays = {
            "lower": lower,
            "upper": upper,
            "max_velocity": max_velocity,
            "max_acceleration": max_acceleration,
        }
        for name, values in arrays.items():
            if len(values) != dof:
                raise ValueError(f"{name} has length {len(values)}, expected dof={dof}")
        # Equal bounds are legal: the six base coordinates are pinned at zero,
        # not actuated. Only an inverted interval is an error.
        if np.any(np.asarray(lower) > np.asarray(upper)):
            raise ValueError("lower must not exceed upper")
        self._config = _mod.OptimizerConfig()
        self._config.limits.lower = np.asarray(lower, dtype=float)
        self._config.limits.upper = np.asarray(upper, dtype=float)
        self._config.max_velocity = np.asarray(max_velocity, dtype=float)
        self._config.max_acceleration = np.asarray(max_acceleration, dtype=float)
        self._config.weights.tracking = tracking
        self._config.weights.velocity = velocity
        self._config.weights.acceleration = acceleration
        self._config.weights.jerk = jerk
        self._config.weights.collision = collision
        self._config.weights.limits = limits_weight
        self._config.max_iterations = max_iterations
        self._config.convergence_tolerance = convergence_tolerance
        self._config.step_size = step_size
        self._config.noise_scale = noise_scale
        self._config.seed = seed
        self.dof = dof

    @property
    def raw(self):
        return self._config


class OptimizerResult:
    """Result of one C++ optimization call."""

    def __init__(self, raw) -> None:
        self.trajectory: np.ndarray = raw.trajectory.positions
        self.initial_cost: float = raw.initial_cost
        self.projected_initial_cost: float = raw.projected_initial_cost
        self.improvement_pct: float = raw.improvement_pct
        self.final_cost: float = raw.final_cost
        self.iterations: int = raw.iterations
        self.converged: bool = raw.converged


def optimize_trajectory(
    initial: np.ndarray,
    reference: np.ndarray,
    dt: float,
    config: OptimizerConfig,
) -> OptimizerResult:
    """Run the C++ optimizer on a (T, dof) trajectory array."""
    traj_in = _mod.trajectory_from_numpy(dt, np.asarray(initial, dtype=float))
    traj_ref = _mod.trajectory_from_numpy(dt, np.asarray(reference, dtype=float))
    optimizer = _mod.TrajectoryOptimizer(config.raw)
    return OptimizerResult(optimizer.optimize(traj_in, traj_ref))


def trajectory_to_numpy(trajectory) -> np.ndarray:
    return _mod.trajectory_to_numpy(trajectory)


def linear_resample(positions: np.ndarray, dt: float, target_dt: float) -> np.ndarray:
    traj = _mod.trajectory_from_numpy(dt, np.asarray(positions, dtype=float))
    return _mod.trajectory_to_numpy(_mod.linear_resample(traj, target_dt))


def cubic_resample(positions: np.ndarray, dt: float, target_dt: float) -> np.ndarray:
    traj = _mod.trajectory_from_numpy(dt, np.asarray(positions, dtype=float))
    return _mod.trajectory_to_numpy(_mod.cubic_resample(traj, target_dt))


def moving_average_smooth(positions: np.ndarray, dt: float, window: int) -> np.ndarray:
    traj = _mod.trajectory_from_numpy(dt, np.asarray(positions, dtype=float))
    return _mod.trajectory_to_numpy(_mod.moving_average_smooth(traj, window))


def project_all(
    positions: np.ndarray,
    dt: float,
    lower: np.ndarray,
    upper: np.ndarray,
    max_velocity: np.ndarray,
    max_acceleration: np.ndarray,
) -> np.ndarray:
    traj = _mod.trajectory_from_numpy(dt, np.asarray(positions, dtype=float))
    limits = _mod.JointLimits()
    limits.lower = np.asarray(lower, dtype=float)
    limits.upper = np.asarray(upper, dtype=float)
    out = _mod.project_all(
        traj,
        limits,
        np.asarray(max_velocity, dtype=float),
        np.asarray(max_acceleration, dtype=float),
    )
    return _mod.trajectory_to_numpy(out)


def compute_metrics(
    positions: np.ndarray, dt: float, reference: np.ndarray | None = None
) -> dict:
    traj = _mod.trajectory_from_numpy(dt, np.asarray(positions, dtype=float))
    ref = None if reference is None else np.asarray(reference, dtype=float)
    return _mod.compute_metrics(traj, ref)


__all__ = [
    "OptimizerConfig",
    "OptimizerResult",
    "compute_metrics",
    "cubic_resample",
    "linear_resample",
    "moving_average_smooth",
    "optimize_trajectory",
    "project_all",
    "trajectory_to_numpy",
]
