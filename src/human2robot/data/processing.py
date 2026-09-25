"""Trajectory smoothing and differentiation (agents/08 §2)."""

import numpy as np


def smooth(values: np.ndarray, window: int = 7) -> np.ndarray:
    """Hanning-window smoothing along axis 0."""
    values = np.asarray(values, dtype=float)
    if window <= 2 or len(values) < window:
        return values
    pad = window // 2
    kernel = np.hanning(window)
    kernel /= kernel.sum()
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    return np.array(
        [
            np.convolve(padded[:, col], kernel, mode="valid")
            for col in range(values.shape[1])
        ]
    ).T[: len(values)]


def differentiate(
    positions: np.ndarray, dt: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference velocity and acceleration."""
    positions = np.asarray(positions, dtype=float)
    vel = np.zeros_like(positions)
    acc = np.zeros_like(positions)
    vel[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    vel[0] = (positions[1] - positions[0]) / dt
    vel[-1] = (positions[-1] - positions[-2]) / dt
    acc[1:-1] = (positions[2:] - 2 * positions[1:-1] + positions[:-2]) / (dt**2)
    return vel, acc


def demo_actions(
    joint_positions: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """Normalized delta actions in [-1,1] for position actuators."""
    q = np.asarray(joint_positions, dtype=float)
    deltas = np.diff(q, axis=0, prepend=q[:1])
    scale = (high - low) / 2.0
    return np.clip(deltas / np.maximum(scale, 1e-6), -1.0, 1.0)
