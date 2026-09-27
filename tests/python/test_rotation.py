"""Regression tests for the rotation-matrix to axis-angle conversion.

The previous implementation recovered the rotation axis by dividing the skew
part of the matrix by ``2 * sin(angle)``. That is singular at 180 degrees, and
a free box resting on the floor reaches that angle routinely, so observations
contained values of order 1e8 and every SAC critic target was corrupted.
"""

import numpy as np
import pytest

from human2robot.utils.rotation import _matrix_to_quaternion, rotation_vector


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Build a rotation matrix from a unit axis and an angle."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Build a rotation matrix from a ``(w, x, y, z)`` quaternion."""
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _skew_division(matrix: np.ndarray) -> np.ndarray:
    """The original singular implementation, kept as the regression reference."""
    trace = float(np.trace(matrix))
    cosine = float(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-7:
        return np.zeros(3)
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    axis /= 2.0 * np.sin(angle)
    return axis * angle


@pytest.mark.parametrize(
    "angle",
    [1e-8, 1e-4, 0.1, 1.0, 3.0, np.pi - 1e-3, np.pi - 1e-6, np.pi],
)
def test_norm_equals_angle(angle: float) -> None:
    """The magnitude of the axis-angle vector is the rotation angle."""
    matrix = _rotation_matrix(np.array([0.3, 0.5, 0.81]), angle)
    assert np.linalg.norm(rotation_vector(matrix)) == pytest.approx(angle, abs=1e-6)


@pytest.mark.parametrize("offset", [0.0, 1e-9, 1e-7, 1e-5, 1e-3])
def test_bounded_near_half_turn(offset: float) -> None:
    """Values stay bounded by pi where the old division diverged."""
    matrix = _rotation_matrix(np.array([0.3, 0.5, 0.81]), np.pi - offset)
    assert np.linalg.norm(rotation_vector(matrix)) <= np.pi + 1e-9


def test_old_implementation_diverges_at_half_turn() -> None:
    """Pin the original defect so the regression cannot silently return."""
    matrix = _rotation_matrix(np.array([0.3, 0.5, 0.81]), np.pi - 1e-9)
    assert np.linalg.norm(_skew_division(matrix)) > 1e6


def test_identity_and_small_angles_return_zero() -> None:
    """Degenerate angles map to a zero vector rather than dividing by zero."""
    assert np.allclose(rotation_vector(np.eye(3)), 0.0)
    tiny = _rotation_matrix(np.array([0.0, 0.0, 1.0]), 1e-12)
    assert np.allclose(rotation_vector(tiny), 0.0, atol=1e-6)


def test_round_trips_through_quaternion() -> None:
    """The axis-angle vector recovers the original rotation matrix."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        quaternion = rng.normal(size=4)
        matrix = _quaternion_matrix(quaternion)
        vector = rotation_vector(matrix)
        recovered = _rotation_matrix(vector, float(np.linalg.norm(vector)))
        assert np.allclose(recovered, matrix, atol=1e-8)


def test_round_trips_for_negative_scalar_quaternions() -> None:
    """q and -q are the same rotation, so the axis must not flip sign."""
    rng = np.random.default_rng(3)
    checked = 0
    while checked < 200:
        quaternion = rng.normal(size=4)
        if quaternion[0] >= 0.0:
            continue
        matrix = _quaternion_matrix(quaternion)
        vector = rotation_vector(matrix)
        recovered = _rotation_matrix(vector, float(np.linalg.norm(vector)))
        assert np.allclose(recovered, matrix, atol=1e-8)
        checked += 1


def test_random_rotations_stay_bounded() -> None:
    """No random rotation produces an unbounded or non-finite result."""
    rng = np.random.default_rng(1)
    for _ in range(5000):
        matrix = _quaternion_matrix(rng.normal(size=4))
        vector = rotation_vector(matrix)
        assert np.isfinite(vector).all()
        assert np.linalg.norm(vector) <= np.pi + 1e-9


def test_quaternion_is_normalized() -> None:
    """The intermediate quaternion is a unit quaternion on every branch."""
    rng = np.random.default_rng(2)
    for _ in range(1000):
        matrix = _quaternion_matrix(rng.normal(size=4))
        quaternion = _matrix_to_quaternion(matrix)
        assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1e-9)
        assert quaternion[0] >= 0.0


@pytest.mark.parametrize(
    ("quaternion", "expected"),
    [
        ([0.0, 1.0, 0.0, 0.0], np.pi),
        ([0.0, 0.0, 1.0, 0.0], np.pi),
        ([0.0, 0.0, 0.0, 1.0], np.pi),
        ([0.7071067811865476, 0.7071067811865476, 0.0, 0.0], np.pi / 2),
    ],
)
def test_axis_aligned_rotations(quaternion: list[float], expected: float) -> None:
    """Axis-aligned rotations, including the degenerate 180 degree case."""
    matrix = _quaternion_matrix(np.array(quaternion))
    assert np.linalg.norm(rotation_vector(matrix)) == pytest.approx(expected, abs=1e-6)
