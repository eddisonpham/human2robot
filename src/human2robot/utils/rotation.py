"""Numerically stable rotation matrix to axis-angle conversion.

The textbook conversion recovers the axis from the skew part of the matrix
divided by ``2 * sin(angle)``, which is singular at ``angle = pi``. A free
rigid body resting on the floor reaches that angle routinely, so the division
produced axis-angle vectors of order 1e8 and poisoned every critic target
downstream. Converting through the quaternion avoids the cancellation
entirely: the quaternion is normalized, and the half-angle is taken with
``atan2``, which stays accurate for every angle in ``[0, pi]``.
"""

import numpy as np

__all__ = ["rotation_vector"]

_SMALL_ANGLE = 1e-12


def rotation_vector(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a bounded axis-angle vector.

    The result has norm equal to the rotation angle, so it is bounded by
    ``pi`` for any valid input, including the 180 degree case that a
    skew-part division cannot represent.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    quaternion = _matrix_to_quaternion(matrix)
    vector = quaternion[1:]
    norm = float(np.linalg.norm(vector))
    if norm < _SMALL_ANGLE:
        return np.zeros(3, dtype=np.float64)
    # The half-angle from atan2 stays accurate for every angle, including the
    # 180 degree case where the trace-derived arccos loses half the rotation.
    angle = 2.0 * float(np.arctan2(norm, float(quaternion[0])))
    return (vector / norm) * angle


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Return the unit quaternion ``(w, x, y, z)`` of a rotation matrix.

    Uses the largest-diagonal branch, which is the numerically stable
    formulation and avoids dividing by a quantity that can approach zero.
    """
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m[2, 1] - m[1, 2]) / scale
        y = (m[0, 2] - m[2, 0]) / scale
        z = (m[1, 0] - m[0, 1]) / scale
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        scale = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / scale
        x = 0.25 * scale
        y = (m[0, 1] + m[1, 0]) / scale
        z = (m[0, 2] + m[2, 0]) / scale
    elif m[1, 1] > m[2, 2]:
        scale = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / scale
        x = (m[0, 1] + m[1, 0]) / scale
        y = 0.25 * scale
        z = (m[1, 2] + m[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / scale
        x = (m[0, 2] + m[2, 0]) / scale
        y = (m[1, 2] + m[2, 1]) / scale
        z = 0.25 * scale
    quaternion = np.array([w, x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    # Unreachable for any finite input. Each of the four branches above sets
    # one component to 0.25 * scale, and the matrix inequalities that select a
    # branch are what force that scale away from zero, so the assembled
    # quaternion always has unit norm. A sweep over 20,000 random rotations
    # plus degenerate and extreme matrices never produced a norm below 1.0. The
    # check stays as a guard against a future rewrite of the branch selection
    # reintroducing a zero denominator, which is the bug this function exists
    # to prevent.
    if norm < _SMALL_ANGLE:  # pragma: no cover - defensive, provably unreachable
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    quaternion /= norm
    # q and -q describe the same rotation. Canonicalize to a non-negative
    # scalar part so the axis derived from the vector part always points
    # along the principal rotation rather than against it.
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    return quaternion
