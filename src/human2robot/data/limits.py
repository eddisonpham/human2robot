"""Canonical Allegro bounds, and the optimizer bounds derived from them.

Every retargeting and optimization path needs the same joint limits. They were
written out four times (`optimization/pipeline.py`, both DexYCB scripts, and
`compare_dexycb_synthetic.py`), each copy spelled differently, so they drifted
independently with no test noticing. They had already drifted from the robot.

**15 of the 16 finger entries were wrong**, in both bounds at once. The copied
values assumed each finger was `[0.196, 0.196, 0.196]` for its three proximal
joints and that the thumb lay in `[-0.8, 0]`. The model says otherwise: the
three fingers are `[-0.196, -0.174, -0.227]` to `[1.61, 1.709, 1.618]`, and the
thumb runs `[0.263, -0.105, -0.189, -0.162]` to `[1.396, 1.163, 1.644, 1.719]`.
The copied thumb upper bound was 0.0 against a real 1.719, so every trajectory
the optimizer produced was projected to hold three thumb joints at or below zero
while the real thumb needs them up to 1.72. Measured against the model, the
optimized trajectories violated the real limits by 0.75 radians.

The correct source is the MuJoCo model, which reads `actuator_ctrlrange` at
runtime (`envs/allegro.py`). This module is the MuJoCo-free mirror of that, so
retargeting can run where MuJoCo is not installed. `test_invariants.py` pins the
two together and will fail if they differ.

Coordinate layout of the 22-dimensional `q`:

- 6 base coordinates. DexYCB records hand pose, not arm pose, so nothing here
  fills them and they are left at zero. The bounds for those six are `[0, 0]`,
  which is not "unbounded": it *pins* the base to zero. That is deliberate, so
  a trajectory cannot drift its base to satisfy a cost term, but it means the
  optimizer cannot represent arm motion at all.
- 16 finger actuators: index, middle, ring, then thumb, four joints each.

`MAX_VELOCITY` and `MAX_ACCELERATION` are not robot properties. They are the
optimizer's smoothness limits chosen for `CONTROL_DT`, and `MAX_VELOCITY` is
what makes the "max velocity" row saturate in every published table.
"""

from __future__ import annotations

import numpy as np

#: Number of actuators in the demo and action schema.
DOF = 22

#: Leading coordinates that are base motion rather than finger actuators.
BASE_DOF = 6

#: Number of finger actuators.
FINGER_DOF = DOF - BASE_DOF

#: DexYCB captures at 30 Hz; the Allegro control period is 20 ms.
CAPTURE_DT = 1.0 / 30.0

#: The Allegro control period, and the period every trajectory is resampled to.
CONTROL_DT = 0.02

# Per-finger joint bounds, copied from the MuJoCo model's actuator_ctrlrange
# and verified against it by test_invariants.py.
_FINGER_LOWER_BLOCK = np.array([-0.47, -0.196, -0.174, -0.227], dtype=np.float64)
_FINGER_UPPER_BLOCK = np.array([0.47, 1.61, 1.709, 1.618], dtype=np.float64)
_THUMB_LOWER = np.array([0.263, -0.105, -0.189, -0.162], dtype=np.float64)
_THUMB_UPPER = np.array([1.396, 1.163, 1.644, 1.719], dtype=np.float64)

#: Finger actuator bounds, index then middle then ring then thumb.
FINGER_LOWER = np.concatenate([_FINGER_LOWER_BLOCK] * 3 + [_THUMB_LOWER])
FINGER_UPPER = np.concatenate([_FINGER_UPPER_BLOCK] * 3 + [_THUMB_UPPER])

#: Base coordinate bounds. Zero width, which pins the base rather than freeing
#: it; see the module docstring.
_BASE_LOWER = np.zeros(BASE_DOF, dtype=np.float64)
_BASE_UPPER = np.zeros(BASE_DOF, dtype=np.float64)

#: All 22 actuator bounds.
ACTUATOR_LOWER = np.concatenate([_BASE_LOWER, FINGER_LOWER])
ACTUATOR_UPPER = np.concatenate([_BASE_UPPER, FINGER_UPPER])

#: Per-actuator speed limit used by the optimizer. Saturates in every published
#: result, so "max velocity" measures constraint enforcement, not headroom.
MAX_VELOCITY = np.full(DOF, 2.0, dtype=np.float64)

#: Per-actuator acceleration limit used by the optimizer. Tuning value chosen
#: for `CONTROL_DT`, not a robot property.
MAX_ACCELERATION = np.full(DOF, 20.0, dtype=np.float64)


#: Column order of `dex-retargeting`'s Allegro output, expressed as the model
#:
#: actuator index each column should be moved to. DexPilot emits [ff, th, mf,
#: rf]; the MuJoCo model is [ff, mf, rf, th]. Applying the two orders as if
#: they agreed drives the thumb with the middle finger's motion and vice
#: versa, and projects each joint against the wrong limit. Identified from the
#: data: the demo column ranges match the permuted joints to within 0.001 rad,
#: which no other assignment does.
DEXRETARGET_TO_MODEL_FINGER = (0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7)

#: The inverse mapping: model actuator index to dex-retargeting column.
_inverse_order = np.argsort(np.asarray(DEXRETARGET_TO_MODEL_FINGER))
MODEL_TO_DEXRETARGET_FINGER = tuple(int(_inverse_order[i]) for i in range(FINGER_DOF))


def fingers_from_dexretarget(dex_fingers: np.ndarray) -> np.ndarray:
    """Reorder (T, 16) dex-retargeting fingers into model actuator order."""
    dex_fingers = np.asarray(dex_fingers)
    if dex_fingers.ndim != 2 or dex_fingers.shape[1] != FINGER_DOF:
        raise ValueError(f"angles shape {dex_fingers.shape} != (T, {FINGER_DOF})")
    return dex_fingers[:, list(DEXRETARGET_TO_MODEL_FINGER)]


def finger_bounds() -> tuple[np.ndarray, np.ndarray]:
    """The 16 finger actuator bounds, with the base coordinates dropped."""
    return FINGER_LOWER.copy(), FINGER_UPPER.copy()


def normalized_targets_to_finger_angles(targets: np.ndarray) -> np.ndarray:
    """Map finger targets in [-1, 1] of shape (T, 16) to actuator angles."""
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != FINGER_DOF:
        raise ValueError(f"targets shape {targets.shape} != (T, {FINGER_DOF})")
    center = 0.5 * (FINGER_LOWER + FINGER_UPPER)
    span = 0.5 * (FINGER_UPPER - FINGER_LOWER)
    return center + np.clip(targets, -1.0, 1.0) * span


def to_q22(finger_angles: np.ndarray) -> np.ndarray:
    """Embed (T, 16) finger angles into a (T, 22) configuration with zero base."""
    finger_angles = np.asarray(finger_angles, dtype=np.float64)
    if finger_angles.ndim != 2 or finger_angles.shape[1] != FINGER_DOF:
        raise ValueError(f"angles shape {finger_angles.shape} != (T, {FINGER_DOF})")
    q22 = np.zeros((len(finger_angles), DOF), dtype=np.float64)
    q22[:, BASE_DOF:] = finger_angles
    return q22


def clamp_finger_angles(finger_angles: np.ndarray) -> np.ndarray:
    """Project (T, 16) finger angles onto the model's actuator bounds."""
    return np.clip(
        np.asarray(finger_angles, dtype=np.float64), FINGER_LOWER, FINGER_UPPER
    )
