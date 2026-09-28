"""Feasibility of a trajectory in the simulated Allegro hand.

Every other quality number in this project measures how *smooth* or how
*learnable* a trajectory is. Neither answers the question the project is
actually about: can the robot perform this motion? A trajectory can be smooth,
respect every joint limit, and be trivially predictable, and still be
impossible for the hand to execute, because the hand has inertia, the fingers
couple, and the actuators cannot reach an arbitrary target inside one control
period.

This closes that gap. A trajectory is scored by replaying it **open-loop** in
the simulator and measuring how closely the hand actually tracks it. The
trajectory is fed in as actuator targets, one per control period, and the drift
between target and achieved configuration is the result. Nothing is corrected
during replay, so a trajectory that is feasible here is one the hand genuinely
performs.

Three quantities are reported, and they answer different questions:

- `tracking_drift_max`, in radians. The worst single-frame gap between the
  target and what the hand reached. The headline number: how far off does the
  hand end up at its worst moment.
- `tracking_drift_rms`, in radians. Average tracking error. Distinguishes a
  trajectory that is uniformly slightly loose from one that fails in a burst.
- `sustained_limit_violation`, in radians. How far outside the actuator bounds
  the trajectory asks the hand to go, summed over time. A trajectory can be
  feasible to track and still demand an impossible configuration.

`is_feasible` is the conjunction of all three against thresholds, and the
thresholds are arguments rather than constants, because what counts as
acceptable depends on the task.

This replaces `data.allegro_demos.validate_open_loop_replay`, which was named
for a simulator replay but checked `a_demo` arithmetic against stub bounds of
`zeros(22)` and `ones(22)` and never touched MuJoCo. It was removed rather than
kept, so there is one definition of executable and it is a real one. See
`docs/FINDINGS_retargeting.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from human2robot.data.limits import (
    ACTUATOR_LOWER,
    ACTUATOR_UPPER,
    BASE_DOF,
    DOF,
    FINGER_DOF,
)

__all__ = [
    "FeasibilityReport",
    "DEFAULT_MAX_DRIFT",
    "DEFAULT_MAX_RMS_DRIFT",
    "DEFAULT_MAX_LIMIT_VIOLATION",
    "score_trajectory",
    "score_demo_directory",
]

#: Worst-case tracking error, radians. Roughly 10 degrees at the fingertip.
#: Set above the hand's ~0.06 rad steady-state floor so a well-behaved
#: trajectory passes, but well below the ~1.1 rad a genuinely unreachable
#: target produces.
DEFAULT_MAX_DRIFT = 0.35

#: Root-mean-square tracking error, radians.
DEFAULT_MAX_RMS_DRIFT = 0.20

#: Total out-of-bounds demand, radians. Zero is exact compliance.
DEFAULT_MAX_LIMIT_VIOLATION = 0.0

#: Frames to hold the first target before measuring. See `score_trajectory`.
DEFAULT_SETTLE_STEPS = 25

#: Below this an out-of-bounds value is rounding noise from resampling and
#: smoothing, not a demand the robot cannot meet. 0.01 rad is 0.57 degrees.
DEFAULT_LIMIT_TOLERANCE = 0.01


@dataclass(frozen=True)
class FeasibilityReport:
    """The result of replaying one trajectory in the simulator."""

    trajectory_id: str
    horizon: int
    tracking_drift_max: float
    tracking_drift_rms: float
    sustained_limit_violation: float
    diverged: bool
    is_feasible: bool
    out_of_bounds_frames: int = 0


def _finger_target_action(targets: np.ndarray) -> np.ndarray:
    """Map desired finger angles onto the normalized action the env expects.

    The environment maps `action[6:]` from [-1, 1] onto the actuator
    `ctrlrange` linearly, so this is the inverse of that mapping. Clipping
    happens on both sides deliberately: a target outside the range is
    unreachable and must not be silently scaled into range, or the feasibility
    number would hide the very thing it is measuring.
    """
    action = np.zeros(DOF, dtype=np.float32)
    low = ACTUATOR_LOWER[BASE_DOF:]
    high = ACTUATOR_UPPER[BASE_DOF:]
    span = np.maximum(high - low, 1e-6)
    action[BASE_DOF:] = np.clip(2.0 * (targets - low) / span - 1.0, -1.0, 1.0)
    return action


def score_trajectory(
    q: np.ndarray,
    env,
    trajectory_id: str = "trajectory",
    max_drift: float = DEFAULT_MAX_DRIFT,
    max_rms_drift: float = DEFAULT_MAX_RMS_DRIFT,
    max_limit_violation: float = DEFAULT_MAX_LIMIT_VIOLATION,
    settle_steps: int = DEFAULT_SETTLE_STEPS,
    limit_tolerance: float = DEFAULT_LIMIT_TOLERANCE,
) -> FeasibilityReport:
    """Replay a (T, 22) configuration sequence open-loop and score it.

    `env` must be an `AllegroPickupEnv`. It is passed in rather than created
    here so a caller can score many trajectories against one model, which
    matters because constructing the environment loads a MuJoCo model and
    dominates the cost otherwise.

    The hand starts at its reset configuration, so the first frames of a
    replay measure how far the hand was from the target rather than how well
    it follows the motion. `settle_steps` holds the first target for a while
    before measuring, so the reported drift is tracking error rather than
    initial displacement. Measured on a constant in-bounds target, the hand
    needs about 25 steps (0.5 s) to reach its ~0.06 rad floor; without this
    prelude a perfectly feasible trajectory reports 0.53 rad.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != DOF:
        raise ValueError(f"q shape {q.shape} != (T, {DOF})")
    if len(q) == 0:
        raise ValueError("empty trajectory")

    # A frame that is not finite cannot be scored, and treating it as an
    # in-bounds demand would report an impossible trajectory as feasible. It is
    # marked diverged with unbounded violation instead of raising, so one bad
    # sequence cannot abort a whole directory.
    finite = np.isfinite(q).all(axis=1)
    finger = np.where(finite[:, None], q[:, BASE_DOF:], 0.0)

    # Out-of-bounds demand is measured before the replay, because the
    # normalized action clamps and the simulator never sees the excess. Values
    # within `limit_tolerance` of the bound are rounding noise from resampling
    # and smoothing rather than a demand the robot cannot meet.
    below = np.maximum(ACTUATOR_LOWER[BASE_DOF:] - finger, 0.0)
    above = np.maximum(finger - ACTUATOR_UPPER[BASE_DOF:], 0.0)
    excess = below + above
    excess = np.where(excess > limit_tolerance, excess, 0.0)
    excess[~finite] = np.inf
    violation = float(excess.max())
    total_violation = float(excess.sum())
    out_of_bounds = int((excess.max(axis=1) > 0).sum())

    diverged = not bool(finite.all())
    drifts: list[float] = []
    if not diverged:
        env.reset(seed=0)
        first_action = _finger_target_action(q[0, BASE_DOF:])
        for _ in range(settle_steps):
            env.step(first_action)
        for frame in q:
            action = _finger_target_action(frame[BASE_DOF:])
            env.step(action)
            if not np.isfinite(env.data.qpos).all():
                diverged = True
                break
            achieved = env.data.qpos[env._finger_qpos][:FINGER_DOF]
            drifts.append(
                float(np.abs(achieved - frame[BASE_DOF : BASE_DOF + FINGER_DOF]).max())
            )

    if diverged or not drifts:
        return FeasibilityReport(
            trajectory_id=trajectory_id,
            horizon=len(q),
            tracking_drift_max=float("inf"),
            tracking_drift_rms=float("inf"),
            sustained_limit_violation=total_violation,
            diverged=True,
            is_feasible=False,
            out_of_bounds_frames=out_of_bounds,
        )

    drift = np.array(drifts, dtype=np.float64)
    drift_max = float(drift.max())
    drift_rms = float(np.sqrt(np.mean(drift**2)))
    feasible = (
        drift_max <= max_drift
        and drift_rms <= max_rms_drift
        and violation <= max_limit_violation
    )
    return FeasibilityReport(
        trajectory_id=trajectory_id,
        horizon=len(q),
        tracking_drift_max=drift_max,
        tracking_drift_rms=drift_rms,
        sustained_limit_violation=total_violation,
        diverged=False,
        is_feasible=bool(feasible),
        out_of_bounds_frames=out_of_bounds,
    )


def score_demo_directory(
    directory: str | Path,
    env,
    max_drift: float = DEFAULT_MAX_DRIFT,
    max_rms_drift: float = DEFAULT_MAX_RMS_DRIFT,
    max_limit_violation: float = DEFAULT_MAX_LIMIT_VIOLATION,
    settle_steps: int = DEFAULT_SETTLE_STEPS,
    limit_tolerance: float = DEFAULT_LIMIT_TOLERANCE,
    suffix: str = "",
) -> list[FeasibilityReport]:
    """Score demos in a directory against one simulator instance.

    `suffix` selects a variant, because an optimized directory holds the raw
    resampled arm and the optimized arm together. Callers comparing arms must
    pass `suffix="_opt"` or `suffix="_resampled"`; the default scores every
    `*.npz` present.
    """
    from human2robot.data.allegro_demos import load_demo_npz

    paths = sorted(Path(directory).glob(f"*{suffix}.npz"))
    if not paths:
        raise FileNotFoundError(f"no demos in {directory} matching *{suffix}.npz")
    reports = []
    for path in paths:
        demo = load_demo_npz(path)
        reports.append(
            score_trajectory(
                demo.q,
                env,
                trajectory_id=path.stem,
                max_drift=max_drift,
                max_rms_drift=max_rms_drift,
                max_limit_violation=max_limit_violation,
                settle_steps=settle_steps,
                limit_tolerance=limit_tolerance,
            )
        )
    return reports


def summarize(reports: list[FeasibilityReport]) -> dict[str, float]:
    """Aggregate a set of feasibility reports into one table row."""
    if not reports:
        raise ValueError("no reports to summarize")
    feasible = [r for r in reports if r.is_feasible]
    return {
        "count": len(reports),
        "feasible": len(feasible),
        "feasible_fraction": len(feasible) / len(reports),
        "tracking_drift_max_mean": float(
            np.mean([r.tracking_drift_max for r in reports])
        ),
        "tracking_drift_max_worst": float(
            np.max([r.tracking_drift_max for r in reports])
        ),
        "tracking_drift_rms_mean": float(
            np.mean([r.tracking_drift_rms for r in reports])
        ),
        "limit_violation_total": float(
            np.sum([r.sustained_limit_violation for r in reports])
        ),
        "out_of_bounds_frames": int(np.sum([r.out_of_bounds_frames for r in reports])),
        "diverged": int(sum(1 for r in reports if r.diverged)),
    }
