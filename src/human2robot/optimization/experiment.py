"""Shared machinery for the trajectory experiments.

The four experiment scripts each carried their own optimizer config, split
helpers, and metric aggregation, spelled slightly differently in each file.
Two of those copies had already drifted. `compare_dexycb_synthetic.py` used
`step_size=0.5` while `validate_optimizer_split.py` selected `0.35`, so the
kinematic tables and the descent-rate table were produced under different
hyperparameters while the documentation described one setting.

Everything here is the single definition. The scripts import it.

Two conventions worth knowing before changing anything:

- `CONTROL_DT` is the Allegro control period. DexYCB captures at 30 Hz, so
  every trajectory is resampled to `CONTROL_DT` before it is optimized, and
  metrics are computed at `CONTROL_DT`. Comparing a resampled arm against a
  non-resampled arm confounds the optimizer with the resampling step, which is
  exactly the mistake that produced a 66.5 percent result in this project.
- `noise_scale` is 0. Independent per-timestep noise raises jerk faster than
  the tracking pull lowers it, so the first candidate of a noisy search trips
  the cost-doubling abort guard and the search returns at iteration 1. This is
  a structural argument rather than a tuned value.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from human2robot.cpp_bindings import OptimizerConfig
from human2robot.data.limits import (
    ACTUATOR_LOWER,
    ACTUATOR_UPPER,
    CAPTURE_DT,
    CONTROL_DT,
    MAX_ACCELERATION,
    MAX_VELOCITY,
)

__all__ = [
    "CAPTURE_DT",
    "CONTROL_DT",
    "DEFAULT_STEP_SIZE",
    "HOLDOUT_FRACTION",
    "PREFIX_FRACTION",
    "METRIC_KEYS",
    "NOISE_SCALE",
    "aggregate_metrics",
    "make_optimizer_config",
    "pairwise_reductions",
    "split_prefix",
    "split_transition",
    "split_trajectory",
    "to_transitions",
]

#: Selected by `scripts/validate_optimizer_split.py` from the first 50
#: subject-01 sequences alone, then confirmed on the last 50 and on all 100
#: subject-02 sequences. Every experiment uses this one value.
DEFAULT_STEP_SIZE = 0.35

#: See the module docstring. 0 is a structural fix, not a tuning choice.
NOISE_SCALE = 0.0

#: Holdout fraction shared by all three splits. The prefix split is scored
#: against a trivial baseline computed on the same slice, so the fraction is
#: needed outside the split itself.
HOLDOUT_FRACTION = 0.1

#: Alias for `HOLDOUT_FRACTION` at the prefix split's cut, so a caller scoring
#: the held-out tail does not have to restate the number.
PREFIX_FRACTION = HOLDOUT_FRACTION

#: The metrics every kinematic table reports.
METRIC_KEYS = ("max_velocity", "max_acceleration", "max_jerk", "smoothness")


def make_optimizer_config(
    step_size: float = DEFAULT_STEP_SIZE,
    noise_scale: float = NOISE_SCALE,
    max_iterations: int = 300,
    seed: int = 0,
) -> OptimizerConfig:
    """The one optimizer configuration the experiments share.

    Weighting is deliberately light on velocity and acceleration relative to
    tracking, and light on jerk again, because the cost is dominated by
    smoothness as soon as those weights rise. `limits_weight` is what actually
    keeps the trajectory inside the actuator bounds.
    """
    return OptimizerConfig(
        dof=len(ACTUATOR_LOWER),
        lower=ACTUATOR_LOWER,
        upper=ACTUATOR_UPPER,
        max_velocity=MAX_VELOCITY,
        max_acceleration=MAX_ACCELERATION,
        tracking=1.0,
        velocity=0.05,
        acceleration=0.05,
        jerk=0.02,
        collision=0.0,
        limits_weight=10.0,
        max_iterations=max_iterations,
        convergence_tolerance=1e-4,
        step_size=step_size,
        noise_scale=noise_scale,
        seed=seed,
    )


def aggregate_metrics(rows: Sequence[dict]) -> dict:
    """Mean, standard deviation and maximum of each metric over a set."""
    out: dict[str, dict[str, float]] = {}
    for key in METRIC_KEYS:
        values = np.array([row[key] for row in rows], dtype=float)
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "max": float(values.max()),
        }
    return out


def pairwise_reductions(before: dict, after: dict) -> dict[str, float]:
    """Percentage change of each metric's mean, positive when it improved.

    A baseline of zero has no headroom to take a percentage of, so it raises
    rather than returning a number. The alternative, clamping the denominator,
    reports an unchanging metric as a 100 percent improvement.
    """
    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        start = before[key]["mean"]
        end = after[key]["mean"]
        if start <= 0:
            raise ValueError(
                f"{key} has non-positive baseline {start}: a percentage "
                f"reduction is undefined and the input is degenerate"
            )
        out[f"{key}_reduction_pct"] = 100.0 * (1.0 - end / start)
    return out


def to_transitions(
    trajectories: Sequence[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Adjacent configurations as (q_t, q_{t+1} - q_t) regression pairs.

    A trajectory of fewer than two configurations has no adjacent pair and is
    dropped. Keeping it would contribute a pair of zero-row arrays, which is
    truthy, so `_pack` would silently concatenate it into an empty data set
    instead of raising.
    """
    return [(t[:-1], t[1:] - t[:-1]) for t in trajectories if len(t) >= 2]


def _pack(trajectories: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pairs = to_transitions(trajectories)
    if not pairs:
        raise ValueError("no trajectories to pack")
    return (
        np.concatenate([obs for obs, _ in pairs]),
        np.concatenate([act for _, act in pairs]),
    )


def split_transition(
    trajectories: Sequence[np.ndarray],
    rng: np.random.Generator,
    fraction: float = HOLDOUT_FRACTION,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Random holdout of pooled transitions.


    Adjacent transitions are near-duplicates, so a holdout item sits one step
    from a training item. This is the optimistic split and the one the published
    tables use; `split_trajectory` is the honest one.
    """
    obs, act = _pack(trajectories)
    order = rng.permutation(len(obs))
    n_holdout = max(1, int(fraction * len(obs)))
    return (obs[order[n_holdout:]], act[order[n_holdout:]]), (
        obs[order[:n_holdout]],
        act[order[:n_holdout]],
    )


def split_trajectory(
    trajectories: Sequence[np.ndarray],
    rng: np.random.Generator,
    fraction: float = HOLDOUT_FRACTION,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Hold out whole trajectories, so nothing adjacent can leak across."""
    order = rng.permutation(len(trajectories))
    n_test = max(1, int(fraction * len(trajectories)))
    test = [trajectories[i] for i in order[:n_test]]
    train = [trajectories[i] for i in order[n_test:]]
    return _pack(train), _pack(test)


def split_prefix(
    trajectories: Sequence[np.ndarray],
    rng: np.random.Generator,
    fraction: float = HOLDOUT_FRACTION,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Hold out the tail of every trajectory: a temporal extrapolation.

    The strictest of the three, and on real retargeted motion it is the one
    whose controlled advantage reverses. The cause is measured, not guessed:
    real hand motion decelerates to a stop, so the held-out tail is nearly
    motionless and predicting "no movement" already scores close to the fitted
    model. The optimizer redistributes that deceleration, which raises the
    tail's motion without improving it, so the arm whose tail is quietest wins
    a test whose answer is close to zero. `scripts/diagnose_tail_extrapolation.py`
    measures the terminal speed and the trivial-baseline score; read this
    split's number only against that baseline.

    Each side has to leave at least one adjacent pair behind, so a trajectory
    needs two configurations on each side of the cut. A one-configuration tail
    would contribute no test transition at all and `_pack` would raise on the
    test set rather than splitting anything.
    """
    del rng
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction {fraction} must be in (0, 1)")
    train, test = [], []
    for traj in trajectories:
        cut = int((1.0 - fraction) * len(traj))
        if cut < 1 or len(traj) - cut < 2:
            raise ValueError(
                f"trajectory of length {len(traj)} is too short to split "
                f"at fraction {fraction}: need 2 held-out and 2 kept steps"
            )
        train.append(traj[:cut])
        test.append(traj[cut:])
    return _pack(train), _pack(test)
