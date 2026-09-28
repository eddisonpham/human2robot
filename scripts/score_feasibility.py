"""How many of these trajectories can the robot actually execute?

Every other number in the conversion pipeline measures how smooth or how
learnable a trajectory is. None of them answers the question the pipeline
exists to answer: can the hand perform this motion? This script answers it by
replaying each trajectory open-loop in MuJoCo and measuring tracking drift, via
`human2robot.evaluation.feasibility`.

Three arms are scored per data set, and the middle one is the reason the table
is worth having:

- ``raw``: the demonstrations as retargeted, at their native capture rate
- ``resampled``: cubic-resampled to the 20 ms control period, no optimizer.
  This is the control that separates resampling from optimization.
- ``optimized``: resampled and optimized, which is what the pipeline ships.

Without the control arm the raw-to-optimized gap is uninterpretable, because
resampling alone accounts for most of it (see `docs/RESULTS.md`).

Usage:

    uv run python scripts/score_feasibility.py [--set all|dexycb|dexycb-s2|synthetic]
"""

import json
import sys
from pathlib import Path

from human2robot.evaluation.feasibility import (
    DEFAULT_LIMIT_TOLERANCE,
    DEFAULT_MAX_DRIFT,
    DEFAULT_MAX_LIMIT_VIOLATION,
    DEFAULT_MAX_RMS_DRIFT,
    DEFAULT_SETTLE_STEPS,
    score_demo_directory,
    summarize,
)

#: ``(raw_dir, optimized_dir, output_path)``. The optimized directory holds
#: both the ``_resampled`` control arm and the ``_opt`` arm when the pipeline
#: that produced it wrote the control; ``data/demonstrations_optimized`` does
#: not, because synthetic data is already at the control rate, so that arm is
#: reported as absent rather than fabricated.
SETS = {
    "dexycb": (
        Path("data/demonstrations_dexycb_ik"),
        Path("data/demonstrations_dexycb_ik_optimized"),
        Path("results/trajectory_optimization/feasibility_dexycb.json"),
    ),
    "dexycb-s2": (
        Path("data/demonstrations_dexycb_ik_s2"),
        Path("data/demonstrations_dexycb_ik_s2_optimized"),
        Path("results/trajectory_optimization/feasibility_dexycb_s2.json"),
    ),
    "synthetic": (
        Path("data/demonstrations"),
        Path("data/demonstrations_optimized"),
        Path("results/trajectory_optimization/feasibility_synthetic.json"),
    ),
}


def score_arm(directory: Path, env, suffix: str, thresholds: dict) -> dict:
    """Score one arm and record the thresholds it was scored against.

    The thresholds are stored beside the numbers because `is_feasible` is their
    conjunction: a bare count of feasible trajectories is meaningless without
    the cut that produced it.
    """
    if not sorted(directory.glob(f"*{suffix}.npz")):
        return {"present": False}
    reports = score_demo_directory(directory, env, suffix=suffix, **thresholds)
    summary = summarize(reports)
    summary["present"] = True
    return summary


def score_set(name: str, thresholds: dict) -> dict:
    """Score the three arms of one data set against a single simulator."""
    from human2robot.envs.allegro import AllegroPickupEnv

    raw_dir, opt_dir, _ = SETS[name]
    if not sorted(raw_dir.glob("*.npz")):
        raise FileNotFoundError(f"no demos in {raw_dir}")
    env = AllegroPickupEnv()
    try:
        arms = {
            "raw": score_arm(raw_dir, env, "", thresholds),
            "resampled": score_arm(opt_dir, env, "_resampled", thresholds),
            "optimized": score_arm(opt_dir, env, "_opt", thresholds),
        }
    finally:
        env.close()
    return {
        "set": name,
        "raw_dir": str(raw_dir),
        "optimized_dir": str(opt_dir),
        "thresholds": thresholds,
        "arms": arms,
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    name = "all"
    if "--set" in argv:
        name = argv[argv.index("--set") + 1]
    if name not in SETS and name != "all":
        print(
            f"unknown set {name!r}; choose from ['all', *{sorted(SETS)}]",
            file=sys.stderr,
        )
        return 1

    thresholds = {
        "max_drift": DEFAULT_MAX_DRIFT,
        "max_rms_drift": DEFAULT_MAX_RMS_DRIFT,
        "max_limit_violation": DEFAULT_MAX_LIMIT_VIOLATION,
        "settle_steps": DEFAULT_SETTLE_STEPS,
        "limit_tolerance": DEFAULT_LIMIT_TOLERANCE,
    }

    names = sorted(SETS) if name == "all" else [name]
    reports = {}
    for set_name in names:
        _, _, out_path = SETS[set_name]
        print(f"scoring {set_name}...")
        report = score_set(set_name, thresholds)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        reports[set_name] = report
        print(f"  written to {out_path}")

    print(
        f"\n{'set':<12}{'arm':<12}{'executable':>14}"
        f"{'drift mean':>12}{'drift worst':>13}{'limit viol':>12}"
    )
    for set_name, report in reports.items():
        for arm, summary in report["arms"].items():
            if not summary.get("present"):
                print(f"{set_name:<12}{arm:<12}{'no arm':>14}")
                continue
            print(
                f"{set_name:<12}{arm:<12}"
                f"{summary['feasible']:>7}/{summary['count']:<6}"
                f"{summary['tracking_drift_max_mean']:>12.3f}"
                f"{summary['tracking_drift_max_worst']:>13.3f}"
                f"{summary['limit_violation_total']:>12.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
