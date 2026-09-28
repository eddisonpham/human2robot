"""Does the BC holdout's split granularity change the optimizer's advantage?

The published BC comparison holds out a random 10 percent of pooled
transitions. Consecutive transitions of one trajectory are nearly identical, so
a holdout transition sits one step away from a training transition and the
absolute error understates how hard generalization is. This script re-runs the
comparison with three split granularities so the claim can be stated at the
level it actually supports:

1. transition: random 10 percent of pooled transitions (what is published)
2. trajectory: whole trajectories held out, so nothing adjacent leaks
3. prefix:   the last 10 percent of every trajectory, the strictest option,
             since it is a temporal extrapolation rather than interpolation

Each granularity is run over several seeds, because a single split produces a
number with no error bar and the granularity finding is precisely a comparison
between splits whose sizes differ. Reporting one seed made the spread
invisible; the whole-trajectory and prefix splits differ from the transition
split by tens of points, which is larger than seed-to-seed noise but not
obviously so from a single draw.

The reported advantage is always against the resampled control arm where one
exists, so that the number isolates the optimizer from the resampling step. The
synthetic set is already at the control rate and has no control arm, so there
the comparison is against raw.

Usage:

    uv run python scripts/check_bc_split_granularity.py [--set dexycb] [--seeds 5]
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_downstream_bc import (  # noqa: E402
    DEMO_SETS,
    evaluate,
    load_positions,
    train_bc,
)

# The three splits are defined once in `optimization.experiment`. They used to
# be copied here with their own spelling, including a hardcoded 22 for the
# degree-of-freedom count and a prefix cut that silently produced an empty
# holdout on a short trajectory. This script produced the published granularity
# finding, so its splits have to be the tested ones rather than private
# duplicates that can drift away from them.
from human2robot.optimization.experiment import (  # noqa: E402
    PREFIX_FRACTION,
    split_prefix,
    split_trajectory,
    split_transition,
)

#: The holdout fraction each split uses, needed to reproduce the prefix split's
#: tail for the trivial baseline below.
SPLIT_FRACTIONS = {"transition": 0.1, "trajectory": 0.1, "prefix": PREFIX_FRACTION}

SPLITS = {
    "transition": split_transition,
    "trajectory": split_trajectory,
    "prefix": split_prefix,
}

#: Reported when the same script produced a single number per granularity. The
#: spread is what makes the prefix result interpretable.
DEFAULT_SEEDS = 5


def arms_for(name: str) -> dict[str, list[np.ndarray]]:
    raw_dir, opt_dir, _ = DEMO_SETS[name]
    raw = load_positions(raw_dir)
    optimized = load_positions(opt_dir, suffix="_opt")
    out = {"raw": raw, "optimized": optimized}
    control = sorted(opt_dir.glob("*_resampled.npz"))
    if control:
        out["resampled_control"] = load_positions(opt_dir, suffix="_resampled")
    return out


def baseline_for(arms: dict[str, list[np.ndarray]]) -> str:
    """The arm the optimizer's advantage is measured against.

    The resampled control when the pipeline wrote one, because DexYCB captures
    at 30 Hz and resampling alone accounts for most of the raw-to-optimized
    gap. Raw otherwise.
    """
    return "resampled_control" if "resampled_control" in arms else "raw"


def trivial_baseline(trajectories: list[np.ndarray], fraction: float) -> float:
    """MSE of predicting no state change, the score of "the hand has stopped".

    Reported beside every split so an advantage can be read against the floor.
    A split whose baseline arm already scores at this floor is measuring how
    quiet that arm's holdout is, not how well motion extrapolates, and the
    optimizer cannot look better on a test set whose answer is nearly zero.
    The prefix split is exactly that case on real data, which is why its
    controlled advantage reverses there. See `diagnose_tail_extrapolation.py`.
    """
    deltas = [
        np.diff(t[int((1.0 - fraction) * len(t)) :, 6:], axis=0) for t in trajectories
    ]
    usable = [d for d in deltas if len(d) > 0]
    if not usable:
        return float("nan")
    return float(np.mean(np.concatenate(usable) ** 2))


def run_set(name: str, seeds: int) -> dict:
    """Score every arm at every granularity over `seeds` independent splits."""
    arms = arms_for(name)
    baseline = baseline_for(arms)
    results: dict[str, dict] = {}
    for split_name, splitter in SPLITS.items():
        fraction = SPLIT_FRACTIONS[split_name]
        per_arm: dict[str, list[float]] = {arm: [] for arm in arms}
        gains: list[float] = []
        positive = 0
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            for arm, trajs in arms.items():
                train, holdout = splitter(trajs, rng)
                model, _ = train_bc(train, holdout, seed=seed)
                per_arm[arm].append(evaluate(model, holdout)["mse"])
            gain = 100.0 * (1.0 - per_arm["optimized"][-1] / per_arm[baseline][-1])
            gains.append(gain)
            positive += int(gain > 0.0)
        row = {
            arm: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
            for arm, v in per_arm.items()
        }
        row["baseline"] = baseline
        row["trivial_baseline_mse"] = {
            arm: trivial_baseline(trajs, fraction) for arm, trajs in arms.items()
        }
        row["advantage_pct"] = {
            "mean": float(np.mean(gains)),
            "sd": float(np.std(gains)),
            "min": float(np.min(gains)),
            "max": float(np.max(gains)),
            "seeds_positive": positive,
            "seeds": seeds,
        }
        results[split_name] = row
    return {"set": name, "seeds": seeds, "results": results}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    name = "dexycb"
    if "--set" in argv:
        name = argv[argv.index("--set") + 1]
    seeds = DEFAULT_SEEDS
    if "--seeds" in argv:
        seeds = int(argv[argv.index("--seeds") + 1])
    if seeds < 1:
        print(f"--seeds must be at least 1, got {seeds}", file=sys.stderr)
        return 1
    if name not in DEMO_SETS:
        print(
            f"unknown demo set {name!r}; choose from {sorted(DEMO_SETS)}",
            file=sys.stderr,
        )
        return 1

    report = run_set(name, seeds)
    _, _, out_path = DEMO_SETS[name]
    out = out_path.parent / f"bc_split_granularity_{name.replace('-', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(
        f"{'split':<12}{'baseline':>12}{'opt':>12}{'advantage':>14}{'pos':>6}"
        f"{'trivial base':>15}"
    )
    for split_name, row in report["results"].items():
        base = row["baseline"]
        adv = row["advantage_pct"]
        print(
            f"{split_name:<12}"
            f"{row[base]['mean']:>12.3e}"
            f"{row['optimized']['mean']:>12.3e}"
            f"{adv['mean']:>9.1f} +/- {adv['sd']:>4.1f}%"
            f"{adv['seeds_positive']:>4}/{adv['seeds']}"
            f"{row['trivial_baseline_mse'][base]:>15.3e}"
        )
    print("\n'trivial base' is the MSE of predicting no motion at all.")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
