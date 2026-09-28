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

Usage:

    uv run python scripts/check_bc_split_granularity.py [--set dexycb]
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
    split_prefix,
    split_trajectory,
    split_transition,
)

SPLITS = {
    "transition": split_transition,
    "trajectory": split_trajectory,
    "prefix": split_prefix,
}


def arms_for(name: str) -> dict[str, list[np.ndarray]]:
    raw_dir, opt_dir, _ = DEMO_SETS[name]
    raw = load_positions(raw_dir)
    optimized = load_positions(opt_dir, suffix="_opt")
    out = {"raw": raw, "optimized": optimized}
    control = sorted(opt_dir.glob("*_resampled.npz"))
    if control:
        out["resampled_control"] = load_positions(opt_dir, suffix="_resampled")
    return out


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    name = "dexycb"
    if "--set" in argv:
        name = argv[argv.index("--set") + 1]
    if name not in DEMO_SETS:
        print(
            f"unknown demo set {name!r}; choose from {sorted(DEMO_SETS)}",
            file=sys.stderr,
        )
        return 1

    results: dict[str, dict[str, float]] = {}
    for split_name, splitter in SPLITS.items():
        rng = np.random.default_rng(0)
        row: dict[str, float] = {}
        for arm, trajs in arms_for(name).items():
            train, holdout = splitter(trajs, rng)
            model, _ = train_bc(train, holdout, seed=0)
            row[arm] = evaluate(model, holdout)["mse"]
        results[split_name] = row

    print(f"{'split':<12}{'raw':>12}{'control':>12}{'opt':>12}{'controlled%':>14}")
    for split_name, row in results.items():
        ctrl = row.get("resampled_control")
        base = ctrl if ctrl is not None else row["raw"]
        gain = 100.0 * (1.0 - row["optimized"] / base)
        ctrl_s = f"{ctrl:.3e}" if ctrl is not None else "n/a"
        print(
            f"{split_name:<12}{row['raw']:>12.3e}{ctrl_s:>12}"
            f"{row['optimized']:>12.3e}{gain:>13.1f}%"
        )

    out = Path("results/trajectory_optimization/bc_split_granularity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
