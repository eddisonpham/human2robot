"""Diagnose which trajectories the C++ optimizer fails to converge on, and why.

The real DexYCB set converges on 85 of 100 sequences while the synthetic set
converges on all 100, so something about real retargeted motion produces cases
the optimizer cannot satisfy. This script characterizes the failures instead of
leaving the gap as a bare number.

Usage:

    uv run python scripts/diagnose_convergence.py [--set dexycb|synthetic] [--limit N]
"""

import argparse

# Import the experiment's own config so the two can never disagree about
# limits or weights. An earlier copy of this script used a different
# acceleration limit and reported a convergence rate the experiment does not
# reproduce.
import importlib.util
import json
from pathlib import Path

import numpy as np

from human2robot.cpp_bindings import (
    compute_metrics,
    cubic_resample,
    optimize_trajectory,
)
from human2robot.data.allegro_demos import load_demo_npz

_spec = importlib.util.spec_from_file_location(
    "h2r_compare", Path(__file__).resolve().parent / "compare_dexycb_synthetic.py"
)
_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_compare)
make_config = _compare.make_config
LOWER = _compare._LOWER
UPPER = _compare._UPPER

DEXYCB_DIR = Path("data/demonstrations_dexycb")
SYNTH_DIR = Path("data/demonstrations")
OUT_PATH = Path("results/trajectory_optimization/convergence_diagnosis.json")

CAPTURE_DT = 1.0 / 30.0
CONTROL_DT = 0.02


def describe(q: np.ndarray) -> dict:
    """Summarize the properties most likely to defeat the optimizer."""
    metrics = compute_metrics(q, CONTROL_DT)
    out_of_range = int(((q < LOWER - 1e-6) | (q > UPPER + 1e-6)).sum())
    return {
        "horizon": int(len(q)),
        "max_velocity": float(metrics["max_velocity"]),
        "max_acceleration": float(metrics["max_acceleration"]),
        "max_jerk": float(metrics["max_jerk"]),
        "smoothness": float(metrics["smoothness"]),
        "out_of_range_samples": out_of_range,
        "out_of_range_fraction": float(out_of_range / max(q.size, 1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="dexycb", choices=["dexycb", "synthetic"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=300)
    args = parser.parse_args()

    source = DEXYCB_DIR if args.set == "dexycb" else SYNTH_DIR
    resample = args.set == "dexycb"
    paths = sorted(p for p in source.glob("*.npz") if "_opt" not in p.stem)
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        print(f"no demos in {source}")
        return 1

    config = make_config(0, max_iterations=args.max_iterations)
    rows = []
    for path in paths:
        demo = load_demo_npz(path)
        q = demo.q
        if resample:
            q16 = cubic_resample(demo.q[:, 6:], CAPTURE_DT, CONTROL_DT)
            q = np.concatenate([np.zeros((len(q16), 6)), q16], axis=1)
        result = optimize_trajectory(q, q, CONTROL_DT, config)
        row = {
            "name": path.stem,
            "converged": bool(result.converged),
            "iterations": int(result.iterations),
            "initial_cost": float(result.initial_cost),
            "final_cost": float(result.final_cost),
        }
        row.update(describe(q))
        rows.append(row)

    converged = [r for r in rows if r["converged"]]
    failed = [r for r in rows if not r["converged"]]
    print(f"{len(rows)} sequences, {len(converged)} converged, {len(failed)} failed")

    def summarize(group, label):
        if not group:
            print(f"  {label}: none")
            return
        print(f"  {label}:")
        for key in (
            "horizon",
            "max_velocity",
            "max_acceleration",
            "max_jerk",
            "out_of_range_fraction",
        ):
            values = np.array([r[key] for r in group], dtype=float)
            print(
                f"    {key:<24} mean={values.mean():>12.4f} max={values.max():>12.4f}"
            )

    summarize(converged, "converged")
    summarize(failed, "failed")

    if failed:
        print("  failing sequences (first 10):")
        for r in sorted(failed, key=lambda r: -r["smoothness"])[:10]:
            print(
                f"    {r['name']:<20} iters={r['iterations']:<5} "
                f"jerk={r['max_jerk']:>10.1f} oob={r['out_of_range_fraction']:.3f} "
                f"cost={r['initial_cost']:.3g}->{r['final_cost']:.3g}"
            )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"demo_set": args.set, "rows": rows}, indent=2))
    print(f"written to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
