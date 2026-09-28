"""Tune the optimizer on one subject and report on another.

The descent rate is easy to inflate by choosing `step_size` while looking at
the same sequences the rate is then reported on. This script fixes the protocol
in two stages:

1. The sweep runs on the first half of subject-01 only, and the chosen setting
   is evaluated on the second half, which no selection decision has seen.
2. The same setting is then evaluated on subject-02 in full. Subject-02 is a
   different person's hand and no hyperparameter was ever selected on it, so it
   is a genuine cross-subject test rather than a within-subject split.

Usage:

    uv run python scripts/validate_optimizer_split.py [--out results/....json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from human2robot.cpp_bindings import (
    cubic_resample,
    optimize_trajectory,
)
from human2robot.data.allegro_demos import load_demo_npz

DEXYCB_DIR = Path("data/demonstrations_dexycb_ik")
DEXYCB_S2_DIR = Path("data/demonstrations_dexycb_ik_s2")
SYNTH_DIR = Path("data/demonstrations")
DEFAULT_OUT = Path("results/trajectory_optimization/optimizer_split.json")

from human2robot.optimization.experiment import (  # noqa: E402
    CAPTURE_DT,
    CONTROL_DT,
    NOISE_SCALE,
    make_optimizer_config,
)

STEP_GRID = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0)


def load_dexycb(directory: Path = DEXYCB_DIR) -> list[np.ndarray]:
    """Load real trajectories, resampled to the 20 ms control period."""
    out = []
    for path in sorted(directory.glob("*.npz")):
        demo = load_demo_npz(path)
        q16 = cubic_resample(demo.q[:, 6:], CAPTURE_DT, CONTROL_DT)
        out.append(np.concatenate([np.zeros((len(q16), 6)), q16], axis=1))
    return out


def load_synthetic() -> list[np.ndarray]:
    return [
        load_demo_npz(p).q.astype(float)
        for p in sorted(SYNTH_DIR.glob("synth_*.npz"))
        if "_opt" not in p.stem
    ]


def score(trajectories: list[np.ndarray], step_size: float) -> dict:
    """Run the optimizer and summarize the cost reduction it achieved."""
    results = [
        optimize_trajectory(
            q, q, CONTROL_DT, make_optimizer_config(step_size=step_size)
        )
        for q in trajectories
    ]
    improvements = np.array([r.improvement_pct for r in results], dtype=float)
    return {
        "n": len(trajectories),
        "improved": int((improvements > 0).sum()),
        "mean_improvement_pct": float(improvements.mean()),
        "median_improvement_pct": float(np.median(improvements)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.5,
        help="Fraction of sequences reserved for reporting",
    )
    args = parser.parse_args(argv)

    dexycb = load_dexycb()
    if len(dexycb) < 4:
        print(f"need DexYCB demos in {DEXYCB_DIR}, found {len(dexycb)}")
        return 1
    split = int(len(dexycb) * (1.0 - args.holdout_frac))
    tune, holdout = dexycb[:split], dexycb[split:]
    print(
        f"{len(dexycb)} DexYCB sequences: {len(tune)} for tuning, "
        f"{len(holdout)} held out"
    )

    sweep = {}
    for step in STEP_GRID:
        sweep[str(step)] = score(tune, step)
        s = sweep[str(step)]
        print(
            f"  step_size={step:<5} tune improved={s['improved']}/{s['n']} "
            f"mean={s['mean_improvement_pct']:.2f}% "
            f"median={s['median_improvement_pct']:.2f}%"
        )

    best_step = max(STEP_GRID, key=lambda s: sweep[str(s)]["mean_improvement_pct"])
    print(f"\nselected step_size={best_step} on the tuning half alone")

    heldout = score(holdout, best_step)
    synth = score(load_synthetic(), best_step)
    print(
        f"  HELD-OUT DexYCB   improved={heldout['improved']}/{heldout['n']} "
        f"mean={heldout['mean_improvement_pct']:.2f}% "
        f"median={heldout['median_improvement_pct']:.2f}%"
    )
    print(
        f"  synthetic         improved={synth['improved']}/{synth['n']} "
        f"mean={synth['mean_improvement_pct']:.2f}% "
        f"median={synth['median_improvement_pct']:.2f}%"
    )

    subject2 = None
    if DEXYCB_S2_DIR.is_dir() and list(DEXYCB_S2_DIR.glob("*.npz")):
        subject2 = score(load_dexycb(DEXYCB_S2_DIR), best_step)
        print(
            f"  CROSS-SUBJECT s-2 improved={subject2['improved']}/{subject2['n']} "
            f"mean={subject2['mean_improvement_pct']:.2f}% "
            f"median={subject2['median_improvement_pct']:.2f}%"
        )
    else:
        print(f"  CROSS-SUBJECT s-2 skipped, no demos in {DEXYCB_S2_DIR}")

    payload = {
        "noise_scale": NOISE_SCALE,
        "selected_step_size": best_step,
        "selection_rule": "highest mean improvement_pct on the tuning half",
        "tune": sweep,
        "holdout_dexycb": heldout,
        "synthetic_all": synth,
        "cross_subject_dexycb_s2": subject2,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
