"""Optimize real DexYCB demos with the C++ core and compare to synthetic.

DexYCB MANO capture runs at 30 Hz; trajectories are cubic-resampled to the
20 ms control period of the Allegro environment before optimization so both
demo sets are processed under identical pipeline settings.

The optimized DexYCB trajectories are written to
``data/demonstrations_dexycb_optimized/`` as ``<stem>_opt.npz`` files, so
``scripts/run_downstream_bc.py`` can run the same imitation comparison on real
data that it runs on the synthetic set. Pass ``--no-write`` to skip that.
"""

import json
import sys
from pathlib import Path

import numpy as np

from human2robot.cpp_bindings import (
    OptimizerConfig,
    compute_metrics,
    cubic_resample,
    optimize_trajectory,
)
from human2robot.data.allegro_demos import load_demo_npz
from human2robot.data.processing import demo_actions, differentiate
from human2robot.data.schema import DEMO_SCHEMA_VERSION

DEXYCB_DIR = Path("data/demonstrations_dexycb")
DEXYCB_OPT_DIR = Path("data/demonstrations_dexycb_optimized")
SYNTH_DIR = Path("data/demonstrations")
OUT_PATH = Path("results/trajectory_optimization/real_vs_synthetic.json")

DEXYCB_CAPTURE_DT = 1.0 / 30.0
CONTROL_DT = 0.02

_LOWER = np.array([0.0] * 6 + [-0.47] + [0.196] * 3 + [-0.175] + [0.0] * 8 + [-0.8] * 3)
_UPPER = np.array([0.0] * 6 + [0.47] + [1.61] * 3 + [1.72] + [1.57] * 8 + [0.0] * 3)


def make_config(seed: int, max_iterations: int = 300) -> OptimizerConfig:
    return OptimizerConfig(
        dof=22,
        lower=_LOWER,
        upper=_UPPER,
        max_velocity=np.full(22, 2.0),
        max_acceleration=np.full(22, 20.0),
        tracking=1.0,
        velocity=0.05,
        acceleration=0.05,
        jerk=0.02,
        limits_weight=10.0,
        max_iterations=max_iterations,
        convergence_tolerance=1e-4,
        step_size=0.05,
        seed=seed,
    )


def aggregate(metrics_rows: list[dict]) -> dict:
    keys = ("max_velocity", "max_acceleration", "max_jerk", "smoothness")
    out = {}
    for key in keys:
        values = np.array([row[key] for row in metrics_rows], dtype=float)
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "max": float(values.max()),
        }
    return out


def process_set(
    demo_paths: list[Path], seed: int, resample: bool, out_dir: Path | None = None
) -> dict:
    config = make_config(seed)
    raw_rows, opt_rows, converged = [], [], 0
    for path in demo_paths:
        demo = load_demo_npz(path)
        q = demo.q
        if resample:
            q16 = cubic_resample(demo.q[:, 6:], DEXYCB_CAPTURE_DT, CONTROL_DT)
            q = np.concatenate([np.zeros((len(q16), 6)), q16], axis=1)
        raw_rows.append(compute_metrics(q, CONTROL_DT))
        result = optimize_trajectory(q, q, CONTROL_DT, config)
        q_opt = np.asarray(result.trajectory, dtype=float)
        opt_rows.append(compute_metrics(q_opt, CONTROL_DT))
        if result.converged:
            converged += 1
        if out_dir is not None:
            if resample:
                write_optimized_demo(
                    out_dir / f"{path.stem}_resampled.npz", q, demo, tag="resampled"
                )
            write_optimized_demo(out_dir / f"{path.stem}_opt.npz", q_opt, demo)
    return {
        "count": len(demo_paths),
        "converged": converged,
        "raw": aggregate(raw_rows),
        "optimized": aggregate(opt_rows),
    }


def write_optimized_demo(
    out_path: Path, q_opt: np.ndarray, demo, tag: str = "opt"
) -> None:
    """Write a trajectory as a demo-schema NPZ.

    Resampling changes the trajectory length, so the derived velocity and action
    fields are recomputed from the joint path being written rather than copied
    from the raw demo, which would leave the file internally inconsistent. The
    ``resampled`` tag marks the pre-optimization control arm, so the imitation
    comparison can isolate the optimizer from the resampling step.
    """
    q_opt = np.asarray(q_opt, dtype=np.float32)
    qdot, _ = differentiate(q_opt, CONTROL_DT)
    a_demo = demo_actions(q_opt, _LOWER, _UPPER)
    length = len(q_opt)
    np.savez_compressed(
        out_path,
        q=q_opt,
        qdot=qdot.astype(np.float32),
        a_demo=a_demo.astype(np.float32),
        object_pose=np.zeros((length, 7), dtype=np.float32),
        object_vel=np.zeros((length, 6), dtype=np.float32),
        contact=np.zeros((length, 5), dtype=np.float32),
        trajectory_id=np.array(f"{demo.trajectory_id}_{tag}"),
        task_id=np.array(demo.task_id),
        source=np.array(f"{demo.source}_{tag}"),
        schema_version=np.array(DEMO_SCHEMA_VERSION),
    )


def reductions(report: dict) -> dict:
    out = {}
    for key in ("max_velocity", "max_acceleration", "max_jerk", "smoothness"):
        raw_mean = report["raw"][key]["mean"]
        opt_mean = report["optimized"][key]["mean"]
        out[f"{key}_reduction_pct"] = 100.0 * (1.0 - opt_mean / max(raw_mean, 1e-12))
    return out


def main(argv: list[str] | None = None) -> int:
    write = "--no-write" not in (argv or sys.argv[1:])
    dexycb_paths = sorted(DEXYCB_DIR.glob("dexycb_*.npz"))
    synth_paths = sorted(p for p in SYNTH_DIR.glob("*.npz") if "_opt" not in p.stem)
    if not dexycb_paths:
        raise FileNotFoundError(
            "no DexYCB demos; run: uv run python -m human2robot.data.dexycb"
        )
    if not synth_paths:
        raise FileNotFoundError("run generate_synthetic_demos first")

    out_dir = DEXYCB_OPT_DIR if write else None
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"processing {len(dexycb_paths)} DexYCB demos (30 Hz -> 20 ms)...")
    dexycb_report = process_set(dexycb_paths, seed=0, resample=True, out_dir=out_dir)
    print(f"processing {len(synth_paths)} synthetic demos...")
    synth_report = process_set(synth_paths, seed=0, resample=False)

    dexycb_report.update(reductions(dexycb_report))
    synth_report.update(reductions(synth_report))
    combined = {
        "control_dt": CONTROL_DT,
        "dexycb_capture_dt": DEXYCB_CAPTURE_DT,
        "dexycb": dexycb_report,
        "synthetic": synth_report,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(combined, indent=2))

    header = (
        f"{'metric':<20}{'DEXYCB raw':>12}{'-> opt':>10}{'red%':>8}"
        f" | {'SYNTH raw':>11}{'-> opt':>10}{'red%':>8}"
    )
    print(header)
    for key in ("max_velocity", "max_acceleration", "max_jerk", "smoothness"):
        d, s = dexycb_report, synth_report
        print(
            f"{key:<20}"
            f"{d['raw'][key]['mean']:>12.3f}{d['optimized'][key]['mean']:>10.3f}"
            f"{d[f'{key}_reduction_pct']:>7.1f}% | "
            f"{s['raw'][key]['mean']:>11.3f}{s['optimized'][key]['mean']:>10.3f}"
            f"{s[f'{key}_reduction_pct']:>7.1f}%"
        )
    print(
        f"convergence: dexycb {dexycb_report['converged']}/{dexycb_report['count']}, "
        f"synthetic {synth_report['converged']}/{synth_report['count']}"
    )
    print(f"report written to {OUT_PATH}")
    if write:
        print(f"optimized DexYCB demos written to {DEXYCB_OPT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
