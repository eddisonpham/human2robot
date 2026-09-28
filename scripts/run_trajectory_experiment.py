"""Phase G experiment runner: baseline versus optimized demonstrations."""

import sys
import time
from pathlib import Path

from human2robot.evaluation.traj_compare import compare_directory, write_report
from human2robot.optimization import (
    OptimizationConfig,
    optimize_demo_directory,
)

RAW_DIR = Path("data/demonstrations")
OPT_DIR = Path("data/demonstrations_optimized")
REPORT_PATH = Path("results/trajectory_optimization/report.json")


def main() -> int:

    config = OptimizationConfig(
        dt=0.02,
        tracking=1.0,
        velocity=0.05,
        acceleration=0.05,
        jerk=0.02,
        limits_weight=10.0,
        max_iterations=300,
        convergence_tolerance=1e-4,
        step_size=0.5,
        noise_scale=0.0,
        seed=0,
    )
    started = time.perf_counter()
    manifest = optimize_demo_directory(RAW_DIR, OPT_DIR, config)
    wall = time.perf_counter() - started

    report = compare_directory(RAW_DIR, OPT_DIR, dt=config.dt)
    report["wall_seconds"] = wall
    report["config"] = config.model_dump()
    report["total_demo_count"] = manifest["count"]
    converged = sum(1 for m in manifest["per_file"].values() if m["converged"])
    report["converged_count"] = converged

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_report(report, REPORT_PATH)
    print(f"optimized {manifest['count']} demos in {wall:.1f} s")
    print(f"converged: {converged}/{manifest['count']}")
    for key in ("max_velocity", "max_acceleration", "max_jerk", "smoothness"):
        raw_mean = report["raw"][key]["mean"]
        opt_mean = report["optimized"][key]["mean"]
        print(
            f"{key}: {raw_mean:.3f} -> {opt_mean:.3f} "
            f"({report[f'{key}_reduction_pct']:.1f}%)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
