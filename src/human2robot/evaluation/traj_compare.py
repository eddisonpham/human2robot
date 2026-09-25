"""Baseline versus optimized trajectory comparison reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from human2robot.cpp_bindings import compute_metrics
from human2robot.data.allegro_demos import load_demo_npz


def compare_directory(
    raw_dir: str | Path,
    optimized_dir: str | Path,
    dt: float = 0.01,
) -> dict:
    """Aggregate geometric metrics over raw and optimized demo trees."""
    raw_paths = sorted(Path(raw_dir).glob("*.npz"))
    opt_paths = sorted(Path(optimized_dir).glob("*_opt.npz"))
    if not raw_paths or not opt_paths:
        raise FileNotFoundError("both raw and optimized demo sets are required")
    pairs = []
    for opt_path in opt_paths:
        stem = opt_path.stem.removeprefix("demo_").removesuffix("_opt")
        for raw_path in raw_paths:
            if raw_path.stem == stem or raw_path.stem == "demo_" + stem:
                pairs.append((raw_path, opt_path))
                break
    if not pairs:
        raise FileNotFoundError("no raw/optimized pairs matched")

    def aggregate(paths) -> dict:
        rows = []
        for path in paths:
            demo = load_demo_npz(path)
            metrics = compute_metrics(demo.q, dt)
            rows.append(metrics)
        keys = (
            "max_velocity",
            "max_acceleration",
            "max_jerk",
            "smoothness",
        )
        out = {}
        for key in keys:
            values = np.array([row[key] for row in rows], dtype=float)
            out[key] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "max": float(values.max()),
            }
        return out

    report = {
        "count": len(pairs),
        "dt": dt,
        "raw": aggregate([pair[0] for pair in pairs]),
        "optimized": aggregate([pair[1] for pair in pairs]),
    }
    for key in ("max_velocity", "max_acceleration", "max_jerk", "smoothness"):
        report[f"{key}_reduction_pct"] = 100.0 * (
            1.0
            - report["optimized"][key]["mean"] / max(report["raw"][key]["mean"], 1e-12)
        )
    return report


def write_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2))
