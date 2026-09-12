"""Audited multi-seed benchmark summaries."""

import json
from pathlib import Path

import numpy as np

from dynhand.evaluation.audit import audit_metrics


def _series(path: str | Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Read and validate one evaluation metric stream."""
    report = audit_metrics(path, metric=metric)
    if not report.healthy:
        raise ValueError(f"unhealthy metrics stream: {path}: {report}")
    steps, values = [], []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            record = __import__("json").loads(text)
            if metric in record:
                steps.append(float(record["step"]))
                values.append(float(record[metric]))
    if not steps:
        raise KeyError(f"missing {metric} in {path}")
    return np.asarray(steps), np.asarray(values)


def area_under_curve(path: str | Path, metric: str = "eval_return_mean") -> float:
    """Compute trapezoidal area under one audited learning curve."""
    steps, values = _series(path, metric)
    return float(np.trapezoid(values, steps))


def summarize_runs(
    paths: list[str | Path], metric: str = "eval_return_mean"
) -> dict[str, object]:
    """Summarize final metrics and AUC across independent seed runs."""
    if not paths:
        raise ValueError("at least one run is required")
    series = [_series(path, metric) for path in paths]
    finals = np.asarray([values[-1] for _, values in series], dtype=float)
    aucs = np.asarray([area_under_curve(path, metric) for path in paths], dtype=float)
    result = {
        "runs": [str(path) for path in paths],
        "metric": metric,
        "seed_count": int(len(paths)),
        "final_mean": float(finals.mean()),
        "final_std": float(finals.std(ddof=1)) if len(finals) > 1 else 0.0,
        "final_min": float(finals.min()),
        "final_max": float(finals.max()),
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0,
    }
    return result


def main() -> None:
    """Write a summary from condition=run_dir arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Summarize audited RL runs")
    parser.add_argument(
        "--group", action="append", required=True, help="LABEL=DIR[,DIR]"
    )

    parser.add_argument("--metric", default="eval_return_mean")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    groups = {}
    for item in args.group:
        label, separator, paths = item.partition("=")
        if not separator or not label or not paths:
            raise ValueError(f"invalid group: {item}")
        groups[label] = [Path(path) / "metrics.jsonl" for path in paths.split(",")]
    print(write_summary(groups, args.out, metric=args.metric))


def write_summary(
    groups: dict[str, list[str | Path]],
    output: str | Path,
    metric: str = "eval_return_mean",
) -> Path:
    """Write JSON summaries for named benchmark conditions."""
    result = {
        label: summarize_runs(paths, metric=metric) for label, paths in groups.items()
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return output
