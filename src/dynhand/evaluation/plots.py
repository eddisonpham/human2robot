"""Learning curve plotting from recorded metrics.

Reads the metrics.jsonl files written by RunRecorder and plots metric
curves, optionally aggregating multiple seeds per condition into a mean
band. No single-seed curve is presented as a stable estimate without the
user seeing the label say so.
"""

import json
from pathlib import Path

import numpy as np

from dynhand.evaluation.audit import audit_metrics


def read_metrics(path: str | Path) -> dict[str, np.ndarray]:
    """Read a metrics.jsonl file into column arrays keyed by metric name."""
    columns: dict[str, list[float]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for key, value in record.items():
                columns.setdefault(key, []).append(value)
    return {key: np.asarray(values) for key, values in columns.items()}


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing moving average; output is shorter than input by window-1."""
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_learning_curves(
    runs: dict[str, list[Path]],
    metric: str = "eval_return_mean",
    step_key: str = "step",
    smooth: int = 1,
    title: str = "",
    out_path: str | Path = "results/plots/learning_curves.png",
) -> Path:
    """Plot metric vs steps for each label, aggregating seed runs into a band.

    runs maps a condition label to a list of metrics.jsonl paths, one per
    seed. Multiple seeds produce a mean line with a min-max band.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, paths in runs.items():
        series = []
        for path in paths:
            report = audit_metrics(path, metric=metric)
            if not report.healthy:
                raise ValueError(f"unhealthy metrics stream {path}: {report}")
            data = read_metrics(path)
            if metric not in data:
                raise KeyError(f"{metric} not found in {path}")
            values = moving_average(data[metric], smooth)
            steps = data[step_key][-len(values) :]
            series.append((steps, values))
        if len(series) == 1:
            steps, values = series[0]
            ax.plot(steps, values, label=label)
        else:
            length = min(len(v) for _, v in series)
            stacked = np.stack([v[-length:] for _, v in series])
            steps = series[0][0][-length:]
            mean = stacked.mean(axis=0)
            low = stacked.min(axis=0)
            high = stacked.max(axis=0)
            ax.plot(steps, mean, label=label)
            ax.fill_between(steps, low, high, alpha=0.2)

    ax.set_xlabel("environment steps")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
