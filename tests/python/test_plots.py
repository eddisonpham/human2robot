"""Tests for metrics reading and learning curve plotting."""

import json
from pathlib import Path

import numpy as np
import pytest

from dynhand.evaluation.plots import moving_average, plot_learning_curves, read_metrics


def write_metrics(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_read_metrics_real(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(
        path,
        [
            {"step": 0, "eval_return_mean": 1.0},
            {"step": 100, "eval_return_mean": 2.0},
            {"step": 200, "eval_return_mean": 3.0},
        ],
    )
    data = read_metrics(path)
    assert set(data.keys()) == {"step", "eval_return_mean"}
    assert np.allclose(data["step"], [0, 100, 200])
    assert np.allclose(data["eval_return_mean"], [1, 2, 3])


def test_read_metrics_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1, "x": 0.5}\n\n{"step": 2, "x": 1.5}\n')
    data = read_metrics(path)
    assert np.allclose(data["x"], [0.5, 1.5])


def test_moving_average_basic() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    smoothed = moving_average(values, window=2)
    assert np.allclose(smoothed, [1.5, 2.5, 3.5])


def test_moving_average_window_one_returns_input() -> None:
    values = np.array([1.0, 2.0, 3.0])
    assert moving_average(values, window=1) is values


def test_moving_average_short_input() -> None:
    values = np.array([1.0, 2.0])
    assert np.allclose(moving_average(values, window=5), values)


def test_plot_learning_curves_single_seed(tmp_path: Path) -> None:
    path = tmp_path / "run_a" / "metrics.jsonl"
    write_metrics(
        path,
        [{"step": s, "eval_return_mean": float(s)} for s in [0, 100, 200, 300]],
    )
    out = tmp_path / "curve.png"
    plot_learning_curves(
        {"A": [path]},
        smooth=2,
        out_path=out,
        title="test",
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_learning_curves_multi_seed_band(tmp_path: Path) -> None:
    seeds = []
    for seed in range(3):
        path = tmp_path / f"run_s{seed}" / "metrics.jsonl"
        rng = np.random.default_rng(seed)
        write_metrics(
            path,
            [
                {"step": s, "eval_return_mean": float(s) + rng.normal(0, 10)}
                for s in [0, 100, 200, 300]
            ],
        )
        seeds.append(path)
    out = tmp_path / "band.png"
    plot_learning_curves({"B": seeds}, smooth=1, out_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_learning_curves_missing_metric_raises(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_metrics(path, [{"step": 0, "other": 1.0}])
    with pytest.raises(KeyError):
        plot_learning_curves(
            {"A": [path]}, metric="eval_return_mean", out_path=tmp_path / "x.png"
        )
