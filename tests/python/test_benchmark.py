"""Tests for multi-seed benchmark summaries."""

import json

import pytest

from dynhand.evaluation.benchmark import area_under_curve, summarize_runs, write_summary


def write_metrics(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps({"step": step, "eval_return_mean": value})
            for step, value in enumerate(values)
        )
        + "\n"
    )


def test_benchmark_summary_and_auc(tmp_path) -> None:
    first = tmp_path / "a" / "metrics.jsonl"
    second = tmp_path / "b" / "metrics.jsonl"
    write_metrics(first, [0.0, 2.0, 4.0])
    write_metrics(second, [1.0, 3.0, 5.0])
    assert area_under_curve(first) == 4.0
    summary = summarize_runs([first, second])
    assert summary["seed_count"] == 2
    assert summary["final_mean"] == 4.5
    output = write_summary({"A": [first, second]}, tmp_path / "summary.json")
    assert json.loads(output.read_text())["A"]["auc_mean"] == 5.0


def test_benchmark_rejects_unhealthy_stream(tmp_path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"step": 1, "eval_return_mean": 1}\n{"step": 1, "eval_return_mean": 2}\n'
    )

    with pytest.raises(ValueError, match="unhealthy"):
        summarize_runs([path])
