"""Tests for evaluation CLI entry points and robustness sweeps."""

import importlib
import json
import sys
from pathlib import Path

import pytest

from human2robot.evaluation import benchmark, plot_cli, robustness


def _run_domain_randomized_eval(sac, env_id, **kwargs):
    """Call domain_randomized_eval with a patched evaluate in place.

    robustness imports evaluate lazily inside the function body, so the
    monkeypatch has to target the defining module rather than this one.
    """
    return robustness.domain_randomized_eval(sac, env_id, **kwargs)


def _write_metrics(path: Path, steps=(0, 10, 20), value=-5.0) -> None:
    path.write_text(
        "\n".join(
            json.dumps({"step": step, "eval_return_mean": value + step})
            for step in steps
        )
        + "\n",
        encoding="utf-8",
    )


def test_domain_randomized_eval_averages_across_scales(monkeypatch) -> None:
    calls = []

    def fake_evaluate(sac, env_id, episodes, seed):
        calls.append((env_id, episodes, seed))
        return {"eval_return_mean": float(seed)}

    evaluate_module = importlib.import_module("human2robot.evaluation.evaluate")

    monkeypatch.setattr(evaluate_module, "evaluate", fake_evaluate)
    result = _run_domain_randomized_eval(
        object(), "Pendulum-v1", episodes=3, seed=0, scales=(0.8, 1.0)
    )
    assert set(result) == {"robustness_mean", "scale_0.8", "scale_1.0"}
    assert result["scale_0.8"] == 8.0
    assert result["scale_1.0"] == 10.0
    assert result["robustness_mean"] == pytest.approx(9.0)
    assert calls[0] == ("Pendulum-v1", 3, 8)


def test_domain_randomized_eval_uses_default_scales(monkeypatch) -> None:
    evaluate_module = importlib.import_module("human2robot.evaluation.evaluate")

    seen = []

    def fake_evaluate(sac, env_id, episodes, seed):
        seen.append(seed)
        return {"eval_return_mean": 1.0}

    monkeypatch.setattr(evaluate_module, "evaluate", fake_evaluate)
    result = _run_domain_randomized_eval(object(), "Pendulum-v1", episodes=1)
    assert seen == [8, 10, 12]
    assert result["robustness_mean"] == pytest.approx(1.0)


def test_generalization_split_counts_unique_ids() -> None:
    result = robustness.generalization_split(["a", "a", "b"], ["c", "c", "d"])
    assert result == {"train_size": 2, "test_size": 2, "leakage": False}


def test_plot_cli_main_writes_requested_output(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    _write_metrics(run_dir / "metrics.jsonl")
    out = tmp_path / "plots" / "curves.png"
    monkeypatch.setattr(
        sys, "argv", ["human2robot-plot", "--runs", f"A={run_dir}", "--out", str(out)]
    )
    plot_cli.main()
    assert out.exists()


def test_plot_cli_main_forwards_metric_and_title(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "run_b"
    run_dir.mkdir()
    _write_metrics(run_dir / "metrics.jsonl", value=-1.0)
    out = tmp_path / "curves.png"
    captured = {}
    real_plot = plot_cli.plot_learning_curves

    def spy(runs, **kwargs):
        captured.update(kwargs)
        captured["runs"] = runs
        return real_plot(runs, **kwargs)

    monkeypatch.setattr(plot_cli, "plot_learning_curves", spy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "human2robot-plot",
            "--runs",
            f"Label={run_dir}",
            "--metric",
            "eval_return_mean",
            "--smooth",
            "2",
            "--title",
            "Tier B",
            "--out",
            str(out),
        ],
    )
    plot_cli.main()
    assert captured["metric"] == "eval_return_mean"
    assert captured["smooth"] == 2
    assert captured["title"] == "Tier B"
    assert list(captured["runs"]) == ["Label"]


def test_benchmark_cli_writes_summary(tmp_path, monkeypatch, capsys) -> None:
    run_dir = tmp_path / "run_c"
    run_dir.mkdir()
    _write_metrics(run_dir / "metrics.jsonl")
    out = tmp_path / "summary" / "bench.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "human2robot-benchmark",
            "--group",
            f"A={run_dir}",
            "--out",
            str(out),
        ],
    )
    benchmark.main()
    assert out.exists()
    summary = json.loads(out.read_text())
    assert "A" in summary
    assert "auc_mean" in summary["A"]
    assert str(out) in capsys.readouterr().out


def test_benchmark_cli_merges_multiple_dirs_per_group(tmp_path, monkeypatch) -> None:
    first, second = tmp_path / "s0", tmp_path / "s1"
    first.mkdir()
    second.mkdir()
    _write_metrics(first / "metrics.jsonl", value=-5.0)
    _write_metrics(second / "metrics.jsonl", value=-40.0)
    out = tmp_path / "bench.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "human2robot-benchmark",
            "--group",
            f"A={first},{second}",
            "--metric",
            "eval_return_mean",
            "--out",
            str(out),
        ],
    )
    benchmark.main()
    summary = json.loads(out.read_text())
    assert summary["A"]["seed_count"] == 2
    assert summary["A"]["auc_std"] > 0.0


@pytest.mark.parametrize("group", ["no-equals", "=missing_label", "A="])
def test_benchmark_cli_rejects_malformed_groups(tmp_path, monkeypatch, group) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["human2robot-benchmark", "--group", group, "--out", str(tmp_path / "b.json")],
    )
    with pytest.raises(ValueError, match="invalid group"):
        benchmark.main()
