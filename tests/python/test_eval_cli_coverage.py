"""Tests for evaluation CLI entry points and robustness sweeps."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from human2robot.evaluation import benchmark, plot_cli, robustness
from human2robot.evaluation.evaluate import build_single_env

ALLEGRO_ENV_ID = "Human2Robot-AllegroPickup-v0"


class _RecordingPolicy:
    """A policy that notes what the model looked like when it acted.

    `domain_randomized_eval` builds the environment it evaluates in, so the
    policy is the only observer that runs while a condition is active. Recording
    the mass it sees is therefore the direct way to prove a scale was really
    applied rather than merely passed to a function that ignored it.
    """

    def __init__(self, env, masses: list[float]) -> None:
        self._env = env
        self._masses = masses

    def act(self, obs, deterministic: bool = True):
        self._masses.append(float(self._env.model.body_mass[0]))
        return np.zeros(self._env.action_space.shape[0], dtype=np.float32)


def _write_metrics(path: Path, steps=(0, 10, 20), value=-5.0) -> None:
    path.write_text(
        "\n".join(
            json.dumps({"step": step, "eval_return_mean": value + step})
            for step in steps
        )
        + "\n",
        encoding="utf-8",
    )


def test_domain_randomization_scales_the_model_and_restores_it() -> None:
    env = build_single_env(ALLEGRO_ENV_ID)
    try:
        model = env.model
        nominal = model.body_mass.copy()
        friction = model.geom_friction.copy()
        damping = model.dof_damping.copy()
        with robustness.domain_randomization(model, 0.5):
            assert np.allclose(model.body_mass, nominal * 0.5)
            assert np.allclose(model.geom_friction, friction * 0.5)
            assert np.allclose(model.dof_damping, damping * 0.5)
        assert np.allclose(model.body_mass, nominal)
        assert np.allclose(model.geom_friction, friction)
        assert np.allclose(model.dof_damping, damping)
    finally:
        env.close()


def test_domain_randomization_restores_after_an_exception() -> None:
    """A failed condition must not leave the perturbation behind.

    Without this the next condition is evaluated against the previous one's
    physics, turning a sweep into an ordering effect that looks like a trend.
    """
    env = build_single_env(ALLEGRO_ENV_ID)
    try:
        model = env.model
        nominal = model.body_mass.copy()
        with (
            pytest.raises(RuntimeError, match="boom"),
            robustness.domain_randomization(model, 2.0),
        ):
            raise RuntimeError("boom")
        assert np.allclose(model.body_mass, nominal)
    finally:
        env.close()


def test_domain_randomization_rejects_a_bad_scale() -> None:
    env = build_single_env(ALLEGRO_ENV_ID)
    try:
        with (
            pytest.raises(ValueError, match="must be positive"),
            robustness.domain_randomization(env.model, 0.0),
        ):
            pass
    finally:
        env.close()


def test_domain_randomization_rejects_a_black_box_model() -> None:
    with (
        pytest.raises(AttributeError, match="not a MuJoCo model"),
        robustness.domain_randomization(object(), 1.1),
    ):
        pass


def test_domain_randomized_eval_perturbs_the_model_per_scale(monkeypatch) -> None:
    """Each condition must really see a different model.

    The previous implementation looped over scales and re-evaluated the same
    unmodified environment, so every condition returned an identical number and
    the sweep measured nothing. Grouping the observed mass by condition and
    asserting the groups are the nominal mass times their own scale is the
    property that would have caught it.
    """
    env = build_single_env(ALLEGRO_ENV_ID)
    evaluate_module = importlib.import_module("human2robot.evaluation.evaluate")
    monkeypatch.setattr(evaluate_module, "build_single_env", lambda env_id: env)

    masses: list[float] = []
    nominal = float(env.model.body_mass[0])
    scales = (0.8, 1.0, 1.2)
    result = robustness.domain_randomized_eval(
        _RecordingPolicy(env, masses),
        ALLEGRO_ENV_ID,
        episodes=1,
        seed=0,
        scales=scales,
    )

    assert set(result) == {
        "robustness_mean",
        "robustness_spread",
        "scale_0.8",
        "scale_1.0",
        "scale_1.2",
    }
    # The model is observed on every policy call, and conditions need not last
    # the same number of steps, so the property is the set of states visited
    # rather than how many times each was reached.
    assert masses, "the policy never ran, so nothing was observed"
    assert {round(v, 9) for v in masses} == {round(nominal * s, 9) for s in scales}


def test_domain_randomized_eval_uses_default_scales(monkeypatch) -> None:
    env = build_single_env(ALLEGRO_ENV_ID)
    evaluate_module = importlib.import_module("human2robot.evaluation.evaluate")
    monkeypatch.setattr(evaluate_module, "build_single_env", lambda env_id: env)

    masses: list[float] = []
    result = robustness.domain_randomized_eval(
        _RecordingPolicy(env, masses), ALLEGRO_ENV_ID, episodes=1
    )
    assert masses, "the policy never ran, so nothing was observed"
    assert robustness.DEFAULT_SCALES == (0.8, 1.0, 1.2)
    assert {k for k in result if k.startswith("scale_")} == {
        "scale_0.8",
        "scale_1.0",
        "scale_1.2",
    }
    assert result["robustness_mean"] == pytest.approx(
        np.mean([result[f"scale_{s:.1f}"] for s in robustness.DEFAULT_SCALES])
    )


def test_domain_randomized_eval_rejects_a_black_box_env() -> None:
    """There is nothing to perturb, so there is no honest number to return."""
    with pytest.raises(ValueError, match="no MuJoCo model"):
        robustness.domain_randomized_eval(object(), "Pendulum-v1", episodes=1)


def test_domain_randomized_eval_rejects_empty_scales() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        robustness.domain_randomized_eval(object(), ALLEGRO_ENV_ID, scales=())


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
