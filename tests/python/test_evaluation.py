"""Tests for evaluation helpers and the run recorder."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from dynhand.config.schema import ExperimentConfig
from dynhand.envs.record import RunRecorder
from dynhand.evaluation.evaluate import build_single_env, evaluate


def test_build_single_env_flattens_dict_obs() -> None:
    pytest.importorskip("gymnasium_robotics")
    env = build_single_env("AdroitHandRelocate-v1")
    assert len(env.observation_space.shape) == 1
    env.close()
    env2 = build_single_env("Pendulum-v1")
    assert env2.observation_space.shape == (3,)
    env2.close()


def test_evaluate_runs_deterministic_episodes() -> None:
    torch = __import__("torch")
    from dynhand.config.schema import SACConfig
    from dynhand.rl.sac import SAC

    config = SACConfig(hidden_dim=32)
    sac = SAC(
        3,
        1,
        np.array([-2.0], dtype=np.float32),
        np.array([2.0], dtype=np.float32),
        config,
        torch.device("cpu"),
    )
    m1 = evaluate(sac, "Pendulum-v1", episodes=2, seed=0)
    m2 = evaluate(sac, "Pendulum-v1", episodes=2, seed=0)
    assert m1 == m2
    assert np.isfinite(m1["eval_return_mean"])


def test_run_recorder_creates_manifest_files(tmp_path: Path) -> None:
    config = ExperimentConfig(experiment_id="recorder_test", results_dir=str(tmp_path))
    recorder = RunRecorder(config, config.results_dir)
    assert recorder.run_dir == tmp_path / "recorder_test"
    assert (recorder.run_dir / "config.yaml").exists()
    assert (recorder.run_dir / "git_commit.txt").exists()
    assert (recorder.run_dir / "system_info.json").exists()
    assert (recorder.run_dir / "metrics.jsonl").exists()
    saved = yaml.safe_load((recorder.run_dir / "config.yaml").read_text())
    assert saved["experiment_id"] == "recorder_test"


def test_run_recorder_logs_metrics(tmp_path: Path) -> None:
    config = ExperimentConfig(experiment_id="metrics_test", results_dir=str(tmp_path))
    recorder = RunRecorder(config, config.results_dir)
    recorder.log_metrics(100, {"reward": 1.5})
    recorder.log_metrics(200, {"reward": 2.5})
    lines = (recorder.run_dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[1])
    assert record == {"step": 200, "reward": 2.5}


def test_run_recorder_saves_checkpoint(tmp_path: Path) -> None:
    torch = __import__("torch")
    config = ExperimentConfig(experiment_id="ckpt_test", results_dir=str(tmp_path))
    recorder = RunRecorder(config, config.results_dir)
    path = recorder.save_checkpoint(500, {"tensor": torch.zeros(3)})
    assert path.exists()
    loaded = torch.load(path, weights_only=True)
    assert torch.equal(loaded["tensor"], torch.zeros(3))


def test_latest_checkpoint_sorts_numerically(tmp_path: Path) -> None:
    config = ExperimentConfig(experiment_id="ckpt_order", results_dir=str(tmp_path))
    recorder = RunRecorder(config, config.results_dir)
    recorder.save_checkpoint(50000, {"step": 50000})
    recorder.save_checkpoint(100000, {"step": 100000})
    recorder.save_checkpoint(5000, {"step": 5000})
    latest = recorder.latest_checkpoint()
    assert latest is not None and latest.stem == "step_100000"


def test_latest_checkpoint_empty_dir(tmp_path: Path) -> None:
    config = ExperimentConfig(experiment_id="ckpt_empty", results_dir=str(tmp_path))
    recorder = RunRecorder(config, config.results_dir)
    assert recorder.latest_checkpoint() is None
