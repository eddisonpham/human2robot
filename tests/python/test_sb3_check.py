"""Tests for the SB3 cross-check wrapper."""

import sys
from pathlib import Path

from human2robot.evaluation import sb3_check


class FakeModel:
    saved_to = None

    def learn(self, total_timesteps: int, reset_num_timesteps: bool = True) -> None:
        assert total_timesteps > 0
        assert reset_num_timesteps is False

    def save(self, path: str) -> None:
        FakeModel.saved_to = path


def test_run_sb3_check_writes_manifest_and_metrics(tmp_path: Path, monkeypatch) -> None:
    import gymnasium

    fake_sb3 = type("sb3", (), {})()
    fake_sb3.SAC = lambda *args, **kwargs: FakeModel()

    class FakeVecEnv(list):
        def close(self) -> None:
            pass

    fake_sb3_common = type("sb3_common", (), {})()
    fake_sb3_common.make_vec_env = lambda *args, **kwargs: FakeVecEnv()
    fake_sb3_common.Monitor = lambda env: env
    fake_sb3_common.evaluate_policy = (
        lambda model, env, n_eval_episodes, deterministic: (3.5, 0.5)
    )

    class FakeGymEnv:
        def close(self) -> None:
            pass

    monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", type("m", (), {})())
    monkeypatch.setitem(
        sys.modules, "stable_baselines3.common.evaluation", fake_sb3_common
    )
    monkeypatch.setitem(
        sys.modules, "stable_baselines3.common.monitor", fake_sb3_common
    )
    monkeypatch.setitem(
        sys.modules, "stable_baselines3.common.vec_env", fake_sb3_common
    )
    monkeypatch.setattr(gymnasium, "make", lambda env_id: FakeGymEnv())

    metrics = sb3_check.run_sb3_check(
        env_id="Pendulum-v1",
        total_timesteps=1000,
        seed=0,
        num_envs=2,
        output_dir=str(tmp_path),
        eval_interval_steps=400,
    )
    assert metrics["eval_return_mean"] == 3.5
    assert metrics["eval_return_std"] == 0.5
    run_dir = tmp_path / "sb3_check"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert len((run_dir / "metrics.jsonl").read_text().strip().splitlines()) == 3
    assert FakeModel.saved_to.startswith(str(run_dir))


def test_sb3_cli_main_forwards_config_values(tmp_path, monkeypatch, capsys) -> None:
    """The CLI must translate a Human2Robot config into run_sb3_check kwargs."""
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"eval_return_mean": 1.0}

    monkeypatch.setattr(sb3_check, "run_sb3_check", fake_run)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "experiment_id: sb3_cli\n"
        "env_id: Pendulum-v1\n"
        "seed: 3\n"
        "total_env_steps: 12345\n"
        "num_envs: 4\n"
        "start_steps: 500\n"
        "checkpoint_interval: 1000\n"
        "sac:\n"
        "  batch_size: 32\n"
        "  buffer_size: 2000\n"
        "  utd_ratio: 2\n"
        "eval:\n"
        "  interval_steps: 777\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "human2robot-sb3",
            "--config",
            str(config_path),
            "--run-name",
            "cli_run",
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    sb3_check.main()
    assert captured["env_id"] == "Pendulum-v1"
    assert captured["total_timesteps"] == 12345
    assert captured["seed"] == 3
    assert captured["num_envs"] == 4
    assert captured["eval_interval_steps"] == 777
    assert captured["learning_starts"] == 500
    assert captured["batch_size"] == 32
    assert captured["buffer_size"] == 2000
    assert captured["gradient_steps"] == 2
    assert captured["run_name"] == "cli_run"
    assert captured["output_dir"] == str(tmp_path / "out")
    assert "eval_return_mean" in capsys.readouterr().out


def test_sb3_cli_timestep_override_wins(tmp_path, monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        sb3_check,
        "run_sb3_check",
        lambda **kwargs: captured.update(kwargs) or {"eval_return_mean": 0.0},
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "experiment_id: sb3_cli\nenv_id: Pendulum-v1\nseed: 0\n"
        "total_env_steps: 1000\nnum_envs: 1\nstart_steps: 10\n"
        "checkpoint_interval: 100\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "human2robot-sb3",
            "--config",
            str(config_path),
            "--total-timesteps",
            "999",
        ],
    )
    sb3_check.main()
    assert captured["total_timesteps"] == 999


def test_sb3_cli_defaults_output_dir_to_config(tmp_path, monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        sb3_check,
        "run_sb3_check",
        lambda **kwargs: captured.update(kwargs) or {"eval_return_mean": 0.0},
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "experiment_id: sb3_cli\nenv_id: Pendulum-v1\nseed: 0\n"
        "total_env_steps: 1000\nnum_envs: 1\nstart_steps: 10\n"
        f"checkpoint_interval: 100\nresults_dir: {tmp_path / 'defaulted'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["human2robot-sb3", "--config", str(config_path)])
    sb3_check.main()
    assert captured["output_dir"] == str(tmp_path / "defaulted")
    assert captured["run_name"] is None
