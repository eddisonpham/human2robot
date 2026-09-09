"""Integration tests: demo-integrated training, eval CLI, BC early stopping."""

import importlib
import sys

import numpy as np
import torch
import yaml

from dynhand.config.schema import DemoConfig, EvalConfig, ExperimentConfig, SACConfig
from dynhand.rl.bc import BCTrainer
from dynhand.rl.sac import SAC
from dynhand.rl.train import train


def make_demo_arrays(n: int = 64) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "obs": rng.normal(size=(n, 3)).astype(np.float32),
        "acts": rng.uniform(-2, 2, size=(n, 1)).astype(np.float32),
        "next_obs": rng.normal(size=(n, 3)).astype(np.float32),
        "rewards": rng.normal(size=(n, 1)).astype(np.float32),
        "dones": np.zeros((n, 1), dtype=np.float32),
    }


def test_train_condition_b_demo_path(tmp_path, monkeypatch) -> None:
    train_mod = importlib.import_module("dynhand.rl.train")

    monkeypatch.setattr(
        train_mod, "load_minari_transitions", lambda dataset_id: make_demo_arrays()
    )
    config = ExperimentConfig(
        experiment_id="smoke_b",
        env_id="Pendulum-v1",
        seed=0,
        condition="B",
        total_env_steps=200,
        num_envs=2,
        start_steps=40,
        results_dir=str(tmp_path),
        checkpoint_interval=100,
        sac=SACConfig(batch_size=32, buffer_size=2000, hidden_dim=32),
        demo=DemoConfig(
            minari_dataset="fake/id",
            bc_epochs=2,
            bc_batch_size=32,
            demo_ratio_start=0.5,
            demo_ratio_anneal_steps=100,
        ),
        eval=EvalConfig(interval_steps=100, episodes=2),
    )
    assert config.demo.enabled is True
    metrics = train(config)
    assert "eval_return_mean" in metrics
    lines = (tmp_path / "smoke_B" / "metrics.jsonl").read_text().strip().splitlines()
    assert any("bc_train_loss" in line for line in lines)


def test_train_checkpoint_written_mid_run(tmp_path) -> None:
    config = ExperimentConfig(
        experiment_id="smoke_ckpt",
        env_id="Pendulum-v1",
        seed=0,
        condition="A",
        total_env_steps=200,
        num_envs=2,
        start_steps=40,
        results_dir=str(tmp_path),
        checkpoint_interval=100,
        sac=SACConfig(batch_size=32, buffer_size=2000, hidden_dim=32),
        eval=EvalConfig(interval_steps=100, episodes=2),
    )
    train(config)
    checkpoints = sorted((tmp_path / "smoke_ckpt" / "checkpoints").glob("*.pt"))
    assert len(checkpoints) >= 2
    loaded = torch.load(checkpoints[0], weights_only=False)
    assert "sac" in loaded and "global_step" in loaded


def test_eval_cli_prints_json(tmp_path, monkeypatch, capsys) -> None:
    from dynhand.evaluation.cli import main

    sac = SAC(
        3,
        1,
        np.array([-2.0], dtype=np.float32),
        np.array([2.0], dtype=np.float32),
        SACConfig(hidden_dim=32),
        torch.device("cpu"),
    )
    checkpoint = tmp_path / "ckpt.pt"
    torch.save({"sac": sac.state_dict()}, checkpoint)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "cli_test",
                "env_id": "Pendulum-v1",
                "sac": {"hidden_dim": 32},
                "eval": {"episodes": 2},
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["dynhand-eval", "--config", str(config_path), "--checkpoint", str(checkpoint)],
    )
    main()
    out = capsys.readouterr().out
    assert "eval_return_mean" in out


def test_bc_early_stopping_breaks(tmp_path) -> None:
    rng = np.random.default_rng(0)
    sac = SAC(
        3,
        1,
        np.array([-2.0], dtype=np.float32),
        np.array([2.0], dtype=np.float32),
        SACConfig(hidden_dim=32),
        torch.device("cpu"),
    )
    bc = BCTrainer(
        sac,
        rng.normal(size=(64, 3)).astype(np.float32),
        rng.uniform(-2, 2, size=(64, 1)).astype(np.float32),
        lr=0.0,
        holdout_fraction=0.25,
        rng=rng,
    )
    metrics = bc.train_epochs(epochs=10, batch_size=16, patience=1)
    assert metrics["bc_epochs_run"] < 10
