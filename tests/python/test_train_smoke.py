"""End-to-end training smoke test on a small classic control task."""

import numpy as np

from dynhand.config.schema import (
    DemoConfig,
    DynamicsAugConfig,
    EvalConfig,
    ExperimentConfig,
    SACConfig,
)
from dynhand.rl.train import _subsample, train


def make_smoke_config(tmp_path, condition: str = "A") -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=f"smoke_test_{condition}",
        env_id="Pendulum-v1",
        seed=0,
        condition=condition,
        total_env_steps=300,
        num_envs=2,
        start_steps=50,
        results_dir=str(tmp_path),
        checkpoint_interval=200,
        sac=SACConfig(batch_size=32, buffer_size=2000, hidden_dim=32),
        demo=DemoConfig(),
        dynamics_aug=DynamicsAugConfig(),
        eval=EvalConfig(interval_steps=150, episodes=2),
    )


def test_train_condition_a_end_to_end(tmp_path) -> None:
    config = make_smoke_config(tmp_path)
    metrics = train(config)
    assert "eval_return_mean" in metrics
    assert np.isfinite(metrics["eval_return_mean"])
    run_dir = tmp_path / "smoke_test_A"
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "git_commit.txt").exists()
    assert (run_dir / "system_info.json").exists()
    assert (run_dir / "metrics.jsonl").exists()
    checkpoints = list((run_dir / "checkpoints").glob("*.pt"))
    assert len(checkpoints) >= 1


def test_train_rejects_residual_dynamics_until_physics_provider(tmp_path) -> None:
    config = make_smoke_config(tmp_path, condition="D")
    import pytest

    with pytest.raises(NotImplementedError, match="physics provider"):
        train(config)


def test_subsample_keeps_all_when_under_limit() -> None:
    rng = np.random.default_rng(0)
    demos = {"obs": np.zeros((10, 2)), "rewards": np.zeros(10)}
    out = _subsample(demos, 20, rng)
    assert len(out["obs"]) == 10


def test_subsample_caps_at_limit() -> None:
    rng = np.random.default_rng(0)
    demos = {"obs": np.arange(100).reshape(100, 1), "rewards": np.zeros(100)}
    out = _subsample(demos, 30, rng)
    assert len(out["obs"]) == 30
    assert set(out["obs"].ravel().tolist()).issubset(set(range(100)))
