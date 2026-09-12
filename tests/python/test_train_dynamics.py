"""Integration test for Condition C black-box augmentation."""

import json

from dynhand.config.schema import (
    DynamicsAugConfig,
    EvalConfig,
    ExperimentConfig,
    SACConfig,
)
from dynhand.rl.train import train


def test_condition_c_generates_synthetic_transitions(tmp_path) -> None:
    config = ExperimentConfig(
        experiment_id="smoke_C",
        env_id="Pendulum-v1",
        seed=0,
        condition="C",
        total_env_steps=220,
        num_envs=2,
        start_steps=40,
        results_dir=str(tmp_path),
        checkpoint_interval=200,
        sac=SACConfig(batch_size=16, buffer_size=1000, hidden_dim=16),
        dynamics_aug=DynamicsAugConfig(
            ensemble_size=1,
            retrain_every_steps=50,
            batch_size=16,
            synthetic_ratio=0.5,
        ),
        eval=EvalConfig(interval_steps=110, episodes=1),
    )
    metrics = train(config)
    assert "eval_return_mean" in metrics
    records = [
        json.loads(line)
        for line in (tmp_path / "smoke_C" / "metrics.jsonl").read_text().splitlines()
    ]
    assert any("dynamics_model_transitions" in record for record in records)
