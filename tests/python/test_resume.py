"""Tests for resuming training from checkpoints."""

import numpy as np
import torch

from dynhand.config.schema import EvalConfig, ExperimentConfig, SACConfig
from dynhand.envs.record import RunRecorder
from dynhand.rl.replay import ReplayBuffer
from dynhand.rl.train import _load_resume_checkpoint, _rng_state, train
from dynhand.utils.seed import seed_everything


def make_config(
    tmp_path, total_steps: int, exp_id: str = "resume_test"
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=exp_id,
        env_id="Pendulum-v1",
        seed=0,
        condition="A",
        total_env_steps=total_steps,
        num_envs=2,
        start_steps=40,
        results_dir=str(tmp_path),
        checkpoint_interval=100,
        sac=SACConfig(batch_size=32, buffer_size=2000, hidden_dim=32),
        eval=EvalConfig(interval_steps=100, episodes=2),
    )


def test_buffer_state_roundtrip() -> None:
    seed_everything(0)
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(2, 1, 10, rng)
    for i in range(6):
        buf.add(
            np.full(2, i, dtype=np.float32), np.zeros(1), np.zeros(1), float(i), 0.0
        )
    state = buf.state_dict()
    buf2 = ReplayBuffer(2, 1, 10, np.random.default_rng(1))
    buf2.load_state(state)
    assert buf2.size == 6
    assert buf2.ptr == 6
    assert np.allclose(buf2.obs[:6], buf.obs[:6])
    assert np.allclose(buf2.rewards[:6], buf.rewards[:6])


def test_buffer_load_state_rejects_shape_mismatch() -> None:
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(2, 1, 10, rng)
    bad = ReplayBuffer(3, 1, 10, rng).state_dict()
    import pytest

    with pytest.raises(ValueError):
        buf.load_state(bad)


def test_load_resume_checkpoint_restores_sac(tmp_path) -> None:
    from dynhand.config.schema import SACConfig
    from dynhand.rl.sac import SAC

    seed_everything(0)
    sac = SAC(
        3,
        1,
        np.array([-2.0], dtype=np.float32),
        np.array([2.0], dtype=np.float32),
        SACConfig(hidden_dim=32),
        torch.device("cpu"),
    )
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(3, 1, 100, rng)
    recorder = RunRecorder(
        ExperimentConfig(experiment_id="ckpt_resume", results_dir=str(tmp_path)),
        str(tmp_path),
    )
    recorder.save_checkpoint(
        500,
        {
            "sac": sac.state_dict(),
            "buffer": buffer.state_dict(),
            "global_step": 500,
            "replay_rng": rng.bit_generator.state,
            "rng": _rng_state(),
        },
    )
    sac2 = SAC(
        3,
        1,
        np.array([-2.0], dtype=np.float32),
        np.array([2.0], dtype=np.float32),
        SACConfig(hidden_dim=32),
        torch.device("cpu"),
    )
    buffer2 = ReplayBuffer(3, 1, 100, rng)
    path = recorder.latest_checkpoint()
    assert path is not None
    step = _load_resume_checkpoint(path, sac2, buffer2, rng)
    assert step == 500
    checkpoint = torch.load(path, weights_only=False)
    assert "rng" in checkpoint
    obs = np.zeros(3, dtype=np.float32)
    assert np.allclose(
        sac.act(obs, deterministic=True), sac2.act(obs, deterministic=True)
    )


def test_train_resume_continues_and_skips_bc(tmp_path) -> None:
    config = make_config(tmp_path, total_steps=200, exp_id="resume_flow")
    train(config)
    checkpoints = sorted((tmp_path / "resume_flow" / "checkpoints").glob("step_*.pt"))
    assert len(checkpoints) >= 1

    config2 = make_config(tmp_path, total_steps=400, exp_id="resume_flow")
    metrics = train(config2, resume=True)
    assert "eval_return_mean" in metrics
    lines = (
        (tmp_path / "resume_flow" / "metrics.jsonl").read_text().strip().splitlines()
    )
    steps = [int(__import__("json").loads(line)["step"]) for line in lines]
    assert max(steps) == 400
    assert any(s == 300 for s in steps)


def test_train_resume_without_checkpoint_starts_fresh(tmp_path) -> None:
    config = make_config(tmp_path, total_steps=120, exp_id="resume_fresh")
    metrics = train(config, resume=True)
    assert "eval_return_mean" in metrics
