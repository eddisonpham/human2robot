"""Tests for pausing and resuming a training run.

Pausing must not cost the run its trained dynamics ensemble: without that, a
resume refits every member from scratch, which dominates the wall-clock cost of
resuming at all.
"""

import json
import signal
from pathlib import Path

import numpy as np
import pytest
import torch

from human2robot.config.schema import (
    DynamicsAugConfig,
    EvalConfig,
    ExperimentConfig,
    SACConfig,
)
from human2robot.dynamics.ensemble import DynamicsEnsemble
from human2robot.rl.replay import ReplayBuffer
from human2robot.rl.sac import SAC
from human2robot.rl.train import _checkpoint_state, _load_resume_checkpoint, train

ROOT = Path(__file__).resolve().parents[2]


def _ensemble(seed: int = 0) -> DynamicsEnsemble:
    torch.manual_seed(seed)
    return DynamicsEnsemble(
        state_dim=6, action_dim=3, ensemble_size=2, mode="blackbox", hidden_dim=16
    )


def test_ensemble_state_dict_round_trips_predictions() -> None:
    ensemble = _ensemble()
    states = np.random.default_rng(0).normal(size=(64, 6)).astype(np.float32)
    actions = np.random.default_rng(1).normal(size=(64, 3)).astype(np.float32)
    next_states = np.random.default_rng(2).normal(size=(64, 6)).astype(np.float32)
    rewards = np.random.default_rng(3).normal(size=64).astype(np.float32)
    ensemble.fit(states, actions, next_states, rewards=rewards, epochs=1, batch_size=16)

    restored = _ensemble()
    restored.load_state_dict(ensemble.state_dict())

    probe = np.random.default_rng(4).normal(size=(8, 6)).astype(np.float32)
    probe_actions = np.random.default_rng(5).normal(size=(8, 3)).astype(np.float32)
    assert np.allclose(
        ensemble.predict(probe, probe_actions),
        restored.predict(probe, probe_actions),
        atol=0.0,
    )
    assert np.allclose(
        ensemble.predict_rewards(probe, probe_actions),
        restored.predict_rewards(probe, probe_actions),
        atol=0.0,
    )


def test_ensemble_restores_fit_flags_and_normalizers() -> None:
    ensemble = _ensemble()
    states = np.random.default_rng(0).normal(size=(64, 6)).astype(np.float32)
    actions = np.random.default_rng(1).normal(size=(64, 3)).astype(np.float32)
    next_states = np.random.default_rng(2).normal(size=(64, 6)).astype(np.float32)
    rewards = np.random.default_rng(3).normal(size=64).astype(np.float32)
    ensemble.fit(states, actions, next_states, rewards=rewards, epochs=1, batch_size=16)

    restored = _ensemble()
    assert not restored.state_dict()["fitted"]
    restored.load_state_dict(ensemble.state_dict())

    assert restored.state_dict()["fitted"] is True
    assert np.allclose(restored.state_mean, ensemble.state_mean)
    assert np.allclose(restored.delta_std, ensemble.delta_std)
    assert restored.reward_mean == pytest.approx(ensemble.reward_mean)


def test_checkpoint_state_includes_dynamics_and_loads_it(tmp_path) -> None:
    torch.manual_seed(0)
    sac = SAC(
        6,
        3,
        np.full(3, -1.0, np.float32),
        np.full(3, 1.0, np.float32),
        SACConfig(hidden_dim=16),
        torch.device("cpu"),
    )
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(6, 3, 100, rng)
    buffer.add_batch(
        np.zeros((10, 6), np.float32),
        np.zeros((10, 3), np.float32),
        np.zeros((10, 6), np.float32),
        np.zeros(10, np.float32),
        np.zeros(10, np.float32),
    )
    ensemble = _ensemble()

    state = _checkpoint_state(sac, buffer, ensemble, 4321, rng)
    assert "dynamics" in state
    path = tmp_path / "ckpt.pt"
    torch.save(state, path)

    fresh_sac = SAC(
        6,
        3,
        np.full(3, -1.0, np.float32),
        np.full(3, 1.0, np.float32),
        SACConfig(hidden_dim=16),
        torch.device("cpu"),
    )
    fresh_buffer = ReplayBuffer(6, 3, 100, np.random.default_rng(1))
    fresh_ensemble = _ensemble(seed=99)
    step = _load_resume_checkpoint(
        path, fresh_sac, fresh_buffer, np.random.default_rng(1), fresh_ensemble
    )
    assert step == 4321
    assert fresh_ensemble.state_dict()["fitted"] == ensemble.state_dict()["fitted"]


def test_checkpoint_omits_dynamics_when_absent() -> None:
    torch.manual_seed(0)
    sac = SAC(
        6,
        3,
        np.full(3, -1.0, np.float32),
        np.full(3, 1.0, np.float32),
        SACConfig(hidden_dim=16),
        torch.device("cpu"),
    )
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(6, 3, 100, rng)
    assert "dynamics" not in _checkpoint_state(sac, buffer, None, 1, rng)


def test_resume_from_a_checkpoint_without_dynamics_still_works(tmp_path) -> None:
    """Older checkpoints predate ensemble persistence and must remain loadable."""
    torch.manual_seed(0)
    sac = SAC(
        6,
        3,
        np.full(3, -1.0, np.float32),
        np.full(3, 1.0, np.float32),
        SACConfig(hidden_dim=16),
        torch.device("cpu"),
    )
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(6, 3, 100, rng)
    path = tmp_path / "old.pt"
    torch.save(_checkpoint_state(sac, buffer, None, 777, rng), path)

    fresh = _ensemble()
    step = _load_resume_checkpoint(path, sac, buffer, rng, fresh)
    assert step == 777
    assert fresh.state_dict()["fitted"] is False


def test_sigterm_stops_training_at_its_current_step(tmp_path) -> None:
    """A signalled run must stop cleanly and leave a resumable checkpoint."""
    import human2robot.rl.train as train_module

    handlers = {}
    original = train_module.signal.signal
    sigterm = getattr(signal, "SIGTERM", None)

    config = ExperimentConfig(
        experiment_id="pause_sig",
        env_id="Pendulum-v1",
        seed=0,
        condition="C",
        total_env_steps=400_000,
        num_envs=2,
        start_steps=10,
        results_dir=str(tmp_path),
        checkpoint_interval=200_000,
        sac=SACConfig(batch_size=16, buffer_size=1000, hidden_dim=16),
        dynamics_aug=DynamicsAugConfig(
            ensemble_size=1, retrain_every_steps=50, batch_size=16, synthetic_ratio=0.5
        ),
        eval=EvalConfig(interval_steps=200_000, episodes=1),
    )

    try:
        # Raise the pause flag as soon as the trainer installs its handler,
        # which mimics SIGTERM arriving mid-run.
        def arm(sig, handler):
            result = original(sig, handler)
            # Arm only on install; the later restore passes a Handlers object.
            if sig == sigterm and callable(handler):
                handlers["handler"] = handler
            return result

        train_module.signal.signal = arm
        result = train(config, _pause_after=40)
    finally:
        train_module.signal.signal = original

    assert "handler" in handlers, "SIGTERM handler was not installed"
    status = json.loads((tmp_path / "pause_sig" / "run_status.json").read_text())
    assert status["status"] == "paused"
    assert status["step"] >= 40
    assert status["step"] < 400_000
    checkpoints = sorted((tmp_path / "pause_sig" / "checkpoints").glob("step_*.pt"))
    assert checkpoints
    assert int(checkpoints[-1].stem.split("_")[1]) == status["step"]
    assert result == {}


def test_paused_status_is_written_with_the_step(tmp_path) -> None:
    from human2robot.config.schema import ExperimentConfig as EC
    from human2robot.envs.record import RunRecorder

    config = EC(
        experiment_id="paused_status", env_id="Pendulum-v1", results_dir=str(tmp_path)
    )
    recorder = RunRecorder(config, str(tmp_path))
    try:
        recorder.mark_paused(12345)
    finally:
        recorder.close()
    status = json.loads((tmp_path / "paused_status" / "run_status.json").read_text())
    assert status["status"] == "paused"
    assert status["step"] == 12345


def test_training_loop_installs_and_restores_signal_handlers() -> None:
    """Handlers must be restored so a second train() call is not left paused."""
    before = signal.getsignal(signal.SIGTERM)
    config = ExperimentConfig(
        experiment_id="sig_restore",
        env_id="Pendulum-v1",
        seed=0,
        total_env_steps=60,
        num_envs=2,
        start_steps=10,
        results_dir="results",
        checkpoint_interval=50,
        sac=SACConfig(batch_size=16, buffer_size=500, hidden_dim=16),
        eval=EvalConfig(interval_steps=50, episodes=1),
    )
    train(config)
    assert signal.getsignal(signal.SIGTERM) is before


def test_pause_sentinel_is_polled_and_cleared_on_resume(tmp_path) -> None:
    """The PAUSE file is the only pause channel Windows offers a detached run."""
    from human2robot.config.schema import ExperimentConfig as EC
    from human2robot.envs.record import RunRecorder

    config = EC(
        experiment_id="sentinel", env_id="Pendulum-v1", results_dir=str(tmp_path)
    )
    recorder = RunRecorder(config, str(tmp_path))
    try:
        assert recorder.pause_requested() is False
        recorder.pause_path.write_text("", encoding="utf-8")
        assert recorder.pause_requested() is True
        recorder.clear_pause()
        assert recorder.pause_requested() is False
        # Clearing twice must not raise, since resume may run on a clean dir.
        recorder.clear_pause()
    finally:
        recorder.close()


def test_rewind_metrics_drops_records_after_the_checkpoint(tmp_path) -> None:
    """A resume must not leave duplicate or out-of-order metric records."""
    from human2robot.config.schema import ExperimentConfig as EC
    from human2robot.envs.record import RunRecorder

    config = EC(experiment_id="rewind", env_id="Pendulum-v1", results_dir=str(tmp_path))
    recorder = RunRecorder(config, str(tmp_path))
    try:
        for step in (0, 1000, 2000, 3000, 4000):
            recorder.log_metrics(step, {"eval_return_mean": float(step)})
        assert recorder.rewind_metrics(2000) == 2
        steps = [
            json.loads(line)["step"]
            for line in (tmp_path / "rewind" / "metrics.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        assert steps == [0, 1000, 2000]
        # A second rewind at the same step is a no-op.
        assert recorder.rewind_metrics(2000) == 0
    finally:
        recorder.close()


def test_rewind_metrics_keeps_malformed_lines_for_the_audit(tmp_path) -> None:
    """Unparseable lines are preserved so the audit still reports them."""
    from human2robot.config.schema import ExperimentConfig as EC
    from human2robot.envs.record import RunRecorder

    config = EC(
        experiment_id="malformed", env_id="Pendulum-v1", results_dir=str(tmp_path)
    )
    recorder = RunRecorder(config, str(tmp_path))
    try:
        recorder.log_metrics(100, {"eval_return_mean": -1.0})
        with open(tmp_path / "malformed" / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write("not json\n")
        recorder.log_metrics(500, {"eval_return_mean": -2.0})
        assert recorder.rewind_metrics(100) == 1
        text = (tmp_path / "malformed" / "metrics.jsonl").read_text(encoding="utf-8")
        assert "not json" in text
        assert text.count('"step": 500') == 0
    finally:
        recorder.close()
