"""Checkpoint size and retention tests.

Checkpoints must stay small: a full-capacity replay buffer is hundreds of
megabytes, so serializing the whole preallocation makes runs fill the disk
and spike memory during every save.
"""

import numpy as np
import pytest

from human2robot.config.schema import ExperimentConfig
from human2robot.envs.record import RunRecorder
from human2robot.rl.replay import ReplayBuffer


def _filled_buffer(rows: int = 200, capacity: int = 10_000) -> ReplayBuffer:
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(64, 22, capacity, rng)
    for i in range(rows):
        buffer.add(
            np.full(64, i, dtype=np.float32),
            np.full(22, i, dtype=np.float32),
            np.full(64, i + 1, dtype=np.float32),
            float(i),
            0.0,
        )
    return buffer


def test_state_dict_slices_to_filled_region() -> None:
    buffer = _filled_buffer(rows=200, capacity=10_000)
    state = buffer.state_dict()
    assert state["obs"].shape == (200, 64)
    assert state["acts"].shape == (200, 22)
    assert state["next_obs"].shape == (200, 64)
    assert state["size"] == 200


def test_state_dict_is_far_smaller_than_capacity() -> None:
    buffer = _filled_buffer(rows=1_000, capacity=1_000_000)
    payload = sum(
        value.nbytes
        for value in buffer.state_dict().values()
        if hasattr(value, "nbytes")
    )
    full = sum(array.nbytes for array in (buffer.obs, buffer.acts, buffer.next_obs))
    assert payload < full / 10


def test_load_state_round_trips_into_larger_capacity() -> None:
    source = _filled_buffer(rows=200, capacity=10_000)
    target = ReplayBuffer(64, 22, 1_000_000, np.random.default_rng(1))
    target.load_state(source.state_dict())
    assert target.size == 200
    assert target.ptr == source.ptr
    assert np.array_equal(target.obs[:200], source.obs[:200])
    assert np.array_equal(target.acts[:200], source.acts[:200])
    assert np.array_equal(target.next_obs[:200], source.next_obs[:200])


def test_load_state_rejects_oversized_checkpoint() -> None:
    source = _filled_buffer(rows=200, capacity=10_000)
    target = ReplayBuffer(64, 22, 100, np.random.default_rng(1))
    with pytest.raises(ValueError, match="buffer shape mismatch"):
        target.load_state(source.state_dict())


def test_load_state_accepts_legacy_full_capacity_format() -> None:
    """Checkpoints written before the trim fix stored the whole capacity."""
    source = _filled_buffer(rows=200, capacity=10_000)
    legacy = {
        "obs": np.zeros((10_000, 64), dtype=np.float32),
        "acts": np.zeros((10_000, 22), dtype=np.float32),
        "next_obs": np.zeros((10_000, 64), dtype=np.float32),
        "rewards": np.zeros((10_000, 1), dtype=np.float32),
        "dones": np.zeros((10_000, 1), dtype=np.float32),
        "ptr": 200,
        "size": 200,
    }
    legacy["obs"][:200] = source.obs[:200]
    target = ReplayBuffer(64, 22, 1_000_000, np.random.default_rng(1))
    target.load_state(legacy)
    assert target.size == 200
    assert np.array_equal(target.obs[:200], source.obs[:200])


def test_save_checkpoint_keeps_only_newest(tmp_path) -> None:
    config = ExperimentConfig(experiment_id="prune", env_id="Pendulum-v1")
    recorder = RunRecorder(config, str(tmp_path))
    for step in range(100, 1000, 100):
        recorder.save_checkpoint(step, {"global_step": step})
    kept = sorted(
        path.name for path in (tmp_path / "prune" / "checkpoints").glob("step_*.pt")
    )
    assert kept == ["step_700.pt", "step_800.pt", "step_900.pt"]
    assert recorder.latest_checkpoint().name == "step_900.pt"
    recorder.close()
