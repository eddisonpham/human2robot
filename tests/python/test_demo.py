"""Tests for demo loading and replay seeding using a synthetic dataset."""

import numpy as np
import pytest

from dynhand.rl.demo import _flatten_obs, seed_replay_buffer
from dynhand.rl.replay import ReplayBuffer


class FakeEpisode:
    def __init__(self, length: int, obs_dim: int, act_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.observations = rng.normal(size=(length + 1, obs_dim)).astype(np.float32)
        self.actions = rng.uniform(-1, 1, size=(length, act_dim)).astype(np.float32)
        self.rewards = rng.normal(size=length).astype(np.float32)
        self.terminations = np.zeros(length, dtype=bool)
        self.terminations[-1] = True
        self.truncations = np.zeros(length, dtype=bool)


class FakeDataset:
    def __init__(self, episodes: list[FakeEpisode]) -> None:
        self._episodes = episodes

    def __iter__(self):
        return iter(self._episodes)


def test_flatten_obs_2d() -> None:
    obs = np.zeros((5, 3), dtype=np.float32)
    assert _flatten_obs(obs).shape == (5, 3)


def test_flatten_obs_dict() -> None:
    obs = {"a": np.zeros((5, 2)), "b": np.zeros((5, 4))}
    flat = _flatten_obs(obs)
    assert flat.shape == (5, 6)


def test_seed_replay_buffer_counts_and_content() -> None:
    rng = np.random.default_rng(0)
    n = 12
    demos = {
        "obs": rng.normal(size=(n, 2)).astype(np.float32),
        "acts": rng.normal(size=(n, 1)).astype(np.float32),
        "next_obs": rng.normal(size=(n, 2)).astype(np.float32),
        "rewards": rng.normal(size=(n, 1)).astype(np.float32),
        "dones": np.zeros((n, 1), dtype=np.float32),
    }
    buf = ReplayBuffer(2, 1, 100, np.random.default_rng(1))
    count = seed_replay_buffer(buf, demos)
    assert count == n
    assert len(buf) == n
    batch = buf.sample(5)
    assert batch["obs"].shape == (5, 2)


def test_load_minari_transitions_with_monkeypatched_minari(monkeypatch) -> None:
    import dynhand.rl.demo as demo_module

    episodes = [FakeEpisode(length=6, obs_dim=3, act_dim=2, seed=s) for s in range(2)]
    fake_minari = type("M", (), {})()
    fake_minari.load_dataset = lambda dataset_id, download=True: FakeDataset(episodes)
    monkeypatch.setattr(demo_module, "load_dataset", None, raising=False)
    import sys

    monkeypatch.setitem(sys.modules, "minari", fake_minari)
    data = demo_module.load_minari_transitions("fake/id")
    assert data["obs"].shape == (12, 3)
    assert data["acts"].shape == (12, 2)
    assert data["next_obs"].shape == (12, 3)
    assert data["rewards"].shape == (12, 1)
    assert data["dones"].shape == (12, 1)
    assert data["dones"].sum() == 2


@pytest.mark.network
def test_minari_relocate_dataset_loads() -> None:
    minari = pytest.importorskip("minari")
    dataset = minari.load_dataset("D4RL/relocate/human-v2", download=True)
    episodes = list(dataset)
    assert len(episodes) > 0
    assert episodes[0].observations is not None
