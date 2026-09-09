"""Tests for seeding, git info, and environment construction."""

import numpy as np
import pytest
import torch

from dynhand.envs.vec import make_env_fn, make_vec_env
from dynhand.utils.git_info import get_git_commit
from dynhand.utils.seed import seed_everything, set_torch_threads, worker_seed


def test_seed_everything_is_deterministic() -> None:
    seed_everything(7)
    a = [np.random.rand() for _ in range(3)]
    t = torch.rand(3)
    seed_everything(7)
    b = [np.random.rand() for _ in range(3)]
    t2 = torch.rand(3)
    assert a == b
    assert torch.equal(t, t2)


def test_worker_seeds_are_distinct() -> None:
    seeds = [worker_seed(0, i) for i in range(8)]
    assert len(set(seeds)) == 8
    assert worker_seed(5, 3) == 8


def test_set_torch_threads_caps(monkeypatch) -> None:
    set_torch_threads(4)
    assert torch.get_num_threads() <= 4


def test_get_git_commit_returns_string() -> None:
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) > 0


def test_get_git_commit_unknown_dir() -> None:
    assert get_git_commit("/nonexistent/repo") == "unknown"


def test_make_env_fn_creates_seeded_env() -> None:
    pytest.importorskip("gymnasium")
    env = make_env_fn("Pendulum-v1", seed=3)()
    obs, _ = env.reset(seed=3)
    assert env.observation_space.contains(obs)


def test_vec_env_worker_seeding_determinism() -> None:
    pytest.importorskip("gymnasium")
    envs = make_vec_env("Pendulum-v1", num_envs=2, seed=42)
    obs1, _ = envs.reset(seed=42)
    envs.close()
    envs2 = make_vec_env("Pendulum-v1", num_envs=2, seed=42)
    obs2, _ = envs2.reset(seed=42)
    envs2.close()
    assert np.allclose(obs1, obs2)


def test_vec_env_distinct_worker_seeds_differ() -> None:
    pytest.importorskip("gymnasium")
    envs = make_vec_env("Pendulum-v1", num_envs=3, seed=0)
    obs_a, _ = envs.reset(seed=0)
    envs.close()
    envs_b = make_vec_env("Pendulum-v1", num_envs=3, seed=1)
    obs_b, _ = envs_b.reset(seed=1)
    envs_b.close()
    assert not np.allclose(obs_a[:, 0], obs_b[:, 0])
