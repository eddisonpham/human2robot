"""Tests for the replay buffer and mixing schedules."""

import numpy as np

from dynhand.rl.replay import ReplayBuffer
from dynhand.rl.schedules import demo_ratio, sample_mixed


def make_rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_add_and_len() -> None:
    buf = ReplayBuffer(obs_dim=2, act_dim=1, capacity=10, rng=make_rng())
    assert len(buf) == 0
    buf.add(np.zeros(2), np.zeros(1), np.ones(2), 1.0, 0.0)
    assert len(buf) == 1


def test_capacity_wraparound() -> None:
    buf = ReplayBuffer(obs_dim=1, act_dim=1, capacity=4, rng=make_rng())
    for i in range(7):
        buf.add(np.full(1, i, dtype=np.float32), np.zeros(1), np.zeros(1), 0.0, 0.0)
    assert len(buf) == 4
    assert buf.ptr == 3
    assert np.all(buf.obs.ravel() == [4, 5, 6, 3])


def test_sample_shapes_and_values() -> None:
    buf = ReplayBuffer(obs_dim=2, act_dim=2, capacity=8, rng=make_rng())
    for i in range(8):
        buf.add(
            np.full(2, i, dtype=np.float32),
            np.full(2, i, dtype=np.float32),
            np.zeros(2),
            float(i),
            0.0,
        )
    batch = buf.sample(5)
    assert batch["obs"].shape == (5, 2)
    assert batch["acts"].shape == (5, 2)
    assert batch["next_obs"].shape == (5, 2)
    assert batch["rewards"].shape == (5, 1)
    assert batch["dones"].shape == (5, 1)
    assert set(batch["obs"][:, 0]).issubset(range(8))


def test_add_batch_matches_sequential_adds() -> None:
    rng = make_rng()
    obs = rng.normal(size=(5, 3)).astype(np.float32)
    acts = rng.normal(size=(5, 2)).astype(np.float32)
    next_obs = rng.normal(size=(5, 3)).astype(np.float32)
    rews = rng.normal(size=5).astype(np.float32)
    dones = np.zeros(5)

    buf_batch = ReplayBuffer(3, 2, 10, make_rng())
    buf_batch.add_batch(obs, acts, next_obs, rews, dones)

    buf_seq = ReplayBuffer(3, 2, 10, make_rng())
    for i in range(5):
        buf_seq.add(obs[i], acts[i], next_obs[i], float(rews[i]), float(dones[i]))

    assert np.allclose(buf_batch.obs[:5], buf_seq.obs[:5])
    assert np.allclose(buf_batch.acts[:5], buf_seq.acts[:5])


def test_demo_ratio_anneals_linearly() -> None:
    assert demo_ratio(0, 0.5, 1000) == 0.5
    assert demo_ratio(500, 0.5, 1000) == pytest_approx(0.25)
    assert demo_ratio(1000, 0.5, 1000) == 0.0
    assert demo_ratio(5000, 0.5, 1000) == 0.0


def test_demo_ratio_zero_anneal_steps() -> None:
    assert demo_ratio(0, 0.5, 0) == 0.0


def test_demo_ratio_zero_start() -> None:
    assert demo_ratio(100, 0.0, 1000) == 0.0


def pytest_approx(value: float) -> float:
    return value


def test_sample_mixed_ratios() -> None:
    main = ReplayBuffer(1, 1, 100, make_rng())
    demo = ReplayBuffer(1, 1, 100, make_rng())
    for i in range(10):
        main.add(
            np.full(1, 10 + i, dtype=np.float32), np.zeros(1), np.zeros(1), 0.0, 0.0
        )
        demo.add(np.full(1, i, dtype=np.float32), np.zeros(1), np.zeros(1), 0.0, 0.0)

    batch = sample_mixed(main, demo, batch_size=10, ratio=0.3, rng=make_rng())
    assert batch["obs"].shape[0] == 10
    values = batch["obs"].ravel()
    demo_count = int((values < 10).sum())
    assert 1 <= demo_count <= 5

    batch_all = sample_mixed(main, demo, batch_size=4, ratio=1.0, rng=make_rng())
    assert (batch_all["obs"] < 10).all()

    batch_none = sample_mixed(main, demo, batch_size=4, ratio=0.0, rng=make_rng())
    assert (batch_none["obs"] >= 10).all()
