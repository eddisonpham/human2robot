"""Sanity tests for the floating Allegro Tier B environment."""

import numpy as np

from dynhand.envs.allegro import ENV_ID, AllegroPickupEnv


def test_allegro_env_contract_and_registration() -> None:
    import gymnasium

    assert ENV_ID in gymnasium.registry
    env = gymnasium.make(ENV_ID)
    obs, info = env.reset(seed=0)
    assert env.action_space.shape == (22,)
    assert env.observation_space.shape == (64,)
    assert obs.shape == (64,)
    assert np.isfinite(obs).all()
    next_obs, reward, terminated, truncated, info = env.step(
        np.zeros(22, dtype=np.float32)
    )
    assert next_obs.shape == (64,)
    assert np.isfinite(next_obs).all()
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    env.close()


def test_allegro_env_is_seed_deterministic() -> None:
    first = AllegroPickupEnv()
    second = AllegroPickupEnv()
    obs_a, _ = first.reset(seed=7)
    obs_b, _ = second.reset(seed=7)
    assert np.array_equal(obs_a, obs_b)
    action = np.linspace(-1.0, 1.0, 22, dtype=np.float32)
    for _ in range(10):
        step_a = first.step(action)
        step_b = second.step(action)
        assert np.allclose(step_a[0], step_b[0])
        assert step_a[1:] == step_b[1:]
    first.close()
    second.close()


def test_allegro_random_rollout_is_finite() -> None:
    env = AllegroPickupEnv(max_episode_steps=100)
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(100):
        obs, reward, terminated, truncated, _ = env.step(
            rng.uniform(-1.0, 1.0, size=22).astype(np.float32)
        )
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        if terminated or truncated:
            env.reset(seed=3)
    env.close()
