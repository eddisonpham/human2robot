"""Deterministic policy evaluation in a fresh single environment."""

import numpy as np

from dynhand.rl.sac import SAC


def build_single_env(env_id: str):
    import gymnasium
    import gymnasium_robotics  # noqa: F401  (registers Adroit envs)
    from gymnasium.wrappers import FlattenObservation

    env = gymnasium.make(env_id)
    if isinstance(env.observation_space, gymnasium.spaces.Dict):
        env = FlattenObservation(env)
    return env


def evaluate(sac: SAC, env_id: str, episodes: int, seed: int) -> dict[str, float]:
    """Run deterministic evaluation episodes and return summary metrics."""
    env = build_single_env(env_id)
    env.action_space.seed(seed)
    returns, lengths = [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        total, steps = 0.0, 0
        while not done:
            action = sac.act(np.asarray(obs, dtype=np.float32), deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            total += float(reward)
            steps += 1
            done = term or trunc
        returns.append(total)
        lengths.append(steps)
    env.close()
    return {
        "eval_return_mean": float(np.mean(returns)),
        "eval_return_std": float(np.std(returns)),
        "eval_episode_steps": float(np.mean(lengths)),
    }
