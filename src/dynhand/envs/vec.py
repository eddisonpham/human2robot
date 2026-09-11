"""Gymnasium environment construction with per-worker seeding."""

import gymnasium
import gymnasium_robotics  # noqa: F401  (registers Adroit envs)
from gymnasium.vector import SyncVectorEnv, VectorEnv
from gymnasium.wrappers import FlattenObservation

from dynhand.utils.seed import worker_seed


def _worker(env_id: str, seed: int):
    def _init():
        env = gymnasium.make(env_id)
        if isinstance(env.observation_space, gymnasium.spaces.Dict):
            env = FlattenObservation(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return _init


def make_env_fn(env_id: str, seed: int = 0):
    """Return a zero-argument factory creating one seeded environment."""
    return _worker(env_id, seed)


def make_vec_env(env_id: str, num_envs: int, seed: int = 0) -> VectorEnv:
    """Create a synchronous vector env, worker i seeded with seed + i."""
    env_fns = [_worker(env_id, worker_seed(seed, i)) for i in range(num_envs)]
    return SyncVectorEnv(env_fns)
