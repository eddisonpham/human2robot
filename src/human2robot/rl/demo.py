"""Demonstration loading for Minari datasets and local .npz demos.

Transitions are flattened to plain arrays so they can be inserted into the
replay buffer and consumed by BC without any environment-specific handling.
"""

from __future__ import annotations

import numpy as np


def _flatten_obs(obs: np.ndarray | dict[str, np.ndarray]) -> np.ndarray:
    """Flatten a Minari observation into a 2-D array of (T, D)."""
    if isinstance(obs, np.ndarray):
        return obs
    if isinstance(obs, dict):
        return np.concatenate([np.asarray(v) for v in obs.values()], axis=-1)
    raise TypeError(f"unexpected observation type: {type(obs)}")


def load_minari_transitions(dataset_id: str) -> dict[str, np.ndarray]:
    """Load transitions for demo-seeded replay buffers.

    Tries Minari dataset first; falls back to local .npz demo files when
    dataset_id is not a known Minari ID.
    """
    import minari

    dataset = None
    try:
        dataset = minari.load_dataset(dataset_id, download=True)
    except Exception:
        dataset = None

    if dataset is None:
        return _load_local_demos(dataset_id)

    obs_list, act_list, next_obs_list, rew_list, done_list = (
        [],
        [],
        [],
        [],
        [],
    )
    for episode in dataset:
        observations = _flatten_obs(episode.observations)
        actions = np.asarray(episode.actions, dtype=np.float32).reshape(
            len(episode.actions), -1
        )
        terminations = np.asarray(episode.terminations, dtype=bool)
        truncations = np.asarray(episode.terminations, dtype=bool)
        obs_list.append(observations[:-1])
        next_obs_list.append(observations[1:])
        act_list.append(actions)
        rew_list.append(np.asarray(episode.rewards, dtype=np.float32))
        done_list.append(terminations | truncations)
    return {
        "obs": np.concatenate(obs_list, axis=0),
        "acts": np.concatenate(act_list, axis=0),
        "next_obs": np.concatenate(next_obs_list, axis=0),
        "rewards": np.concatenate(rew_list, axis=0).reshape(-1, 1),
        "dones": np.concatenate(done_list, axis=0).reshape(-1, 1).astype(np.float32),
    }


def _load_local_demos(demo_dir: str) -> dict[str, np.ndarray]:
    """Load every .npz demo in *demo_dir* and flatten to transitions.

    Each stored trajectory is replayed through the environment's MuJoCo data
    so the returned observations match what the live env would produce.
    """
    from human2robot.rl._local_demos import _load_local_demos as _impl

    return _impl(demo_dir)


def seed_replay_buffer(buffer, demos: dict[str, np.ndarray]) -> int:
    """Insert demonstration transitions into a replay buffer, return count."""
    count = len(demos["obs"])
    buffer.add_batch(
        demos["obs"],
        demos["acts"],
        demos["next_obs"],
        demos["rewards"],
        demos["dones"],
    )
    return count
