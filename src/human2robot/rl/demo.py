"""Demonstration loading from Minari datasets.

Used for Tier A validation runs. Transitions are flattened to plain arrays
so they can be inserted into the replay buffer and consumed by BC without
any environment-specific handling.
"""

import numpy as np


def _flatten_obs(obs: object) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [
            np.asarray(v, dtype=np.float32).reshape(len(next(iter(obs.values()))), -1)
            for v in obs.values()
        ]
        return np.concatenate(parts, axis=-1)
    return np.asarray(obs, dtype=np.float32).reshape(len(obs), -1)


def load_minari_transitions(dataset_id: str) -> dict[str, np.ndarray]:
    """Load a Minari dataset and flatten episodes into transition arrays."""
    import minari

    dataset = minari.load_dataset(dataset_id, download=True)
    obs_list, act_list, next_obs_list, rew_list, done_list = [], [], [], [], []
    for episode in dataset:
        observations = _flatten_obs(episode.observations)
        actions = np.asarray(episode.actions, dtype=np.float32).reshape(
            len(episode.actions), -1
        )
        terminations = np.asarray(episode.terminations, dtype=bool)
        truncations = np.asarray(episode.truncations, dtype=bool)
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
