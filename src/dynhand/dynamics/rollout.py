"""Synthetic transition generation for model-based SAC."""

import numpy as np

from dynhand.dynamics.ensemble import DynamicsEnsemble


def synthetic_transitions(
    model: DynamicsEnsemble,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray | None = None,
    dones: np.ndarray | None = None,
    physics_deltas: np.ndarray | None = None,
    member: int | None = None,
) -> dict[str, np.ndarray]:
    """Predict a replay-compatible transition batch."""
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    next_states = model.predict(
        states,
        actions,
        physics_deltas=physics_deltas,
        member=member,
    )
    if rewards is None:
        rewards = model.predict_rewards(states, actions, member=member)
    if dones is None:
        dones = np.zeros(len(states), dtype=np.float32)
    return {
        "obs": states,
        "acts": actions,
        "next_obs": next_states.astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
        "dones": np.asarray(dones, dtype=np.float32).reshape(-1, 1),
    }
