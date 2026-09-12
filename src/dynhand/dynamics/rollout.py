"""Synthetic transition generation for model-based SAC."""

import numpy as np

from dynhand.dynamics.ensemble import DynamicsEnsemble


def synthetic_transitions(
    model: DynamicsEnsemble,
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    physics_deltas: np.ndarray | None = None,
    member: int | None = None,
) -> dict[str, np.ndarray]:
    """Predict next states and return a replay-compatible transition batch."""
    next_states = model.predict(
        states,
        actions,
        physics_deltas=physics_deltas,
        member=member,
    )
    return {
        "obs": np.asarray(states, dtype=np.float32),
        "acts": np.asarray(actions, dtype=np.float32),
        "next_obs": next_states.astype(np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32).reshape(-1, 1),
        "dones": np.asarray(dones, dtype=np.float32).reshape(-1, 1),
    }
