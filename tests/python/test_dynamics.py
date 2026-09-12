"""Tests for standalone learned dynamics."""

import numpy as np
import pytest

from dynhand.dynamics.ensemble import DynamicsEnsemble
from dynhand.dynamics.rollout import synthetic_transitions


def dataset(size: int = 256, state_dim: int = 4, action_dim: int = 2):
    rng = np.random.default_rng(2)
    states = rng.normal(size=(size, state_dim)).astype(np.float32)
    actions = rng.normal(size=(size, action_dim)).astype(np.float32)
    delta = np.zeros_like(states)
    delta[:, :action_dim] = actions
    next_states = states + delta
    return states, actions, next_states, delta


def test_blackbox_ensemble_learns_transition_and_rollout() -> None:
    states, actions, next_states, _ = dataset()
    model = DynamicsEnsemble(4, 2, ensemble_size=2, hidden_dim=32)
    rewards = actions[:, 0] - actions[:, 1]
    losses = model.fit(
        states,
        actions,
        next_states,
        rewards=rewards,
        epochs=100,
        batch_size=64,
    )

    assert len(losses) == 2
    metrics = model.evaluate(states, actions, next_states, horizons=(1, 5))
    assert metrics.one_step_mse < 0.05
    assert set(metrics.rollout_errors) == {1, 5}


def test_residual_requires_physics_deltas() -> None:
    states, actions, next_states, delta = dataset()
    model = DynamicsEnsemble(4, 2, ensemble_size=1, hidden_dim=16, mode="residual")
    with pytest.raises(ValueError, match="physics_deltas"):
        model.fit(states, actions, next_states, epochs=1)
    model.fit(states, actions, next_states, epochs=3, physics_deltas=delta)
    prediction = model.predict(states[:4], actions[:4], physics_deltas=delta[:4])
    assert prediction.shape == (4, 4)


def test_synthetic_transitions_are_replay_compatible() -> None:
    states, actions, next_states, _ = dataset(size=16)
    model = DynamicsEnsemble(4, 2, ensemble_size=1, hidden_dim=16)
    model.fit(
        states,
        actions,
        next_states,
        rewards=actions[:, 0] - actions[:, 1],
        epochs=2,
    )
    batch = synthetic_transitions(model, states[:8], actions[:8])
    assert set(batch) == {"obs", "acts", "next_obs", "rewards", "dones"}
    assert batch["next_obs"].shape == (8, 4)
    assert batch["rewards"].shape == (8, 1)


def test_invalid_mode_and_shapes_fail() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        DynamicsEnsemble(2, 1, mode="bad")
    states, actions, next_states, _ = dataset()
    model = DynamicsEnsemble(4, 2, ensemble_size=1)
    with pytest.raises(ValueError, match="wrong shape"):
        model.fit(states[:, :3], actions, next_states)
