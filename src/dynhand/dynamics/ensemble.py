"""Small ensemble models for MBPO-style state-delta prediction."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class _MLP(nn.Module):
    """MLP used for normalized state-delta and reward prediction."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Predict normalized targets."""
        return self.net(values)


@dataclass
class DynamicsMetrics:
    """Prediction and rollout error by horizon."""

    one_step_mse: float
    rollout_errors: dict[int, float]


class DynamicsEnsemble:
    """Bootstrap ensemble predicting next state and reward."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        ensemble_size: int = 5,
        mode: str = "blackbox",
        hidden_dim: int = 256,
        device: torch.device | None = None,
    ) -> None:
        if mode not in {"blackbox", "residual"}:
            raise ValueError(f"unsupported dynamics mode: {mode}")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mode = mode
        self.device = device or torch.device("cpu")
        input_dim = state_dim + action_dim
        self.models = [
            _MLP(input_dim, state_dim, hidden_dim).to(self.device)
            for _ in range(ensemble_size)
        ]
        self.reward_models = [
            _MLP(input_dim, 1, hidden_dim).to(self.device) for _ in range(ensemble_size)
        ]
        self.optimizers = [
            torch.optim.Adam(
                list(model.parameters()) + list(reward_model.parameters()), lr=3e-4
            )
            for model, reward_model in zip(self.models, self.reward_models, strict=True)
        ]
        self.state_mean = np.zeros(state_dim, dtype=np.float32)
        self.state_std = np.ones(state_dim, dtype=np.float32)
        self.delta_mean = np.zeros(state_dim, dtype=np.float32)
        self.delta_std = np.ones(state_dim, dtype=np.float32)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self._fitted = False
        self._reward_fitted = False

    def fit(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        epochs: int = 10,
        batch_size: int = 256,
        physics_deltas: np.ndarray | None = None,
        rewards: np.ndarray | None = None,
        seed: int = 0,
    ) -> list[float]:
        """Fit each model on a bootstrap sample of state transitions."""
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError("states have the wrong shape")
        if actions.shape != (len(states), self.action_dim):
            raise ValueError("actions have the wrong shape")
        if next_states.shape != states.shape:
            raise ValueError("next_states have the wrong shape")
        deltas = next_states - states
        if self.mode == "residual":
            if physics_deltas is None:
                raise ValueError("residual mode requires physics_deltas")
            deltas = deltas - np.asarray(physics_deltas, dtype=np.float32)
        self.state_mean = states.mean(axis=0)
        self.state_std = np.maximum(states.std(axis=0), 1e-6)
        self.delta_mean = deltas.mean(axis=0)
        self.delta_std = np.maximum(deltas.std(axis=0), 1e-6)
        features = np.concatenate(
            ((states - self.state_mean) / self.state_std, actions), axis=1
        )
        targets = (deltas - self.delta_mean) / self.delta_std
        reward_targets = None
        if rewards is not None:
            rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
            if len(rewards) != len(states):
                raise ValueError("rewards have the wrong shape")
            self.reward_mean = float(rewards.mean())
            self.reward_std = float(max(rewards.std(), 1e-6))
            reward_targets = (rewards - self.reward_mean) / self.reward_std
        rng = np.random.default_rng(seed)
        losses: list[float] = []
        for model, reward_model, optimizer in zip(
            self.models, self.reward_models, self.optimizers, strict=True
        ):
            indices = rng.integers(0, len(states), size=len(states))
            final_loss = 0.0
            for _ in range(epochs):
                order = rng.permutation(indices)
                for start in range(0, len(order), batch_size):
                    batch = order[start : start + batch_size]
                    x = torch.from_numpy(features[batch]).to(self.device)
                    y = torch.from_numpy(targets[batch]).to(self.device)
                    prediction = model(x)
                    loss = nn.functional.mse_loss(prediction, y)
                    if reward_targets is not None:
                        reward = torch.from_numpy(reward_targets[batch]).to(self.device)
                        predicted_reward = reward_model(x).squeeze(-1)
                        loss = loss + nn.functional.mse_loss(predicted_reward, reward)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    final_loss = float(loss.item())
            losses.append(final_loss)
        self._fitted = True
        self._reward_fitted = reward_targets is not None
        return losses

    def _features(self, states: np.ndarray, actions: np.ndarray) -> torch.Tensor:
        """Normalize state-action inputs for inference."""
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != self.state_dim:
            raise ValueError("states have the wrong shape")
        if actions.shape != (len(states), self.action_dim):
            raise ValueError("actions have the wrong shape")
        values = np.concatenate(
            ((states - self.state_mean) / self.state_std, actions), axis=1
        )
        return torch.from_numpy(values).to(self.device)

    def predict(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        physics_deltas: np.ndarray | None = None,
        member: int | None = None,
    ) -> np.ndarray:
        """Predict next states using one member or the ensemble mean."""
        if not self._fitted:
            raise RuntimeError("fit the dynamics ensemble before prediction")
        x = self._features(states, actions)
        selected = self.models if member is None else [self.models[member]]
        with torch.no_grad():
            predictions = [model(x).cpu().numpy() for model in selected]
        delta = np.mean(predictions, axis=0) * self.delta_std + self.delta_mean
        if self.mode == "residual":
            if physics_deltas is None:
                raise ValueError("residual mode requires physics_deltas")
            delta = delta + np.asarray(physics_deltas, dtype=np.float32)
        return np.asarray(states, dtype=np.float32) + delta

    def predict_rewards(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        member: int | None = None,
    ) -> np.ndarray:
        """Predict rewards for model-generated transitions."""
        if not self._reward_fitted:
            raise RuntimeError("fit the ensemble with rewards before reward prediction")
        x = self._features(states, actions)
        selected = (
            self.reward_models if member is None else [self.reward_models[member]]
        )
        with torch.no_grad():
            predictions = [model(x).squeeze(-1).cpu().numpy() for model in selected]
        return np.mean(predictions, axis=0) * self.reward_std + self.reward_mean

    def evaluate(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        horizons: tuple[int, ...] = (1, 5, 10, 20, 50),
        physics_deltas: np.ndarray | None = None,
    ) -> DynamicsMetrics:
        """Report one-step MSE and open-loop errors at requested horizons."""
        predicted = self.predict(states, actions, physics_deltas=physics_deltas)
        one_step = float(np.mean((predicted - next_states) ** 2))
        rollout_errors = {
            1: float(np.mean(np.linalg.norm(predicted - next_states, axis=1)))
        }
        current = np.asarray(states, dtype=np.float32).copy()
        for horizon in range(2, max(horizons) + 1):
            current = self.predict(current, actions, physics_deltas=physics_deltas)
            if horizon in horizons:
                rollout_errors[horizon] = float(
                    np.mean(np.linalg.norm(current - next_states, axis=1))
                )
        return DynamicsMetrics(one_step, rollout_errors)
