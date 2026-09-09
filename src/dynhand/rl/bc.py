"""Behavior cloning pretraining on demonstration transitions.

Fits the actor's mean head by supervised regression before RL starts,
with early stopping on a held-out split of the demonstrations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from dynhand.rl.sac import SAC


class BCTrainer:
    """Supervised pretraining of the SAC actor on demo (obs, action) pairs."""

    def __init__(
        self,
        sac: SAC,
        demo_obs: np.ndarray,
        demo_acts: np.ndarray,
        lr: float,
        holdout_fraction: float,
        rng: np.random.Generator,
    ) -> None:
        self.sac = sac
        self.device = sac.device
        self.rng = rng
        n = len(demo_obs)
        if n < 2:
            raise ValueError("BC requires at least 2 demonstration transitions")
        idx = self.rng.permutation(n)
        n_holdout = max(1, int(holdout_fraction * n))
        self.holdout = (demo_obs[idx[:n_holdout]], demo_acts[idx[:n_holdout]])
        self.train = (demo_obs[idx[n_holdout:]], demo_acts[idx[n_holdout:]])
        self.optimizer = Adam(self.sac.actor.parameters(), lr=lr)

    def _holdout_loss(self) -> float:
        obs = torch.as_tensor(self.holdout[0], dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(self.holdout[1], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self.sac.actor.deterministic(obs)
            return float(F.mse_loss(pred, acts).item())

    def train_epochs(
        self, epochs: int, batch_size: int, patience: int = 10
    ) -> dict[str, float]:
        """Run BC epochs with early stopping, restoring the best weights."""
        best_holdout = float("inf")
        best_state = {k: v.clone() for k, v in self.sac.actor.state_dict().items()}
        epochs_run = 0
        steps_without_improvement = 0
        final_train_loss = 0.0
        for _ in range(epochs):
            epochs_run += 1
            order = self.rng.permutation(len(self.train[0]))
            losses = []
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                obs = torch.as_tensor(
                    self.train[0][idx], dtype=torch.float32, device=self.device
                )
                acts = torch.as_tensor(
                    self.train[1][idx], dtype=torch.float32, device=self.device
                )
                pred = self.sac.actor.deterministic(obs)
                loss = F.mse_loss(pred, acts)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.item()))
            final_train_loss = float(np.mean(losses))
            holdout = self._holdout_loss()
            if holdout < best_holdout - 1e-6:
                best_holdout = holdout
                best_state = {
                    k: v.clone() for k, v in self.sac.actor.state_dict().items()
                }
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1
                if steps_without_improvement >= patience:
                    break
        self.sac.actor.load_state_dict(best_state)
        return {
            "bc_train_loss": final_train_loss,
            "bc_holdout_loss": best_holdout,
            "bc_epochs_run": float(epochs_run),
        }
