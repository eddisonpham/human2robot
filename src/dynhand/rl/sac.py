"""Soft Actor-Critic learner.

Forked conceptually from CleanRL's sac_continuous_action.py and restructured
into a reusable class: twin Q networks, tanh-squashed Gaussian actor, and
automatic entropy temperature tuning.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from dynhand.config.schema import SACConfig
from dynhand.rl.networks import GaussianActor, QNetwork


class SAC:
    """Owns the actor, critics, optimizers, and one gradient update step."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        sac_config: SACConfig,
        device: torch.device,
    ) -> None:
        self.config = sac_config
        self.device = device
        self.gamma = sac_config.gamma
        self.tau = sac_config.tau
        self.target_entropy = sac_cfg_target_entropy(sac_config, act_dim)

        self.actor = GaussianActor(obs_dim, act_dim, sac_config.hidden_dim).to(device)
        self.actor.set_action_range(action_low, action_high)
        self.qf = QNetwork(obs_dim, act_dim, sac_config.hidden_dim).to(device)
        self.qf_target = QNetwork(obs_dim, act_dim, sac_config.hidden_dim).to(device)
        self.qf_target.load_state_dict(self.qf.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr=sac_config.actor_lr)
        self.qf_optimizer = Adam(self.qf.parameters(), lr=sac_config.critic_lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=sac_config.alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select an action for a single observation or a batch."""
        single = obs.ndim == 1
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if single:
            tensor = tensor.unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(tensor)
            else:
                action, _, _ = self.actor(tensor)
        out = action.cpu().numpy()
        return out[0] if single else out

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Run one SAC gradient update on a minibatch and return metrics."""
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        acts = torch.as_tensor(batch["acts"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(
            batch["next_obs"], dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor(next_obs)
            q1_target, q2_target = self.qf_target(next_obs, next_actions)
            min_q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_pi
            next_qs = rewards + (1.0 - dones) * self.gamma * min_q_target

        q1, q2 = self.qf(obs, acts)
        qf_loss = F.mse_loss(q1, next_qs) + F.mse_loss(q2, next_qs)
        self.qf_optimizer.zero_grad()
        qf_loss.backward()
        self.qf_optimizer.step()

        actions, log_pi, _ = self.actor(obs)
        q1_pi, q2_pi = self.qf(obs, actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_pi - min_q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            for param, target_param in zip(
                self.qf.parameters(),
                self.qf_target.parameters(),
                strict=True,
            ):
                target_param.mul_(1.0 - self.tau).add_(param, alpha=self.tau)

        return {
            "qf_loss": float(qf_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "q_mean": float(q1.mean().item()),
        }

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "qf": self.qf.state_dict(),
            "qf_target": self.qf_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "qf_optimizer": self.qf_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.qf.load_state_dict(state["qf"])
        self.qf_target.load_state_dict(state["qf_target"])
        with torch.no_grad():
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.qf_optimizer.load_state_dict(state["qf_optimizer"])
        self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])


def sac_cfg_target_entropy(sac_config: SACConfig, act_dim: int) -> float:
    """Standard heuristic: target entropy scales with negative action dim."""
    return sac_config.target_entropy_scale * act_dim
