"""Actor and critic networks for SAC, CleanRL-style."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianActor(nn.Module):
    """Tanh-squashed diagonal Gaussian policy with automatic temperature."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden_dim, act_dim)
        self.fc_logstd = nn.Linear(hidden_dim, act_dim)
        self.register_buffer("action_scale", torch.ones(act_dim))
        self.register_buffer("action_bias", torch.zeros(act_dim))

    def set_action_range(self, low: np.ndarray, high: np.ndarray) -> None:
        """Map the tanh output onto the environment action space bounds."""
        device = self.fc_mean.weight.device
        self.action_scale = (
            torch.as_tensor(high, dtype=torch.float32, device=device)
            - torch.as_tensor(low, dtype=torch.float32, device=device)
        ) / 2.0
        self.action_bias = (
            torch.as_tensor(high, dtype=torch.float32, device=device)
            + torch.as_tensor(low, dtype=torch.float32, device=device)
        ) / 2.0

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (squashed action, log prob, mean action before squashing)."""
        h = self.net(obs)
        mean = self.fc_mean(h)
        log_std = self.fc_logstd(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        scaled = y_t * self.action_scale + self.action_bias
        return scaled, log_prob, y_t

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Mean action, used for evaluation and ONNX export."""
        h = self.net(obs)
        mean = self.fc_mean(h)
        return torch.tanh(mean) * self.action_scale + self.action_bias


class QNetwork(nn.Module):
    """Twin Q networks taking concatenated (obs, action)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.q1 = self._build(obs_dim, act_dim, hidden_dim)
        self.q2 = self._build(obs_dim, act_dim, hidden_dim)

    @staticmethod
    def _build(obs_dim: int, act_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)
