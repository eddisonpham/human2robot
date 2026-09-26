"""Uniform replay buffer for off-policy RL."""

import numpy as np


class ReplayBuffer:
    """Fixed-size circular buffer storing flat transitions in CPU RAM."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        capacity: int,
        rng: np.random.Generator,
    ) -> None:
        self.capacity = capacity
        self.rng = rng
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts = np.zeros((capacity, act_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        done: float,
    ) -> None:
        """Insert one transition."""
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.next_obs[self.ptr] = next_obs
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        obs: np.ndarray,
        acts: np.ndarray,
        next_obs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Insert a batch of transitions in order."""
        rewards = np.asarray(rewards).reshape(len(obs), -1)[:, 0]
        dones = np.asarray(dones).reshape(len(obs), -1)[:, 0]
        for i in range(len(obs)):
            self.add(obs[i], acts[i], next_obs[i], float(rewards[i]), float(dones[i]))

    def __len__(self) -> int:
        return self.size

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a uniform random minibatch."""
        idx = self.rng.integers(0, self.size, size=batch_size)
        return {
            "obs": self.obs[idx],
            "acts": self.acts[idx],
            "next_obs": self.next_obs[idx],
            "rewards": self.rewards[idx],
            "dones": self.dones[idx],
        }

    def state_dict(self) -> dict[str, np.ndarray | int]:
        """Return only the filled region of the arrays for checkpointing.

        Serializing the full preallocated capacity wastes hundreds of
        megabytes per checkpoint early in a run, so slice to ``size`` rows.
        """
        size = self.size
        return {
            "obs": self.obs[:size],
            "acts": self.acts[:size],
            "next_obs": self.next_obs[:size],
            "rewards": self.rewards[:size],
            "dones": self.dones[:size],
            "ptr": self.ptr,
            "size": size,
        }

    def load_state(self, state: dict[str, np.ndarray | int]) -> None:
        """Restore arrays and pointers from a checkpoint.

        Accepts both the current trimmed format and the older format that
        serialized the whole preallocated capacity.
        """
        rows = int(state["size"])
        if rows > self.obs.shape[0]:
            raise ValueError(
                f"buffer shape mismatch: checkpoint has {rows} rows"
                f" but capacity is {self.obs.shape[0]}"
            )
        for name in ("obs", "acts", "next_obs", "rewards", "dones"):
            values = np.asarray(state[name])
            source = values if values.shape[0] == rows else values[:rows]
            getattr(self, name)[:rows] = source
        self.ptr = int(state["ptr"])
        self.size = rows
