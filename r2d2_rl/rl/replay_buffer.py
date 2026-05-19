"""Simple FIFO replay buffer for off-policy RL training.

Stores ``(obs, action, reward, next_obs, done)`` transitions and samples
uniform random batches as torch tensors on the requested device.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TransitionBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Numpy-backed FIFO replay buffer with float32 storage."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if obs_dim <= 0 or act_dim <= 0:
            raise ValueError("obs_dim and act_dim must be positive.")

        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    @property
    def size(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self._ptr
        self._obs[idx] = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        self._actions[idx] = np.asarray(action, dtype=np.float32).reshape(self.act_dim)
        self._rewards[idx] = float(reward)
        self._next_obs[idx] = np.asarray(next_obs, dtype=np.float32).reshape(self.obs_dim)
        self._dones[idx] = 1.0 if done else 0.0

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> TransitionBatch:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        idx = np.random.randint(0, self._size, size=batch_size)
        device = torch.device(device)
        return TransitionBatch(
            obs=torch.as_tensor(self._obs[idx], dtype=torch.float32, device=device),
            actions=torch.as_tensor(self._actions[idx], dtype=torch.float32, device=device),
            rewards=torch.as_tensor(self._rewards[idx], dtype=torch.float32, device=device),
            next_obs=torch.as_tensor(self._next_obs[idx], dtype=torch.float32, device=device),
            dones=torch.as_tensor(self._dones[idx], dtype=torch.float32, device=device),
        )
