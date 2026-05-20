"""Replay buffer for LeRobot-style visual SAC observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .lerobot_compat import IMAGE_KEY, STATE_KEY


@dataclass
class VisualTransitionBatch:
    images: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_images: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class VisualReplayBuffer:
    """Numpy-backed FIFO buffer for wrist image + 24D state observations."""

    def __init__(
        self,
        capacity: int,
        image_shape: tuple[int, int, int],
        state_dim: int,
        act_dim: int,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self.image_shape = tuple(int(v) for v in image_shape)
        self.state_dim = int(state_dim)
        self.act_dim = int(act_dim)

        self._images = np.zeros((self.capacity, *self.image_shape), dtype=np.uint8)
        self._states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._actions = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self._rewards = np.zeros((self.capacity,), dtype=np.float32)
        self._next_images = np.zeros((self.capacity, *self.image_shape), dtype=np.uint8)
        self._next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self._dones = np.zeros((self.capacity,), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs: dict[str, np.ndarray],
        done: bool,
    ) -> None:
        idx = self._ptr
        self._images[idx] = np.asarray(obs[IMAGE_KEY], dtype=np.uint8).reshape(self.image_shape)
        self._states[idx] = np.asarray(obs[STATE_KEY], dtype=np.float32).reshape(self.state_dim)
        self._actions[idx] = np.asarray(action, dtype=np.float32).reshape(self.act_dim)
        self._rewards[idx] = float(reward)
        self._next_images[idx] = np.asarray(next_obs[IMAGE_KEY], dtype=np.uint8).reshape(self.image_shape)
        self._next_states[idx] = np.asarray(next_obs[STATE_KEY], dtype=np.float32).reshape(self.state_dim)
        self._dones[idx] = 1.0 if done else 0.0

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> VisualTransitionBatch:
        if self._size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        idx = np.random.randint(0, self._size, size=int(batch_size))
        device = torch.device(device)
        return VisualTransitionBatch(
            images=torch.as_tensor(self._images[idx], dtype=torch.uint8, device=device),
            states=torch.as_tensor(self._states[idx], dtype=torch.float32, device=device),
            actions=torch.as_tensor(self._actions[idx], dtype=torch.float32, device=device),
            rewards=torch.as_tensor(self._rewards[idx], dtype=torch.float32, device=device),
            next_images=torch.as_tensor(self._next_images[idx], dtype=torch.uint8, device=device),
            next_states=torch.as_tensor(self._next_states[idx], dtype=torch.float32, device=device),
            dones=torch.as_tensor(self._dones[idx], dtype=torch.float32, device=device),
        )
