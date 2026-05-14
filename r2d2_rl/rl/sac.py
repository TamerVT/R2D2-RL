"""Soft Actor-Critic agent for the local ``align_grasp`` policy.

Standard SAC: a tanh-squashed Gaussian actor, a double Q critic with a
target network, and automatic temperature tuning against a target entropy.
Reuses ``RL_envs.networks.SquashedGaussianActor`` and ``DoubleQNet`` so the
trainable modules match the shapes from the HW4 RL building blocks.

The agent is intentionally framework-light: numpy obs/actions in,
gradient updates out. It does not own the replay buffer or env loop; those
live in ``rl.replay_buffer`` and ``scripts/train_align_grasp.py``.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from RL_envs.networks import DoubleQNet, SquashedGaussianActor
from rl.replay_buffer import TransitionBatch


@dataclass
class SACConfig:
    hidden_sizes: Sequence[int] = field(default_factory=lambda: (256, 256, 128))
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_log_alpha: float = 0.0
    target_entropy: float | None = None  # defaults to -act_dim
    device: str = "cpu"


class SACAgent:
    """Soft Actor-Critic with double-Q critic + auto temperature."""

    def __init__(self, obs_dim: int, act_dim: int, config: SACConfig | None = None):
        if obs_dim <= 0 or act_dim <= 0:
            raise ValueError("obs_dim and act_dim must be positive.")
        cfg = config or SACConfig()
        self.cfg = cfg
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = torch.device(cfg.device)

        hidden = [int(size) for size in cfg.hidden_sizes]
        if not hidden or any(size <= 0 for size in hidden):
            raise ValueError("SACConfig.hidden_sizes must contain positive layer sizes.")
        self.actor = SquashedGaussianActor(obs_dim, act_dim, hidden).to(self.device)
        self.critic = DoubleQNet(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optim = Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=cfg.critic_lr)

        target_entropy = cfg.target_entropy if cfg.target_entropy is not None else -float(act_dim)
        self.target_entropy = float(target_entropy)
        self.log_alpha = torch.tensor(
            float(cfg.init_log_alpha), dtype=torch.float32, device=self.device, requires_grad=True
        )
        self.alpha_optim = Adam([self.log_alpha], lr=cfg.alpha_lr)

    # ------------------------------------------------------------------ act

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(1, -1)
        if deterministic:
            action = self.actor.act_inference(obs_t)
        else:
            action, _ = self.actor.act(obs_t)
        return action.detach().cpu().numpy().reshape(self.act_dim)

    # ---------------------------------------------------------------- update

    def update(self, batch: TransitionBatch) -> dict[str, float]:
        obs = batch.obs.to(self.device)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        next_obs = batch.next_obs.to(self.device)
        dones = batch.dones.to(self.device)

        alpha = self.log_alpha.exp().detach()

        # Critic loss.
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.act(next_obs)
            next_q1, next_q2 = self.critic_target(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2) - alpha * next_log_prob
            target = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # Actor loss.
        sampled_actions, log_prob = self.actor.act(obs)
        q1_pi, q2_pi = self.critic(obs, sampled_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (alpha * log_prob - q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Temperature loss.
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        # Soft-update target critic.
        with torch.no_grad():
            for tgt, src in zip(self.critic_target.parameters(), self.critic.parameters()):
                tgt.data.mul_(1.0 - self.cfg.tau).add_(src.data, alpha=self.cfg.tau)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
            "q1_mean": float(q1.mean().item()),
            "entropy": float((-log_prob).mean().item()),
        }

    # --------------------------------------------------------------- save / load

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": float(self.log_alpha.detach().cpu().item()),
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "config": {**asdict(self.cfg), "hidden_sizes": list(self.cfg.hidden_sizes)},
                "hidden_sizes": list(self.cfg.hidden_sizes),
                "target_entropy": self.target_entropy,
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "alpha_optim": self.alpha_optim.state_dict(),
            },
            str(path),
        )

    def load(self, path: str | Path) -> None:
        path = Path(path)
        ckpt = torch.load(str(path), map_location=self.device)
        if "obs_dim" in ckpt and int(ckpt["obs_dim"]) != self.obs_dim:
            raise ValueError(
                f"Checkpoint obs_dim={ckpt['obs_dim']} does not match agent obs_dim={self.obs_dim}."
            )
        if "act_dim" in ckpt and int(ckpt["act_dim"]) != self.act_dim:
            raise ValueError(
                f"Checkpoint act_dim={ckpt['act_dim']} does not match agent act_dim={self.act_dim}."
            )
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(torch.tensor(float(ckpt["log_alpha"]), device=self.device))
        if "actor_optim" in ckpt:
            self.actor_optim.load_state_dict(ckpt["actor_optim"])
        if "critic_optim" in ckpt:
            self.critic_optim.load_state_dict(ckpt["critic_optim"])
        if "alpha_optim" in ckpt:
            self.alpha_optim.load_state_dict(ckpt["alpha_optim"])


def load_agent_for_inference(path: str | Path, device: str = "cpu") -> SACAgent:
    """Convenience: build an agent matching a checkpoint and load weights."""
    path = Path(path)
    ckpt = torch.load(str(path), map_location=device)
    cfg_data = ckpt.get("config") or {}
    cfg = SACConfig(
        hidden_sizes=tuple(ckpt.get("hidden_sizes", cfg_data.get("hidden_sizes", (256, 256, 128)))),
        device=device,
        gamma=float(cfg_data.get("gamma", SACConfig.gamma)),
        tau=float(cfg_data.get("tau", SACConfig.tau)),
        actor_lr=float(cfg_data.get("actor_lr", SACConfig.actor_lr)),
        critic_lr=float(cfg_data.get("critic_lr", SACConfig.critic_lr)),
        alpha_lr=float(cfg_data.get("alpha_lr", SACConfig.alpha_lr)),
        init_log_alpha=float(cfg_data.get("init_log_alpha", SACConfig.init_log_alpha)),
        target_entropy=float(
            ckpt.get("target_entropy", cfg_data.get("target_entropy", -float(ckpt["act_dim"])))
        ),
    )
    agent = SACAgent(int(ckpt["obs_dim"]), int(ckpt["act_dim"]), cfg)
    agent.load(path)
    return agent
