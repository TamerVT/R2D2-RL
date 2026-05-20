"""Visual SAC for LeRobot-compatible SO-101 local grasp training."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam

from .lerobot_compat import IMAGE_KEY, STATE_KEY
from .visual_replay_buffer import VisualTransitionBatch


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class VisualSACConfig:
    hidden_sizes: Sequence[int] = field(default_factory=lambda: (512, 256))
    image_feature_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_log_alpha: float = 0.0
    target_entropy: float | None = None
    state_mean: Sequence[float] = field(default_factory=lambda: tuple(0.0 for _ in range(24)))
    state_std: Sequence[float] = field(default_factory=lambda: tuple(1.0 for _ in range(24)))
    device: str = "cpu"


def _build_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for hidden in hidden_sizes:
        layers.append(nn.Linear(last, int(hidden)))
        layers.append(nn.ReLU())
        last = int(hidden)
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class WristStateEncoder(nn.Module):
    """Encode uint8 wrist image and normalized 24D state into one feature vector."""

    def __init__(
        self,
        *,
        state_mean: Sequence[float],
        state_std: Sequence[float],
        image_feature_dim: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.image_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, image_feature_dim),
            nn.ReLU(),
        )
        mean = torch.as_tensor(state_mean, dtype=torch.float32).reshape(24)
        std = torch.clamp(torch.as_tensor(state_std, dtype=torch.float32).reshape(24), min=1e-6)
        self.register_buffer("state_mean", mean)
        self.register_buffer("state_std", std)
        self.out_dim = int(image_feature_dim) + 24

    def forward(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if image.dtype == torch.uint8:
            image_f = image.float() / 255.0
        else:
            image_f = image.float()
            if image_f.max().detach() > 2.0:
                image_f = image_f / 255.0
        image_feat = self.image_fc(self.conv(image_f))
        norm_state = (state.float() - self.state_mean) / self.state_std
        return torch.cat([image_feat, norm_state], dim=-1)


class VisualSquashedGaussianActor(nn.Module):
    def __init__(
        self,
        *,
        act_dim: int,
        hidden_sizes: Sequence[int],
        state_mean: Sequence[float],
        state_std: Sequence[float],
        image_feature_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = WristStateEncoder(
            state_mean=state_mean,
            state_std=state_std,
            image_feature_dim=image_feature_dim,
        )
        hidden_sizes = tuple(int(v) for v in hidden_sizes)
        trunk_out = hidden_sizes[-1]
        self.trunk = _build_mlp(self.encoder.out_dim, hidden_sizes[:-1], trunk_out)
        self.mu_layer = nn.Linear(trunk_out, act_dim)
        self.log_std_layer = nn.Linear(trunk_out, act_dim)

    def _distribution(self, image: torch.Tensor, state: torch.Tensor) -> Normal:
        h = self.trunk(self.encoder(image, state))
        mu = self.mu_layer(h)
        log_std = torch.clamp(self.log_std_layer(h), LOG_STD_MIN, LOG_STD_MAX)
        return Normal(mu, torch.exp(log_std))

    def act(self, image: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._distribution(image, state)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def act_inference(self, image: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        dist = self._distribution(image, state)
        return torch.tanh(dist.mean)


class VisualQNet(nn.Module):
    def __init__(
        self,
        *,
        act_dim: int,
        hidden_sizes: Sequence[int],
        state_mean: Sequence[float],
        state_std: Sequence[float],
        image_feature_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = WristStateEncoder(
            state_mean=state_mean,
            state_std=state_std,
            image_feature_dim=image_feature_dim,
        )
        self.net = _build_mlp(self.encoder.out_dim + act_dim, hidden_sizes, 1)

    def forward(self, image: torch.Tensor, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = self.encoder(image, state)
        return self.net(torch.cat([features, action], dim=-1)).squeeze(-1)


class VisualDoubleQNet(nn.Module):
    def __init__(
        self,
        *,
        act_dim: int,
        hidden_sizes: Sequence[int],
        state_mean: Sequence[float],
        state_std: Sequence[float],
        image_feature_dim: int,
    ) -> None:
        super().__init__()
        self.q1 = VisualQNet(
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            state_mean=state_mean,
            state_std=state_std,
            image_feature_dim=image_feature_dim,
        )
        self.q2 = VisualQNet(
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            state_mean=state_mean,
            state_std=state_std,
            image_feature_dim=image_feature_dim,
        )

    def forward(
        self,
        image: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(image, state, action), self.q2(image, state, action)


class VisualSACAgent:
    """SAC actor/critic for LeRobot visual observations and scaled actions."""

    def __init__(
        self,
        act_dim: int = 6,
        config: VisualSACConfig | None = None,
    ) -> None:
        self.cfg = config or VisualSACConfig()
        self.act_dim = int(act_dim)
        self.device = torch.device(self.cfg.device)
        hidden = tuple(int(v) for v in self.cfg.hidden_sizes)
        state_mean = tuple(float(v) for v in self.cfg.state_mean)
        state_std = tuple(float(v) for v in self.cfg.state_std)

        self.actor = VisualSquashedGaussianActor(
            act_dim=self.act_dim,
            hidden_sizes=hidden,
            state_mean=state_mean,
            state_std=state_std,
            image_feature_dim=int(self.cfg.image_feature_dim),
        ).to(self.device)
        self.critic = VisualDoubleQNet(
            act_dim=self.act_dim,
            hidden_sizes=hidden,
            state_mean=state_mean,
            state_std=state_std,
            image_feature_dim=int(self.cfg.image_feature_dim),
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optim = Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
        self.target_entropy = (
            float(self.cfg.target_entropy)
            if self.cfg.target_entropy is not None
            else -float(self.act_dim)
        )
        self.log_alpha = torch.tensor(
            float(self.cfg.init_log_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optim = Adam([self.log_alpha], lr=self.cfg.alpha_lr)

    @torch.no_grad()
    def act(self, obs: dict[str, np.ndarray], deterministic: bool = False) -> np.ndarray:
        image = torch.as_tensor(obs[IMAGE_KEY], dtype=torch.uint8, device=self.device).unsqueeze(0)
        state = torch.as_tensor(obs[STATE_KEY], dtype=torch.float32, device=self.device).reshape(1, 24)
        if deterministic:
            action = self.actor.act_inference(image, state)
        else:
            action, _ = self.actor.act(image, state)
        return action.detach().cpu().numpy().reshape(self.act_dim).astype(np.float32)

    def update(self, batch: VisualTransitionBatch) -> dict[str, float]:
        images = batch.images.to(self.device)
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        next_images = batch.next_images.to(self.device)
        next_states = batch.next_states.to(self.device)
        dones = batch.dones.to(self.device)

        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_actions, next_log_prob = self.actor.act(next_images, next_states)
            next_q1, next_q2 = self.critic_target(next_images, next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - alpha * next_log_prob
            target = rewards + self.cfg.gamma * (1.0 - dones) * next_q

        q1, q2 = self.critic(images, states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        sampled_actions, log_prob = self.actor.act(images, states)
        q1_pi, q2_pi = self.critic(images, states, sampled_actions)
        actor_loss = (alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

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

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_alpha": float(self.log_alpha.detach().cpu().item()),
                "act_dim": self.act_dim,
                "config": {**asdict(self.cfg), "hidden_sizes": list(self.cfg.hidden_sizes)},
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "alpha_optim": self.alpha_optim.state_dict(),
            },
            str(path),
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(str(path), map_location=self.device)
        if int(ckpt["act_dim"]) != self.act_dim:
            raise ValueError(f"Checkpoint act_dim={ckpt['act_dim']} does not match {self.act_dim}.")
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


def load_visual_agent_for_inference(path: str | Path, device: str = "cpu") -> VisualSACAgent:
    ckpt = torch.load(str(path), map_location=device)
    cfg_data = ckpt["config"]
    cfg = VisualSACConfig(
        hidden_sizes=tuple(cfg_data.get("hidden_sizes", (512, 256))),
        image_feature_dim=int(cfg_data.get("image_feature_dim", 256)),
        gamma=float(cfg_data.get("gamma", 0.99)),
        tau=float(cfg_data.get("tau", 0.005)),
        actor_lr=float(cfg_data.get("actor_lr", 3e-4)),
        critic_lr=float(cfg_data.get("critic_lr", 3e-4)),
        alpha_lr=float(cfg_data.get("alpha_lr", 3e-4)),
        init_log_alpha=float(cfg_data.get("init_log_alpha", 0.0)),
        target_entropy=cfg_data.get("target_entropy"),
        state_mean=tuple(cfg_data.get("state_mean", [0.0] * 24)),
        state_std=tuple(cfg_data.get("state_std", [1.0] * 24)),
        device=device,
    )
    agent = VisualSACAgent(act_dim=int(ckpt["act_dim"]), config=cfg)
    agent.load(path)
    return agent
