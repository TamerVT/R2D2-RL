from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from project3_modular.rl_grasp.envs.so101_local_grasp_hil_compat_env import (
    SO101LocalGraspHILCompatEnv,
)
from project3_modular.rl_grasp.scripts.train_real_hil_bc_policy import (
    RealHILBCDataset,
    make_episode_split,
)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class TrainConfig:
    output_dir: str
    normalization_stats: str

    total_timesteps: int = 50_000
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # SAC
    learning_rate: float = 3e-4
    buffer_size: int = 10_000
    learning_starts: int = 1_000
    batch_size: int = 128
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"

    # Real-demo SAC actor BC pretraining.
    bc_dataset_root: str | None = None
    bc_pretrain_epochs: int = 0
    bc_batch_size: int = 128
    bc_lr: float = 3e-4
    bc_val_episode_fraction: float = 0.2
    bc_zero_motor_currents: bool = True

    checkpoint_freq: int = 10_000

    # Optional SB3 SAC checkpoint to resume RL training from.
    # When set, BC pretraining is skipped and `total_timesteps` is interpreted
    # as *additional* RL timesteps.
    resume_from: str | None = None


# ---------------------------------------------------------------------
# CNN feature extractor
# ---------------------------------------------------------------------


class WristImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(image))


class VisualHILFeaturesExtractor(BaseFeaturesExtractor):
    """
    Extract:
        wrist RGB -> 256D CNN feature
        24D state  -> normalized 24D vector
    Concatenate to 280D.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_mean: list[float],
        state_std: list[float],
        image_feature_dim: int = 256,
    ) -> None:
        super().__init__(
            observation_space,
            features_dim=image_feature_dim + 24,
        )

        self.image_encoder = WristImageEncoder(out_dim=image_feature_dim)

        mean = torch.tensor(state_mean, dtype=torch.float32)
        std = torch.tensor(state_std, dtype=torch.float32)
        std = torch.clamp(std, min=1e-6)

        self.register_buffer("state_mean", mean)
        self.register_buffer("state_std", std)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        image = observations["observation.images.wrist"]
        state = observations["observation.state"]

        image_feat = self.image_encoder(image)
        norm_state = (state - self.state_mean) / self.state_std

        return torch.cat([image_feat, norm_state], dim=-1)


# ---------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------


class LocalGraspInfoCallback(BaseCallback):
    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True

        info = infos[0] if isinstance(infos, list) else infos

        for key in [
            "cube_lift",
            "xy_dist",
            "xyz_dist",
        ]:
            if key in info:
                self.logger.record(f"env/{key}", float(info[key]))

        if "local_success" in info:
            self.logger.record("env/local_success", float(bool(info["local_success"])))

        if "sim_is_grasped" in info:
            self.logger.record("env/sim_is_grasped", float(bool(info["sim_is_grasped"])))

        return True


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def load_normalization_stats(path: str | Path) -> dict[str, list[float]]:
    stats = json.loads(Path(path).read_text())

    required = ["state_mean", "state_std"]
    for key in required:
        if key not in stats:
            raise RuntimeError(f"Normalization stats missing key {key!r}")

    return stats


def env_actions_to_scaled_actions(
    actions: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    actions = torch.clamp(actions, action_low, action_high)
    scaled = 2.0 * (actions - action_low) / (action_high - action_low) - 1.0
    return torch.clamp(scaled, -1.0, 1.0)


def scaled_actions_to_env_actions(
    scaled_actions: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> torch.Tensor:
    scaled_actions = torch.clamp(scaled_actions, -1.0, 1.0)
    return action_low + 0.5 * (scaled_actions + 1.0) * (action_high - action_low)


# ---------------------------------------------------------------------
# Actor BC pretraining
# ---------------------------------------------------------------------


@torch.no_grad()
def evaluate_actor_bc(
    *,
    actor: nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> dict[str, float]:
    actor.eval()

    total_scaled_mse = 0.0
    total_raw_mae = 0.0
    total_count = 0

    gripper_true: list[np.ndarray] = []
    gripper_pred: list[np.ndarray] = []

    for batch in loader:
        image_float = batch["image"].to(device)
        image_uint8 = torch.clamp(
            torch.round(image_float * 255.0),
            0,
            255,
        ).to(torch.uint8)

        state = batch["state"].to(device)
        action = batch["action"].to(device)

        obs = {
            "observation.images.wrist": image_uint8,
            "observation.state": state,
        }

        pred_scaled = actor(obs, deterministic=True)
        target_scaled = env_actions_to_scaled_actions(
            action,
            action_low,
            action_high,
        )

        pred_env = scaled_actions_to_env_actions(
            pred_scaled,
            action_low,
            action_high,
        )

        batch_size = action.shape[0]
        total_scaled_mse += float(F.mse_loss(pred_scaled, target_scaled, reduction="sum").item())
        total_raw_mae += float(F.l1_loss(pred_env, action, reduction="sum").item())
        total_count += batch_size

        gripper_true.append(action[:, 5].detach().cpu().numpy())
        gripper_pred.append(pred_env[:, 5].detach().cpu().numpy())

    denom = max(1, total_count * 6)

    y_true = np.concatenate(gripper_true)
    y_pred = np.concatenate(gripper_pred)
    grip_corr = float(np.corrcoef(y_true, y_pred)[0, 1])

    return {
        "scaled_mse": total_scaled_mse / denom,
        "raw_mae": total_raw_mae / denom,
        "gripper_corr": grip_corr,
        "gripper_true_p95": float(np.percentile(y_true, 95)),
        "gripper_pred_p95": float(np.percentile(y_pred, 95)),
    }


def bc_pretrain_sac_actor(
    *,
    model: SAC,
    cfg: TrainConfig,
    output_dir: Path,
) -> None:
    if cfg.bc_dataset_root is None or cfg.bc_pretrain_epochs <= 0:
        print("Skipping real-demo actor BC pretraining.")
        return

    print()
    print("=== Real-demo BC pretraining of SAC actor ===")

    dataset = RealHILBCDataset(
        Path(cfg.bc_dataset_root),
        image_size=128,
        zero_motor_currents=cfg.bc_zero_motor_currents,
    )

    train_indices, val_indices = make_episode_split(
        dataset.episode_indices,
        val_fraction=cfg.bc_val_episode_fraction,
        seed=cfg.seed,
    )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=cfg.bc_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(model.device.type == "cuda"),
    )

    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=cfg.bc_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(model.device.type == "cuda"),
    )

    actor = model.policy.actor
    actor.train()

    optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=cfg.bc_lr,
    )

    action_low = torch.tensor(
        model.action_space.low,
        dtype=torch.float32,
        device=model.device,
    )
    action_high = torch.tensor(
        model.action_space.high,
        dtype=torch.float32,
        device=model.device,
    )

    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard_bc_pretrain"))

    best_val = float("inf")

    for epoch in range(1, cfg.bc_pretrain_epochs + 1):
        actor.train()

        running_loss = 0.0
        running_count = 0

        for batch in train_loader:
            image_float = batch["image"].to(model.device)
            image_uint8 = torch.clamp(
                torch.round(image_float * 255.0),
                0,
                255,
            ).to(torch.uint8)

            state = batch["state"].to(model.device)
            action = batch["action"].to(model.device)

            obs = {
                "observation.images.wrist": image_uint8,
                "observation.state": state,
            }

            pred_scaled = actor(obs, deterministic=True)
            target_scaled = env_actions_to_scaled_actions(
                action,
                action_low,
                action_high,
            )

            loss = F.mse_loss(pred_scaled, target_scaled)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = action.shape[0]
            running_loss += float(loss.item()) * batch_size
            running_count += batch_size

        train_loss = running_loss / max(1, running_count)
        val_metrics = evaluate_actor_bc(
            actor=actor,
            loader=val_loader,
            device=model.device,
            action_low=action_low,
            action_high=action_high,
        )

        writer.add_scalar("bc_actor/train_scaled_mse", train_loss, epoch)
        writer.add_scalar("bc_actor/val_scaled_mse", val_metrics["scaled_mse"], epoch)
        writer.add_scalar("bc_actor/val_raw_mae", val_metrics["raw_mae"], epoch)
        writer.add_scalar("bc_actor/val_gripper_corr", val_metrics["gripper_corr"], epoch)
        writer.add_scalar("bc_actor/val_gripper_true_p95", val_metrics["gripper_true_p95"], epoch)
        writer.add_scalar("bc_actor/val_gripper_pred_p95", val_metrics["gripper_pred_p95"], epoch)

        print(
            f"BC epoch {epoch:03d}/{cfg.bc_pretrain_epochs} | "
            f"train_scaled_mse={train_loss:.6f} | "
            f"val_scaled_mse={val_metrics['scaled_mse']:.6f} | "
            f"val_raw_mae={val_metrics['raw_mae']:.4f} | "
            f"grip_corr={val_metrics['gripper_corr']:.3f}"
        )

        if val_metrics["scaled_mse"] < best_val:
            best_val = val_metrics["scaled_mse"]
            model.save(str(output_dir / "sac_actor_bc_pretrained_best"))

    writer.close()

    model.save(str(output_dir / "sac_actor_bc_pretrained_last"))
    print(f"Best actor BC validation scaled MSE: {best_val:.6f}")


# ---------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--normalization-stats", required=True)

    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", default="auto")

    parser.add_argument("--bc-dataset-root")
    parser.add_argument("--bc-pretrain-epochs", type=int, default=0)
    parser.add_argument("--bc-batch-size", type=int, default=128)
    parser.add_argument("--bc-lr", type=float, default=3e-4)
    parser.add_argument("--bc-val-episode-fraction", type=float, default=0.2)
    parser.add_argument("--bc-zero-motor-currents", action="store_true")

    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument(
        "--resume-from",
        help="Optional SAC .zip checkpoint to resume RL training from.",
    )

    args = parser.parse_args()

    return TrainConfig(
        output_dir=args.output_dir,
        normalization_stats=args.normalization_stats,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        bc_dataset_root=args.bc_dataset_root,
        bc_pretrain_epochs=args.bc_pretrain_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_lr=args.bc_lr,
        bc_val_episode_fraction=args.bc_val_episode_fraction,
        bc_zero_motor_currents=args.bc_zero_motor_currents,
        checkpoint_freq=args.checkpoint_freq,
        resume_from=args.resume_from,
    )


def main() -> None:
    cfg = parse_args()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    stats = load_normalization_stats(cfg.normalization_stats)

    env = Monitor(
        SO101LocalGraspHILCompatEnv(open_gui=False),
        filename=str(output_dir / "monitor.csv"),
    )

    policy_kwargs = {
        "features_extractor_class": VisualHILFeaturesExtractor,
        "features_extractor_kwargs": {
            "state_mean": stats["state_mean"],
            "state_std": stats["state_std"],
            "image_feature_dim": 256,
        },
        "net_arch": {
            "pi": [512, 256],
            "qf": [512, 256],
        },
        "activation_fn": nn.ReLU,
        "normalize_images": True,
    }

    if cfg.resume_from is None:
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts,
            batch_size=cfg.batch_size,
            tau=cfg.tau,
            gamma=cfg.gamma,
            train_freq=cfg.train_freq,
            gradient_steps=cfg.gradient_steps,
            ent_coef=cfg.ent_coef,
            verbose=1,
            tensorboard_log=str(output_dir / "tensorboard_rl"),
            policy_kwargs=policy_kwargs,
            seed=cfg.seed,
            device=cfg.device,
        )
    else:
        print(f"Resuming SAC from checkpoint: {cfg.resume_from}")
        model = SAC.load(
            cfg.resume_from,
            env=env,
            device=cfg.device,
            tensorboard_log=str(output_dir / "tensorboard_rl"),
        )

    print()
    print("=== Visual HIL-compatible SAC setup ===")
    print("Output dir:       ", output_dir)
    print("Device:           ", model.device)
    print("Timesteps:        ", cfg.total_timesteps)
    print("Replay buffer:    ", cfg.buffer_size)
    print("BC pretrain eps:  ", cfg.bc_pretrain_epochs)
    print("Resume from:      ", cfg.resume_from)

    if cfg.resume_from is None:
        bc_pretrain_sac_actor(
            model=model,
            cfg=cfg,
            output_dir=output_dir,
        )
    else:
        print("Skipping BC pretraining because this is an RL resume run.")

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.checkpoint_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="visual_hil_sac",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    callback = CallbackList(
        [
            LocalGraspInfoCallback(),
            checkpoint_callback,
        ]
    )

    print()
    print("=== SAC RL training ===")

    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callback,
        tb_log_name="SAC_visual_hil_compat",
        reset_num_timesteps=(cfg.resume_from is None),
        progress_bar=True,
    )

    model.save(str(output_dir / "final_model"))
    env.close()

    print()
    print("Training finished.")
    print("Saved final model to:", output_dir / "final_model.zip")


if __name__ == "__main__":
    main()
