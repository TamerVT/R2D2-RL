"""Behavior-cloning trainer for SO-101 local-grasp demos (LeRobot dataset).

Trains a CNN + MLP policy from
    (wrist RGB, 24D HIL state) -> 6D absolute joint action
on the multicolor HIL dataset (``p3_local_grasp_hil_multicolor_colorcond_v1``).

Outputs (per run, under ``--output-dir``):
- ``checkpoints/best.pt`` and ``last.pt`` - model state dict + normalization stats.
- ``normalization_stats.json``           - 24D state + 6D action mean/std.
- ``tensorboard/``                       - scalar logs.

The dataset class is reused by ``train_visual_hil_compat_sac.py`` to BC-pretrain
the SAC actor before any sim rollouts.

Run::

    MUJOCO_GL=egl python r2d2_rl/scripts/train_real_hil_bc_policy.py \
        --dataset-root r2d2_rl/outputs/hil_dataset/p3_local_grasp_hil_multicolor_colorcond_v1 \
        --output-dir r2d2_rl/outputs/real_hil_bc_warmstart \
        --epochs 80 --zero-motor-currents
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter


REPO_ROOT = Path(__file__).resolve().parents[2]
R2D2_RL = REPO_ROOT / "r2d2_rl"
if str(R2D2_RL) not in sys.path:
    sys.path.insert(0, str(R2D2_RL))


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class TrainConfig:
    dataset_root: str
    output_dir: str
    image_size: int = 128
    batch_size: int = 128
    epochs: int = 80
    lr: float = 3e-4
    weight_decay: float = 1e-5
    val_episode_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    zero_motor_currents: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------


class RealHILBCDataset(Dataset):
    """Dataset for ``(wrist RGB, 24D HIL state) -> 6D absolute joint action``.

    Scalar features are read from the LeRobot Parquet file. Frames are decoded
    once from the wrist-camera MP4 via system FFmpeg and kept in RAM as uint8
    HWC images. For the current dataset size this is fast enough.
    """

    def __init__(
        self,
        root: Path,
        image_size: int,
        *,
        zero_motor_currents: bool = False,
    ) -> None:
        self.root = root
        self.image_size = image_size
        self.zero_motor_currents = zero_motor_currents

        parquet_files = sorted(root.glob("data/**/*.parquet"))
        if len(parquet_files) != 1:
            raise RuntimeError(
                f"Expected exactly one Parquet file for now, found {len(parquet_files)}: "
                f"{parquet_files}"
            )

        video_files = sorted(root.glob("videos/observation.images.wrist/**/*.mp4"))
        if len(video_files) != 1:
            raise RuntimeError(
                f"Expected exactly one wrist-camera MP4 for now, found {len(video_files)}: "
                f"{video_files}"
            )

        self.parquet_path = parquet_files[0]
        self.video_path = video_files[0]

        self.df = pd.read_parquet(self.parquet_path).reset_index(drop=True)

        self.states = np.stack(self.df["observation.state"].to_numpy()).astype(np.float32)
        if self.zero_motor_currents:
            # Match the sim HIL-compatible env, which exposes zero placeholders
            # for the 6 motor-current channels.
            self.states[:, 12:18] = 0.0
            print("Zeroed observation.state[12:18] motor-current channels.")

        self.actions = np.stack(self.df["action"].to_numpy()).astype(np.float32)
        self.episode_indices = self.df["episode_index"].to_numpy().astype(np.int64)
        self.frame_indices = self.df["frame_index"].to_numpy().astype(np.int64)

        if self.states.shape[1:] != (24,):
            raise RuntimeError(f"Expected state shape [N,24], got {self.states.shape}")
        if self.actions.shape[1:] != (6,):
            raise RuntimeError(f"Expected action shape [N,6], got {self.actions.shape}")

        self.frames = self._decode_video_frames(self.video_path)
        max_frame_index = int(self.frame_indices.max())
        if max_frame_index >= len(self.frames):
            raise RuntimeError(
                f"Parquet references frame_index={max_frame_index}, "
                f"but video only has {len(self.frames)} frames."
            )

        print(f"Loaded rows:   {len(self.df)}")
        print(f"Loaded frames: {len(self.frames)}")
        print(f"Episodes:      {len(np.unique(self.episode_indices))}")
        print(f"State shape:   {self.states.shape}")
        print(f"Action shape:  {self.actions.shape}")

    def _decode_video_frames(self, path: Path) -> list[np.ndarray]:
        """Decode wrist frames with system FFmpeg.

        OpenCV's bundled FFmpeg backend can fail on AV1 MP4s; the system
        ffmpeg binary handles them correctly.
        """
        width = self.image_size
        height = self.image_size
        bytes_per_frame = width * height * 3

        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", str(path),
            "-vf", f"scale={width}:{height}",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ]
        try:
            result = subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg binary not found. Install system FFmpeg.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg failed to decode {path}:\n{stderr}") from exc

        raw = result.stdout
        if len(raw) == 0:
            raise RuntimeError(f"FFmpeg decoded zero bytes from {path}")
        if len(raw) % bytes_per_frame != 0:
            raise RuntimeError(
                f"Unexpected FFmpeg byte count: {len(raw)} not divisible by "
                f"{bytes_per_frame} bytes/frame."
            )

        num_frames = len(raw) // bytes_per_frame
        array = np.frombuffer(raw, dtype=np.uint8).reshape(num_frames, height, width, 3)
        frames = [np.ascontiguousarray(array[i]) for i in range(num_frames)]
        print(f"Decoded {len(frames)} frames from {path} via FFmpeg.")
        return frames

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        frame_idx = int(self.frame_indices[idx])
        image_hwc = self.frames[frame_idx]
        image = torch.from_numpy(image_hwc).permute(2, 0, 1).float() / 255.0
        state = torch.from_numpy(self.states[idx])
        action = torch.from_numpy(self.actions[idx])
        return {"image": image, "state": state, "action": action}


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------


@dataclass
class NormalizationStats:
    state_mean: list[float]
    state_std: list[float]
    action_mean: list[float]
    action_std: list[float]


def compute_normalization_stats(
    dataset: RealHILBCDataset, train_indices: list[int]
) -> NormalizationStats:
    states = dataset.states[train_indices]
    actions = dataset.actions[train_indices]
    state_std = np.maximum(states.std(axis=0), 1e-6)
    action_std = np.maximum(actions.std(axis=0), 1e-6)
    return NormalizationStats(
        state_mean=states.mean(axis=0).astype(float).tolist(),
        state_std=state_std.astype(float).tolist(),
        action_mean=actions.mean(axis=0).astype(float).tolist(),
        action_std=action_std.astype(float).tolist(),
    )


# ---------------------------------------------------------------------
# Model
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class RealHILBCPolicy(nn.Module):
    def __init__(self, state_dim: int = 24, action_dim: int = 6) -> None:
        super().__init__()
        self.image_encoder = WristImageEncoder(out_dim=256)
        self.policy_head = nn.Sequential(
            nn.Linear(256 + state_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

    def forward(self, image: torch.Tensor, normalized_state: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_encoder(image)
        fused = torch.cat([image_feat, normalized_state], dim=-1)
        return self.policy_head(fused)


# ---------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------


def make_episode_split(
    episode_indices: np.ndarray, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    unique_eps = sorted(np.unique(episode_indices).tolist())
    rng = random.Random(seed)
    rng.shuffle(unique_eps)
    n_val = max(1, round(len(unique_eps) * val_fraction))
    val_eps = set(unique_eps[:n_val])

    train_indices: list[int] = []
    val_indices: list[int] = []
    for i, ep in enumerate(episode_indices.tolist()):
        (val_indices if ep in val_eps else train_indices).append(i)
    return train_indices, val_indices


def normalize_batch(
    state: torch.Tensor, action: torch.Tensor, stats_tensors: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    norm_state = (state - stats_tensors["state_mean"]) / stats_tensors["state_std"]
    norm_action = (action - stats_tensors["action_mean"]) / stats_tensors["action_std"]
    return norm_state, norm_action


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    stats_tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_norm_mse = 0.0
    total_raw_mse = 0.0
    total_l1 = 0.0
    total_count = 0
    for batch in loader:
        image = batch["image"].to(device)
        state = batch["state"].to(device)
        action = batch["action"].to(device)
        norm_state, norm_action = normalize_batch(state, action, stats_tensors)
        pred_norm_action = model(image, norm_state)
        norm_mse = F.mse_loss(pred_norm_action, norm_action, reduction="sum")
        pred_action = (
            pred_norm_action * stats_tensors["action_std"] + stats_tensors["action_mean"]
        )
        raw_mse = F.mse_loss(pred_action, action, reduction="sum")
        raw_l1 = F.l1_loss(pred_action, action, reduction="sum")
        batch_size = image.shape[0]
        total_norm_mse += float(norm_mse.item())
        total_raw_mse += float(raw_mse.item())
        total_l1 += float(raw_l1.item())
        total_count += batch_size
    denom = max(1, total_count * 6)
    return {
        "norm_mse": total_norm_mse / denom,
        "raw_mse": total_raw_mse / denom,
        "raw_mae": total_l1 / denom,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(R2D2_RL / "outputs" / "real_hil_bc_warmstart"),
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-episode-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--zero-motor-currents",
        action="store_true",
        help="Set observation.state[12:18] to zero during BC training/validation.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    return TrainConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_episode_fraction=args.val_episode_fraction,
        seed=args.seed,
        num_workers=args.num_workers,
        zero_motor_currents=args.zero_motor_currents,
        device=args.device,
    )


def main() -> None:
    cfg = parse_args()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    with (output_dir / "train_config.json").open("w") as f:
        json.dump(asdict(cfg), f, indent=2)

    dataset = RealHILBCDataset(
        Path(cfg.dataset_root),
        image_size=cfg.image_size,
        zero_motor_currents=cfg.zero_motor_currents,
    )
    train_indices, val_indices = make_episode_split(
        dataset.episode_indices,
        val_fraction=cfg.val_episode_fraction,
        seed=cfg.seed,
    )
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples:   {len(val_indices)}")

    stats = compute_normalization_stats(dataset, train_indices)
    with (output_dir / "normalization_stats.json").open("w") as f:
        json.dump(asdict(stats), f, indent=2)

    device = torch.device(cfg.device)
    print(f"Device: {device}")

    stats_tensors = {
        "state_mean": torch.tensor(stats.state_mean, dtype=torch.float32, device=device),
        "state_std": torch.tensor(stats.state_std, dtype=torch.float32, device=device),
        "action_mean": torch.tensor(stats.action_mean, dtype=torch.float32, device=device),
        "action_std": torch.tensor(stats.action_std, dtype=torch.float32, device=device),
    }

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = RealHILBCPolicy().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
    best_val = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        for batch in train_loader:
            image = batch["image"].to(device)
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            norm_state, norm_action = normalize_batch(state, action, stats_tensors)
            pred_norm_action = model(image, norm_state)
            loss = F.mse_loss(pred_norm_action, norm_action)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = image.shape[0]
            running_loss += float(loss.item()) * batch_size
            running_count += batch_size

        train_norm_mse = running_loss / max(1, running_count)
        val_metrics = evaluate(model, val_loader, stats_tensors, device)
        writer.add_scalar("bc/train_norm_mse", train_norm_mse, epoch)
        writer.add_scalar("bc/val_norm_mse", val_metrics["norm_mse"], epoch)
        writer.add_scalar("bc/val_raw_mse", val_metrics["raw_mse"], epoch)
        writer.add_scalar("bc/val_raw_mae", val_metrics["raw_mae"], epoch)
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train_norm_mse={train_norm_mse:.6f} | "
            f"val_norm_mse={val_metrics['norm_mse']:.6f} | "
            f"val_raw_mae={val_metrics['raw_mae']:.4f}"
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "normalization_stats": asdict(stats),
            "train_config": asdict(cfg),
            "epoch": epoch,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "checkpoints" / "last.pt")
        if val_metrics["norm_mse"] < best_val:
            best_val = val_metrics["norm_mse"]
            torch.save(checkpoint, output_dir / "checkpoints" / "best.pt")

    writer.close()
    print(f"Done. Best validation normalized MSE: {best_val:.6f}")


if __name__ == "__main__":
    main()
