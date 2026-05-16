from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


class PregraspJointMLP(nn.Module):
    """
    Small MLP:
        cube_xyz -> pregrasp joint positions
    """

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 6,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        last_dim = input_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class PregraspNormalization:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray

    def normalize_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def denormalize_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean


@dataclass
class LoadedPregraspRegressor:
    model: PregraspJointMLP
    normalization: PregraspNormalization
    joint_keys: list[str]
    device: torch.device

    @torch.inference_mode()
    def predict(self, cube_xyz: np.ndarray) -> np.ndarray:
        cube_xyz = np.asarray(cube_xyz, dtype=np.float32).reshape(1, -1)

        x_norm = self.normalization.normalize_x(cube_xyz)
        x_tensor = torch.from_numpy(x_norm.astype(np.float32)).to(self.device)

        y_norm = self.model(x_tensor).cpu().numpy()
        y = self.normalization.denormalize_y(y_norm)
        return y.reshape(-1)


def save_pregrasp_checkpoint(
    *,
    path: str | Path,
    model: PregraspJointMLP,
    normalization: PregraspNormalization,
    model_config: dict[str, Any],
    joint_keys: list[str],
    training_summary: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "joint_keys": joint_keys,
        "normalization": {
            "x_mean": normalization.x_mean.tolist(),
            "x_std": normalization.x_std.tolist(),
            "y_mean": normalization.y_mean.tolist(),
            "y_std": normalization.y_std.tolist(),
        },
        "training_summary": training_summary,
    }

    torch.save(payload, path)


def load_pregrasp_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedPregraspRegressor:
    path = Path(path)
    device = torch.device(device)

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model_config = checkpoint["model_config"]
    model = PregraspJointMLP(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    norm = checkpoint["normalization"]
    normalization = PregraspNormalization(
        x_mean=np.asarray(norm["x_mean"], dtype=np.float32),
        x_std=np.asarray(norm["x_std"], dtype=np.float32),
        y_mean=np.asarray(norm["y_mean"], dtype=np.float32),
        y_std=np.asarray(norm["y_std"], dtype=np.float32),
    )

    return LoadedPregraspRegressor(
        model=model,
        normalization=normalization,
        joint_keys=list(checkpoint["joint_keys"]),
        device=device,
    )
