"""Pregrasp joint regressor used to align sim resets with real SO-101 rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

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
    """Small MLP mapping cube xyz to an absolute 6D SO-101 pregrasp pose."""

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
        return (x - self.x_mean) / np.clip(self.x_std, 1e-6, None)

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
        return self.normalization.denormalize_y(y_norm).reshape(-1)


def load_pregrasp_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedPregraspRegressor:
    """Load Flo's pregrasp checkpoint format."""

    path = Path(path)
    device = torch.device(device)
    if path.suffix == ".zip":
        with ZipFile(path) as archive:
            ckpt_names = [name for name in archive.namelist() if name.endswith("best_pregrasp_mlp.pt")]
            if not ckpt_names:
                raise ValueError(f"{path} does not contain best_pregrasp_mlp.pt.")
            checkpoint: dict[str, Any] = torch.load(
                BytesIO(archive.read(ckpt_names[0])),
                map_location=device,
                weights_only=False,
            )
    else:
        checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = PregraspJointMLP(**checkpoint["model_config"])
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
        joint_keys=list(checkpoint.get("joint_keys", JOINT_KEYS)),
        device=device,
    )
