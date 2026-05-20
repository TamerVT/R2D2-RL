"""LeRobot-compatible observation/action constants for SO-101 grasp policies."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


IMAGE_KEY = "observation.images.wrist"
STATE_KEY = "observation.state"

HIL_COLOR_NAMES = ["blue", "green", "purple", "orange", "yellow", "red"]
HIL_COLOR_TO_INDEX = {name: idx for idx, name in enumerate(HIL_COLOR_NAMES)}

# Calibration-derived follower ranges from Flo's HIL-compatible setup.
DEFAULT_ACTION_LOW = np.array(
    [-68.90625, -103.7548828125, -97.470703125, -102.216796875, -179.9560546875, 0.0],
    dtype=np.float32,
)
DEFAULT_ACTION_HIGH = np.array(
    [68.90625, 103.7548828125, 97.470703125, 102.216796875, 179.9560546875, 100.0],
    dtype=np.float32,
)


def target_color_onehot(color: str) -> np.ndarray:
    if color not in HIL_COLOR_TO_INDEX:
        raise ValueError(f"Unknown target color {color!r}. Available: {HIL_COLOR_NAMES}")
    onehot = np.zeros(len(HIL_COLOR_NAMES), dtype=np.float32)
    onehot[HIL_COLOR_TO_INDEX[color]] = 1.0
    return onehot


def scaled_to_lerobot_action(scaled_action: np.ndarray) -> np.ndarray:
    scaled_action = np.asarray(scaled_action, dtype=np.float32).reshape(6)
    scaled_action = np.clip(scaled_action, -1.0, 1.0)
    return DEFAULT_ACTION_LOW + 0.5 * (scaled_action + 1.0) * (
        DEFAULT_ACTION_HIGH - DEFAULT_ACTION_LOW
    )


def lerobot_to_scaled_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(6)
    action = np.clip(action, DEFAULT_ACTION_LOW, DEFAULT_ACTION_HIGH)
    scaled = 2.0 * (action - DEFAULT_ACTION_LOW) / (DEFAULT_ACTION_HIGH - DEFAULT_ACTION_LOW) - 1.0
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def load_state_normalization(path: str | Path | None) -> tuple[np.ndarray, np.ndarray]:
    """Load 24D state normalization from Flo JSON or LeRobot ``meta/stats.json``.

    Returns zero mean / unit std when ``path`` is omitted.
    """

    if path is None:
        return np.zeros(24, dtype=np.float32), np.ones(24, dtype=np.float32)

    path = Path(path)
    if path.suffix == ".zip":
        with ZipFile(path) as archive:
            stats_names = [name for name in archive.namelist() if name.endswith("/meta/stats.json")]
            if not stats_names:
                raise ValueError(f"{path} does not contain a LeRobot meta/stats.json file.")
            data = json.loads(archive.read(stats_names[0]).decode("utf-8"))
    else:
        data = json.loads(path.read_text())
    if "state_mean" in data and "state_std" in data:
        mean = np.asarray(data["state_mean"], dtype=np.float32)
        std = np.asarray(data["state_std"], dtype=np.float32)
    elif STATE_KEY in data and "mean" in data[STATE_KEY] and "std" in data[STATE_KEY]:
        mean = np.asarray(data[STATE_KEY]["mean"], dtype=np.float32)
        std = np.asarray(data[STATE_KEY]["std"], dtype=np.float32)
    else:
        raise ValueError(
            f"{path} does not contain state_mean/state_std or {STATE_KEY}.mean/std."
        )

    if mean.shape != (24,) or std.shape != (24,):
        raise ValueError(f"Expected 24D normalization, got mean={mean.shape} std={std.shape}.")
    return mean.astype(np.float32), np.clip(std.astype(np.float32), 1e-6, None)
