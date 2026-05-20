"""Inference adapter for LeRobot-compatible visual align-grasp checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass
class LeRobotVisualAlignGraspPolicy:
    """Load a visual SAC checkpoint and emit absolute SO-101 follower actions.

    The input observation must contain:

    - ``observation.images.wrist``: uint8 RGB CHW [3, 128, 128]
    - ``observation.state``: float32 [24]

    The returned action is a 6D absolute target in the same calibrated units as
    the real LeRobot SO-101 follower path used by the HIL dataset.
    """

    checkpoint_path: str | Path
    device: str = "cpu"

    def __post_init__(self) -> None:
        from rl.lerobot_compat import scaled_to_lerobot_action
        from rl.visual_sac import load_visual_agent_for_inference

        self._scaled_to_lerobot_action = scaled_to_lerobot_action
        self._agent = load_visual_agent_for_inference(self.checkpoint_path, device=self.device)

    def act(self, observation: Mapping[str, np.ndarray], *, deterministic: bool = True) -> np.ndarray:
        obs = {key: np.asarray(value) for key, value in observation.items()}
        scaled_action = self._agent.act(obs, deterministic=deterministic)
        return self._scaled_to_lerobot_action(scaled_action).astype(np.float32)
