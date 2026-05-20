"""Hybrid-runtime adapter for Flo/SB3 visual align-grasp checkpoints.

``train_visual_hil_compat_sac.py`` trains an SB3 SAC policy with the LeRobot
SO-101 local-grasp interface:

- ``observation.images.wrist``: uint8 RGB, CHW, [3, 128, 128]
- ``observation.state``: 24D float32
- action: 6D absolute follower target in calibrated LeRobot units

The hybrid executor, however, only requires a ``LocalPolicy`` with
``run("align_grasp", target_color, belief) -> bool``. This adapter bridges
those two contracts while leaving the existing observe / approach / lift /
transport / release pipeline intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from control.waypoint_controller import RcsWaypointController
from estimation.block_belief import BlockBelief
from rl.lerobot_compat import (
    DEFAULT_ACTION_HIGH,
    DEFAULT_ACTION_LOW,
    IMAGE_KEY,
    STATE_KEY,
    target_color_onehot,
)
from runtime.rcs_sim_adapters import DEFAULT_WRIST_CAMERA_NAME, wrist_rgb_from_obs


@dataclass
class SB3VisualAlignGraspPolicy:
    """Run an SB3 visual SAC checkpoint as the hybrid ``align_grasp`` phase."""

    controller: RcsWaypointController
    config: dict[str, Any]
    checkpoint_path: str | Path
    device: str = "cpu"
    max_steps: int = 80
    camera_name: str = DEFAULT_WRIST_CAMERA_NAME
    image_size: int = 128
    real_gripper_max: float = 35.0
    compat_dt_s: float = 0.1
    joint_delta_deg: float = 5.0
    action_scale: float = 1.0
    close_gripper_threshold: float = 0.5

    def __post_init__(self) -> None:
        from stable_baselines3 import SAC

        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if self.image_size < 1:
            raise ValueError("image_size must be >= 1.")

        self._model = SAC.load(str(self.checkpoint_path), device=self.device)
        self._prev_real_positions: np.ndarray | None = None

    # ------------------------------------------------------------------ LocalPolicy

    def run(self, phase: str, target_color: str, belief: BlockBelief) -> bool:
        if phase != "align_grasp":
            return False
        if belief is None or not belief.initialized:
            return False
        if self.controller.last_obs is None:
            return False

        self._prev_real_positions = None
        for _ in range(self.max_steps):
            obs = self._build_obs(target_color)
            if obs is None:
                return False

            action, _state = self._model.predict(obs, deterministic=True)
            action = self._clip_lerobot_action(action)

            _obs, _reward, terminated, truncated, info = self._apply_lerobot_action(action)
            if terminated or truncated:
                return False

            gripper_cmd = self._gripper_action_to_sim(action)
            if gripper_cmd <= self.close_gripper_threshold and self._grasp_confirmed(info):
                return True

        return False

    # ------------------------------------------------------------------ observation

    def _build_obs(self, target_color: str) -> dict[str, np.ndarray] | None:
        latest = self.controller.last_obs
        if latest is None:
            return None

        rgb = wrist_rgb_from_obs(latest, camera_name=self.camera_name)
        if rgb is None:
            return None
        image_chw = self._image_to_chw(rgb)

        real_positions = self._real_positions_from_obs(latest)
        if real_positions is None:
            return None
        if self._prev_real_positions is None:
            velocities = np.zeros(6, dtype=np.float32)
        else:
            velocities = (
                (real_positions - self._prev_real_positions) / max(1e-6, self.compat_dt_s)
            ).astype(np.float32)
        self._prev_real_positions = real_positions.copy()

        state = np.concatenate(
            [
                real_positions,
                velocities,
                np.zeros(6, dtype=np.float32),
                target_color_onehot(target_color),
            ],
            axis=0,
        ).astype(np.float32)

        return {
            IMAGE_KEY: image_chw,
            STATE_KEY: state,
        }

    def _image_to_chw(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB wrist image, got shape {rgb.shape}.")
        if rgb.shape[0] != self.image_size or rgb.shape[1] != self.image_size:
            rgb = _resize_rgb(rgb, self.image_size, self.image_size)
        return np.transpose(rgb, (2, 0, 1)).copy()

    def _real_positions_from_obs(self, obs: dict[str, Any]) -> np.ndarray | None:
        robot_obs = obs.get("robot") if isinstance(obs, dict) else None
        if not isinstance(robot_obs, dict):
            return None
        joints = robot_obs.get("joints")
        gripper = robot_obs.get("gripper")
        if joints is None or gripper is None:
            return None

        joints_deg = np.rad2deg(np.asarray(joints, dtype=np.float32).reshape(5)).astype(np.float32)
        gripper_norm = float(np.asarray(gripper, dtype=np.float32).reshape(-1)[0])
        gripper_real = np.array(
            [np.clip(gripper_norm, 0.0, 1.0) * self.real_gripper_max],
            dtype=np.float32,
        )
        return np.concatenate([joints_deg, gripper_real], axis=0).astype(np.float32)

    # ------------------------------------------------------------------ action

    def _clip_lerobot_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(6)
        return np.clip(action, DEFAULT_ACTION_LOW, DEFAULT_ACTION_HIGH).astype(np.float32)

    def _apply_lerobot_action(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        robots = self.controller.env.get_wrapper_attr("robot")
        robot = robots["robot"] if isinstance(robots, dict) else robots
        if not hasattr(robot, "set_joint_position"):
            raise RuntimeError("RCS robot does not expose set_joint_position; cannot apply SB3 action.")

        if self._prev_real_positions is None:
            raise RuntimeError("Cannot apply SB3 action before building the LeRobot observation.")

        current_joints_deg = self._prev_real_positions[:5]
        joint_delta_deg = action[:5] - current_joints_deg
        max_delta_deg = self.joint_delta_deg * self.action_scale
        stepped_joints_deg = current_joints_deg + np.clip(
            joint_delta_deg,
            -max_delta_deg,
            max_delta_deg,
        )
        target_joints_rad = np.deg2rad(stepped_joints_deg).astype(np.float64)
        robot.set_joint_position(target_joints_rad)

        # Advance wrappers/camera and command the gripper through the normal
        # controller path so last_obs/last_info stay coherent for the executor.
        gripper_cmd = self._gripper_action_to_sim(action)
        return self.controller.step_delta(np.zeros(3, dtype=np.float64), gripper=gripper_cmd)

    def _gripper_action_to_sim(self, action: np.ndarray) -> float:
        return float(np.clip(float(action[5]) / max(1e-6, self.real_gripper_max), 0.0, 1.0))

    # ------------------------------------------------------------------ success

    def _grasp_confirmed(self, info: dict[str, Any]) -> bool:
        """Use explicit RCS grasp feedback when present; otherwise fall back.

        The current Project3SO101 hybrid scene does not always expose a PickTask
        grasp flag. In that case, matching the older learned adapter, a close
        command is treated as sufficient for the executor to proceed to the
        classical lift phase.
        """
        if not isinstance(info, dict):
            return True
        if "success" in info and bool(info["success"]):
            return True
        if "is_grasped" in info:
            return bool(info["is_grasped"])
        robot = info.get("robot")
        if isinstance(robot, dict) and "is_grasped" in robot:
            return bool(robot["is_grasped"])
        return True


# Backwards-friendly name matching the team discussion.
FloVisualAlignGraspPolicy = SB3VisualAlignGraspPolicy


def _resize_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA).astype(np.uint8)
    except ImportError:
        src_h, src_w = rgb.shape[:2]
        ys = np.linspace(0, src_h - 1, height).astype(np.int64)
        xs = np.linspace(0, src_w - 1, width).astype(np.int64)
        return np.asarray(rgb[ys][:, xs], dtype=np.uint8)
