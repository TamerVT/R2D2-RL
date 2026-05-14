"""Adapter that exposes a trained SAC actor as the executor's ``LocalPolicy``.

The hybrid task executor calls ``policy.run("align_grasp", target_color, belief)``
and expects ``True`` on success. This adapter:

1. Loads a SAC checkpoint built by ``r2d2_rl/scripts/train_align_grasp.py``.
2. Builds the 11-D observation expected by :class:`AlignGraspEnv`, using
   the belief mean as the target XY (no perception loop here — the
   executor's observer already produced the belief).
3. Iteratively samples actions from the SAC actor and applies them through
   the RCS waypoint controller (delta XYZ + gripper), stepping until the
   gripper closes around a successfully aligned grasp or a step budget is
   exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from control.waypoint_controller import RcsWaypointController
from estimation.block_belief import BlockBelief


@dataclass
class LearnedAlignGraspPolicy:
    """Run a trained SAC actor as the executor's ``align_grasp`` phase."""

    controller: RcsWaypointController
    config: dict[str, Any]
    checkpoint_path: str | Path
    device: str = "cpu"
    max_steps: int = 60
    close_gripper_threshold: float = 0.5
    z_grasp: float | None = None
    delta_xyz_max: float | None = None

    def __post_init__(self) -> None:
        from rl.sac import load_agent_for_inference  # local import; torch is heavy

        rl_cfg = (self.config.get("rl") or {}).get("action_space", {})
        if self.delta_xyz_max is None:
            self.delta_xyz_max = float(rl_cfg.get("delta_xyz_max", 0.015))
        planning_cfg = self.config.get("planning") or {}
        if self.z_grasp is None:
            self.z_grasp = float(planning_cfg.get("z_grasp", 0.025))

        self._agent = load_agent_for_inference(self.checkpoint_path, device=self.device)
        self._prev_action = np.zeros(4, dtype=np.float32)

    # ----------------------------------------------------------------- run

    def run(self, phase: str, target_color: str, belief: BlockBelief) -> bool:
        if phase != "align_grasp":
            return False
        if belief is None or not belief.initialized:
            return False

        self._prev_action = np.zeros(4, dtype=np.float32)
        for _ in range(self.max_steps):
            obs = self._build_obs(belief)
            action = self._agent.act(obs, deterministic=True).astype(np.float32)
            delta_xyz = action[:3] * float(self.delta_xyz_max)
            gripper_cmd = float(np.clip((action[3] + 1.0) * 0.5, 0.0, 1.0))

            _, _, term, trunc, _ = self.controller.step_delta(delta_xyz, gripper=gripper_cmd)
            self._prev_action = action

            if term or trunc:
                return False

            if gripper_cmd <= self.close_gripper_threshold:
                # Treat first commanded close as the grasp attempt; the
                # executor's classical lift will verify whether the cube
                # actually came along, and the watchdog handles loss.
                return True

        return False

    # ----------------------------------------------------------------- obs

    def _build_obs(self, belief: BlockBelief) -> np.ndarray:
        robot = self.controller.last_obs.get("robot") if isinstance(self.controller.last_obs, dict) else None
        if isinstance(robot, dict) and "xyzrpy" in robot:
            ee_xyz = np.asarray(robot["xyzrpy"], dtype=np.float32).reshape(-1)[:3]
        elif isinstance(robot, dict) and "tquat" in robot:
            ee_xyz = np.asarray(robot["tquat"], dtype=np.float32).reshape(-1)[:3]
        else:
            ee_xyz = np.zeros(3, dtype=np.float32)

        gripper = 1.0
        if isinstance(robot, dict):
            try:
                gripper = float(np.asarray(robot.get("gripper"), dtype=np.float32).reshape(-1)[0])
            except (TypeError, ValueError, IndexError):
                pass

        cube_xy = np.asarray(belief.mean_xy, dtype=np.float32).reshape(2)
        cube_z = float(self.config.get("workspace", {}).get("z_object_center", 0.02))

        return np.concatenate(
            [
                ee_xyz[:2],
                ee_xyz[2:3],
                cube_xy,
                np.array([cube_z], dtype=np.float32),
                np.array([gripper], dtype=np.float32),
                self._prev_action,
            ],
            dtype=np.float32,
        )
