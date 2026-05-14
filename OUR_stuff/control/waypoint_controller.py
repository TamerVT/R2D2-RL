"""Waypoint controller adapters.

The RCS adapter is intentionally small: it converts absolute base-frame
waypoints into repeated relative ``tquat`` translation commands for the RCS
Gymnasium action space. Orientation tracking is deferred to the RCS IK/control
stack; for now, waypoints keep the current wrist orientation and command only
translation plus gripper state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from planning.hybrid_waypoint_planner import Waypoint


@dataclass
class RcsWaypointController:
    """Execute :class:`Waypoint` objects on an RCS SO101 Gym env."""

    env: Any
    max_step_m: float = 0.025
    position_tolerance_m: float = 0.025
    max_steps_per_waypoint: int = 30
    strict_position: bool = False

    def __post_init__(self) -> None:
        if self.max_step_m <= 0:
            raise ValueError("max_step_m must be positive.")
        if self.position_tolerance_m <= 0:
            raise ValueError("position_tolerance_m must be positive.")
        if self.max_steps_per_waypoint < 1:
            raise ValueError("max_steps_per_waypoint must be >= 1.")
        self.last_obs: dict[str, Any] | None = None
        self.last_info: dict[str, Any] | None = None
        self.step_count = 0

    def reset(self, seed: int | None = None):
        obs, info = self.env.reset(seed=seed)
        self.last_obs = obs
        self.last_info = info
        self.step_count = 0
        return obs, info

    def step_delta(self, delta_xyz: np.ndarray, gripper: float | None = None):
        action = self._action(delta_xyz, gripper)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_obs = obs
        self.last_info = info
        self.step_count += 1
        return obs, reward, terminated, truncated, info

    def execute(self, waypoints: Sequence[Waypoint]) -> bool:
        if self.last_obs is None:
            self.reset(seed=0)

        for waypoint in waypoints:
            reached = False
            for _ in range(self.max_steps_per_waypoint):
                current_xyz = self.robot_xyz
                error = waypoint.xyz_base - current_xyz
                distance = float(np.linalg.norm(error))
                if distance <= self.position_tolerance_m:
                    reached = True
                    break
                delta_xyz = _proportional_step(error, self.max_step_m)
                _, _, terminated, truncated, _ = self.step_delta(delta_xyz, waypoint.gripper)
                if terminated or truncated:
                    return False

            if waypoint.gripper is not None:
                _, _, terminated, truncated, _ = self.step_delta(np.zeros(3), waypoint.gripper)
                if terminated or truncated:
                    return False

            if self.strict_position and not reached:
                final_error = float(np.linalg.norm(waypoint.xyz_base - self.robot_xyz))
                if final_error > self.position_tolerance_m:
                    return False

        return True

    @property
    def robot_xyz(self) -> np.ndarray:
        if self.last_obs is None:
            raise RuntimeError("Controller has no observation; call reset first.")
        robot = self.last_obs.get("robot") if isinstance(self.last_obs, dict) else None
        if not isinstance(robot, dict):
            raise RuntimeError("RCS observation does not contain a robot dict.")
        if "xyzrpy" in robot:
            return np.asarray(robot["xyzrpy"], dtype=np.float64).reshape(-1)[:3]
        if "tquat" in robot:
            return np.asarray(robot["tquat"], dtype=np.float64).reshape(-1)[:3]
        raise RuntimeError("RCS robot observation has neither xyzrpy nor tquat.")

    def _action(self, delta_xyz: np.ndarray, gripper: float | None) -> dict[str, dict[str, np.ndarray]]:
        delta = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(delta)):
            raise ValueError("delta_xyz must be finite.")

        # ``gripper=None`` means "preserve current state" — read it back from
        # the latest observation. Passing 1.0 (open) by default would drop a
        # held cube during pure-observation steps.
        if gripper is None:
            gripper = self._current_gripper_state()

        action: dict[str, dict[str, np.ndarray]] = {}
        for robot_key, robot_space in self.env.action_space.spaces.items():
            robot_action: dict[str, np.ndarray] = {}
            spaces = getattr(robot_space, "spaces", {})
            if "tquat" in spaces:
                robot_action["tquat"] = np.array(
                    [delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float64,
                )
            if "gripper" in spaces:
                value = float(gripper) if gripper is not None else 1.0
                robot_action["gripper"] = np.array([np.clip(value, 0.0, 1.0)], dtype=np.float32)
            action[robot_key] = robot_action
        return action

    def _current_gripper_state(self) -> float | None:
        """Read the current normalized gripper value from the latest observation."""
        if not isinstance(self.last_obs, dict):
            return None
        robot = self.last_obs.get("robot")
        if not isinstance(robot, dict):
            return None
        gripper = robot.get("gripper")
        try:
            value = float(np.asarray(gripper, dtype=np.float64).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            return None
        return float(np.clip(value, 0.0, 1.0))


def _proportional_step(vector: np.ndarray, max_step: float) -> np.ndarray:
    """Cap the step at ``max_step`` while preserving the direction of ``vector``.

    A previous version used dominant-axis stepping (one axis at a time), which
    produced zigzag Cartesian paths and worked the IK harder than necessary.
    Returning ``vector * (max_step / norm)`` when the norm exceeds the limit
    keeps the gripper moving in a straight line toward the waypoint.
    """
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.zeros_like(vector)
    if norm <= max_step:
        return vector.astype(np.float64, copy=True)
    return vector * (max_step / norm)
