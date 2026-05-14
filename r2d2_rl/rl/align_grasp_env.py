"""Gym env wrapper for training the local ``align_grasp`` policy on SO-101.

Wraps :class:`envs.project3_so101_env.Project3SO101Env` for the contact-rich
final-alignment + grasp phase of Project 3:

- **Observation** (flat, 11-D float32) — ``[ee_xy (2), ee_z (1),
  cube_xy (2), cube_z (1), gripper (1), prev_action (4)]`` in robot-base
  frame, with cube position read directly from the sim free joint (the
  hybrid runtime would substitute the belief mean here at deployment).
- **Action** (4-D float32, each in [-1, 1]) — ``[delta_x, delta_y, delta_z,
  gripper]``. Translation deltas are scaled by ``rl.action_space.delta_xyz_max``
  from the config; the gripper command is rescaled to [0, 1].
- **Reward** — dense alignment shaping (XY distance, Z error) + sparse lift
  bonus when the cube center clears ``lift_height_threshold`` above the
  table, plus a small action-norm cost.
- **Reset** — places the cube at a random workspace XY, snaps the EE to a
  pregrasp pose above it (with a small noise perturbation), opens the
  gripper.

The wrapper is deliberately lightweight — it does not include perception or
belief tracking, since the goal is to train a local manipulation policy
that consumes already-localized state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.project3_so101_env import (
    CubeSpec,
    Project3SO101Config,
    Project3SO101Env,
)


@dataclass
class AlignGraspEnvConfig:
    """Override knobs for the training env."""

    cube_color: str = "green"
    workspace_x: tuple[float, float] = (0.16, 0.28)
    workspace_y: tuple[float, float] = (-0.08, 0.08)
    cube_z_init: float = 0.02
    z_pregrasp: float = 0.10
    z_grasp: float = 0.025
    lift_height_threshold: float = 0.05
    initial_ee_noise_xy: float = 0.02
    initial_ee_noise_z: float = 0.01
    delta_xyz_max: float = 0.015
    max_steps: int = 60
    reward_align_scale: float = 1.0
    reward_z_scale: float = 0.5
    reward_action_cost: float = 0.05
    reward_lift_bonus: float = 5.0
    reward_grasp_bonus: float = 1.0
    success_lift_height: float = 0.06
    end_on_success: bool = True


class AlignGraspEnv(gym.Env):
    """Training env for the SO-101 ``align_grasp`` SAC policy.

    Notes:
        - The underlying RCS env defaults to ``headless=True``, no GUI.
        - The full reset (rebuilding the MJCF scene) is slow, so we only
          rebuild the env on the first reset and re-seed / re-place the cube
          via the sim's data buffers on subsequent resets.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: AlignGraspEnvConfig | None = None):
        super().__init__()
        self.cfg = config or AlignGraspEnvConfig()
        self._np_random = np.random.default_rng()

        # Build a single Project3SO101 env once and reuse it for every episode.
        p3_cfg = Project3SO101Config(
            cubes=[CubeSpec(color=self.cfg.cube_color, xy=(0.20, 0.0), z=self.cfg.cube_z_init)],
            headless=True,
        )
        self._scene = Project3SO101Env(p3_cfg)
        self._env = self._scene.create_env(self._scene.config())

        self._sim = self._env.get_wrapper_attr("sim")
        self._cube_joint_name = self._find_cube_joint_name()

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        self._prev_action = np.zeros(4, dtype=np.float32)
        self._step_count = 0
        self._target_xy = np.zeros(2, dtype=np.float32)
        self._last_obs_dict: dict[str, Any] | None = None

    # ----------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        target_xy = np.array(
            [
                self._np_random.uniform(*self.cfg.workspace_x),
                self._np_random.uniform(*self.cfg.workspace_y),
            ],
            dtype=np.float32,
        )
        self._target_xy = target_xy

        obs_dict, _ = self._env.reset(seed=seed)
        self._last_obs_dict = obs_dict
        self._place_cube(target_xy)
        self._move_ee_to_pregrasp(target_xy)
        obs_dict = self._read_obs_dict()

        self._prev_action = np.zeros(4, dtype=np.float32)
        self._step_count = 0
        self._last_obs_dict = obs_dict
        return self._flat_obs(obs_dict), {"target_xy": target_xy.tolist()}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(4)
        action = np.clip(action, -1.0, 1.0)

        env_action = self._to_env_action(action)
        obs_dict, _, terminated, truncated, info = self._env.step(env_action)

        self._step_count += 1
        self._prev_action = action
        self._last_obs_dict = obs_dict

        flat_obs = self._flat_obs(obs_dict)
        reward, reward_info = self._compute_reward(flat_obs, action)
        info = {**info, **reward_info, "target_xy": self._target_xy.tolist()}

        truncated = bool(truncated) or self._step_count >= self.cfg.max_steps
        if self.cfg.end_on_success and reward_info.get("success", 0.0) >= 1.0:
            terminated = True
        return flat_obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    # ----------------------------------------------------------------- helpers

    def _find_cube_joint_name(self) -> str:
        for joint_id in range(self._sim.model.njnt):
            name = self._sim.model.joint(joint_id).name
            if name and name.endswith("box_joint") and self.cfg.cube_color in name:
                return name
        raise RuntimeError(f"Cube joint for color '{self.cfg.cube_color}' not found in MJCF.")

    def _place_cube(self, xy: np.ndarray) -> None:
        joint = self._sim.data.joint(self._cube_joint_name)
        qpos = joint.qpos
        qpos[:3] = np.array([xy[0], xy[1], self.cfg.cube_z_init], dtype=np.float64)
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        # Zero out the free-joint velocity.
        joint.qvel[:] = 0.0

    def _move_ee_to_pregrasp(self, xy: np.ndarray) -> None:
        """Step the env with a small bias upward; coarse stand-in for IK reset.

        The full reset chain (rebuilding RCS configs, MuJoCo step) does not
        give us a clean joint-space goal, so we drive the EE near the
        pregrasp pose by issuing a few small delta commands. This keeps the
        wrapper compatible with the relative-action RCS env.
        """
        # A few upward + horizontal nudges to settle near the pregrasp pose.
        noise = self._np_random.normal(0.0, self.cfg.initial_ee_noise_xy, size=2)
        z_noise = float(self._np_random.normal(0.0, self.cfg.initial_ee_noise_z))
        target = np.array(
            [xy[0] + noise[0], xy[1] + noise[1], self.cfg.z_pregrasp + z_noise],
            dtype=np.float64,
        )

        for _ in range(8):
            obs = self._read_obs_dict()
            ee_xyz = self._ee_xyz_from_obs(obs)
            error = target - ee_xyz
            step = np.clip(error * 0.5, -self.cfg.delta_xyz_max, self.cfg.delta_xyz_max)
            if np.linalg.norm(error) < 0.005:
                break
            self._env.step(self._cartesian_step(step, gripper=1.0))

    def _read_obs_dict(self) -> dict[str, Any]:
        """Pull a fresh observation by issuing a zero-action step.

        RCS exposes the latest obs via the wrapper return value, so we issue
        a zero-delta no-op to get a current view without altering the state.
        """
        obs, _, _, _, _ = self._env.step(self._cartesian_step(np.zeros(3), gripper=None))
        self._last_obs_dict = obs
        return obs

    def _cartesian_step(self, delta_xyz: np.ndarray, gripper: float | None) -> dict[str, Any]:
        if gripper is None:
            gripper = self._current_gripper_state()

        action: dict[str, dict[str, np.ndarray]] = {}
        for robot_key, robot_space in self._env.action_space.spaces.items():
            sub: dict[str, np.ndarray] = {}
            spaces_dict = getattr(robot_space, "spaces", {})
            if "tquat" in spaces_dict:
                sub["tquat"] = np.array(
                    [delta_xyz[0], delta_xyz[1], delta_xyz[2], 0.0, 0.0, 0.0, 1.0],
                    dtype=np.float64,
                )
            if "gripper" in spaces_dict:
                value = 1.0 if gripper is None else float(np.clip(gripper, 0.0, 1.0))
                sub["gripper"] = np.array([value], dtype=np.float32)
            action[robot_key] = sub
        return action

    def _current_gripper_state(self) -> float | None:
        if not isinstance(self._last_obs_dict, dict):
            return None
        robot = self._last_obs_dict.get("robot")
        if not isinstance(robot, dict):
            return None
        try:
            value = float(np.asarray(robot.get("gripper"), dtype=np.float64).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            return None
        return float(np.clip(value, 0.0, 1.0))

    def _to_env_action(self, normalized: np.ndarray) -> dict[str, Any]:
        delta_xyz = normalized[:3] * self.cfg.delta_xyz_max
        gripper = float((normalized[3] + 1.0) * 0.5)
        return self._cartesian_step(delta_xyz, gripper=gripper)

    def _ee_xyz_from_obs(self, obs_dict: dict[str, Any]) -> np.ndarray:
        robot = obs_dict.get("robot") if isinstance(obs_dict, dict) else None
        if not isinstance(robot, dict):
            return np.zeros(3, dtype=np.float64)
        if "xyzrpy" in robot:
            return np.asarray(robot["xyzrpy"], dtype=np.float64).reshape(-1)[:3]
        if "tquat" in robot:
            return np.asarray(robot["tquat"], dtype=np.float64).reshape(-1)[:3]
        return np.zeros(3, dtype=np.float64)

    def _gripper_from_obs(self, obs_dict: dict[str, Any]) -> float:
        robot = obs_dict.get("robot") if isinstance(obs_dict, dict) else None
        if not isinstance(robot, dict):
            return 1.0
        gripper = robot.get("gripper")
        try:
            return float(np.asarray(gripper, dtype=np.float64).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            return 1.0

    def _cube_xyz_from_sim(self) -> np.ndarray:
        return np.asarray(self._sim.data.joint(self._cube_joint_name).qpos[:3], dtype=np.float32)

    def _flat_obs(self, obs_dict: dict[str, Any]) -> np.ndarray:
        ee_xyz = self._ee_xyz_from_obs(obs_dict).astype(np.float32)
        cube_xyz = self._cube_xyz_from_sim()
        gripper = np.array([self._gripper_from_obs(obs_dict)], dtype=np.float32)
        return np.concatenate(
            [ee_xyz[:2], ee_xyz[2:3], cube_xyz[:2], cube_xyz[2:3], gripper, self._prev_action],
            dtype=np.float32,
        )

    def _compute_reward(self, obs: np.ndarray, action: np.ndarray) -> tuple[float, dict[str, float]]:
        ee_xy = obs[0:2]
        ee_z = obs[2]
        cube_xy = obs[3:5]
        cube_z = obs[5]

        xy_err = float(np.linalg.norm(ee_xy - cube_xy))
        z_err = float(abs(ee_z - self.cfg.z_grasp))
        action_cost = float(np.linalg.norm(action))

        cube_above = float(cube_z - self.cfg.cube_z_init)
        lifted = cube_above > self.cfg.lift_height_threshold
        success = cube_above > self.cfg.success_lift_height

        reward = (
            -self.cfg.reward_align_scale * xy_err
            - self.cfg.reward_z_scale * z_err
            - self.cfg.reward_action_cost * action_cost
        )
        if lifted:
            reward += self.cfg.reward_lift_bonus
        if success:
            reward += self.cfg.reward_grasp_bonus

        return reward, {
            "xy_err": xy_err,
            "z_err": z_err,
            "cube_lift": cube_above,
            "success": 1.0 if success else 0.0,
        }
