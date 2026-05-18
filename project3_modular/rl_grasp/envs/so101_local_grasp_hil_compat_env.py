from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from project3_modular.models.pregrasp_joint_regressor import (
    LoadedPregraspRegressor,
    load_pregrasp_checkpoint,
)
from project3_modular.rl_grasp.envs.so101_local_grasp_env import (
    SO101LocalGraspConfig,
    SO101LocalGraspEnv,
)


# Must match the real HIL actor / dataset color encoding.
HIL_COLOR_NAMES = ["blue", "green", "purple", "orange", "yellow", "red"]
HIL_COLOR_TO_INDEX = {name: i for i, name in enumerate(HIL_COLOR_NAMES)}


# Calibration-derived arm joint limits.
# These are not dataset min/max values.
DEFAULT_ACTION_LOW = (
    -68.90625,
    -103.7548828125,
    -97.470703125,
    -102.216796875,
    -179.9560546875,
    0.0,
)

DEFAULT_ACTION_HIGH = (
    68.90625,
    103.7548828125,
    97.470703125,
    102.216796875,
    179.9560546875,
    100.0,
)


@dataclass(frozen=True)
class SO101LocalGraspHILCompatConfig:
    """
    Learner-facing sim wrapper shaped like the real HIL actor env.
    """

    base_env: SO101LocalGraspConfig = field(
        default_factory=lambda: SO101LocalGraspConfig(
            include_wrist_rgb=True,
            # Internal reset pose before the regressor-guided Phase 1 move.
            # This is just a safe-ish neutral starting point for the sim reset.
            pregrasp_q_home_rad=(
                -0.11353367,
                0.01610939,
                0.36133552,
                1.02187282,
                0.09590584,
            ),
        )
    )

    compat_dt_s: float = 0.1
    real_gripper_max: float = 35.0

    action_low: tuple[float, float, float, float, float, float] = DEFAULT_ACTION_LOW
    action_high: tuple[float, float, float, float, float, float] = DEFAULT_ACTION_HIGH

    # Phase 1: cube_xyz -> absolute 6D follower joint target.
    use_pregrasp_regressor: bool = True
    pregrasp_regressor_checkpoint: str = (
        "project3_modular/outputs/pregrasp_regressor/best_pregrasp_mlp.pt"
    )

    # Internal sim control steps to reach the predicted pregrasp target.
    # The base sim env moves by joint_delta_deg per step, so this needs to be
    # long enough for fairly large moves.
    pregrasp_max_internal_steps: int = 80
    pregrasp_settle_steps: int = 3


class SO101LocalGraspHILCompatEnv(gym.Env):
    """
    Sim wrapper exposing the same observation/action interface as the real HIL env.

    Observation:
      observation.images.wrist : uint8 [3,128,128]
      observation.state        : float32 [24]

    State layout:
      [0:6]    5 arm joints in degrees + gripper in real-like units
      [6:12]   finite-difference velocities
      [12:18]  zero current placeholders
      [18:24]  target-color one-hot

    Action:
      absolute 6D joint/gripper target in real-like units.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: SO101LocalGraspHILCompatConfig | None = None,
        *,
        open_gui: bool = False,
    ) -> None:
        super().__init__()

        self.config = config or SO101LocalGraspHILCompatConfig()

        base_cfg = self.config.base_env
        if not base_cfg.include_wrist_rgb:
            base_cfg = replace(base_cfg, include_wrist_rgb=True)

        if base_cfg.target_color not in HIL_COLOR_TO_INDEX:
            raise ValueError(
                f"Unknown HIL target color {base_cfg.target_color!r}. "
                f"Available: {HIL_COLOR_NAMES}"
            )

        self.env = SO101LocalGraspEnv(base_cfg, open_gui=open_gui)

        self.action_space = spaces.Box(
            low=np.asarray(self.config.action_low, dtype=np.float32),
            high=np.asarray(self.config.action_high, dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "observation.images.wrist": spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        3,
                        base_cfg.wrist_image_height,
                        base_cfg.wrist_image_width,
                    ),
                    dtype=np.uint8,
                ),
                "observation.state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(24,),
                    dtype=np.float32,
                ),
            }
        )

        self._prev_real_positions: np.ndarray | None = None

        self._pregrasp_regressor: LoadedPregraspRegressor | None = None
        if self.config.use_pregrasp_regressor:
            self._pregrasp_regressor = load_pregrasp_checkpoint(
                self.config.pregrasp_regressor_checkpoint,
                device="cpu",
            )

    # ------------------------------------------------------------------
    # Observation/action conversion
    # ------------------------------------------------------------------

    def _target_color_onehot(self) -> np.ndarray:
        color = self.config.base_env.target_color
        onehot = np.zeros(len(HIL_COLOR_NAMES), dtype=np.float32)
        onehot[HIL_COLOR_TO_INDEX[color]] = 1.0
        return onehot

    def _flat_obs_to_real_positions(
        self,
        flat_obs: dict[str, np.ndarray],
    ) -> np.ndarray:
        joints_rad = np.asarray(flat_obs["joints"], dtype=np.float32).reshape(5)
        gripper_sim = float(
            np.asarray(flat_obs["gripper"], dtype=np.float32).reshape(1)[0]
        )

        joints_deg = np.rad2deg(joints_rad).astype(np.float32)
        gripper_real = np.array(
            [np.clip(gripper_sim, 0.0, 1.0) * self.config.real_gripper_max],
            dtype=np.float32,
        )

        return np.concatenate([joints_deg, gripper_real], axis=0).astype(np.float32)

    def _convert_observation(
        self,
        flat_obs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        real_positions = self._flat_obs_to_real_positions(flat_obs)

        if self._prev_real_positions is None:
            velocities = np.zeros(6, dtype=np.float32)
        else:
            velocities = (
                (real_positions - self._prev_real_positions)
                / max(1e-6, self.config.compat_dt_s)
            ).astype(np.float32)

        self._prev_real_positions = real_positions.copy()

        currents_placeholder = np.zeros(6, dtype=np.float32)
        target_color = self._target_color_onehot()

        state = np.concatenate(
            [
                real_positions,
                velocities,
                currents_placeholder,
                target_color,
            ],
            axis=0,
        ).astype(np.float32)

        wrist_hwc = np.asarray(flat_obs["wrist_rgb"], dtype=np.uint8)
        wrist_chw = np.transpose(wrist_hwc, (2, 0, 1)).copy()

        return {
            "observation.images.wrist": wrist_chw,
            "observation.state": state,
        }

    def _real_absolute_action_to_base_action(
        self,
        action: np.ndarray,
    ) -> np.ndarray:
        if self._prev_real_positions is None:
            raise RuntimeError(
                "Cannot convert action before reset(): no current sim pose known."
            )

        action = np.asarray(action, dtype=np.float32).reshape(6)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        current_positions = self._prev_real_positions

        joint_target_deg = action[:5]
        joint_current_deg = current_positions[:5]
        joint_delta_deg = joint_target_deg - joint_current_deg

        max_delta_deg = (
            self.env.config.joint_delta_deg
            * self.env.config.action_scale
        )
        base_joint_action = np.clip(
            joint_delta_deg / max(1e-6, max_delta_deg),
            -1.0,
            1.0,
        )

        gripper_real_target = float(action[5])
        gripper_sim_target = np.clip(
            gripper_real_target / max(1e-6, self.config.real_gripper_max),
            0.0,
            1.0,
        )

        base_gripper_action = np.array(
            [2.0 * gripper_sim_target - 1.0],
            dtype=np.float32,
        )

        return np.concatenate(
            [
                base_joint_action.astype(np.float32),
                base_gripper_action,
            ],
            axis=0,
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Regressor-guided Phase 1 pregrasp move
    # ------------------------------------------------------------------

    def _move_to_regressed_pregrasp(
        self,
        flat_obs: dict[str, np.ndarray],
        info: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self._pregrasp_regressor is None:
            return flat_obs, info

        cube_xyz_sim = np.asarray(flat_obs["cube_xyz"], dtype=np.float32).copy()

        # Convert from the RCS sim shared frame to the real coordinate frame
        # used when collecting the pregrasp-regressor dataset.
        #
        # Calibrated from the visual correspondence:
        #   real cube (x=0.00, y=0.30) m
        #   matches sim cube (x=0.26, y=0.02) m
        #
        # Axis convention:
        #   real x ≈ sim y   (lateral)
        #   real y ≈ sim x   (forward)
        cube_xyz_regressor = np.array(
            [
                cube_xyz_sim[1] - 0.02,
                cube_xyz_sim[0] + 0.04,
                0.0,
            ],
            dtype=np.float32,
        )

        predicted_pose = self._pregrasp_regressor.predict(
            cube_xyz_regressor
        ).astype(np.float32)
        predicted_pose = np.clip(
            predicted_pose,
            self.action_space.low,
            self.action_space.high,
        )

        # For reset distribution construction, do not physically simulate the
        # entire Phase-1 pregrasp transit. Hard-set the sim arm directly to the
        # regressor-predicted 5D joint pose. This avoids expensive convergence
        # loops and ensures Phase 2 starts exactly at the intended pregrasp.
        robots = self.env.env.get_wrapper_attr("robot")
        robot = robots["robot"]

        predicted_arm_q_rad = np.deg2rad(
            predicted_pose[:5].astype(np.float64)
        )
        robot.set_joints_hard(predicted_arm_q_rad)

        sim = self.env.env.get_wrapper_attr("sim")
        import mujoco
        mujoco.mj_forward(sim.model, sim.data)

        # Recompute a fresh base-env observation after the hard reset pose update.
        # A zero arm-delta action keeps the hard-set joint pose unchanged; the
        # gripper command remains closed, matching the start of Phase 2.
        refresh_action = np.zeros(6, dtype=np.float32)
        # Start Phase 2 with an open gripper. This matches the intended
        # local-grasp sequence: approach open -> close near cube -> lift.
        # In the real pipeline we can likewise open the gripper after the
        # pregrasp positioning phase and before HIL/RL control begins.
        refresh_action[5] = 1.0

        latest_flat_obs, _refresh_reward, _refresh_terminated, _refresh_truncated, latest_info = (
            self.env.step(refresh_action)
        )
        latest_info["pregrasp_cube_xyz_sim"] = cube_xyz_sim.copy()
        latest_info["pregrasp_cube_xyz_regressor"] = cube_xyz_regressor.copy()
        latest_info["pregrasp_cube_xyz"] = cube_xyz_regressor.copy()
        latest_info["pregrasp_predicted_pose"] = predicted_pose.copy()
        latest_info["pregrasp_internal_steps"] = 0
        latest_info["pregrasp_reached_pose"] = self._flat_obs_to_real_positions(
            latest_flat_obs
        )

        return latest_flat_obs, latest_info

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self._prev_real_positions = None

        flat_obs, info = self.env.reset(seed=seed, options=options)

        # Phase 1: move to cube-conditioned pregrasp.
        flat_obs, info = self._move_to_regressed_pregrasp(flat_obs, info)

        # Phase 2 starts here. Velocity history should restart at zero.
        self._prev_real_positions = None
        obs = self._convert_observation(flat_obs)

        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        base_action = self._real_absolute_action_to_base_action(action)

        flat_obs, reward, terminated, truncated, info = self.env.step(base_action)
        obs = self._convert_observation(flat_obs)

        info["hil_compat_base_action"] = base_action
        info["hil_compat_absolute_action"] = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(6)

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self) -> None:
        self.env.close()
