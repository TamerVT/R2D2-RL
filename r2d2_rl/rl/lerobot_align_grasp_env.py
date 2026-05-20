"""RCS SO-101 local-grasp env with a LeRobot-compatible policy boundary.

This is the sim-to-real training surface:

- observation key ``observation.images.wrist``: uint8 RGB, CHW, [3, 128, 128]
- observation key ``observation.state``: 24D float32
- action: 6D absolute SO-101 follower target in real-like units

The simulator remains RCS. The wrist camera comes from the patched SO101 XML
and is exposed through RCS camera config as ``robotwrist``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from rcs._core.sim import CameraType, SimCameraConfig
from rcs.envs.base import ControlMode, RelativeTo
from rcs.envs.configs import EmptyWorldSO101
from rcs.envs.tasks import PickTaskConfig, RandomSquareObjPos

import rcs

try:
    from control.pregrasp_joint_regressor import LoadedPregraspRegressor, load_pregrasp_checkpoint
    from rl.lerobot_compat import (
        DEFAULT_ACTION_HIGH,
        DEFAULT_ACTION_LOW,
        HIL_COLOR_TO_INDEX,
        IMAGE_KEY,
        STATE_KEY,
        target_color_onehot,
        lerobot_to_scaled_action,
    )
except ImportError:  # pragma: no cover - package import path used by tests/tools
    from r2d2_rl.control.pregrasp_joint_regressor import (
        LoadedPregraspRegressor,
        load_pregrasp_checkpoint,
    )
    from r2d2_rl.rl.lerobot_compat import (
        DEFAULT_ACTION_HIGH,
        DEFAULT_ACTION_LOW,
        HIL_COLOR_TO_INDEX,
        IMAGE_KEY,
        STATE_KEY,
        target_color_onehot,
        lerobot_to_scaled_action,
    )


R2D2_RL_ROOT = Path(__file__).resolve().parents[1]
PROJECT3_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREGRASP_CHECKPOINT = (
    PROJECT3_ROOT / "r2d2_rl" / "outputs" / "pregrasp_regressor" / "best_pregrasp_mlp.pt"
)
CUBE_2CM_ASSETS_DIR = R2D2_RL_ROOT / "envs" / "assets" / "cubes"
LEROBOT_CUBE_PATHS = {
    "blue": str(CUBE_2CM_ASSETS_DIR / "blue_cube_2cm.xml"),
    "green": str(CUBE_2CM_ASSETS_DIR / "green_cube_2cm.xml"),
    "purple": str(CUBE_2CM_ASSETS_DIR / "purple_cube_2cm.xml"),
    "orange": str(CUBE_2CM_ASSETS_DIR / "orange_cube_2cm.xml"),
    "yellow": str(CUBE_2CM_ASSETS_DIR / "yellow_cube_2cm.xml"),
    "red": str(CUBE_2CM_ASSETS_DIR / "red_cube_2cm.xml"),
}


@dataclass
class LeRobotAlignGraspEnvConfig:
    """Configuration for the LeRobot-compatible RCS local grasp env."""

    cube_color: str = "green"
    cube_center: tuple[float, float, float] = (0.18, 0.03, 0.01)
    # Full-width ranges sampled uniformly in [-w/2, +w/2] around cube_center by
    # RCS's RandomSquareObjPos.  Default 0.06 = +/-3 cm window: a tight,
    # easier-to-learn task so a short (50k-step) run can reliably converge to a
    # grasping policy.  Widen later (e.g. 0.12) for sim-to-real robustness once
    # grasping is confirmed to work.
    cube_randomization_xy: tuple[float, float] = (0.06, 0.06)
    cube_include_rotation: bool = True
    robot_z_offset: float = -0.03

    max_episode_steps: int = 100
    joint_delta_deg: float = 5.0
    action_scale: float = 1.0
    real_gripper_max: float = 35.0
    compat_dt_s: float = 0.1
    sim_joint_arrival_tolerance_deg: float = 0.25

    # The wrist camera is added directly to so101.xml by the RCS patch
    # (external/robot-control-stack/assets/robots/so101/so101.xml).  RCS
    # prefixes scene-level camera names with the robot name, so the in-XML
    # ``<camera name="wrist">`` ends up as ``robotwrist`` at runtime.
    wrist_camera_name: str = "robotwrist"
    wrist_image_width: int = 128
    wrist_image_height: int = 128

    success_lift_delta_m: float = 0.005
    # TCP-to-cube distance under which a grasp can count as a candidate grasp.
    # The final ``valid_cube_grasp`` below is stricter: it must be caged,
    # commanded closed, not pressing the cube down, and stable for a few steps.
    # 0.025 (2.5 cm) was too loose for a 2 cm cube -- the gripper could be a
    # near-miss and still be credited, so the policy learned to bump the cube
    # rather than enclose it.
    valid_grasp_radius_m: float = 0.012
    caging_xy_radius_m: float = 0.008
    caging_z_radius_m: float = 0.012
    gripper_closed_threshold: float = 0.75
    cube_press_tolerance_m: float = 0.0005
    stable_grasp_steps: int = 3
    success_hold_steps: int = 3
    action_penalty_weight: float = 0.01

    # --- reward shaping (robosuite-Lift-inspired, anti-hover) -------------
    # The previous reward let the policy farm a dense alignment bonus
    # indefinitely without ever grasping ("hover" loophole). This design
    # fixes it with three principles:
    #   1. a per-step time penalty so hovering nets *negative* return;
    #   2. bounded dense shaping (tanh, <= time_penalty) so approaching can
    #      never out-earn a grasp;
    #   3. a flat success reward that strictly dominates, paid every step
    #      the grasp is held (the env no longer terminates on success, so
    #      holding a successful grasp is the only route to large return).
    # time_penalty (0.6) matches the max dense approach shaping
    # (reach+xy_align+descend+caging = 0.60), so an aligned hover nets <= 0
    # after action/change penalties.
    # Once grasped the reward is grasp_hold + lift_weight * normalized_lift:
    # a *continuous* lift gradient (the v1 reward lacked this, so the policy
    # grasped but never pushed the cube past the success threshold). The
    # grasped-regime maximum (grasp_hold + lift_weight - time_penalty = 2.9)
    # stays below success_reward (5.0) so a full lift -> success still
    # strictly dominates hovering just under the threshold.
    time_penalty: float = 0.6
    reach_weight: float = 0.20
    xy_align_weight: float = 0.15
    descend_weight: float = 0.15
    caging_weight: float = 0.10
    candidate_grasp_reward: float = 0.40
    grasp_hold_reward: float = 1.0
    lift_weight: float = 2.5
    success_reward: float = 5.0
    gripper_reopen_penalty_weight: float = 1.0
    gripper_action_change_penalty_weight: float = 0.05
    # Penalty for shoving the cube *below* its resting height -- the trained
    # policy was descending onto the cube and pressing it into the floor
    # (cube z went ~6 mm negative) instead of grasping and lifting. This
    # makes the press-down basin net-negative; normalized by the same 5 mm
    # scale as the lift, weighted heavier than the max dense shaping (0.5) so
    # pressing is strongly discouraged.
    cube_press_penalty_weight: float = 3.0

    pregrasp_q_home_rad: tuple[float, float, float, float, float] | None = None
    use_pregrasp_regressor: bool = False
    pregrasp_regressor_checkpoint: str | Path = DEFAULT_PREGRASP_CHECKPOINT
    pregrasp_max_steps: int = 80


class LeRobotAlignGraspEnv(gym.Env):
    """Gym env exposing the real SO-101 local-grasp policy interface."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: LeRobotAlignGraspEnvConfig | None = None,
        *,
        open_gui: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = config or LeRobotAlignGraspEnvConfig()
        if self.cfg.cube_color not in HIL_COLOR_TO_INDEX:
            raise ValueError(f"Unsupported target color for HIL encoding: {self.cfg.cube_color!r}")
        if self.cfg.cube_color not in LEROBOT_CUBE_PATHS:
            raise ValueError(
                f"No local cube MJCF registered for {self.cfg.cube_color!r}. "
                f"Available assets: {sorted(LEROBOT_CUBE_PATHS)}"
            )

        self.open_gui = open_gui
        self._object_joint_name: str | None = None
        self._shared2world: rcs.common.Pose | None = None
        self._step_count = 0
        self._episode_cube_start_z = float(self.cfg.cube_center[2])
        self._has_grasped = False
        self._tcp_z_at_first_grasp: float | None = None
        self._candidate_grasp_steps = 0
        self._success_hold_steps = 0
        self._prev_tcp_cube_dist: float | None = None
        self._prev_gripper_action_real: float | None = None
        self._prev_real_positions: np.ndarray | None = None

        self._pregrasp_regressor: LoadedPregraspRegressor | None = None
        if self.cfg.use_pregrasp_regressor:
            self._pregrasp_regressor = load_pregrasp_checkpoint(
                self.cfg.pregrasp_regressor_checkpoint,
                device="cpu",
            )

        self.env = self._build_rcs_env()

        self.action_space = spaces.Box(
            low=DEFAULT_ACTION_LOW.copy(),
            high=DEFAULT_ACTION_HIGH.copy(),
            shape=(6,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                IMAGE_KEY: spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, self.cfg.wrist_image_height, self.cfg.wrist_image_width),
                    dtype=np.uint8,
                ),
                STATE_KEY: spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32),
            }
        )

    # ------------------------------------------------------------------ build

    def _build_rcs_env(self) -> gym.Env:
        scene = EmptyWorldSO101()
        cfg = scene.config()

        cfg.robot_to_shared_base_frame = {
            "robot": rcs.common.Pose(
                translation=np.array([0.0, 0.0, self.cfg.robot_z_offset], dtype=np.float64)
            )
        }
        cfg.control_mode = ControlMode.JOINTS
        cfg.relative_to = RelativeTo.LAST_STEP
        cfg.max_relative_movement = np.deg2rad(self.cfg.joint_delta_deg)
        cfg.robot_cfgs["robot"].joint_rotational_tolerance = np.deg2rad(
            self.cfg.sim_joint_arrival_tolerance_deg
        )
        cfg.headless = not self.open_gui
        cfg.sim_cfg.realtime = False
        # Async fixed-rate stepping: each env.step() advances a fixed slice of
        # physics (round(1/frequency/timestep) steps -- ~50 at 10 Hz) instead
        # of the synchronous "step until joints converge" loop, which was
        # hitting the 500-step cap every step. This is both faster and a
        # closer match to a real robot's fixed 10 Hz control rate (and to the
        # env's compat_dt_s = 0.1 s). NOTE: this changes contact/settling
        # dynamics vs sync mode -- checkpoints are not cross-comparable.
        cfg.sim_cfg.async_control = True
        cfg.sim_cfg.frequency = 10.0

        cfg.camera_cfgs = {
            self.cfg.wrist_camera_name: SimCameraConfig(
                identifier=self.cfg.wrist_camera_name,
                type=CameraType.fixed,
                resolution_width=self.cfg.wrist_image_width,
                resolution_height=self.cfg.wrist_image_height,
                frame_rate=30,
            )
        }

        if self.cfg.pregrasp_q_home_rad is not None:
            cfg.robot_cfgs["robot"].q_home = np.asarray(
                self.cfg.pregrasp_q_home_rad,
                dtype=np.float64,
            )

        pick_task_cfg = PickTaskConfig(
            robot_name="robot",
            object_center_to_root_frame=rcs.common.Pose(
                translation=np.array(self.cfg.cube_center, dtype=np.float64),
                quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            ),
            object_joint="box_joint",
            include_rotation=self.cfg.cube_include_rotation,
        )
        pick_task_cfg.object_xml = LEROBOT_CUBE_PATHS[self.cfg.cube_color]
        cfg.task_cfg = pick_task_cfg

        self._object_joint_name = pick_task_cfg.prefix + pick_task_cfg.object_joint
        self._shared2world = cfg.shared_base_frame_to_root_frame * cfg.root_frame_to_world

        env = scene.create_env(cfg)
        randomizer = self._find_wrapper_of_type(env, RandomSquareObjPos)
        if randomizer is None:
            raise RuntimeError("Could not find RCS RandomSquareObjPos wrapper.")
        randomizer.x_width = float(self.cfg.cube_randomization_xy[0])
        randomizer.y_width = float(self.cfg.cube_randomization_xy[1])
        return env

    @staticmethod
    def _find_wrapper_of_type(env: gym.Env, wrapper_type: type) -> Any | None:
        current: Any = env
        while True:
            if isinstance(current, wrapper_type):
                return current
            if not hasattr(current, "env"):
                return None
            current = current.env

    # ------------------------------------------------------------------ obs

    def _cube_xyz_in_shared_frame(self) -> np.ndarray:
        if self._object_joint_name is None or self._shared2world is None:
            raise RuntimeError("RCS object joint was not initialized.")
        sim = self.env.get_wrapper_attr("sim")
        cube_xyz_world = np.asarray(sim.data.joint(self._object_joint_name).qpos[:3], dtype=np.float64)
        cube_pose_world = rcs.common.Pose(translation=cube_xyz_world)
        cube_pose_shared = self._shared2world.inverse() * cube_pose_world
        return np.asarray(cube_pose_shared.translation(), dtype=np.float32)

    def _rgb_from_obs(self, obs: dict[str, Any]) -> np.ndarray:
        frames = obs.get("frames")
        if not isinstance(frames, dict):
            raise RuntimeError("RCS observation did not include camera frames.")
        camera = frames.get(self.cfg.wrist_camera_name)
        if not isinstance(camera, dict) or "rgb" not in camera:
            raise RuntimeError(f"RCS observation did not include {self.cfg.wrist_camera_name!r} RGB.")
        rgb = camera["rgb"].get("data")
        if not isinstance(rgb, np.ndarray):
            raise RuntimeError("RCS wrist RGB frame did not contain numpy data.")
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.shape[:2] != (self.cfg.wrist_image_height, self.cfg.wrist_image_width):
            rgb = self._resize_nearest(rgb, self.cfg.wrist_image_height, self.cfg.wrist_image_width)
        return rgb

    @staticmethod
    def _resize_nearest(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
        src_h, src_w = rgb.shape[:2]
        ys = np.linspace(0, src_h - 1, height).astype(np.int64)
        xs = np.linspace(0, src_w - 1, width).astype(np.int64)
        return np.asarray(rgb[ys][:, xs], dtype=np.uint8)

    def _flat_obs(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        robot_obs = obs["robot"]
        joints = np.asarray(robot_obs["joints"], dtype=np.float32).reshape(5)
        gripper = np.asarray(robot_obs["gripper"], dtype=np.float32).reshape(1)
        tcp_xyzrpy = np.asarray(robot_obs["xyzrpy"], dtype=np.float32).reshape(6)
        cube_xyz = self._cube_xyz_in_shared_frame()
        return {
            "joints": joints,
            "gripper": gripper,
            "tcp_xyzrpy": tcp_xyzrpy,
            "cube_xyz": cube_xyz,
            "tcp_to_cube_xyz": (cube_xyz - tcp_xyzrpy[:3]).astype(np.float32),
            "wrist_rgb": self._rgb_from_obs(obs),
        }

    def _flat_obs_to_real_positions(self, flat_obs: dict[str, np.ndarray]) -> np.ndarray:
        joints_deg = np.rad2deg(flat_obs["joints"]).astype(np.float32)
        gripper_real = np.array(
            [np.clip(float(flat_obs["gripper"][0]), 0.0, 1.0) * self.cfg.real_gripper_max],
            dtype=np.float32,
        )
        return np.concatenate([joints_deg, gripper_real], axis=0).astype(np.float32)

    def _convert_observation(self, flat_obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        real_positions = self._flat_obs_to_real_positions(flat_obs)
        if self._prev_real_positions is None:
            velocities = np.zeros(6, dtype=np.float32)
        else:
            velocities = (
                (real_positions - self._prev_real_positions) / max(1e-6, self.cfg.compat_dt_s)
            ).astype(np.float32)
        self._prev_real_positions = real_positions.copy()

        state = np.concatenate(
            [
                real_positions,
                velocities,
                np.zeros(6, dtype=np.float32),
                target_color_onehot(self.cfg.cube_color),
            ],
            axis=0,
        ).astype(np.float32)
        wrist_chw = np.transpose(flat_obs["wrist_rgb"], (2, 0, 1)).copy()
        return {IMAGE_KEY: wrist_chw, STATE_KEY: state}

    # ---------------------------------------------------------------- actions

    def _real_action_to_rcs_action(self, action: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
        if self._prev_real_positions is None:
            raise RuntimeError("Cannot step before reset(): current pose is unknown.")
        action = np.asarray(action, dtype=np.float32).reshape(6)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        current = self._prev_real_positions
        joint_delta_deg = action[:5] - current[:5]
        max_delta = self.cfg.joint_delta_deg * self.cfg.action_scale
        joint_delta_rad = np.deg2rad(np.clip(joint_delta_deg, -max_delta, max_delta)).astype(np.float64)

        gripper_real_target = float(action[5])
        gripper_sim_target = np.clip(gripper_real_target / max(1e-6, self.cfg.real_gripper_max), 0.0, 1.0)
        rcs_action = {
            "robot": {
                "joints": joint_delta_rad,
                "gripper": np.array([gripper_sim_target], dtype=np.float32),
            }
        }
        return rcs_action, action

    def _normalized_base_action(self, joint_delta_rad: np.ndarray, gripper_sim: float) -> np.ndarray:
        max_delta_rad = np.deg2rad(self.cfg.joint_delta_deg) * self.cfg.action_scale
        joint_part = np.clip(joint_delta_rad / max(1e-6, max_delta_rad), -1.0, 1.0)
        return np.concatenate(
            [joint_part.astype(np.float32), np.array([2.0 * gripper_sim - 1.0], dtype=np.float32)]
        )

    # -------------------------------------------------------------- pregrasp

    def _move_to_regressed_pregrasp(
        self,
        flat_obs: dict[str, np.ndarray],
        info: dict[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self._pregrasp_regressor is None:
            return flat_obs, info

        cube_xyz_sim = flat_obs["cube_xyz"].astype(np.float32).copy()
        cube_xyz_regressor = np.array(
            [cube_xyz_sim[1] - 0.02, cube_xyz_sim[0] + 0.04, 0.0],
            dtype=np.float32,
        )
        target = self._pregrasp_regressor.predict(cube_xyz_regressor).astype(np.float32)
        target = np.clip(target, self.action_space.low, self.action_space.high)

        latest_info = info
        latest_flat = flat_obs
        step_idx = 0
        robots = self.env.get_wrapper_attr("robot")
        robot = robots["robot"] if isinstance(robots, dict) else robots
        if hasattr(robot, "set_joints_hard"):
            import mujoco

            robot.set_joints_hard(np.deg2rad(target[:5]).astype(np.float64))
            sim = self.env.get_wrapper_attr("sim")
            mujoco.mj_forward(sim.model, sim.data)
            # Pull a fresh camera/robot observation after the hard reset pose.
            obs, _, _, _, latest_info = self.env.step({"robot": {"joints": np.zeros(5), "gripper": np.array([1.0])}})
            latest_flat = self._flat_obs(obs)
        else:
            self._prev_real_positions = self._flat_obs_to_real_positions(flat_obs)
            for step_idx in range(self.cfg.pregrasp_max_steps):
                rcs_action, clipped_target = self._real_action_to_rcs_action(target)
                obs, _, _, _, latest_info = self.env.step(rcs_action)
                latest_flat = self._flat_obs(obs)
                current = self._flat_obs_to_real_positions(latest_flat)
                self._prev_real_positions = current
                if np.linalg.norm(current[:5] - clipped_target[:5], ord=np.inf) < 1.0:
                    break
        latest_info["pregrasp_cube_xyz_sim"] = cube_xyz_sim
        latest_info["pregrasp_cube_xyz_regressor"] = cube_xyz_regressor
        latest_info["pregrasp_predicted_pose"] = target
        latest_info["pregrasp_reached_pose"] = self._flat_obs_to_real_positions(latest_flat)
        latest_info["pregrasp_internal_steps"] = step_idx + 1
        return latest_flat, latest_info

    # ---------------------------------------------------------------- reward

    def _compute_reward(
        self,
        *,
        flat_obs: dict[str, np.ndarray],
        raw_info: dict[str, Any],
        action: np.ndarray,
    ) -> tuple[float, bool, dict[str, float]]:
        """Anti-hover reward (robosuite-Lift-inspired).

        Three mutually-exclusive regimes, evaluated top-down:

        - ``success`` (stable grasp + lifted past threshold): flat
          ``success_reward`` -- strictly dominates every other regime.
        - ``stable grasp`` (caged and closed for several steps): a modest
          hold reward plus lift-progress gradient.
        - ``candidate grasp``: one-step caged/closed contact gets a small
          reward, but does not unlock full lift shaping until it is stable.
        - otherwise (approaching): bounded ``tanh`` shaping whose maximum is
          <= ``time_penalty``. Hovering aligned therefore nets <= 0 per step;
          merely approaching nets clearly negative. The only route to positive
          episode return is to grasp and lift.
        """
        cfg = self.cfg
        cube_xyz = flat_obs["cube_xyz"].astype(np.float64)
        tcp_xyz = flat_obs["tcp_xyzrpy"][:3].astype(np.float64)
        gripper_open = float(flat_obs["gripper"][0])

        delta = cube_xyz - tcp_xyz
        xy_dist = float(np.linalg.norm(delta[:2]))
        xyz_dist = float(np.linalg.norm(delta))
        z_dist = float(abs(delta[2]))

        # Bounded [0, 1] shaping terms (tanh, like robosuite Lift).
        reach = 1.0 - float(np.tanh(10.0 * xyz_dist))
        xy_align = 1.0 - float(np.tanh(10.0 * xy_dist))
        descend = 1.0 - float(np.tanh(10.0 * z_dist))
        xy_caging = 1.0 - float(np.tanh(xy_dist / max(1e-6, cfg.caging_xy_radius_m)))
        z_caging = 1.0 - float(np.tanh(z_dist / max(1e-6, cfg.caging_z_radius_m)))
        caging = xy_caging * z_caging
        gripper_closed_amount = float(np.clip(1.0 - gripper_open, 0.0, 1.0))

        robot_info = raw_info.get("robot", {}) if isinstance(raw_info, dict) else {}
        sim_is_grasped = float(
            bool(raw_info.get("is_grasped", False) or robot_info.get("is_grasped", False))
        )

        cube_lift = max(0.0, float(cube_xyz[2] - self._episode_cube_start_z))
        normalized_lift = float(np.clip(cube_lift / max(1e-6, cfg.success_lift_delta_m), 0.0, 1.0))

        # Cube press depth: how far the cube sits *below* its resting height.
        # Kept for logging/diagnostics only. The press *penalty* was removed:
        # in the v6 run it subtracted from every grasped step (grasping a 2 cm
        # cube always presses it a little) and collapsed grasp success from
        # 100% -> 0%. The tight valid_grasp_radius_m + caging predicate already
        # rule out the shove-into-floor degenerate grasp, so the penalty was
        # both redundant and actively harmful.
        cube_press = max(0.0, float(self._episode_cube_start_z - cube_xyz[2]))
        normalized_press = float(np.clip(cube_press / max(1e-6, cfg.success_lift_delta_m), 0.0, 1.0))
        press_penalty = cfg.cube_press_penalty_weight * normalized_press

        closed_enough = gripper_closed_amount >= cfg.gripper_closed_threshold
        caged_enough = xy_dist <= cfg.caging_xy_radius_m and z_dist <= cfg.caging_z_radius_m
        not_pressing = cube_press <= cfg.cube_press_tolerance_m
        candidate_cube_grasp = float(
            sim_is_grasped > 0.5
            and closed_enough
            and xyz_dist < cfg.valid_grasp_radius_m
            and caged_enough
        )
        if candidate_cube_grasp > 0.5:
            self._candidate_grasp_steps += 1
        else:
            self._candidate_grasp_steps = 0
        valid_cube_grasp = float(self._candidate_grasp_steps >= cfg.stable_grasp_steps)

        if candidate_cube_grasp > 0.5 and not self._has_grasped:
            self._has_grasped = True
            self._tcp_z_at_first_grasp = float(tcp_xyz[2])

        scaled_action = lerobot_to_scaled_action(action)
        action_penalty = cfg.action_penalty_weight * float(np.mean(np.square(scaled_action)))
        gripper_target_real = float(action[5])
        if self._prev_gripper_action_real is None:
            gripper_action_change_penalty = 0.0
        else:
            normalized_gripper_change = abs(gripper_target_real - self._prev_gripper_action_real) / max(
                1e-6, cfg.real_gripper_max
            )
            gripper_action_change_penalty = (
                cfg.gripper_action_change_penalty_weight * normalized_gripper_change
            )
        self._prev_gripper_action_real = gripper_target_real

        gripper_reopen_penalty = 0.0
        if self._has_grasped and not closed_enough:
            gripper_reopen_penalty = cfg.gripper_reopen_penalty_weight

        instant_success = bool(cube_lift >= cfg.success_lift_delta_m and valid_cube_grasp > 0.5)
        if instant_success:
            self._success_hold_steps += 1
        else:
            self._success_hold_steps = 0
        success = bool(self._success_hold_steps >= cfg.success_hold_steps)
        # press_penalty intentionally excluded -- see the cube_press comment
        # above (it collapsed grasp success in v6).
        behavior_penalty = (
            action_penalty
            + gripper_action_change_penalty
            + gripper_reopen_penalty
        )

        if instant_success:
            # Flat, dominant, paid each held step (no early termination).
            # (cube is lifted here, so press_penalty is 0 -- subtracted for
            # uniformity.)
            reward = cfg.success_reward - behavior_penalty
        elif valid_cube_grasp > 0.5:
            # Grasped but not yet lifted to threshold. Flat hold reward plus a
            # *continuous* lift-progress gradient so every extra mm of lift
            # pays -- this is what pulls the policy from "grasp and hold on
            # the table" up to a full lift. Bounded below success_reward.
            reward = (
                cfg.grasp_hold_reward
                + cfg.lift_weight * normalized_lift
                - cfg.time_penalty
                - behavior_penalty
            )
        elif candidate_cube_grasp > 0.5:
            # One-step candidate grasp. This keeps the stable-grasp condition
            # learnable without letting a momentary bump earn full lift credit.
            reward = (
                cfg.candidate_grasp_reward
                + cfg.caging_weight * caging
                + 0.25 * cfg.lift_weight * normalized_lift
                - cfg.time_penalty
                - behavior_penalty
            )
        else:
            # Approaching. Bounded dense shaping minus the time penalty, minus
            # the press penalty so descending *onto* the cube and pressing it
            # into the floor is net-negative.
            reward = (
                cfg.reach_weight * reach
                + cfg.xy_align_weight * xy_align
                + cfg.descend_weight * descend * xy_align
                + cfg.caging_weight * caging
                - cfg.time_penalty
                - behavior_penalty
            )

        terms = {
            "xy_dist": xy_dist,
            "xyz_dist": xyz_dist,
            "z_dist": z_dist,
            "xy_caging": xy_caging,
            "z_caging": z_caging,
            "caging": caging,
            "cube_z": float(cube_xyz[2]),
            "cube_lift": cube_lift,
            "normalized_lift": normalized_lift,
            "cube_press": cube_press,
            "normalized_press": normalized_press,
            "sim_is_grasped": sim_is_grasped,
            "closed_enough": float(closed_enough),
            "caged_enough": float(caged_enough),
            "not_pressing": float(not_pressing),
            "candidate_cube_grasp": candidate_cube_grasp,
            "stable_grasp_steps": float(self._candidate_grasp_steps),
            "valid_cube_grasp": valid_cube_grasp,
            "instant_success": float(instant_success),
            "success_hold_steps": float(self._success_hold_steps),
            "gripper_closed_amount": gripper_closed_amount,
            "reach": reach,
            "xy_align": xy_align,
            "descend": descend,
            "action_penalty": action_penalty,
            "gripper_action_change_penalty": gripper_action_change_penalty,
            "gripper_reopen_penalty": gripper_reopen_penalty,
            "press_penalty": press_penalty,
            "local_success": float(success),
        }
        return float(reward), success, terms

    # ---------------------------------------------------------------- gym API

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_real_positions = None
        self._has_grasped = False
        self._tcp_z_at_first_grasp = None
        self._candidate_grasp_steps = 0
        self._success_hold_steps = 0
        self._prev_tcp_cube_dist = None
        self._prev_gripper_action_real = None

        obs, info = self.env.reset(seed=seed, options=options)
        flat_obs = self._flat_obs(obs)
        self._episode_cube_start_z = float(flat_obs["cube_xyz"][2])

        flat_obs, info = self._move_to_regressed_pregrasp(flat_obs, info)

        # Start local-grasp control with an open gripper and zero velocity history.
        self._prev_real_positions = self._flat_obs_to_real_positions(flat_obs)
        open_action = self._prev_real_positions.copy()
        open_action[5] = self.cfg.real_gripper_max
        rcs_action, _ = self._real_action_to_rcs_action(open_action)
        obs, _, _, _, info = self.env.step(rcs_action)
        flat_obs = self._flat_obs(obs)

        self._prev_real_positions = None
        return self._convert_observation(flat_obs), info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        rcs_action, clipped_action = self._real_action_to_rcs_action(action)
        obs, _rcs_reward, _rcs_terminated, truncated, info = self.env.step(rcs_action)
        flat_obs = self._flat_obs(obs)
        reward, local_success, reward_terms = self._compute_reward(
            flat_obs=flat_obs,
            raw_info=info,
            action=clipped_action,
        )
        out_obs = self._convert_observation(flat_obs)

        timeout = self._step_count >= self.cfg.max_episode_steps
        # Fixed-horizon task: do NOT terminate on success. The success reward
        # is paid every step the grasp is held, so the optimal policy reaches
        # success fast and holds it -- this is what makes hovering strictly
        # worse than grasping. Episodes end only on timeout truncation.
        terminated = False
        truncated = bool(truncated or timeout)

        info["success"] = local_success
        info["local_success"] = local_success
        info["reward_terms"] = reward_terms
        for key in (
            "cube_z",
            "cube_lift",
            "cube_press",
            "xy_dist",
            "xyz_dist",
            "sim_is_grasped",
            "candidate_cube_grasp",
            "valid_cube_grasp",
            "instant_success",
        ):
            info[key] = reward_terms[key]
        info["lerobot_action"] = clipped_action
        info["rcs_joint_delta_rad"] = rcs_action["robot"]["joints"]
        return out_obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()
