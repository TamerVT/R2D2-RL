from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import rcs
from rcs._core.common import Pose
from rcs.envs.base import ControlMode, RelativeTo
from rcs.envs.configs import EmptyWorldSO101
from rcs.envs.tasks import PickTaskConfig, RandomSquareObjPos


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUBE_XML = PROJECT_ROOT / "rl_grasp" / "assets" / "cubes" / "green_cube_2cm.xml"

COLOR_NAMES = ["green", "red", "blue", "yellow", "purple", "orange"]
COLOR_TO_INDEX = {name: i for i, name in enumerate(COLOR_NAMES)}


@dataclass(frozen=True)
class SO101LocalGraspConfig:
    cube_xml: str = str(DEFAULT_CUBE_XML)

    # The default SO101 sim home TCP was approximately:
    #   x=0.183, y=0.031, z=0.054
    # So place the cube roughly underneath it for a local grasp task.
    cube_center: tuple[float, float, float] = (0.18, 0.03, 0.01)

    # Total randomization width in x/y.
    # 0.06 means ±3 cm around cube_center.
    cube_randomization_xy: tuple[float, float] = (0.06, 0.06)
    cube_include_rotation: bool = True

    max_episode_steps: int = 100
    joint_delta_deg: float = 5.0

    # Needed so the SO101 mounting plate visually rests on the floor plane.
    robot_z_offset: float = -0.03

    target_color: str = "green"
    action_scale: float = 1.0

    # RCS convergence checks whether the low-level sim reaches the commanded
    # joint target within this tolerance. The default is tighter than the
    # residual settling error we observe for SO101 (~0.1-0.16 deg), causing
    # spurious "Max convergence steps reached!" warnings on every RL step.
    sim_joint_arrival_tolerance_deg: float = 0.25

    # Optional simulated wrist RGB observation.
    # Disabled by default so the existing state-based SAC run is unaffected.
    include_wrist_rgb: bool = False
    wrist_camera_name: str = "robotwrist"
    wrist_render_width: int = 640
    wrist_render_height: int = 480
    wrist_image_width: int = 128
    wrist_image_height: int = 128

    # Custom local-grasp reward shaping.
    # Success requires the cube to be lifted this far above its reset height.
    success_lift_delta_m: float = 0.005
    action_penalty_weight: float = 0.01

    # Optional later:
    # Set to a 5D tuple in radians if we want a custom pregrasp reset pose.
    pregrasp_q_home_rad: tuple[float, float, float, float, float] | None = None


class SO101LocalGraspEnv(gym.Env):
    """
    Thin Gym wrapper around RCS SO101 PickTask.

    Current debug version:
      - one green 2 cm cube
      - RCS PickTask reward/success logic
      - small local cube randomization around a pregrasp-ish region
      - flattened RL-friendly observation
      - privileged cube position included for early RL debugging
      - target color one-hot already present for later Task 2 extension

    Later:
      - replace privileged cube state with wrist RGB
      - add multiple colored cubes
      - condition grasping on target color
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: SO101LocalGraspConfig | None = None,
        *,
        open_gui: bool = False,
    ) -> None:
        super().__init__()

        self.config = config or SO101LocalGraspConfig()
        self.open_gui = open_gui
        self._step_count = 0
        self._episode_cube_start_z = 0.01
        self._has_grasped = False
        self._tcp_z_at_first_grasp: float | None = None

        if self.config.target_color not in COLOR_TO_INDEX:
            raise ValueError(
                f"Unknown target color {self.config.target_color!r}. "
                f"Available: {list(COLOR_TO_INDEX.keys())}"
            )

        self._object_joint_name: str | None = None
        self._shared2world: Pose | None = None

        self.env = self._build_rcs_env()

        self._wrist_renderer: mujoco.Renderer | None = None
        if self.config.include_wrist_rgb:
            sim = self.env.get_wrapper_attr("sim")
            self._wrist_renderer = mujoco.Renderer(
                sim.model,
                height=self.config.wrist_render_height,
                width=self.config.wrist_render_width,
            )

        # Flat RL action:
        #   0:5 -> relative joint deltas
        #   5   -> gripper command in [-1,1], remapped to [0,1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )

        # Debug/state observation.
        # The cube state is privileged and will later be replaced by wrist RGB.
        self.observation_space = spaces.Dict(
            {
                "joints": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(5,),
                    dtype=np.float32,
                ),
                "gripper": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "tcp_xyzrpy": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(6,),
                    dtype=np.float32,
                ),
                "cube_xyz": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "tcp_to_cube_xyz": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "target_color": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(len(COLOR_NAMES),),
                    dtype=np.float32,
                ),
            }
        )

        if self.config.include_wrist_rgb:
            self.observation_space.spaces["wrist_rgb"] = spaces.Box(
                low=0,
                high=255,
                shape=(
                    self.config.wrist_image_height,
                    self.config.wrist_image_width,
                    3,
                ),
                dtype=np.uint8,
            )

    # ------------------------------------------------------------------
    # Environment construction
    # ------------------------------------------------------------------

    def _build_rcs_env(self) -> gym.Env:
        scene = EmptyWorldSO101()
        cfg = scene.config()

        # Lower the SO101 so the physical base sits on the floor plane.
        cfg.robot_to_shared_base_frame = {
            "robot": rcs.common.Pose(
                translation=np.array([0.0, 0.0, self.config.robot_z_offset])
            )
        }

        cfg.control_mode = ControlMode.JOINTS
        cfg.relative_to = RelativeTo.LAST_STEP
        cfg.max_relative_movement = np.deg2rad(self.config.joint_delta_deg)

        cfg.robot_cfgs["robot"].joint_rotational_tolerance = np.deg2rad(
            self.config.sim_joint_arrival_tolerance_deg
        )

        if self.config.pregrasp_q_home_rad is not None:
            cfg.robot_cfgs["robot"].q_home = np.asarray(
                self.config.pregrasp_q_home_rad,
                dtype=np.float64,
            )

        pick_task_cfg = PickTaskConfig(
            robot_name="robot",
            object_center_to_root_frame=Pose(
                translation=np.array(self.config.cube_center, dtype=np.float64),
                quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            ),
            object_joint="green_cube_joint",
            include_rotation=self.config.cube_include_rotation,
        )

        # object_xml is a class attribute in RCS's dataclass, not an __init__ arg.
        pick_task_cfg.object_xml = self.config.cube_xml
        cfg.task_cfg = pick_task_cfg

        # Store the same joint name convention RCS PickTask uses internally.
        self._object_joint_name = pick_task_cfg.prefix + pick_task_cfg.object_joint
        self._shared2world = cfg.shared_base_frame_to_root_frame * cfg.root_frame_to_world

        env = scene.create_env(cfg)

        # RCS PickTask internally adds RandomSquareObjPos with default 20x20 cm.
        # Tighten it to our desired local grasp reset range.
        randomizer = self._find_wrapper_of_type(env, RandomSquareObjPos)
        if randomizer is None:
            raise RuntimeError(
                "Could not find RCS RandomSquareObjPos wrapper. "
                "Local cube randomization could not be configured."
            )

        randomizer.x_width = float(self.config.cube_randomization_xy[0])
        randomizer.y_width = float(self.config.cube_randomization_xy[1])

        if self.open_gui:
            env.get_wrapper_attr("sim").open_gui()

        return env

    @staticmethod
    def _find_wrapper_of_type(
        env: gym.Env,
        wrapper_type: type,
    ) -> Any | None:
        current: Any = env

        while True:
            if isinstance(current, wrapper_type):
                return current

            if not hasattr(current, "env"):
                break

            current = current.env

        return None

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _target_color_onehot(self) -> np.ndarray:
        onehot = np.zeros(len(COLOR_NAMES), dtype=np.float32)
        onehot[COLOR_TO_INDEX[self.config.target_color]] = 1.0
        return onehot

    def _cube_xyz_in_shared_frame(self) -> np.ndarray:
        assert self._object_joint_name is not None
        assert self._shared2world is not None

        sim = self.env.get_wrapper_attr("sim")
        cube_xyz_world = np.asarray(
            sim.data.joint(self._object_joint_name).qpos[:3],
            dtype=np.float64,
        )

        cube_pose_world = rcs.common.Pose(
            translation=cube_xyz_world
        )
        cube_pose_shared = self._shared2world.inverse() * cube_pose_world

        return np.asarray(
            cube_pose_shared.translation(),
            dtype=np.float32,
        )

    def _render_wrist_rgb(self) -> np.ndarray:
        if self._wrist_renderer is None:
            raise RuntimeError(
                "Wrist renderer requested, but include_wrist_rgb=False."
            )

        sim = self.env.get_wrapper_attr("sim")
        self._wrist_renderer.update_scene(
            sim.data,
            camera=self.config.wrist_camera_name,
        )
        rgb = self._wrist_renderer.render().copy()

        resized = cv2.resize(
            rgb,
            (
                self.config.wrist_image_width,
                self.config.wrist_image_height,
            ),
            interpolation=cv2.INTER_AREA,
        )

        return np.asarray(resized, dtype=np.uint8)

    def _flatten_obs(
        self,
        obs: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        robot_obs = obs["robot"]

        joints = np.asarray(robot_obs["joints"], dtype=np.float32)
        gripper = np.asarray(robot_obs["gripper"], dtype=np.float32).reshape(1)
        tcp_xyzrpy = np.asarray(robot_obs["xyzrpy"], dtype=np.float32)

        cube_xyz = self._cube_xyz_in_shared_frame()
        tcp_xyz = tcp_xyzrpy[:3]
        tcp_to_cube_xyz = cube_xyz - tcp_xyz

        flat_obs = {
            "joints": joints,
            "gripper": gripper,
            "tcp_xyzrpy": tcp_xyzrpy,
            "cube_xyz": cube_xyz,
            "tcp_to_cube_xyz": tcp_to_cube_xyz.astype(np.float32),
            "target_color": self._target_color_onehot(),
        }

        if self.config.include_wrist_rgb:
            flat_obs["wrist_rgb"] = self._render_wrist_rgb()

        return flat_obs
    
    def _compute_local_grasp_reward(
        self,
        *,
        flat_obs: dict[str, np.ndarray],
        raw_info: dict[str, Any],
        action: np.ndarray,
    ) -> tuple[float, bool, dict[str, float]]:
        """
        Shaped local grasp reward.

        Intended sequence:
          1. align TCP with cube in xy
          2. descend toward cube
          3. close gripper when near
          4. lift cube
        """
        cube_xyz = flat_obs["cube_xyz"].astype(np.float64)
        tcp_xyz = flat_obs["tcp_xyzrpy"][:3].astype(np.float64)
        gripper_open = float(flat_obs["gripper"][0])

        delta = cube_xyz - tcp_xyz
        xy_dist = float(np.linalg.norm(delta[:2]))
        xyz_dist = float(np.linalg.norm(delta))
        z_dist = float(abs(delta[2]))

        # Dense directional shaping: reward only actual progress toward the cube.
        # The policy does not observe cube_xyz; this is privileged sim reward only.
        if self._prev_tcp_cube_dist is None:
            distance_progress = 0.0
        else:
            distance_progress = float(self._prev_tcp_cube_dist - xyz_dist)

        self._prev_tcp_cube_dist = xyz_dist

        # Penalize moving away from the cube only while the TCP is still
        # meaningfully outside the local grasp region. Once it is within a
        # small 4 cm x 4 cm x 4 cm box around the cube center, allow free local
        # repositioning without this progress penalty.
        #
        # Box half-extent = 2 cm in x/y/z.
        free_reposition_half_extent_m = 0.02
        inside_free_reposition_box = bool(
            np.all(np.abs(delta) <= free_reposition_half_extent_m)
        )

        if inside_free_reposition_box:
            progress_reward = 0.0
        else:
            # Moving away by 1 cm -> -1.0 penalty.
            # Moving closer or staying equally close -> 0.0.
            progress_reward = float(np.clip(100.0 * distance_progress, -1.0, 0.0))

        # 1) General proximity reward.
        # Starts moderately high and becomes close to 1 near the cube.
        reach_reward = float(np.exp(-20.0 * xyz_dist))

        # 2) Strong horizontal alignment reward.
        xy_align_reward = float(np.exp(-40.0 * xy_dist))

        # 3) Reward descending toward cube height, but mainly once xy is aligned.
        vertical_align_reward = float(np.exp(-60.0 * z_dist))
        descend_reward = xy_align_reward * vertical_align_reward

        # 4) Reward closing the gripper only when close to the cube.
        # Reset obs has gripper≈1.0 open; closed corresponds to lower values.
        gripper_closed_amount = float(np.clip(1.0 - gripper_open, 0.0, 1.0))
        near_cube = float(np.exp(-60.0 * xyz_dist))
        close_near_cube_reward = near_cube * gripper_closed_amount

        # 5) Grasp/lift signals.
        robot_info = raw_info.get("robot", {})
        sim_is_grasped = float(bool(robot_info.get("is_grasped", False)))

        # RCS's raw grasp flag means the gripper encountered an obstruction;
        # it does not by itself guarantee that the obstruction is the cube.
        # Gate it by TCP-cube proximity to prevent exploiting self/base contact.
        valid_grasp_radius_m = 0.025
        valid_cube_grasp = sim_is_grasped * float(xyz_dist < valid_grasp_radius_m)

        cube_lift = max(0.0, float(cube_xyz[2] - self._episode_cube_start_z))
        normalized_lift = float(
            np.clip(
                cube_lift / max(1e-6, self.config.success_lift_delta_m),
                0.0,
                1.0,
            )
        )

        grasp_reward = valid_cube_grasp
        lift_reward = normalized_lift

        success = bool(
            cube_lift >= self.config.success_lift_delta_m
            and valid_cube_grasp > 0.5
        )
        success_bonus = 1.0 if success else 0.0

        # Latch the task into a post-grasp phase once a valid grasp is detected.
        if valid_cube_grasp > 0.5 and not self._has_grasped:
            self._has_grasped = True
            self._tcp_z_at_first_grasp = float(tcp_xyz[2])

        # Once grasped, reward lifting the TCP while the grasp is still held.
        # This gives a denser directional signal than cube lift alone.
        if self._tcp_z_at_first_grasp is None:
            tcp_lift = 0.0
        else:
            tcp_lift = max(0.0, float(tcp_xyz[2] - self._tcp_z_at_first_grasp))

        normalized_tcp_lift = float(
            np.clip(
                tcp_lift / max(1e-6, self.config.success_lift_delta_m),
                0.0,
                1.0,
            )
        )

        tcp_lift_while_grasped_reward = valid_cube_grasp * normalized_tcp_lift

        action_penalty = self.config.action_penalty_weight * float(
            np.mean(np.square(np.asarray(action, dtype=np.float64)))
        )

        if not self._has_grasped:
            # Phase 1: acquire a good grasp.
            reward = (
                0.50 * reach_reward
                + 0.75 * xy_align_reward
                + 1.00 * descend_reward
                + 0.75 * progress_reward
                + 1.00 * close_near_cube_reward
                + 2.00 * grasp_reward
                - action_penalty
            )
        else: 
            # Phase 2: once a grasp has happened, the task is to hold and lift.
            # Do not keep over-rewarding the "stay near the tabletop" configuration.
            reward = (
                1.50 * valid_cube_grasp
                + 6.00 * lift_reward
                + 3.00 * tcp_lift_while_grasped_reward
                + 10.00 * success_bonus
                - action_penalty
            )

        terms = {
            "xy_dist": xy_dist,
            "xyz_dist": xyz_dist,
            "z_dist": z_dist,
            "cube_z": float(cube_xyz[2]),
            "cube_lift": cube_lift,
            "gripper_closed_amount": gripper_closed_amount,
            "sim_is_grasped": sim_is_grasped,
            "valid_cube_grasp": valid_cube_grasp,
            "valid_grasp_radius_m": valid_grasp_radius_m,
            "distance_progress": distance_progress,
            "progress_reward": progress_reward,
            "inside_free_reposition_box": float(inside_free_reposition_box),
            "free_reposition_half_extent_m": free_reposition_half_extent_m,
            "reach_reward": reach_reward,
            "xy_align_reward": xy_align_reward,
            "descend_reward": descend_reward,
            "close_near_cube_reward": close_near_cube_reward,
            "grasp_reward": grasp_reward,
            "lift_reward": lift_reward,
            "success_bonus": success_bonus,
            "success_lift_delta_m": self.config.success_lift_delta_m,
            "has_grasped_latched": float(self._has_grasped),
            "tcp_lift": tcp_lift,
            "tcp_lift_while_grasped_reward": tcp_lift_while_grasped_reward,
        }

        return float(reward), success, terms

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------

    def _flat_action_to_rcs_action(
        self,
        action: np.ndarray,
    ) -> dict[str, Any]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        joint_bound = np.deg2rad(self.config.joint_delta_deg)
        joint_delta = (
            action[:5].astype(np.float64)
            * joint_bound
            * self.config.action_scale
        )

        # Map [-1, 1] to [0, 1].
        gripper_cmd = np.array(
            [(action[5] + 1.0) / 2.0],
            dtype=np.float32,
        )

        return {
            "robot": {
                "joints": joint_delta,
                "gripper": gripper_cmd,
            }
        }

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)

        self._step_count = 0
        obs, info = self.env.reset(seed=seed, options=options)
        # Match the real local-grasp setup: episodes begin with the gripper closed.
        # The flat action's last component maps -1 -> gripper command 0.0 (closed).
        closed_gripper_action = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            dtype=np.float32,
        )

        for _ in range(10):
            rcs_action = self._flat_action_to_rcs_action(closed_gripper_action)
            obs, _reward, _terminated, _truncated, info = self.env.step(rcs_action)
        flat_obs = self._flatten_obs(obs)

        self._episode_cube_start_z = float(flat_obs["cube_xyz"][2])
        self._has_grasped = False
        self._tcp_z_at_first_grasp = None
        self._prev_tcp_cube_dist = None

        return flat_obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1

        rcs_action = self._flat_action_to_rcs_action(action)
        obs, _rcs_reward, _rcs_terminated, truncated, info = self.env.step(rcs_action)

        flat_obs = self._flatten_obs(obs)

        reward, local_success, reward_terms = self._compute_local_grasp_reward(
            flat_obs=flat_obs,
            raw_info=info,
            action=action,
        )

        timeout = self._step_count >= self.config.max_episode_steps
        truncated = bool(truncated or timeout)
        terminated = bool(local_success)

        # Keep both our task-level success and RCS's lower-level info visible.
        info["success"] = local_success
        info["local_success"] = local_success
        info["reward_terms"] = reward_terms

        # Convenient scalar fields for Monitor / callbacks.
        info["cube_z"] = reward_terms["cube_z"]
        info["cube_lift"] = reward_terms["cube_lift"]
        info["xy_dist"] = reward_terms["xy_dist"]
        info["xyz_dist"] = reward_terms["xyz_dist"]
        info["sim_is_grasped"] = bool(reward_terms["sim_is_grasped"] > 0.5)

        return (
            flat_obs,
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        return None

    def close(self) -> None:
        if self._wrist_renderer is not None:
            self._wrist_renderer.close()
        self.env.close()
