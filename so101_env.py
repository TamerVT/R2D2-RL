"""
Gymnasium environment wrapper for SO-101 pick-and-place with wrist camera.

Integrates:
  - SO101Controller (RCS motor control)
  - MuJoCo 3.8.0 simulation
  - Wrist camera RGB feed (headless-friendly)
  - Perception module (RGB → block positions)
  - LeRobot-p3 compatible observation dict

Usage:
    from so101_env import SO101PickPlaceEnv, SO101EnvConfig

    config = SO101EnvConfig()
    env = SO101PickPlaceEnv(config)
    obs, info = env.reset()

    for _ in range(1000):
        action = policy(obs)  # obs['wrist_image'], obs['block_positions']
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2  # For color segmentation and contour detection

import mujoco
import mujoco.viewer

# Import controller
sys.path.insert(0, str(Path(__file__).parent))
from controller import SO101Controller, ControllerConfig, ControlPhase


@dataclass
class SO101EnvConfig:
    """Configuration for SO101PickPlaceEnv."""
    # Paths
    mujoco_xml_path: Optional[Path] = None  # Auto-detect if None

    # Camera
    wrist_camera_name: str = "wrist_cam"
    wrist_camera_width: int = 640
    wrist_camera_height: int = 480

    # Simulation
    render_mode: Optional[str] = None  # None, "human", "rgb_array"
    sim_timestep: float = 0.001  # MuJoCo timestep
    control_decimation: int = 10  # Steps per control action

    # Task
    max_episode_steps: int = 500
    pick_target_color: str = "red"  # Which color to pick (red, blue, green, etc.)

    # Reward shaping
    dist_reward_scale: float = 1.0
    grasp_reward: float = 10.0
    place_reward: float = 50.0
    collision_penalty: float = -1.0

    # Controller
    controller_config: Optional[ControllerConfig] = None


# perception module (real color-based detection) 

class BlockPerception:
    """
    Real-time perception pipeline: RGB → block 3D positions.

    Pipeline:
      1. HSV color segmentation for each color
      2. Contour detection & centroid calculation
      3. Camera model backprojection (pixel → 3D point)
      4. Transform to world frame
    """

    # Color ranges in HSV (H: 0-180, S/V: 0-255)
    COLOR_RANGES = {
        'red': [
            ((0, 80, 100), (10, 255, 255)),       # Red hue lower range (0-10)
            ((170, 80, 100), (180, 255, 255)),    # Red hue upper range (170-180)
        ],
        'blue': [
            ((100, 80, 100), (130, 255, 255)),    # Blue hue range (100-130)
        ],
        'green': [
            ((40, 80, 100), (80, 255, 255)),      # Green hue range (40-80)
        ],
    }

    def __init__(
        self,
        image_height: int = 480,
        image_width: int = 640,
        focal_length: float = 400.0,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        depth_estimate: float = 0.15,  # Assumed distance to table
    ):
        self.height = image_height
        self.width = image_width
        self.focal_length = focal_length
        self.cx = cx if cx is not None else image_width / 2  # Principal point X
        self.cy = cy if cy is not None else image_height / 2  # Principal point Y
        self.depth_estimate = depth_estimate  # Assume objects at fixed depth

    def detect_blocks(
        self,
        wrist_image: np.ndarray,
        ee_pos: np.ndarray,
        ee_rot: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Detect colored blocks in wrist image and return 3D positions.

        Args:
            wrist_image: RGB image from wrist camera (H, W, 3), uint8.
            ee_pos: End-effector position (3,) in world frame.
            ee_rot: End-effector rotation matrix (3, 3) in world frame.

        Returns:
            Dict mapping color name → 3D position in world frame.
        """
        blocks = {}

        # Convert BGR (OpenCV default) to HSV for robust color detection
        hsv = cv2.cvtColor(wrist_image, cv2.COLOR_RGB2HSV)

        # Detect each color
        for color, ranges in self.COLOR_RANGES.items():
            # Create binary mask by thresholding HSV image
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            for lower, upper in ranges:
                # Threshold HSV to isolate this color range
                submask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                # Combine masks (especially important for red, which wraps HSV)
                mask = cv2.bitwise_or(mask, submask)

            # Morphological operations to clean up noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # Erosion removes small noise
            mask = cv2.erode(mask, kernel, iterations=1)
            # Dilation recovers object boundaries
            mask = cv2.dilate(mask, kernel, iterations=1)

            # Find contours in mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get largest contour (likely the block)
                largest_contour = max(contours, key=cv2.contourArea)
                # Calculate contour area to filter noise
                area = cv2.contourArea(largest_contour)

                if area > 50:  # Minimum area threshold to ignore noise
                    # Compute centroid of contour (center of mass in pixel space)
                    moments = cv2.moments(largest_contour)
                    if moments['m00'] != 0:
                        # Centroid in image coordinates
                        u = int(moments['m10'] / moments['m00'])
                        v = int(moments['m01'] / moments['m00'])

                        # Backproject pixel to 3D using pinhole camera model
                        # Z = depth_estimate (assume objects on table plane)
                        # X = (u - cx) * Z / focal_length
                        # Y = (v - cy) * Z / focal_length
                        pos_camera = self._pixel_to_camera(u, v, self.depth_estimate)

                        # Transform from camera frame to end-effector frame
                        # Assume camera is mounted at EE with identity rotation (hardcoded)
                        pos_ee = pos_camera.copy()

                        # Transform from EE frame to world frame using EE pose
                        # pos_world = ee_pos + ee_rot @ pos_ee
                        pos_world = ee_pos + ee_rot @ pos_ee

                        blocks[color] = pos_world.astype(np.float32)

        # Return detections, or fallback to empty dict if no blocks detected
        return blocks

    def _pixel_to_camera(self, u: int, v: int, depth: float) -> np.ndarray:
        """
        Backproject pixel coordinates to 3D point in camera frame.

        Pinhole camera model:
            X_cam = (u - cx) * Z / f
            Y_cam = (v - cy) * Z / f
            Z_cam = Z (depth)

        Args:
            u, v: Pixel coordinates in image.
            depth: Depth estimate (Z distance from camera).

        Returns:
            3D point in camera frame (x, y, z).
        """
        # Normalize pixel coordinates relative to principal point
        x_norm = (u - self.cx) / self.focal_length
        y_norm = (v - self.cy) / self.focal_length
        # Scale by depth to get 3D point
        x_cam = x_norm * depth
        y_cam = y_norm * depth
        z_cam = depth
        return np.array([x_cam, y_cam, z_cam], dtype=np.float32)


# environment wrapper integrating mujoco, controller, and perception

class SO101PickPlaceEnv(gym.Env):
    """
    Gymnasium environment for SO-101 pick-and-place with visual perception.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config: Optional[SO101EnvConfig] = None):
        self.config = config or SO101EnvConfig()

        # mujoco
        xml_path = self.config.mujoco_xml_path
        if xml_path is None:
            # Auto-detect from hw4_reinforcement_learning
            repo_root = Path(__file__).parent
            hw4_dir = repo_root / "hw4_reinforcement_learning" if not repo_root.name == "hw4_reinforcement_learning" else repo_root
            xml_path = hw4_dir / "assets" / "mujoco" / "so100_pos_ctrl.xml"

        xml_path = Path(xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # controller
        ctrl_cfg = self.config.controller_config or ControllerConfig()
        self.controller = SO101Controller(
            self.model,
            self.data,
            ee_site_name="ee_site",
            config=ctrl_cfg,
        )

        # perception module with camera
        # Create perception module with camera intrinsics (focal length, principal point)
        self.perception = BlockPerception(
            image_height=self.config.wrist_camera_height,  # Camera image height
            image_width=self.config.wrist_camera_width,    # Camera image width
        )

        # camera setup 
        self.wrist_camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.config.wrist_camera_name,
        )
        if self.wrist_camera_id < 0:
            raise ValueError(f"Camera '{self.config.wrist_camera_name}' not found in model")

        #  renderer for offscreen rendering 
        self.renderer = mujoco.Renderer(
            self.model,
            height=self.config.wrist_camera_height,
            width=self.config.wrist_camera_width,
        )

        #  viewer for human rendering 
        self.viewer = None
        self.render_mode = self.config.render_mode

        #  observation/action spaces 
        self.observation_space = spaces.Dict({
            'wrist_image': spaces.Box(
                low=0,
                high=255,
                shape=(self.config.wrist_camera_height, self.config.wrist_camera_width, 3),
                dtype=np.uint8,
            ),
            'block_positions': spaces.Dict({
                'red': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                'blue': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                'green': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            }),
            'ee_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'ee_rot': spaces.Box(low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float32),
            'qpos': spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32),
        })

        self.action_space = spaces.Dict({
            'target_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            'target_rot': spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.float32),
            'grasp': spaces.Discrete(2),  # 0=no grasp, 1=grasp
        })

        self.episode_step = 0
        self.pick_pose = None
        self.place_pose = None

    # Reset

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment and return initial observation."""
        super().reset(seed=seed, options=options)

        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = np.array([0.0, -1.57, 1.0, 1.0, 0.0, 0.02], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

        # Reset controller
        self.controller.reset()

        # Reset episode
        self.episode_step = 0
        self.pick_pose = None
        self.place_pose = None

        obs = self._get_observation()
        info = {"episode_step": self.episode_step}

        return obs, info


    def step(self, action: Dict[str, Any]) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one environment step.

        Args:
            action: Dict with 'target_pos', 'target_rot', 'grasp'.
                    If from policy: motor level commands.
                    If from controller: high-level goals.

        Returns:
            obs, reward, terminated, truncated, info
        """
        # Parse action
        if isinstance(action, dict):
            target_pos = np.asarray(action.get('target_pos', self.controller.get_ee_pos()))
            target_rot = np.asarray(action.get('target_rot', self.controller.get_ee_rot()))
            grasp_cmd = action.get('grasp', 0)
        else:
            # Fallback: use controller's current target
            target_pos = self.controller.state.target_ee_pos
            target_rot = self.controller.state.target_ee_rot
            grasp_cmd = 0

        # Plan motion if new target
        if not np.allclose(target_pos, self.controller.state.target_ee_pos):
            self.controller.plan_to_cartesian(target_pos, target_rot, duration=1.0)

        # Grasp command
        if grasp_cmd > 0:
            self.controller.set_gripper(open=False)
        else:
            self.controller.set_gripper(open=True)

        # Simulate for control_decimation steps
        for _ in range(self.config.control_decimation):
            ctrl_action = self.controller.step(self.config.sim_timestep)
            self.data.ctrl[:6] = ctrl_action
            mujoco.mj_step(self.model, self.data)

        self.episode_step += 1

        # Get observation
        obs = self._get_observation()

        # Compute reward
        reward = self._compute_reward(obs, action)

        # Check termination
        truncated = self.episode_step >= self.config.max_episode_steps
        terminated = False  # Only truncate on max steps, not on task completion (yet)

        info = {
            "episode_step": self.episode_step,
            "controller_phase": self.controller.state.phase.name,
            "safety_ok": self.controller.state.safety_ok,
            "ee_pos_error": self.controller.state.ee_pos_error,
        }

        # Render if requested
        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    # observation with perception

    def _get_observation(self) -> Dict[str, Any]:
        """Get current observation dict with camera feed and perceived blocks."""
        # Disable depth rendering (only RGB needed), update scene from MuJoCo state
        self.renderer.enable_depth_rendering(False)
        self.renderer.update_scene(self.data)
        # Render wrist camera view as RGB image (H, W, 3) uint8
        wrist_image = self.renderer.render()

        # Get end-effector pose (position + rotation matrix)
        ee_pos = self.controller.get_ee_pos()
        ee_rot = self.controller.get_ee_rot()
        # Run perception pipeline: RGB → block 3D positions in world frame
        block_positions = self.perception.detect_blocks(wrist_image, ee_pos, ee_rot)

        # Assemble observation dict for policy/RL
        obs = {
            'wrist_image': wrist_image.astype(np.uint8),  # Raw RGB feed
            'block_positions': {
                k: v.astype(np.float32) for k, v in block_positions.items()  # Detected blocks
            },
            'ee_pos': ee_pos.astype(np.float32),  # End-effector position
            'ee_rot': ee_rot.astype(np.float32),  # End-effector rotation matrix
            'qpos': self.controller.get_qpos().astype(np.float32),  # Joint positions
        }

        return obs


    def _compute_reward(self, obs: Dict, action: Dict) -> float:
        """Compute reward."""
        reward = 0.0

        # Distance to target
        if self.controller.state.phase == ControlPhase.MOVING_TO_TARGET:
            pos_error = self.controller.state.ee_pos_error
            reward -= self.config.dist_reward_scale * pos_error

        # Grasp bonus
        if self.controller.state.phase == ControlPhase.GRASPING:
            reward += self.config.grasp_reward / 100.0  # per step

        # Safety penalty
        if not self.controller.state.safety_ok:
            reward += self.config.collision_penalty

        return float(reward)


    def render(self) -> Optional[np.ndarray]:
        """Render environment."""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()

        elif self.render_mode == "rgb_array":
            self.renderer.update_scene(self.data)
            return self.renderer.render()

        return None

    def close(self) -> None:
        """Cleanup."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.renderer.close()


