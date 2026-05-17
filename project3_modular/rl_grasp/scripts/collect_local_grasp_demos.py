from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import rcs
from rcs._core.common import Pose
from rcs.envs.base import ControlMode, RelativeTo
from rcs.envs.configs import EmptyWorldSO101
from rcs.envs.tasks import PickTaskConfig, RandomSquareObjPos

from project3_modular.rl_grasp.envs.so101_local_grasp_env import (
    COLOR_NAMES,
    COLOR_TO_INDEX,
    SO101LocalGraspConfig,
)


# ---------------------------------------------------------------------
# Keyboard defaults
# ---------------------------------------------------------------------

DEFAULT_KEY_BINDINGS: dict[str, list[int]] = {
    "move_forward": [ord("w")],
    "move_backward": [ord("s")],
    "move_left": [ord("a")],
    "move_right": [ord("d")],
    "move_up": [ord("i")],
    "move_down": [ord("k")],
    "rot_x_pos": [ord("u")],
    "rot_x_neg": [ord("j")],
    "rot_y_pos": [ord("o")],
    "rot_y_neg": [ord("l")],
    "rot_z_pos": [ord("q")],
    "rot_z_neg": [ord("e")],
    "gripper_open": [ord("v")],
    "gripper_close": [ord("c")],
    "reset": [ord("r")],
    "record": [32],       # SPACE
    "end_episode": [13, 10],  # ENTER variants
    "escape": [27],       # ESC
}


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

OBS_KEYS = [
    "joints",
    "gripper",
    "tcp_xyzrpy",
    "cube_xyz",
    "tcp_to_cube_xyz",
    "target_color",
]


@dataclass
class DemoConfig:
    output_dir: str
    control_hz: float
    max_episode_steps: int
    translation_step_m: float
    rotation_step_deg: float
    joint_delta_deg_for_sac: float
    cube_center: tuple[float, float, float]
    cube_randomization_xy: tuple[float, float]
    success_lift_delta_m: float
    target_color: str


class EpisodeBuffer:
    def __init__(self) -> None:
        self.obs: dict[str, list[np.ndarray]] = {key: [] for key in OBS_KEYS}
        self.next_obs: dict[str, list[np.ndarray]] = {key: [] for key in OBS_KEYS}

        self.cartesian_tquat_actions: list[np.ndarray] = []
        self.gripper_commands: list[np.ndarray] = []
        self.rcs_rewards: list[float] = []

        self.sim_is_grasped: list[bool] = []
        self.cube_lift_m: list[float] = []
        self.local_success: list[bool] = []
        self.terminated: list[bool] = []
        self.truncated: list[bool] = []

    def __len__(self) -> int:
        return len(self.rcs_rewards)

    def clear(self) -> None:
        self.__init__()

    def append(
        self,
        *,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        cartesian_tquat_action: np.ndarray,
        gripper_command: np.ndarray,
        rcs_reward: float,
        sim_is_grasped: bool,
        cube_lift_m: float,
        local_success: bool,
        terminated: bool,
        truncated: bool,
    ) -> None:
        for key in OBS_KEYS:
            self.obs[key].append(np.asarray(obs[key], dtype=np.float32).copy())
            self.next_obs[key].append(np.asarray(next_obs[key], dtype=np.float32).copy())

        self.cartesian_tquat_actions.append(
            np.asarray(cartesian_tquat_action, dtype=np.float32).copy()
        )
        self.gripper_commands.append(
            np.asarray(gripper_command, dtype=np.float32).reshape(1).copy()
        )
        self.rcs_rewards.append(float(rcs_reward))

        self.sim_is_grasped.append(bool(sim_is_grasped))
        self.cube_lift_m.append(float(cube_lift_m))
        self.local_success.append(bool(local_success))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect expert local-grasp demos in the RCS SO101 simulation."
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT
        / "project3_modular"
        / "rl_grasp"
        / "demos"
        / "raw_local_grasp",
    )
    parser.add_argument(
        "--keymap-json",
        type=Path,
        default=None,
        help=(
            "Optional HW3-style keymap JSON from configure_keys.py. "
            "If omitted, WASD/I/K-style defaults are used."
        ),
    )

    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--max-episode-steps", type=int, default=160)

    parser.add_argument("--translation-step-m", type=float, default=0.005)
    parser.add_argument("--rotation-step-deg", type=float, default=5.0)

    parser.add_argument(
        "--joint-delta-deg-for-sac",
        type=float,
        default=5.0,
        help=(
            "Normalization scale used to convert observed joint changes into "
            "the flat SAC action format."
        ),
    )

    parser.add_argument("--cube-x", type=float, default=0.18)
    parser.add_argument("--cube-y", type=float, default=0.03)
    parser.add_argument("--cube-z", type=float, default=0.01)
    parser.add_argument(
        "--cube-randomization-width",
        type=float,
        default=0.04,
        help="Total x/y width. 0.04 means ±2 cm.",
    )

    parser.add_argument(
        "--success-lift-delta-m",
        type=float,
        default=0.005,
        help="Displayed success threshold: cube lift above reset height.",
    )

    parser.add_argument("--target-color", type=str, default="green")

    return parser.parse_args()


def load_keymap(path: Path | None) -> tuple[dict[int, str], dict[str, str]]:
    """
    Return:
      raw_key -> action_name
      action_name -> human-readable label
    """
    if path is None:
        raw_to_action: dict[int, str] = {}
        labels: dict[str, str] = {}

        for action, raw_codes in DEFAULT_KEY_BINDINGS.items():
            for code in raw_codes:
                raw_to_action[code] = action

            labels[action] = default_label_for_codes(raw_codes)

        return raw_to_action, labels

    data = json.loads(path.read_text())

    raw_to_action = {}
    labels = {}

    for action, entry in data.items():
        raw = int(entry["raw"])
        raw_to_action[raw] = action
        labels[action] = str(entry.get("label", f"raw:{raw}"))

    return raw_to_action, labels


def default_label_for_codes(codes: list[int]) -> str:
    if not codes:
        return "?"
    code = codes[0]
    if code == 32:
        return "SPACE"
    if code in (10, 13):
        return "ENTER"
    if code == 27:
        return "ESC"
    if 32 <= code <= 126:
        return chr(code)
    return f"raw:{code}"


def axis_angle_to_quat_xyzw(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(1e-12, float(np.linalg.norm(axis)))

    half = 0.5 * angle_rad
    sin_half = np.sin(half)
    x, y, z = axis * sin_half
    w = np.cos(half)

    return np.array([x, y, z, w], dtype=np.float64)


def identity_tquat() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def find_wrapper_of_type(env: gym.Env, wrapper_type: type) -> Any | None:
    current: Any = env

    while True:
        if isinstance(current, wrapper_type):
            return current

        if not hasattr(current, "env"):
            return None

        current = current.env


def target_color_onehot(target_color: str) -> np.ndarray:
    if target_color not in COLOR_TO_INDEX:
        raise ValueError(
            f"Unknown target color {target_color!r}. "
            f"Available: {list(COLOR_TO_INDEX)}"
        )

    onehot = np.zeros(len(COLOR_NAMES), dtype=np.float32)
    onehot[COLOR_TO_INDEX[target_color]] = 1.0
    return onehot


# ---------------------------------------------------------------------
# Teleop environment
# ---------------------------------------------------------------------

class LocalGraspCartesianTeleopEnv:
    """
    RCS SO101 local grasp environment in Cartesian action mode.

    This is intentionally separate from SO101LocalGraspEnv:
      - teleop uses intuitive Cartesian tquat controls
      - collected demos are converted afterward to joint-delta SAC actions
    """

    def __init__(
        self,
        *,
        env_config: SO101LocalGraspConfig,
        translation_step_m: float,
        rotation_step_deg: float,
        target_color: str,
    ) -> None:
        self.env_config = env_config
        self.translation_step_m = float(translation_step_m)
        self.rotation_step_rad = float(np.deg2rad(rotation_step_deg))
        self.target_color = target_color

        self.object_joint_name: str | None = None
        self.shared2world: Pose | None = None

        self.env = self._build_env()

    def _build_env(self) -> gym.Env:
        scene = EmptyWorldSO101()
        cfg = scene.config()

        cfg.robot_to_shared_base_frame = {
            "robot": rcs.common.Pose(
                translation=np.array([0.0, 0.0, self.env_config.robot_z_offset])
            )
        }

        cfg.control_mode = ControlMode.CARTESIAN_TQuat
        cfg.relative_to = RelativeTo.LAST_STEP
        cfg.max_relative_movement = (
            self.translation_step_m,
            self.rotation_step_rad,
        )

        pick_task_cfg = PickTaskConfig(
            robot_name="robot",
            object_center_to_root_frame=Pose(
                translation=np.array(self.env_config.cube_center, dtype=np.float64),
                quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            ),
            object_joint="green_cube_joint",
            include_rotation=True,
        )
        pick_task_cfg.object_xml = self.env_config.cube_xml
        cfg.task_cfg = pick_task_cfg

        self.object_joint_name = pick_task_cfg.prefix + pick_task_cfg.object_joint
        self.shared2world = cfg.shared_base_frame_to_root_frame * cfg.root_frame_to_world

        env = scene.create_env(cfg)

        randomizer = find_wrapper_of_type(env, RandomSquareObjPos)
        if randomizer is None:
            raise RuntimeError("Could not find RandomSquareObjPos wrapper.")

        randomizer.x_width = float(self.env_config.cube_randomization_xy[0])
        randomizer.y_width = float(self.env_config.cube_randomization_xy[1])

        env.get_wrapper_attr("sim").open_gui()
        return env

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)
        return self.flatten_obs(obs), info

    def step(
        self,
        *,
        tquat_action: np.ndarray,
        gripper_command: float,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = {
            "robot": {
                "tquat": np.asarray(tquat_action, dtype=np.float64),
                "gripper": np.array([gripper_command], dtype=np.float32),
            }
        }

        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.flatten_obs(obs), float(reward), bool(terminated), bool(truncated), info

    def cube_xyz_in_shared_frame(self) -> np.ndarray:
        assert self.object_joint_name is not None
        assert self.shared2world is not None

        sim = self.env.get_wrapper_attr("sim")
        cube_xyz_world = np.asarray(
            sim.data.joint(self.object_joint_name).qpos[:3],
            dtype=np.float64,
        )

        cube_pose_world = rcs.common.Pose(translation=cube_xyz_world)
        cube_pose_shared = self.shared2world.inverse() * cube_pose_world

        return np.asarray(cube_pose_shared.translation(), dtype=np.float32)

    def flatten_obs(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        robot_obs = obs["robot"]

        joints = np.asarray(robot_obs["joints"], dtype=np.float32)
        gripper = np.asarray(robot_obs["gripper"], dtype=np.float32).reshape(1)
        tcp_xyzrpy = np.asarray(robot_obs["xyzrpy"], dtype=np.float32)

        cube_xyz = self.cube_xyz_in_shared_frame()
        tcp_to_cube_xyz = cube_xyz - tcp_xyzrpy[:3]

        return {
            "joints": joints,
            "gripper": gripper,
            "tcp_xyzrpy": tcp_xyzrpy,
            "cube_xyz": cube_xyz,
            "tcp_to_cube_xyz": tcp_to_cube_xyz.astype(np.float32),
            "target_color": target_color_onehot(self.target_color),
        }

    def close(self) -> None:
        self.env.close()


# ---------------------------------------------------------------------
# Teleop control mapping
# ---------------------------------------------------------------------

def build_tquat_action(
    *,
    action_name: str | None,
    translation_step_m: float,
    rotation_step_rad: float,
) -> np.ndarray:
    act = identity_tquat()

    if action_name is None:
        return act

    # RCS/SO101 frame:
    #   +x roughly forward away from the robot base
    #   +y lateral
    #   +z upward
    if action_name == "move_forward":
        act[0] = +translation_step_m
    elif action_name == "move_backward":
        act[0] = -translation_step_m
    elif action_name == "move_left":
        act[1] = +translation_step_m
    elif action_name == "move_right":
        act[1] = -translation_step_m
    elif action_name == "move_up":
        act[2] = +translation_step_m
    elif action_name == "move_down":
        act[2] = -translation_step_m
    elif action_name == "rot_x_pos":
        act[3:] = axis_angle_to_quat_xyzw(
            np.array([1.0, 0.0, 0.0]), +rotation_step_rad
        )
    elif action_name == "rot_x_neg":
        act[3:] = axis_angle_to_quat_xyzw(
            np.array([1.0, 0.0, 0.0]), -rotation_step_rad
        )
    elif action_name == "rot_y_pos":
        act[3:] = axis_angle_to_quat_xyzw(
            np.array([0.0, 1.0, 0.0]), +rotation_step_rad
        )
    elif action_name == "rot_y_neg":
        act[3:] = axis_angle_to_quat_xyzw(
            np.array([0.0, 1.0, 0.0]), -rotation_step_rad
        )
    elif action_name == "rot_z_pos":
        act[3:] = axis_angle_to_quat_xyzw(
            np.array([0.0, 0.0, 1.0]), +rotation_step_rad
        )
    elif action_name == "rot_z_neg":
        act[3:] = axis_angle_to_quat_xyzw(
            np.array([0.0, 0.0, 1.0]), -rotation_step_rad
        )

    return act


# ---------------------------------------------------------------------
# Saving demos
# ---------------------------------------------------------------------

def stack_obs_dict(buffers: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.stack(values, axis=0).astype(np.float32)
        for key, values in buffers.items()
    }


def save_episode(
    *,
    episode_idx: int,
    output_dir: Path,
    episode: EpisodeBuffer,
    demo_config: DemoConfig,
) -> dict[str, Any]:
    if len(episode) == 0:
        raise ValueError("Cannot save an empty episode.")

    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / f"episode_{episode_idx:04d}.npz"

    obs = stack_obs_dict(episode.obs)
    next_obs = stack_obs_dict(episode.next_obs)

    tquat_actions = np.stack(episode.cartesian_tquat_actions, axis=0).astype(np.float32)
    gripper_commands = np.stack(episode.gripper_commands, axis=0).astype(np.float32)

    obs_joints = obs["joints"]
    next_joints = next_obs["joints"]

    joint_bound_rad = float(np.deg2rad(demo_config.joint_delta_deg_for_sac))
    raw_joint_deltas = next_joints - obs_joints
    normalized_joint_deltas = raw_joint_deltas / max(1e-12, joint_bound_rad)

    joint_clip_mask = np.abs(normalized_joint_deltas) > 1.0
    clipped_joint_fraction = float(np.mean(joint_clip_mask))
    normalized_joint_deltas_clipped = np.clip(
        normalized_joint_deltas,
        -1.0,
        1.0,
    ).astype(np.float32)

    normalized_gripper_actions = (2.0 * gripper_commands - 1.0).astype(np.float32)

    flat_sac_actions = np.concatenate(
        [normalized_joint_deltas_clipped, normalized_gripper_actions],
        axis=1,
    ).astype(np.float32)

    rcs_rewards = np.asarray(episode.rcs_rewards, dtype=np.float32)
    sim_is_grasped = np.asarray(episode.sim_is_grasped, dtype=np.bool_)
    cube_lift_m = np.asarray(episode.cube_lift_m, dtype=np.float32)
    local_success = np.asarray(episode.local_success, dtype=np.bool_)
    terminated = np.asarray(episode.terminated, dtype=np.bool_)
    truncated = np.asarray(episode.truncated, dtype=np.bool_)

    np.savez_compressed(
        episode_path,
        # obs
        obs_joints=obs["joints"],
        obs_gripper=obs["gripper"],
        obs_tcp_xyzrpy=obs["tcp_xyzrpy"],
        obs_cube_xyz=obs["cube_xyz"],
        obs_tcp_to_cube_xyz=obs["tcp_to_cube_xyz"],
        obs_target_color=obs["target_color"],
        # next obs
        next_obs_joints=next_obs["joints"],
        next_obs_gripper=next_obs["gripper"],
        next_obs_tcp_xyzrpy=next_obs["tcp_xyzrpy"],
        next_obs_cube_xyz=next_obs["cube_xyz"],
        next_obs_tcp_to_cube_xyz=next_obs["tcp_to_cube_xyz"],
        next_obs_target_color=next_obs["target_color"],
        # actions
        teleop_cartesian_tquat_actions=tquat_actions,
        teleop_gripper_commands=gripper_commands,
        raw_joint_deltas=raw_joint_deltas.astype(np.float32),
        normalized_flat_sac_actions=flat_sac_actions,
        # metadata / diagnostics
        rcs_rewards=rcs_rewards,
        sim_is_grasped=sim_is_grasped,
        cube_lift_m=cube_lift_m,
        local_success=local_success,
        terminated=terminated,
        truncated=truncated,
    )

    summary = {
        "episode_idx": int(episode_idx),
        "path": str(episode_path),
        "num_steps": int(len(episode)),
        "any_grasped": bool(np.any(sim_is_grasped)),
        "final_grasped": bool(sim_is_grasped[-1]),
        "max_cube_lift_m": float(np.max(cube_lift_m)),
        "final_cube_lift_m": float(cube_lift_m[-1]),
        "any_local_success": bool(np.any(local_success)),
        "final_local_success": bool(local_success[-1]),
        "clipped_joint_action_fraction": clipped_joint_fraction,
    }

    return summary


def update_manifest(
    *,
    output_dir: Path,
    demo_config: DemoConfig,
    summaries: list[dict[str, Any]],
) -> None:
    manifest = {
        "demo_config": asdict(demo_config),
        "episodes": summaries,
        "num_episodes": len(summaries),
        "num_successful_episodes": int(
            sum(bool(ep["any_local_success"]) for ep in summaries)
        ),
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


# ---------------------------------------------------------------------
# UI overlay
# ---------------------------------------------------------------------

def make_status_image(
    *,
    recording: bool,
    episode_idx: int,
    steps_buffered: int,
    latest_grasped: bool,
    latest_lift_m: float,
    latest_success: bool,
    labels: dict[str, str],
) -> np.ndarray:
    h, w = 330, 1180
    img = np.zeros((h, w, 3), dtype=np.uint8)

    status = (
        f"{'REC' if recording else 'IDLE'} | "
        f"saved episodes {episode_idx} | "
        f"buffered steps {steps_buffered}"
    )
    cv2.putText(
        img,
        status,
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255) if recording else (220, 220, 220),
        2,
    )

    diagnostics = (
        f"grasped={latest_grasped} | "
        f"cube lift={latest_lift_m * 1000:.1f} mm | "
        f"local success={latest_success}"
    )
    cv2.putText(
        img,
        diagnostics,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if latest_success else (255, 255, 255),
        2,
    )

    move_hint = (
        f"Move: {labels.get('move_forward', '?')}/{labels.get('move_backward', '?')} "
        f"forward/back, {labels.get('move_left', '?')}/{labels.get('move_right', '?')} "
        f"left/right, {labels.get('move_up', '?')}/{labels.get('move_down', '?')} up/down"
    )
    cv2.putText(
        img,
        move_hint,
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )

    grip_hint = (
        f"Gripper: {labels.get('gripper_open', '?')} open, "
        f"{labels.get('gripper_close', '?')} close"
    )
    cv2.putText(
        img,
        grip_hint,
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )

    session_hint = (
        f"{labels.get('record', '?')} rec toggle | "
        f"{labels.get('end_episode', '?')} save & reset | "
        f"{labels.get('reset', '?')} discard/reset | "
        f"{labels.get('escape', '?')} quit"
    )
    cv2.putText(
        img,
        session_hint,
        (20, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 220, 255),
        2,
    )

    tiny_hint = (
        "Focus this OpenCV window for keys; steer by watching the MuJoCo GUI."
    )
    cv2.putText(
        img,
        tiny_hint,
        (20, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (170, 170, 170),
        1,
    )

    return img


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"demos_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    env_config = SO101LocalGraspConfig(
        cube_center=(args.cube_x, args.cube_y, args.cube_z),
        cube_randomization_xy=(
            args.cube_randomization_width,
            args.cube_randomization_width,
        ),
        max_episode_steps=args.max_episode_steps,
        target_color=args.target_color,
        success_lift_delta_m=args.success_lift_delta_m,
    )

    demo_config = DemoConfig(
        output_dir=str(run_dir),
        control_hz=float(args.control_hz),
        max_episode_steps=int(args.max_episode_steps),
        translation_step_m=float(args.translation_step_m),
        rotation_step_deg=float(args.rotation_step_deg),
        joint_delta_deg_for_sac=float(args.joint_delta_deg_for_sac),
        cube_center=env_config.cube_center,
        cube_randomization_xy=env_config.cube_randomization_xy,
        success_lift_delta_m=float(args.success_lift_delta_m),
        target_color=str(args.target_color),
    )

    (run_dir / "demo_config.json").write_text(
        json.dumps(asdict(demo_config), indent=2) + "\n"
    )

    raw_to_action, labels = load_keymap(args.keymap_json)

    teleop_env = LocalGraspCartesianTeleopEnv(
        env_config=env_config,
        translation_step_m=args.translation_step_m,
        rotation_step_deg=args.rotation_step_deg,
        target_color=args.target_color,
    )

    current_obs, _ = teleop_env.reset()
    episode_cube_start_z = float(current_obs["cube_xyz"][2])

    gripper_command = 1.0  # open
    episode = EpisodeBuffer()
    summaries: list[dict[str, Any]] = []

    recording = False
    running = True
    latest_grasped = False
    latest_lift_m = 0.0
    latest_success = False

    cv2.namedWindow("SO101 Local Grasp Demo Collector", cv2.WINDOW_AUTOSIZE)

    dt = 1.0 / float(args.control_hz)
    episode_env_steps = 0

    print("\n=== SO101 local grasp demo collector ===")
    print(f"Output directory: {run_dir}")
    print("Focus the OpenCV status window to send keys.")
    print("Watch the MuJoCo GUI to teleoperate.")
    print("Try to save successful grasp-and-lift episodes.\n")

    try:
        while running:
            loop_start = time.perf_counter()

            key_raw = cv2.waitKeyEx(1)
            action_name = raw_to_action.get(key_raw) if key_raw != -1 else None

            if key_raw != -1:
                print(f"key_raw={key_raw}, action_name={action_name}")

            if action_name == "escape":
                running = False
                continue

            if action_name == "record":
                recording = not recording
                print("RECORDING ON" if recording else "RECORDING OFF")
                action_name = None

            elif action_name == "reset":
                if recording and len(episode) > 0:
                    print("Discarded current recorded episode.")
                recording = False
                episode.clear()

                current_obs, _ = teleop_env.reset()
                episode_cube_start_z = float(current_obs["cube_xyz"][2])
                gripper_command = 1.0
                episode_env_steps = 0

                latest_grasped = False
                latest_lift_m = 0.0
                latest_success = False
                action_name = None

            elif action_name == "end_episode":
                if recording and len(episode) > 0:
                    summary = save_episode(
                        episode_idx=len(summaries),
                        output_dir=run_dir,
                        episode=episode,
                        demo_config=demo_config,
                    )
                    summaries.append(summary)
                    update_manifest(
                        output_dir=run_dir,
                        demo_config=demo_config,
                        summaries=summaries,
                    )

                    print(
                        f"Saved episode {summary['episode_idx']:04d}: "
                        f"{summary['num_steps']} steps, "
                        f"success={summary['any_local_success']}, "
                        f"max lift={summary['max_cube_lift_m'] * 1000:.1f} mm, "
                        f"joint clip frac={summary['clipped_joint_action_fraction']:.3f}"
                    )
                else:
                    print("No recorded steps to save.")

                recording = False
                episode.clear()

                current_obs, _ = teleop_env.reset()
                episode_cube_start_z = float(current_obs["cube_xyz"][2])
                gripper_command = 1.0
                episode_env_steps = 0

                latest_grasped = False
                latest_lift_m = 0.0
                latest_success = False
                action_name = None

            if action_name == "gripper_open":
                gripper_command = 1.0
                action_name = None

            elif action_name == "gripper_close":
                gripper_command = 0.0
                action_name = None

            tquat_action = build_tquat_action(
                action_name=action_name,
                translation_step_m=args.translation_step_m,
                rotation_step_rad=float(np.deg2rad(args.rotation_step_deg)),
            )

            next_obs, rcs_reward, terminated, truncated, info = teleop_env.step(
                tquat_action=tquat_action,
                gripper_command=gripper_command,
            )

            robot_info = info.get("robot", {})
            sim_is_grasped = bool(robot_info.get("is_grasped", False))

            latest_lift_m = float(next_obs["cube_xyz"][2] - episode_cube_start_z)
            latest_grasped = sim_is_grasped
            latest_success = bool(
                sim_is_grasped
                and latest_lift_m >= args.success_lift_delta_m
            )

            if recording:
                episode.append(
                    obs=current_obs,
                    next_obs=next_obs,
                    cartesian_tquat_action=tquat_action,
                    gripper_command=np.array([gripper_command], dtype=np.float32),
                    rcs_reward=rcs_reward,
                    sim_is_grasped=sim_is_grasped,
                    cube_lift_m=latest_lift_m,
                    local_success=latest_success,
                    terminated=terminated,
                    truncated=truncated,
                )

            current_obs = next_obs
            episode_env_steps += 1

            if episode_env_steps >= args.max_episode_steps:
                print("Reached max episode steps; resetting. Unsaved buffer is kept.")
                current_obs, _ = teleop_env.reset()
                episode_cube_start_z = float(current_obs["cube_xyz"][2])
                gripper_command = 1.0
                episode_env_steps = 0

            status_img = make_status_image(
                recording=recording,
                episode_idx=len(summaries),
                steps_buffered=len(episode),
                latest_grasped=latest_grasped,
                latest_lift_m=latest_lift_m,
                latest_success=latest_success,
                labels=labels,
            )
            cv2.imshow("SO101 Local Grasp Demo Collector", status_img)

            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    finally:
        update_manifest(
            output_dir=run_dir,
            demo_config=demo_config,
            summaries=summaries,
        )
        teleop_env.close()
        cv2.destroyAllWindows()
        print("\nDemo collection finished.")
        print(f"Saved {len(summaries)} episode(s) to:")
        print(f"  {run_dir}")


if __name__ == "__main__":
    main()
