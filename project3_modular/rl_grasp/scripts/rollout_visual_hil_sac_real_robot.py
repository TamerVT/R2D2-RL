from __future__ import annotations

import argparse
import csv
import json
import signal
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from lerobot.robots.so_follower import SOFollower, SO101FollowerConfig



JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

COLOR_NAMES = ["blue", "green", "purple", "orange", "yellow", "red"]
COLOR_TO_INDEX = {name: i for i, name in enumerate(COLOR_NAMES)}

# Same arm-joint limits we used in the HIL-compatible sim wrapper.
ACTION_LOW = np.array(
    [
        -68.90625,
        -103.7548828125,
        -97.470703125,
        -102.216796875,
        -179.9560546875,
        0.0,
    ],
    dtype=np.float32,
)

ACTION_HIGH = np.array(
    [
        68.90625,
        103.7548828125,
        97.470703125,
        102.216796875,
        179.9560546875,
        30.0,  # Real HIL config used max_gripper_pos=30.
    ],
    dtype=np.float32,
)


# ---------------------------------------------------------------------
# SAC feature extractor definitions
# ---------------------------------------------------------------------
#
# These must match the classes used during sim training. They are copied here
# rather than imported from train_visual_hil_compat_sac.py because that training
# script imports the RCS simulation environment, which is unavailable in the
# Python 3.12 LeRobot runtime used for real-arm control.


class WristImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(image))


class VisualHILFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        state_mean: list[float],
        state_std: list[float],
        image_feature_dim: int = 256,
    ) -> None:
        super().__init__(
            observation_space,
            features_dim=image_feature_dim + 24,
        )

        self.image_encoder = WristImageEncoder(out_dim=image_feature_dim)

        mean = torch.tensor(state_mean, dtype=torch.float32)
        std = torch.tensor(state_std, dtype=torch.float32)
        std = torch.clamp(std, min=1e-6)

        self.register_buffer("state_mean", mean)
        self.register_buffer("state_std", std)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        image = observations["observation.images.wrist"]
        state = observations["observation.state"]

        image_feat = self.image_encoder(image)
        norm_state = (state - self.state_mean) / self.state_std

        return torch.cat([image_feat, norm_state], dim=-1)


STOP_REQUESTED = False


def handle_sigint(_sig, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\nStop requested. Finishing current step safely...")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="SB3 SAC .zip checkpoint.",
    )
    parser.add_argument(
        "--normalization-stats",
        required=True,
        help="JSON normalization stats used by the visual SAC feature extractor.",
    )
    parser.add_argument("--robot-port", default="/dev/ttyACM2")
    parser.add_argument("--robot-id", default="my_awesome_follower_arm")

    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)

    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--target-color", default="green", choices=COLOR_NAMES)

    parser.add_argument(
        "--max-joint-step-deg",
        type=float,
        default=3.0,
        help="Safety cap on per-control-step joint change.",
    )
    parser.add_argument(
        "--max-gripper-step",
        type=float,
        default=5.0,
        help="Safety cap on per-control-step gripper change.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run perception + policy, print actions, but do not send commands.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic SAC actions instead of deterministic policy mean.",
    )

    parser.add_argument(
        "--output-dir",
        default="third_party/robot-control-stack/project3_modular/rl_grasp/outputs/real_sac_rollout_debug",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save the real wrist-camera input video.",
    )

    return parser.parse_args()


def read_frame(cap: cv2.VideoCapture, retries: int = 10) -> np.ndarray:
    for _ in range(retries):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.02)
    raise RuntimeError("Could not read a frame from the wrist camera.")


def preprocess_frame_bgr_to_obs_chw(frame_bgr: np.ndarray) -> np.ndarray:
    # HIL config cropped the full image and resized to 128x128.
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_128 = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    chw = np.transpose(rgb_128, (2, 0, 1)).copy()
    return chw.astype(np.uint8)


def observation_positions(robot_obs: dict[str, float]) -> np.ndarray:
    return np.array([float(robot_obs[k]) for k in JOINT_KEYS], dtype=np.float32)


def make_color_onehot(color: str) -> np.ndarray:
    onehot = np.zeros(6, dtype=np.float32)
    onehot[COLOR_TO_INDEX[color]] = 1.0
    return onehot


def build_policy_state(
    positions: np.ndarray,
    prev_positions: np.ndarray | None,
    dt_s: float,
    color_onehot: np.ndarray,
) -> np.ndarray:
    if prev_positions is None:
        velocities = np.zeros(6, dtype=np.float32)
    else:
        velocities = ((positions - prev_positions) / max(1e-6, dt_s)).astype(np.float32)

    # The sim SAC policy was trained with zero current placeholders.
    currents_placeholder = np.zeros(6, dtype=np.float32)

    state = np.concatenate(
        [
            positions.astype(np.float32),
            velocities,
            currents_placeholder,
            color_onehot,
        ],
        axis=0,
    ).astype(np.float32)

    assert state.shape == (24,)
    return state


def clip_policy_action_safely(
    raw_action: np.ndarray,
    current_positions: np.ndarray,
    max_joint_step_deg: float,
    max_gripper_step: float,
) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float32).reshape(6)
    action = np.clip(action, ACTION_LOW, ACTION_HIGH)

    safe = action.copy()

    joint_delta = safe[:5] - current_positions[:5]
    joint_delta = np.clip(joint_delta, -max_joint_step_deg, max_joint_step_deg)
    safe[:5] = current_positions[:5] + joint_delta

    gripper_delta = float(safe[5] - current_positions[5])
    gripper_delta = float(np.clip(gripper_delta, -max_gripper_step, max_gripper_step))
    safe[5] = current_positions[5] + gripper_delta

    safe = np.clip(safe, ACTION_LOW, ACTION_HIGH)
    return safe.astype(np.float32)


def action_array_to_robot_dict(action: np.ndarray) -> dict[str, float]:
    return {k: float(v) for k, v in zip(JOINT_KEYS, action, strict=True)}


def overlay_debug(
    frame_bgr: np.ndarray,
    *,
    step: int,
    dry_run: bool,
    current: np.ndarray,
    safe_action: np.ndarray,
) -> np.ndarray:
    out = frame_bgr.copy()

    mode = "DRY RUN" if dry_run else "LIVE CONTROL"
    cv2.putText(
        out,
        f"{mode} | step {step}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255) if not dry_run else (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        f"curr q: {np.array2string(current, precision=1, suppress_small=True)}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        f"send q: {np.array2string(safe_action, precision=1, suppress_small=True)}",
        (12, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return out


def main() -> None:
    global STOP_REQUESTED

    args = parse_args()
    signal.signal(signal.SIGINT, handle_sigint)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    deterministic = not args.stochastic

    print("Loading SAC policy...")

    stats = json.loads(Path(args.normalization_stats).read_text())

    policy_kwargs = {
        "features_extractor_class": VisualHILFeaturesExtractor,
        "features_extractor_kwargs": {
            "state_mean": stats["state_mean"],
            "state_std": stats["state_std"],
            "image_feature_dim": 256,
        },
        "net_arch": {
            "pi": [512, 256],
            "qf": [512, 256],
        },
        "activation_fn": torch.nn.ReLU,
        "normalize_images": True,
    }

    # The checkpoint was saved under Python 3.11, while this real rollout must
    # run in the Python 3.12 LeRobot environment. Replace the serialized
    # policy_kwargs so SB3 does not try to unpickle the old custom extractor.
    model = SAC.load(
        args.checkpoint,
        device=device,
        custom_objects={"policy_kwargs": policy_kwargs},
    )
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Deterministic actions: {deterministic}")

    print()
    print("Connecting follower arm...")
    robot = SOFollower(
        SO101FollowerConfig(
            port=args.robot_port,
            id=args.robot_id,
            use_degrees=True,
        )
    )
    robot.connect()

    print("Opening wrist camera...")
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        robot.disconnect()
        raise RuntimeError(f"Could not open camera index {args.camera_index}.")

    # Camera warmup.
    for _ in range(10):
        _ = read_frame(cap)
        time.sleep(0.03)

    first_obs = robot.get_observation()
    first_positions = observation_positions(first_obs)

    print()
    print("Initial real robot positions:")
    for k, v in zip(JOINT_KEYS, first_positions, strict=True):
        print(f"  {k}: {float(v):.3f}")

    print()
    print("Important:")
    print("  - This checkpoint was trained with a CLOSED Phase-2 start.")
    print("  - For this first real diagnostic, place the robot in a safe local pregrasp")
    print("    with the cube visible and the gripper closed/nearly closed.")
    print("  - Press Ctrl+C at any time to stop.")
    print()
    print("Starting in 5 seconds...")
    time.sleep(5.0)

    dt_s = 1.0 / max(1e-6, args.fps)
    color_onehot = make_color_onehot(args.target_color)
    prev_positions: np.ndarray | None = None

    log_path = output_dir / ("dry_run_log.csv" if args.dry_run else "live_run_log.csv")
    log_file = log_path.open("w", newline="")
    writer_csv = csv.writer(log_file)
    writer_csv.writerow(
        ["step", "timestamp"]
        + [f"current_{k}" for k in JOINT_KEYS]
        + [f"raw_action_{k}" for k in JOINT_KEYS]
        + [f"safe_action_{k}" for k in JOINT_KEYS]
    )

    video_writer = None
    if args.save_video:
        video_path = output_dir / ("dry_run_wrist.mp4" if args.dry_run else "live_run_wrist.mp4")
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps,
            (args.camera_width, args.camera_height),
        )
        print(f"Saving wrist video to: {video_path}")

    try:
        for step in range(args.steps):
            if STOP_REQUESTED:
                break

            t0 = time.perf_counter()

            robot_obs = robot.get_observation()
            current_positions = observation_positions(robot_obs)

            frame_bgr = read_frame(cap)
            image_chw = preprocess_frame_bgr_to_obs_chw(frame_bgr)

            state = build_policy_state(
                positions=current_positions,
                prev_positions=prev_positions,
                dt_s=dt_s,
                color_onehot=color_onehot,
            )

            policy_obs = {
                "observation.images.wrist": image_chw,
                "observation.state": state,
            }

            raw_action, _ = model.predict(
                policy_obs,
                deterministic=deterministic,
            )
            raw_action = np.asarray(raw_action, dtype=np.float32).reshape(6)

            safe_action = clip_policy_action_safely(
                raw_action=raw_action,
                current_positions=current_positions,
                max_joint_step_deg=args.max_joint_step_deg,
                max_gripper_step=args.max_gripper_step,
            )

            print(
                f"step {step:02d} | "
                f"curr={np.array2string(current_positions, precision=2, suppress_small=True)} | "
                f"raw={np.array2string(raw_action, precision=2, suppress_small=True)} | "
                f"send={np.array2string(safe_action, precision=2, suppress_small=True)}"
            )

            if not args.dry_run:
                robot.send_action(action_array_to_robot_dict(safe_action))

            writer_csv.writerow(
                [step, time.time()]
                + current_positions.tolist()
                + raw_action.tolist()
                + safe_action.tolist()
            )
            log_file.flush()

            if video_writer is not None:
                overlay = overlay_debug(
                    frame_bgr,
                    step=step,
                    dry_run=args.dry_run,
                    current=current_positions,
                    safe_action=safe_action,
                )
                video_writer.write(overlay)

            prev_positions = current_positions.copy()

            elapsed = time.perf_counter() - t0
            sleep_s = max(0.0, dt_s - elapsed)
            time.sleep(sleep_s)

    finally:
        if video_writer is not None:
            video_writer.release()
        log_file.close()
        cap.release()
        robot.disconnect()

    print()
    print("Rollout finished.")
    print(f"Saved log to: {log_path}")


if __name__ == "__main__":
    main()
