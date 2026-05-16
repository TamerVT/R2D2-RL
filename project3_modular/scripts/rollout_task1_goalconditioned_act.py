from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Allow imports from project3_modular/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from control.robot_session import RobotSession, RobotSessionConfig

from lerobot.datasets import LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy
from lerobot.policies.utils import build_inference_frame, make_robot_action


def parse_camera_identifier(value: str) -> int | str:
    """Allow either `0` or `/dev/video0`."""
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Roll out a goal-conditioned ACT Task 1 policy on the SO101 follower, "
            "injecting the specified bowl xyz as observation.environment_state."
        )
    )

    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Path to the ACT pretrained_model checkpoint directory.",
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        default="frainer/Task1_goalconditioned_envstate",
        help="Dataset repo id used to load feature metadata and normalization stats.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path.home()
        / ".cache"
        / "huggingface"
        / "lerobot"
        / "frainer"
        / "Task1_goalconditioned_envstate",
        help="Local root of the augmented goal-conditioned dataset.",
    )

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--robot-port", type=str, default="/dev/ttyACM1")
    parser.add_argument("--robot-id", type=str, default="my_awesome_follower_arm")

    parser.add_argument("--camera-name", type=str, default="front")
    parser.add_argument("--camera-index-or-path", type=parse_camera_identifier, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-fourcc", type=str, default="MJPG")

    parser.add_argument("--bowl-x", type=float, required=True)
    parser.add_argument("--bowl-y", type=float, required=True)
    parser.add_argument("--bowl-z", type=float, default=0.0)

    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Rollout duration in seconds.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Control loop target frequency.",
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=2.0,
        help="Wait time after robot/camera connection before rollout starts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run policy inference and print actions, but do not send commands to the robot.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=30,
        help="Print one status line every N control steps.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.duration <= 0:
        raise ValueError("--duration must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")

    device = torch.device(args.device)

    print("Loading ACT policy...")
    policy = ACTPolicy.from_pretrained(args.policy_path)
    policy.to(device)
    policy.eval()
    policy.reset()

    print("Loading dataset metadata/stats...")
    dataset_metadata = LeRobotDatasetMetadata(
        args.dataset_repo_id,
        root=args.dataset_root,
    )

    required_feature = "observation.environment_state"
    if required_feature not in dataset_metadata.features:
        raise RuntimeError(
            f"Dataset metadata does not contain {required_feature!r}. "
            "Use the augmented goal-conditioned dataset."
        )

    feature_spec = dataset_metadata.features[required_feature]
    expected_names = list(feature_spec.get("names") or [])
    if expected_names != ["bowl_x", "bowl_y", "bowl_z"]:
        raise RuntimeError(
            f"Unexpected {required_feature} names: {expected_names}. "
            "Expected ['bowl_x', 'bowl_y', 'bowl_z']."
        )

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        dataset_stats=dataset_metadata.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    robot_config = RobotSessionConfig(
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        camera_name=args.camera_name,
        camera_index_or_path=args.camera_index_or_path,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        camera_fourcc=args.camera_fourcc,
    )

    n_steps = max(1, int(round(args.duration * args.fps)))
    target_dt = 1.0 / args.fps

    print("\nGoal-conditioned ACT rollout")
    print(f"  policy:       {args.policy_path}")
    print(f"  dataset root: {args.dataset_root}")
    print(f"  bowl xyz:     [{args.bowl_x:.4f}, {args.bowl_y:.4f}, {args.bowl_z:.4f}]")
    print(f"  duration:     {args.duration:.1f}s ({n_steps} steps @ {args.fps:.1f} Hz)")
    print(f"  dry run:      {args.dry_run}")

    print("\nConnecting to robot and camera...")
    with RobotSession(robot_config) as robot:
        time.sleep(args.warmup_s)
        policy.reset()

        print("Starting rollout...")
        rollout_start = time.perf_counter()

        with torch.inference_mode():
            for step_idx in range(n_steps):
                step_start = time.perf_counter()

                raw_obs = robot.get_observation()

                # build_inference_frame(...) will create
                # observation.environment_state by looking up these raw feature names.
                raw_obs["bowl_x"] = float(args.bowl_x)
                raw_obs["bowl_y"] = float(args.bowl_y)
                raw_obs["bowl_z"] = float(args.bowl_z)

                obs_frame = build_inference_frame(
                    observation=raw_obs,
                    ds_features=dataset_metadata.features,
                    device=device,
                )
                policy_input = preprocess(obs_frame)

                action = policy.select_action(policy_input)
                action = postprocess(action)
                robot_action = make_robot_action(action, dataset_metadata.features)

                if not args.dry_run:
                    robot.send_action(robot_action)

                if args.print_every > 0 and (
                    step_idx % args.print_every == 0 or step_idx == n_steps - 1
                ):
                    elapsed = time.perf_counter() - rollout_start
                    gripper = robot_action.get("gripper.pos", None)
                    gripper_str = f" gripper={gripper:.3f}" if gripper is not None else ""
                    print(
                        f"step {step_idx + 1:4d}/{n_steps} "
                        f"elapsed={elapsed:6.2f}s{gripper_str}"
                    )

                step_elapsed = time.perf_counter() - step_start
                sleep_s = target_dt - step_elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)

    print("Rollout finished.")


if __name__ == "__main__":
    main()
