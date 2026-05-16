from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from control.ee_motion import extract_joint_positions, move_to_joint_pose
from control.poses import OBSERVATION_POSES, PARK_POSE
from control.robot_session import RobotSession, RobotSessionConfig


def parse_camera_identifier(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move through named observation poses and capture one image at each pose."
    )

    parser.add_argument("--robot-port", type=str, default="/dev/ttyACM1")
    parser.add_argument("--robot-id", type=str, default="my_awesome_follower_arm")

    parser.add_argument("--camera-name", type=str, default="front")
    parser.add_argument("--camera-index-or-path", type=parse_camera_identifier, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-fourcc", type=str, default="MJPG")

    parser.add_argument(
        "--poses",
        nargs="+",
        default=["center"],
        help=(
            "Observation pose names to visit in order. "
            f"Currently available: {sorted(OBSERVATION_POSES.keys())}"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "observation_scans",
    )

    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--pose-settle-s", type=float, default=0.7)
    parser.add_argument("--max-step-deg", type=float, default=2.0)
    parser.add_argument("--step-time-s", type=float, default=0.04)

    return parser.parse_args()


def extract_camera_image(observation: dict[str, Any], camera_name: str) -> np.ndarray:
    candidate_keys = [
        camera_name,
        f"observation.images.{camera_name}",
    ]

    for key in candidate_keys:
        if key in observation:
            image = np.asarray(observation[key])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"Camera image under key {key!r} has unexpected shape {image.shape}."
                )
            return image

    raise KeyError(
        f"Could not find camera image for {camera_name!r}. "
        f"Available observation keys: {list(observation.keys())}"
    )


def save_rgb_image(image: np.ndarray, path: Path) -> None:
    Image.fromarray(image.astype(np.uint8, copy=False)).save(path)


def main() -> None:
    args = parse_args()

    unknown_poses = [name for name in args.poses if name not in OBSERVATION_POSES]
    if unknown_poses:
        raise ValueError(
            f"Unknown pose(s): {unknown_poses}. "
            f"Available poses: {sorted(OBSERVATION_POSES.keys())}"
        )

    scan_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scan_dir = args.output_dir / f"scan_{scan_stamp}"
    scan_dir.mkdir(parents=True, exist_ok=False)

    config = RobotSessionConfig(
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        camera_name=args.camera_name,
        camera_index_or_path=args.camera_index_or_path,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        camera_fourcc=args.camera_fourcc,
    )

    metadata: dict[str, Any] = {
        "scan_id": scan_stamp,
        "pose_order": args.poses,
        "captures": [],
    }

    print("Connecting to robot and camera...")
    with RobotSession(config) as robot:
        time.sleep(args.warmup_s)

        for pose_name in args.poses:
            print(f"\nMoving to observation pose: {pose_name}")
            target_pose = OBSERVATION_POSES[pose_name]

            final_obs = move_to_joint_pose(
                robot,
                target_pose,
                max_step_deg=args.max_step_deg,
                step_time_s=args.step_time_s,
                settle_time_s=args.pose_settle_s,
            )

            image = extract_camera_image(final_obs, args.camera_name)
            joints = extract_joint_positions(final_obs)

            image_path = scan_dir / f"{pose_name}.png"
            save_rgb_image(image, image_path)

            metadata["captures"].append(
                {
                    "pose_name": pose_name,
                    "image_path": str(image_path),
                    "image_shape": list(image.shape),
                    "joint_positions": joints,
                }
            )

            print(f"Saved {pose_name} image: {image_path}")

        print("\nMoving to park pose before disconnecting...")
        move_to_joint_pose(
            robot,
            PARK_POSE,
            max_step_deg=args.max_step_deg,
            step_time_s=args.step_time_s,
            settle_time_s=args.pose_settle_s,
        )
        print("Reached park pose.")

    metadata_path = scan_dir / "scan_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print("\nObservation scan complete.")
    print(f"Scan folder: {scan_dir}")
    print(f"Metadata:    {metadata_path}")


if __name__ == "__main__":
    main()