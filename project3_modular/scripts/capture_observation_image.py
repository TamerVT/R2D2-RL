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

# Allow imports like `from control.robot_session import ...`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from control.robot_session import RobotSession, RobotSessionConfig


def parse_camera_identifier(value: str) -> int | str:
    """
    Allow either:
      --camera-index-or-path 0
    or:
      --camera-index-or-path /dev/video0
    """
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one wrist-camera observation image from the SO101 follower."
    )

    parser.add_argument("--robot-port", type=str, default="/dev/ttyACM2")
    parser.add_argument("--robot-id", type=str, default="my_awesome_follower_arm")

    parser.add_argument("--camera-name", type=str, default="front")
    parser.add_argument("--camera-index-or-path", type=parse_camera_identifier, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-fourcc", type=str, default="MJPG")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "observations",
        help="Directory in which to save the image and metadata JSON.",
    )
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=2.0,
        help="Wait time after connecting before reading the observation.",
    )

    return parser.parse_args()


def extract_camera_image(observation: dict[str, Any], camera_name: str) -> np.ndarray:
    """
    LeRobot robot observations usually expose raw camera frames under the camera key,
    e.g. 'front'. This function also accepts the dataset-style key as a fallback.
    """
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


def extract_robot_state(observation: dict[str, Any]) -> dict[str, float]:
    """
    Extract scalar motor-position entries such as:
      shoulder_pan.pos, shoulder_lift.pos, ..., gripper.pos
    """
    state: dict[str, float] = {}

    for key, value in observation.items():
        if not key.endswith(".pos"):
            continue

        try:
            state[key] = float(value)
        except (TypeError, ValueError):
            pass

    return state


def save_rgb_image(image: np.ndarray, path: Path) -> None:
    """
    LeRobot's OpenCV camera observations are configured as RGB in your setup,
    so Pillow can save them directly.
    """
    image_uint8 = image.astype(np.uint8, copy=False)
    Image.fromarray(image_uint8).save(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = args.output_dir / f"observation_{timestamp}.png"
    metadata_path = args.output_dir / f"observation_{timestamp}.json"

    print("Connecting to robot and camera...")
    with RobotSession(config) as robot:
        time.sleep(args.warmup_s)

        observation = robot.get_observation()

        print("\nObservation keys:")
        for key in observation.keys():
            print(f"  - {key}")

        image = extract_camera_image(observation, args.camera_name)
        robot_state = extract_robot_state(observation)

        save_rgb_image(image, image_path)

        metadata = {
            "image_path": str(image_path),
            "image_shape": list(image.shape),
            "camera_name": args.camera_name,
            "robot_state": robot_state,
            "observation_keys": list(observation.keys()),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print("\nSaved:")
    print(f"  image:    {image_path}")
    print(f"  metadata: {metadata_path}")
    print(f"  shape:    {tuple(image.shape)}")

    if robot_state:
        print("\nRobot state:")
        for key, value in robot_state.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()