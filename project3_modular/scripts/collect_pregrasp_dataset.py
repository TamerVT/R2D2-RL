from __future__ import annotations

import argparse
import json
import os
import select
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from control.ee_motion import move_to_joint_pose
from control.poses import OBSERVATION_POSES, PARK_POSE
from control.robot_session import RobotSession, RobotSessionConfig
from perception.cube_detector import CubeDetectorConfig
from perception.multiview_cube_localizer import (
    WorkspaceBounds,
    localize_cube_from_scan,
    localization_result_to_jsonable,
)


DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "data" / "calibration"
DEFAULT_SCAN_DIR = PROJECT_ROOT / "data" / "pregrasp_collection_scans"
DEFAULT_DATASET_JSON = PROJECT_ROOT / "data" / "pregrasp_dataset" / "pregrasp_examples.json"

JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def parse_camera_identifier(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive collection of cube-position → pregrasp-joint examples."
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
        default=["center", "left", "right"],
    )

    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
    )
    parser.add_argument(
        "--scan-output-dir",
        type=Path,
        default=DEFAULT_SCAN_DIR,
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=DEFAULT_DATASET_JSON,
    )

    parser.add_argument(
        "--cube-z",
        type=float,
        default=0.0,
        help="Cube table z coordinate stored in the dataset.",
    )

    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--pose-settle-s", type=float, default=0.7)
    parser.add_argument("--max-step-deg", type=float, default=2.0)
    parser.add_argument("--step-time-s", type=float, default=0.04)

    parser.add_argument("--target-color", type=str, default=None)

    parser.add_argument("--min-area-px", type=float, default=80.0)
    parser.add_argument("--max-area-px", type=float, default=12000.0)
    parser.add_argument("--min-aspect-ratio", type=float, default=0.35)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.8)

    parser.add_argument("--workspace-x-min", type=float, default=-0.30)
    parser.add_argument("--workspace-x-max", type=float, default=0.30)
    parser.add_argument("--workspace-y-min", type=float, default=0.05)
    parser.add_argument("--workspace-y-max", type=float, default=0.65)

    parser.add_argument("--cluster-radius-m", type=float, default=0.02)
    parser.add_argument("--reliable-min-views", type=int, default=2)

    return parser.parse_args()


def wait_for_key() -> str:
    """
    Wait for one keypress in the terminal without requiring Enter.
    Falls back to input() if stdin is not a TTY.
    """
    if not sys.stdin.isatty():
        return input().strip()[:1]

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def save_rgb_image(image_rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(image_rgb.astype(np.uint8, copy=False)).save(path)


def extract_camera_image(
    observation: dict[str, Any],
    camera_name: str,
) -> np.ndarray:
    for key in [camera_name, f"observation.images.{camera_name}"]:
        if key in observation:
            image = np.asarray(observation[key])
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"Unexpected image shape under {key!r}: {image.shape}")
            return image

    raise KeyError(
        f"Could not find camera image for {camera_name!r}. "
        f"Available keys: {list(observation.keys())}"
    )


def make_robot_config(args: argparse.Namespace) -> RobotSessionConfig:
    return RobotSessionConfig(
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        camera_name=args.camera_name,
        camera_index_or_path=args.camera_index_or_path,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        camera_fourcc=args.camera_fourcc,
    )


def capture_scan(
    *,
    args: argparse.Namespace,
    robot_config: RobotSessionConfig,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    scan_dir = args.scan_output_dir / f"scan_{timestamp}"
    scan_dir.mkdir(parents=True, exist_ok=False)

    metadata: dict[str, Any] = {
        "timestamp": timestamp,
        "pose_order": args.poses,
        "captures": [],
    }

    print("\nConnecting for scan...")
    with RobotSession(robot_config) as robot:
        time.sleep(args.warmup_s)

        for pose_name in args.poses:
            if pose_name not in OBSERVATION_POSES:
                raise ValueError(
                    f"Unknown observation pose {pose_name!r}. "
                    f"Available: {sorted(OBSERVATION_POSES.keys())}"
                )

            print(f"  Moving to {pose_name}...")
            observation = move_to_joint_pose(
                robot,
                OBSERVATION_POSES[pose_name],
                max_step_deg=args.max_step_deg,
                step_time_s=args.step_time_s,
                settle_time_s=args.pose_settle_s,
            )

            image = extract_camera_image(observation, args.camera_name)
            image_path = scan_dir / f"{pose_name}.png"
            save_rgb_image(image, image_path)

            metadata["captures"].append(
                {
                    "pose_name": pose_name,
                    "image_path": str(image_path),
                    "image_shape": list(image.shape),
                }
            )

        print("  Moving to park pose...")
        move_to_joint_pose(
            robot,
            PARK_POSE,
            max_step_deg=args.max_step_deg,
            step_time_s=args.step_time_s,
            settle_time_s=args.pose_settle_s,
        )

    metadata_path = scan_dir / "scan_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Scan saved: {scan_dir}")
    return scan_dir


def localize_cube(
    *,
    args: argparse.Namespace,
    scan_dir: Path,
) -> dict[str, Any]:
    detector_config = CubeDetectorConfig(
        min_area_px=args.min_area_px,
        max_area_px=args.max_area_px,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
    )

    workspace_bounds = WorkspaceBounds(
        x_min=args.workspace_x_min,
        x_max=args.workspace_x_max,
        y_min=args.workspace_y_min,
        y_max=args.workspace_y_max,
    )

    result = localize_cube_from_scan(
        scan_dir=scan_dir,
        calibration_dir=args.calibration_dir,
        poses=args.poses,
        target_color=args.target_color,
        detector_config=detector_config,
        workspace_bounds=workspace_bounds,
        cluster_radius_m=args.cluster_radius_m,
        reliable_min_views=args.reliable_min_views,
    )

    cube_xy = result["cube_xy"]
    best_cluster = result["best_cluster"]

    if cube_xy is None or best_cluster is None:
        print("\nNo cube localization found.")
        return result

    print("\nCube localization:")
    print(f"  x = {cube_xy[0]:.4f} m")
    print(f"  y = {cube_xy[1]:.4f} m")
    print(f"  reliable = {result['is_reliable']}")
    print(f"  views = {best_cluster['distinct_views']}")
    print(f"  disagreement = {best_cluster['max_residual_m'] * 100:.2f} cm")

    return result


def load_or_create_dataset(path: Path) -> dict[str, Any]:
    if path.exists():
        data = json.loads(path.read_text())
        if "examples" not in data:
            raise ValueError(f"Existing dataset JSON has no 'examples' field: {path}")
        return data

    return {
        "metadata": {
            "description": "Cube/table position to manually chosen pregrasp joint pose.",
            "coordinate_convention": {
                "cube_xyz": "table coordinates in meters, as produced by multiview localization",
                "pregrasp_joints": "SO101 follower joint positions in LeRobot degree convention",
            },
            "joint_keys": JOINT_KEYS,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "examples": [],
    }


def save_dataset_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp_path, path)


def read_current_joint_pose_and_park(
    *,
    args: argparse.Namespace,
    robot_config: RobotSessionConfig,
) -> tuple[list[float], dict[str, float]]:
    print("\nConnecting to read pregrasp joints...")
    print("Keep hands clear: after saving, the arm will move to PARK_POSE.")

    with RobotSession(robot_config) as robot:
        observation = robot.get_observation()

        joint_dict: dict[str, float] = {}
        for key in JOINT_KEYS:
            if key not in observation:
                raise KeyError(
                    f"Missing joint key {key!r}. "
                    f"Available keys: {list(observation.keys())}"
                )
            joint_dict[key] = float(observation[key])

        joint_list = [joint_dict[key] for key in JOINT_KEYS]

        print("Captured pregrasp joints:")
        for key in JOINT_KEYS:
            print(f"  {key:20s} {joint_dict[key]: .4f}")

        print("Moving to park pose...")
        move_to_joint_pose(
            robot,
            PARK_POSE,
            max_step_deg=args.max_step_deg,
            step_time_s=args.step_time_s,
            settle_time_s=args.pose_settle_s,
        )

    print("Reached park pose.")
    return joint_list, joint_dict


def append_example(
    *,
    dataset: dict[str, Any],
    args: argparse.Namespace,
    localization_result: dict[str, Any],
    scan_dir: Path,
    joint_list: list[float],
    joint_dict: dict[str, float],
) -> dict[str, Any]:
    cube_xy = localization_result["cube_xy"]
    if cube_xy is None:
        raise RuntimeError("Cannot save example without a cube localization.")

    example_index = len(dataset["examples"])

    example = {
        "example_index": example_index,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cube_xyz": [
            float(cube_xy[0]),
            float(cube_xy[1]),
            float(args.cube_z),
        ],
        "pregrasp_joints": joint_list,
        "pregrasp_joint_dict": joint_dict,
        "scan_dir": str(scan_dir),
        "localization": localization_result_to_jsonable(localization_result),
    }

    dataset["examples"].append(example)
    return example


def print_idle_prompt(num_examples: int) -> None:
    print("\n" + "=" * 72)
    print(f"Saved examples: {num_examples}")
    print("Place the cube at a new position.")
    print("Controls:")
    print("  SPACE  localize cube and start a new example")
    print("  Q      quit")
    print("=" * 72)


def print_after_localization_prompt() -> None:
    print("\nNow manually move the follower arm to a good pregrasp position.")
    print("Then press:")
    print("  S      save cube + current joint pose, then park arm")
    print("  R      redo localization")
    print("  K      skip this sample")
    print("  Q      quit")


def main() -> None:
    args = parse_args()

    args.scan_output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_or_create_dataset(args.dataset_json)
    save_dataset_atomic(args.dataset_json, dataset)

    robot_config = make_robot_config(args)

    print("Interactive pregrasp dataset collector")
    print(f"Dataset JSON: {args.dataset_json}")

    while True:
        print_idle_prompt(len(dataset["examples"]))
        key = wait_for_key().lower()

        if key == "q":
            print("\nExiting.")
            break

        if key != " ":
            continue

        while True:
            scan_dir = capture_scan(
                args=args,
                robot_config=robot_config,
            )
            localization_result = localize_cube(
                args=args,
                scan_dir=scan_dir,
            )

            if localization_result["cube_xy"] is None:
                print("\nLocalization failed.")
                print("Press R to retry, K to skip, or Q to quit.")
            else:
                print_after_localization_prompt()

            key = wait_for_key().lower()

            if key == "q":
                print("\nExiting.")
                return

            if key == "r":
                print("\nRedoing localization...")
                continue

            if key == "k":
                print("\nSkipping this sample.")
                break

            if key == "s":
                if localization_result["cube_xy"] is None:
                    print("\nCannot save: no cube localization available.")
                    continue

                joint_list, joint_dict = read_current_joint_pose_and_park(
                    args=args,
                    robot_config=robot_config,
                )

                example = append_example(
                    dataset=dataset,
                    args=args,
                    localization_result=localization_result,
                    scan_dir=scan_dir,
                    joint_list=joint_list,
                    joint_dict=joint_dict,
                )
                save_dataset_atomic(args.dataset_json, dataset)

                cube_xyz = example["cube_xyz"]
                print("\nSaved example:")
                print(f"  index = {example['example_index']}")
                print(
                    f"  cube_xyz = "
                    f"({cube_xyz[0]:.4f}, {cube_xyz[1]:.4f}, {cube_xyz[2]:.4f})"
                )
                print(f"  total examples = {len(dataset['examples'])}")
                break


if __name__ == "__main__":
    main()
