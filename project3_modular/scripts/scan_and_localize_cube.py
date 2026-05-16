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
from perception.cube_detector import CubeDetectorConfig
from perception.multiview_cube_localizer import (
    WorkspaceBounds,
    localize_cube_from_scan,
    localization_result_to_jsonable,
)


DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "data" / "calibration"


def parse_camera_identifier(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a live multi-view wrist-camera scan and localize the cube "
            "in table / robot XY coordinates."
        )
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
        help=(
            "Observation pose names to visit in order. "
            f"Available: {sorted(OBSERVATION_POSES.keys())}"
        ),
    )

    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
    )

    parser.add_argument(
        "--scan-output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "observation_scans",
    )
    parser.add_argument(
        "--localization-output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "live_localization",
    )

    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--pose-settle-s", type=float, default=0.7)
    parser.add_argument("--max-step-deg", type=float, default=2.0)
    parser.add_argument("--step-time-s", type=float, default=0.04)

    parser.add_argument(
        "--target-color",
        type=str,
        default=None,
        help="Optional cube color filter, e.g. blue, green, red.",
    )

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


def extract_camera_image(
    observation: dict[str, Any],
    camera_name: str,
) -> np.ndarray:
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
        f"Could not find camera image for camera {camera_name!r}. "
        f"Available observation keys: {list(observation.keys())}"
    )


def save_rgb_image(image_rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(image_rgb.astype(np.uint8, copy=False)).save(path)


def capture_live_scan(
    *,
    args: argparse.Namespace,
    robot_config: RobotSessionConfig,
    scan_dir: Path,
) -> None:
    metadata: dict[str, Any] = {
        "pose_order": args.poses,
        "captures": [],
    }

    print("Connecting to robot and camera...")
    with RobotSession(robot_config) as robot:
        time.sleep(args.warmup_s)

        for pose_name in args.poses:
            print(f"\nMoving to observation pose: {pose_name}")

            target_pose = OBSERVATION_POSES[pose_name]
            observation = move_to_joint_pose(
                robot,
                target_pose,
                max_step_deg=args.max_step_deg,
                step_time_s=args.step_time_s,
                settle_time_s=args.pose_settle_s,
            )

            image_rgb = extract_camera_image(observation, args.camera_name)
            joint_positions = extract_joint_positions(observation)

            image_path = scan_dir / f"{pose_name}.png"
            save_rgb_image(image_rgb, image_path)

            metadata["captures"].append(
                {
                    "pose_name": pose_name,
                    "image_path": str(image_path),
                    "image_shape": list(image_rgb.shape),
                    "joint_positions": joint_positions,
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
    print(f"Saved scan metadata: {metadata_path}")


def print_localization_result(result: dict[str, Any]) -> None:
    cube_xy = result["cube_xy"]
    best_cluster = result["best_cluster"]

    if cube_xy is None or best_cluster is None:
        print("\nNo cube position found.")
        return

    print("\nDetected cube position:")
    print(f"  x = {cube_xy[0]:.4f} m")
    print(f"  y = {cube_xy[1]:.4f} m")
    print(f"  reliable = {result['is_reliable']}")

    print("\nBest cluster:")
    print(f"  views = {best_cluster['distinct_views']}")
    print(f"  num views = {best_cluster['num_distinct_views']}")
    print(
        f"  max disagreement = "
        f"{best_cluster['max_residual_m'] * 100:.2f} cm"
    )

    print("\nCluster members:")
    for member in best_cluster["members"]:
        xy = member["table_xy"]
        uv = member["centroid_uv"]
        print(
            f"  {member['pose_name']:>6s}: "
            f"color={member['color']:<7s} "
            f"xy=({xy[0]:.4f}, {xy[1]:.4f}) "
            f"uv=({uv[0]:.1f}, {uv[1]:.1f})"
        )


def main() -> None:
    args = parse_args()

    unknown_poses = [pose for pose in args.poses if pose not in OBSERVATION_POSES]
    if unknown_poses:
        raise ValueError(
            f"Unknown observation pose(s): {unknown_poses}. "
            f"Available poses: {sorted(OBSERVATION_POSES.keys())}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    scan_dir = args.scan_output_dir / f"scan_{timestamp}"
    scan_dir.mkdir(parents=True, exist_ok=False)

    localization_dir = args.localization_output_dir / f"localization_{timestamp}"
    localization_dir.mkdir(parents=True, exist_ok=False)

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

    capture_live_scan(
        args=args,
        robot_config=robot_config,
        scan_dir=scan_dir,
    )

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

    print("\nRunning multi-view cube localization...")
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

    print_localization_result(result)

    summary = localization_result_to_jsonable(result)
    summary["scan_dir"] = str(scan_dir)

    summary_path = localization_dir / "live_localization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nSaved localization summary: {summary_path}")
    print(f"Saved scan directory: {scan_dir}")


if __name__ == "__main__":
    main()

