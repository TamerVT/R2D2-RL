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

from control.ee_motion import move_to_joint_pose
from control.poses import OBSERVATION_POSES, JointPose, PARK_POSE
from control.robot_session import RobotSession, RobotSessionConfig
from models.pregrasp_joint_regressor import load_pregrasp_checkpoint
from perception.cube_detector import CubeDetectorConfig
from perception.multiview_cube_localizer import (
    WorkspaceBounds,
    localize_cube_from_scan,
    localization_result_to_jsonable,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "pregrasp_regressor" / "best_pregrasp_mlp.pt"
)
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "data" / "calibration"
DEFAULT_SCAN_OUTPUT_DIR = PROJECT_ROOT / "data" / "pregrasp_regressor_test_scans"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "pregrasp_regressor_test_outputs"


def parse_camera_identifier(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end pregrasp regressor test: "
            "scan scene, localize cube, predict pregrasp joint pose, "
            "optionally execute it."
        )
    )

    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)

    # Optional: reuse an existing scan instead of capturing a live one.
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=None,
        help="Optional existing scan directory. If omitted, a live scan is captured.",
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

    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--scan-output-dir", type=Path, default=DEFAULT_SCAN_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    # Scan motion parameters
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--pose-settle-s", type=float, default=0.7)
    parser.add_argument("--max-step-deg", type=float, default=2.0)
    parser.add_argument("--step-time-s", type=float, default=0.04)

    # Localization parameters
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

    # Execution
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move the follower arm to the predicted pregrasp joint pose.",
    )
    parser.add_argument(
        "--allow-unreliable-localization",
        action="store_true",
        help="Allow prediction/execution even if localization is not reliable.",
    )
    parser.add_argument(
        "--park-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return to PARK_POSE after executing the predicted pregrasp move.",
    )

    parser.add_argument(
        "--hold-at-target-s",
        type=float,
        default=2.0,
        help="Wait this many seconds at the predicted pregrasp pose before parking.",
    )

    return parser.parse_args()


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


def save_rgb_image(image_rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(image_rgb.astype(np.uint8, copy=False)).save(path)


def capture_live_scan(
    *,
    args: argparse.Namespace,
    robot_config: RobotSessionConfig,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scan_dir = args.scan_output_dir / f"scan_{timestamp}"
    scan_dir.mkdir(parents=True, exist_ok=False)

    metadata: dict[str, Any] = {
        "timestamp": timestamp,
        "pose_order": args.poses,
        "captures": [],
    }

    print("Connecting to robot and camera for scan...")
    with RobotSession(robot_config) as robot:
        time.sleep(args.warmup_s)

        for pose_name in args.poses:
            if pose_name not in OBSERVATION_POSES:
                raise ValueError(
                    f"Unknown observation pose {pose_name!r}. "
                    f"Available: {sorted(OBSERVATION_POSES.keys())}"
                )

            print(f"\nMoving to observation pose: {pose_name}")
            observation = move_to_joint_pose(
                robot,
                OBSERVATION_POSES[pose_name],
                max_step_deg=args.max_step_deg,
                step_time_s=args.step_time_s,
                settle_time_s=args.pose_settle_s,
            )

            image_rgb = extract_camera_image(observation, args.camera_name)
            image_path = scan_dir / f"{pose_name}.png"
            save_rgb_image(image_rgb, image_path)

            metadata["captures"].append(
                {
                    "pose_name": pose_name,
                    "image_path": str(image_path),
                    "image_shape": list(image_rgb.shape),
                }
            )

            print(f"Saved {pose_name} image: {image_path}")

        print("\nMoving to park pose before localization...")
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

    cube_xy = result["cube_xy"]
    best_cluster = result["best_cluster"]

    if cube_xy is None or best_cluster is None:
        print("\nNo cube localization found.")
        return result

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

    return result


def make_joint_pose(pred: np.ndarray) -> JointPose:
    return JointPose(
        shoulder_pan=float(pred[0]),
        shoulder_lift=float(pred[1]),
        elbow_flex=float(pred[2]),
        wrist_flex=float(pred[3]),
        wrist_roll=float(pred[4]),
        gripper=float(pred[5]),
    )


def print_prediction(pred: np.ndarray) -> None:
    names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    print("\nPredicted pregrasp joints:")
    for name, value in zip(names, pred):
        print(f"  {name:16s} {value: .4f}")


def predict_pregrasp(
    *,
    args: argparse.Namespace,
    localization_result: dict[str, Any],
) -> tuple[np.ndarray, JointPose]:
    cube_xy = localization_result["cube_xy"]
    if cube_xy is None:
        raise RuntimeError("Cannot predict pregrasp pose: cube localization is missing.")

    cube_xyz = np.array(
        [float(cube_xy[0]), float(cube_xy[1]), 0.0],
        dtype=np.float32,
    )

    regressor = load_pregrasp_checkpoint(args.checkpoint, device="cpu")
    pred = regressor.predict(cube_xyz)
    joint_pose = make_joint_pose(pred)

    print(
        f"\nRegressor input cube xyz: "
        f"({cube_xyz[0]:.4f}, {cube_xyz[1]:.4f}, {cube_xyz[2]:.4f})"
    )
    print_prediction(pred)

    return pred, joint_pose


def execute_pregrasp_motion(
    *,
    args: argparse.Namespace,
    robot_config: RobotSessionConfig,
    joint_pose: JointPose,
) -> None:
    print("\nConnecting to robot for pregrasp execution...")
    with RobotSession(robot_config) as robot:
        print("Moving to predicted pregrasp joint pose...")
        move_to_joint_pose(
            robot,
            joint_pose,
            max_step_deg=args.max_step_deg,
            step_time_s=args.step_time_s,
            settle_time_s=args.pose_settle_s,
        )
        print("Reached predicted pregrasp pose.")

        if args.hold_at_target_s > 0:
            print(f"Holding at target pose for {args.hold_at_target_s:.1f}s...")
            time.sleep(args.hold_at_target_s)

        if args.park_after:
            print("Moving to park pose...")
            move_to_joint_pose(
                robot,
                PARK_POSE,
                max_step_deg=args.max_step_deg,
                step_time_s=args.step_time_s,
                settle_time_s=args.pose_settle_s,
            )
            print("Reached park pose.")


def save_run_summary(
    *,
    args: argparse.Namespace,
    scan_dir: Path,
    localization_result: dict[str, Any],
    prediction: np.ndarray | None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    summary = {
        "scan_dir": str(scan_dir),
        "checkpoint": str(args.checkpoint),
        "execute": bool(args.execute),
        "localization": localization_result_to_jsonable(localization_result),
        "predicted_pregrasp_joints": None
        if prediction is None
        else prediction.tolist(),
    }

    summary_path = run_dir / "pregrasp_regressor_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Regressor checkpoint not found: {args.checkpoint}\n"
            "Train the pregrasp regressor first."
        )

    args.scan_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    robot_config = make_robot_config(args)

    if args.scan_dir is None:
        scan_dir = capture_live_scan(
            args=args,
            robot_config=robot_config,
        )
    else:
        scan_dir = args.scan_dir
        print(f"Using existing scan directory: {scan_dir}")

    localization_result = localize_cube(
        args=args,
        scan_dir=scan_dir,
    )

    cube_xy = localization_result["cube_xy"]
    is_reliable = localization_result["is_reliable"]

    prediction: np.ndarray | None = None
    joint_pose: JointPose | None = None

    if cube_xy is None:
        print("\nStopping: no cube was localized.")
    elif not is_reliable and not args.allow_unreliable_localization:
        print(
            "\nStopping: localization was not reliable. "
            "Pass --allow-unreliable-localization to override."
        )
    else:
        prediction, joint_pose = predict_pregrasp(
            args=args,
            localization_result=localization_result,
        )

        if args.execute:
            assert joint_pose is not None
            execute_pregrasp_motion(
                args=args,
                robot_config=robot_config,
                joint_pose=joint_pose,
            )
        else:
            print("\nDry run only. Pass --execute to move the robot.")

    summary_path = save_run_summary(
        args=args,
        scan_dir=scan_dir,
        localization_result=localization_result,
        prediction=prediction,
    )
    print(f"\nSaved run summary: {summary_path}")


if __name__ == "__main__":
    main()