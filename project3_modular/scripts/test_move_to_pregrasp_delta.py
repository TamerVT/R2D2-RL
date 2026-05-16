from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from control.ee_delta_motion import (
    current_ee_pose,
    make_ee_delta_motion_stack,
    move_toward_xyz,
    move_toward_xy,
)
from control.ee_motion import move_to_joint_pose
from control.poses import PARK_POSE
from control.robot_session import RobotSession, RobotSessionConfig


def parse_camera_identifier(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test LeRobot-native EE-delta motion."
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
        "--urdf-path",
        type=Path,
        default=Path("project3_modular/assets/SO101/so101_new_calib.urdf"),
    )
    parser.add_argument(
        "--target-frame-name",
        type=str,
        default="gripper_frame_link",
    )

    # For first testing, default is lift-only.
    parser.add_argument(
        "--lift-dz",
        type=float,
        default=0.01,
        help="Lift current EE pose by this many meters.",
    )

    parser.add_argument(
        "--x",
        type=float,
        default=None,
        help="Optional URDF-frame target x after the lift stage.",
    )
    parser.add_argument(
        "--y",
        type=float,
        default=None,
        help="Optional URDF-frame target y after the lift stage.",
    )

    parser.add_argument(
        "--control-hz",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--tolerance-m",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--proportional-gain",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run processors but do not send motor commands.",
    )
    parser.add_argument(
        "--park-after",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--min-safe-z",
        type=float,
        default=None,
        help=(
            "Optional abort threshold during XY motion. "
            "If the measured EE z falls below this value, stop the XY move."
        ),
    )
    parser.add_argument(
        "--z-hold-gain",
        type=float,
        default=0.25,
        help="How strongly Stage 2 tries to keep its starting EE z height.",
    )
    parser.add_argument(
        "--max-abs-z-command",
        type=float,
        default=0.35,
        help="Maximum normalized z delta command during XY motion.",
    )

    return parser.parse_args()


def print_pose(label: str, pose) -> None:
    print(label)
    print(f"  xyz:     ({pose.x:.4f}, {pose.y:.4f}, {pose.z:.4f})")
    print(f"  rotvec:  ({pose.wx:.4f}, {pose.wy:.4f}, {pose.wz:.4f})")
    print(f"  gripper: {pose.gripper_pos:.4f}")


def run_stage(
    *,
    name: str,
    robot,
    motion_stack,
    target_xyz: np.ndarray,
    args: argparse.Namespace,
) -> None:
    print(f"\n{name}")
    print(
        f"  target xyz = "
        f"({target_xyz[0]:.4f}, {target_xyz[1]:.4f}, {target_xyz[2]:.4f})"
    )

    final_pose, final_distance_m, reached = move_toward_xyz(
        robot=robot,
        motion_stack=motion_stack,
        target_xyz=target_xyz,
        tolerance_m=args.tolerance_m,
        timeout_s=args.timeout_s,
        control_hz=args.control_hz,
        proportional_gain=args.proportional_gain,
        dry_run=args.dry_run,
        verbose=True,
    )

    print_pose("  final measured pose:", final_pose)
    print(f"  final distance = {final_distance_m * 100:.2f} cm")
    print(f"  reached = {reached}")


def main() -> None:
    args = parse_args()

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

    print("Connecting to robot...")
    with RobotSession(robot_config) as robot:
        motion_stack = make_ee_delta_motion_stack(
            robot=robot,
            urdf_path=args.urdf_path,
            target_frame_name=args.target_frame_name,
        )

        start_pose = current_ee_pose(robot=robot, motion_stack=motion_stack)
        print_pose("\nCurrent EE pose:", start_pose)

        # Stage 1: safe small vertical lift.
        lift_target_xyz = np.array(
            [
                start_pose.x,
                start_pose.y,
                start_pose.z + args.lift_dz,
            ],
            dtype=np.float64,
        )
        run_stage(
            name="Stage 1: small upward EE-delta lift",
            robot=robot,
            motion_stack=motion_stack,
            target_xyz=lift_target_xyz,
            args=args,
        )

        # Stage 2: optional horizontal translation at the lifted z.
        if args.x is not None or args.y is not None:
            if args.x is None or args.y is None:
                raise ValueError("Provide both --x and --y, or neither.")

            print("\nStage 2: XY-only move toward target")
            target_xy = np.array([args.x, args.y], dtype=np.float64)
            print(f"  target xy = ({target_xy[0]:.4f}, {target_xy[1]:.4f})")
            stage2_start_pose = current_ee_pose(robot=robot, motion_stack=motion_stack)
            hold_z = stage2_start_pose.z
            print(f"  softly holding z near {hold_z:.4f} m")

            final_pose, final_xy_distance_m, reached = move_toward_xy(
                robot=robot,
                motion_stack=motion_stack,
                target_xy=target_xy,
                tolerance_m=args.tolerance_m,
                timeout_s=args.timeout_s,
                control_hz=args.control_hz,
                proportional_gain=args.proportional_gain,
                hold_z=hold_z,
                z_hold_gain=args.z_hold_gain,
                max_abs_z_command=args.max_abs_z_command,
                min_safe_z=args.min_safe_z,
                dry_run=args.dry_run,
                verbose=True,
            )

            print_pose("  final measured pose:", final_pose)
            print(f"  final XY distance = {final_xy_distance_m * 100:.2f} cm")
            print(f"  reached = {reached}")

        if args.park_after:
            print("\nMoving to park pose...")
            move_to_joint_pose(robot, PARK_POSE)
            print("Reached park pose.")


if __name__ == "__main__":
    main()
