from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from control.ee_motion import move_to_joint_pose
from control.ee_target_motion import (
    EEPose,
    current_ee_pose,
    make_ee_motion_stack,
    move_to_ee_pose_feedback_controlled,
)
from control.poses import PARK_POSE
from control.robot_session import RobotSession, RobotSessionConfig


def parse_camera_identifier(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test LeRobot end-effector IK motion toward a safe pre-grasp pose."
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
        "--urdf-path",
        type=Path,
        default=Path("SO-ARM100/Simulation/SO101/so101_new_calib.urdf"),
    )
    parser.add_argument(
        "--target-frame-name",
        type=str,
        default="gripper_frame_link",
    )

    parser.add_argument("--x", type=float, default=None)
    parser.add_argument("--y", type=float, default=None)
    parser.add_argument(
        "--clearance-z",
        type=float,
        default=0.20,
        help="Safe overhead EE height used for lift and horizontal translation.",
    )

    parser.add_argument(
        "--print-current-only",
        action="store_true",
        help="Only print the current LeRobot FK EE pose, then park.",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move the robot. Without this flag, only dry-run IK.",
    )

    parser.add_argument(
        "--max-translation-step-m",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--step-time-s",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--park-after",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--settle-time-s",
        type=float,
        default=2.0,
        help="Wait after the Cartesian trajectory before measuring final pose / parking.",
    )
    parser.add_argument(
        "--lift-only",
        action="store_true",
        help="Only perform the initial vertical lift to clearance height, then park.",
    )

    parser.add_argument(
        "--max-waypoint-step-m",
        type=float,
        default=0.01,
        help="Maximum Cartesian distance between feedback-controlled waypoints.",
    )
    parser.add_argument(
        "--reach-tolerance-m",
        type=float,
        default=0.005,
        help="Measured EE distance required before advancing to the next waypoint.",
    )
    parser.add_argument(
        "--waypoint-timeout-s",
        type=float,
        default=3.0,
        help="Abort if a waypoint is not reached within this time.",
    )
    parser.add_argument(
        "--command-period-s",
        type=float,
        default=0.10,
        help="How often to resend the current waypoint command while waiting.",
    )

    return parser.parse_args()


def print_ee_pose(label: str, pose: EEPose) -> None:
    print(label)
    print(f"  xyz:        ({pose.x:.4f}, {pose.y:.4f}, {pose.z:.4f})")
    print(f"  rotvec:     ({pose.wx:.4f}, {pose.wy:.4f}, {pose.wz:.4f})")
    print(f"  gripper:    {pose.gripper_pos:.4f}")


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
        motion_stack = make_ee_motion_stack(
            robot=robot,
            urdf_path=args.urdf_path,
            target_frame_name=args.target_frame_name,
            safety_max_ee_step_m=0.10,
        )

        current_pose = current_ee_pose(
            robot=robot,
            motion_stack=motion_stack,
        )
        print_ee_pose("\nCurrent EE pose from LeRobot FK:", current_pose)

        if args.print_current_only:
            if args.park_after:
                print("\nMoving to park pose...")
                move_to_joint_pose(robot, PARK_POSE)
                print("Reached park pose.")
            return

        # Stage 1: lift vertically from the current pose to a safe clearance height.
        lift_pose = EEPose(
            x=current_pose.x,
            y=current_pose.y,
            z=args.clearance_z,
            wx=current_pose.wx,
            wy=current_pose.wy,
            wz=current_pose.wz,
            gripper_pos=current_pose.gripper_pos,
        )

        print_ee_pose("\nStage 1 target: vertical lift to clearance height", lift_pose)

        print()
        move_to_ee_pose_feedback_controlled(
            robot=robot,
            motion_stack=motion_stack,
            target_pose=lift_pose,
            max_waypoint_step_m=args.max_waypoint_step_m,
            reach_tolerance_m=args.reach_tolerance_m,
            waypoint_timeout_s=args.waypoint_timeout_s,
            command_period_s=args.command_period_s,
            dry_run=not args.execute,
            verbose=True,
        )

        if args.lift_only:
            print("\nLift-only mode complete.")
        else:
            if args.x is None or args.y is None:
                raise ValueError(
                    "Provide --x and --y for overhead translation, "
                    "or use --lift-only."
                )

        # Stage 2: translate horizontally at the same safe clearance height.
        overhead_pose = EEPose(
            x=args.x,
            y=args.y,
            z=args.clearance_z,
            wx=current_pose.wx,
            wy=current_pose.wy,
            wz=current_pose.wz,
            gripper_pos=current_pose.gripper_pos,
        )

        print_ee_pose(
            "\nStage 2 target: horizontal move above target XY at clearance height",
            overhead_pose,
        )

        print()
        move_to_ee_pose_feedback_controlled(
            robot=robot,
            motion_stack=motion_stack,
            target_pose=overhead_pose,
            max_waypoint_step_m=args.max_waypoint_step_m,
            reach_tolerance_m=args.reach_tolerance_m,
            waypoint_timeout_s=args.waypoint_timeout_s,
            command_period_s=args.command_period_s,
            dry_run=not args.execute,
            verbose=True,
        )

        if args.execute:
            final_pose = current_ee_pose(
                robot=robot,
                motion_stack=motion_stack,
            )
            print_ee_pose("\nFinal EE pose from LeRobot FK:", final_pose)
        else:
            print("\nDry run only. No robot motion sent.")
            print("Pass --execute to perform the movement.")

        if args.park_after:
            print("\nMoving to park pose...")
            move_to_joint_pose(robot, PARK_POSE)
            print("Reached park pose.")


if __name__ == "__main__":
    main()