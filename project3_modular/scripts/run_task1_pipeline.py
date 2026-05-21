from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from control.ee_motion import move_to_joint_pose
from control.poses import JointPose, PARK_POSE
from control.robot_session import RobotSession, RobotSessionConfig

from test_pregrasp_regressor import (
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_SCAN_OUTPUT_DIR,
    DEFAULT_OUTPUT_DIR as DEFAULT_PREGRASP_OUTPUT_DIR,
    capture_live_scan,
    execute_pregrasp_motion,
    localize_cube,
    make_robot_config as make_pregrasp_robot_config,
    predict_pregrasp,
)

from test_bowl_transfer_regressor import BowlTransferRegressor
from collect_bowl_transfer_dataset import read_joint_positions


DEFAULT_PREGRASP_CHECKPOINT = (
    PROJECT_ROOT / "data" / "pregrasp_dataset" / "pregrasp_regressor.pt"
)
DEFAULT_BOWL_TRANSFER_CHECKPOINT = (
    PROJECT_ROOT / "data" / "bowl_transfer_dataset" / "bowl_transfer_regressor.pt"
)


def parse_camera_identifier(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_xy(value: str) -> tuple[float, float]:
    """Parse '(0.10, 0.38)', '0.10,0.38', or '0.10 0.38'."""
    cleaned = (
        value.strip()
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace(",", " ")
    )
    parts = cleaned.split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected two numbers, got {value!r}. "
            "Examples: --bowl_xy '(0.10, 0.38)' or --bowl_xy 0.10,0.38"
        )
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Could not parse xy value {value!r} as two floats."
        ) from e


def parse_joint_offsets(value: str) -> tuple[float, float, float, float, float, float]:
    """Parse six comma/space separated joint offsets in degrees."""
    cleaned = (
        value.strip()
        .replace("(", "")
        .replace(")", "")
        .replace("[", "")
        .replace("]", "")
        .replace(",", " ")
    )
    parts = cleaned.split()
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            f"Expected six joint offsets, got {value!r}. "
            "Example: --pregrasp-joint-offsets '0,-5,5,0,0,0'"
        )
    try:
        return tuple(float(x) for x in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Could not parse joint offsets {value!r} as floats."
        ) from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end Task 1 pipeline: "
            "cube localization -> pregrasp regressor -> ACT grasp/lift -> "
            "bowl transfer regressor -> release."
        )
    )

    # Robot/camera.
    parser.add_argument("--robot-port", type=str, default="/dev/ttyACM2")
    parser.add_argument("--robot-id", type=str, default="my_awesome_follower_arm")

    parser.add_argument("--camera-name", type=str, default="wrist")
    parser.add_argument("--camera-index-or-path", type=parse_camera_identifier, default=1)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-fourcc", type=str, default="MJPG")

    # Target bowl input.
    parser.add_argument(
        "--bowl_xy",
        type=parse_xy,
        required=True,
        help="Bowl xy in robot frame, e.g. --bowl_xy '(0.10, 0.38)'",
    )
    parser.add_argument("--bowl-z", type=float, default=0.0)

    # Pregrasp/localization.
    parser.add_argument("--pregrasp-checkpoint", type=Path, default=DEFAULT_PREGRASP_CHECKPOINT)
    parser.add_argument("--target-color", type=str, default=None)
    parser.add_argument("--allow-unreliable-localization", action="store_true")

    parser.add_argument("--poses", nargs="+", default=["center", "left", "right"])
    parser.add_argument("--calibration-dir", type=Path, default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--scan-output-dir", type=Path, default=DEFAULT_SCAN_OUTPUT_DIR)
    parser.add_argument("--pregrasp-output-dir", type=Path, default=DEFAULT_PREGRASP_OUTPUT_DIR)

    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--pose-settle-s", type=float, default=0.7)
    parser.add_argument("--max-step-deg", type=float, default=2.0)
    parser.add_argument("--step-time-s", type=float, default=0.04)

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

    parser.add_argument("--pregrasp-hold-s", type=float, default=0.5)
    parser.add_argument(
        "--pregrasp-joint-offsets",
        type=parse_joint_offsets,
        default=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        help=(
            "Add these six joint offsets to the predicted pregrasp before execution. "
            "Useful for quick demo-time correction, e.g. '0,-5,5,0,0,0' to lift slightly."
        ),
    )

    # ACT local grasp/lift.
    parser.add_argument(
        "--act-policy-path",
        type=str,
        default=None,
        help="Path to trained ACT policy/checkpoint. Required unless --skip-act or --manual-act.",
    )
    parser.add_argument(
        "--manual-act",
        action="store_true",
        help="Do not call lerobot-rollout; instead prompt user to run/perform ACT manually.",
    )
    parser.add_argument("--skip-act", action="store_true")
    parser.add_argument(
        "--act-extra-args",
        type=str,
        default="",
        help="Extra raw args appended to lerobot-rollout, e.g. \"--some.flag=value\".",
    )
    parser.add_argument(
        "--act-timeout-s",
        type=float,
        default=0.0,
        help="Optional timeout for lerobot-rollout. 0 means no timeout.",
    )

    # Bowl transfer.
    parser.add_argument("--bowl-transfer-checkpoint", type=Path, default=DEFAULT_BOWL_TRANSFER_CHECKPOINT)
    parser.add_argument("--bowl-device", type=str, default="cpu")
    parser.add_argument("--transfer-hold-s", type=float, default=0.5)
    parser.add_argument("--release-gripper", type=float, default=None)
    parser.add_argument("--no-release", action="store_true")
    parser.add_argument("--park-after", action="store_true")

    # Debug/control.
    parser.add_argument("--dry-run", action="store_true", help="Predict all poses but do not move/run ACT.")
    parser.add_argument("--skip-pregrasp", action="store_true")
    parser.add_argument("--skip-transfer", action="store_true")

    return parser.parse_args()


def make_pregrasp_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build an argparse-like object expected by test_pregrasp_regressor.py helpers."""
    return argparse.Namespace(
        checkpoint=args.pregrasp_checkpoint,
        scan_dir=None,
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        camera_name=args.camera_name,
        camera_index_or_path=args.camera_index_or_path,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        camera_fourcc=args.camera_fourcc,
        poses=args.poses,
        calibration_dir=args.calibration_dir,
        scan_output_dir=args.scan_output_dir,
        output_dir=args.pregrasp_output_dir,
        warmup_s=args.warmup_s,
        pose_settle_s=args.pose_settle_s,
        max_step_deg=args.max_step_deg,
        step_time_s=args.step_time_s,
        target_color=args.target_color,
        min_area_px=args.min_area_px,
        max_area_px=args.max_area_px,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        workspace_x_min=args.workspace_x_min,
        workspace_x_max=args.workspace_x_max,
        workspace_y_min=args.workspace_y_min,
        workspace_y_max=args.workspace_y_max,
        cluster_radius_m=args.cluster_radius_m,
        reliable_min_views=args.reliable_min_views,
        execute=False,
        allow_unreliable_localization=args.allow_unreliable_localization,
        park_after=False,
        hold_at_target_s=args.pregrasp_hold_s,
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


def apply_joint_offsets(pose: JointPose, offsets: tuple[float, float, float, float, float, float]) -> JointPose:
    return JointPose(
        shoulder_pan=pose.shoulder_pan + offsets[0],
        shoulder_lift=pose.shoulder_lift + offsets[1],
        elbow_flex=pose.elbow_flex + offsets[2],
        wrist_flex=pose.wrist_flex + offsets[3],
        wrist_roll=pose.wrist_roll + offsets[4],
        gripper=pose.gripper + offsets[5],
    )


def run_localization_and_pregrasp(args: argparse.Namespace) -> None:
    if args.pregrasp_checkpoint is None or not args.pregrasp_checkpoint.exists():
        raise FileNotFoundError(f"Missing pregrasp checkpoint: {args.pregrasp_checkpoint}")

    pg_args = make_pregrasp_args(args)
    robot_config = make_pregrasp_robot_config(pg_args)

    print("\n" + "=" * 80)
    print("STAGE 1: Scan, localize cube, predict pregrasp")
    print("=" * 80)

    scan_dir = capture_live_scan(args=pg_args, robot_config=robot_config)
    localization_result = localize_cube(args=pg_args, scan_dir=scan_dir)

    cube_xy = localization_result["cube_xy"]
    is_reliable = localization_result["is_reliable"]

    if cube_xy is None:
        raise RuntimeError("No cube localized; cannot continue.")

    if not is_reliable and not args.allow_unreliable_localization:
        raise RuntimeError(
            "Cube localization was marked unreliable. "
            "Use --allow-unreliable-localization to override."
        )

    _, pregrasp_joint_pose = predict_pregrasp(args=pg_args, localization_result=localization_result)

    if any(abs(x) > 1e-9 for x in args.pregrasp_joint_offsets):
        print("\nApplying pregrasp joint offsets:")
        names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        for name, off in zip(names, args.pregrasp_joint_offsets):
            print(f"  {name:16s} {off:+.3f}")
        pregrasp_joint_pose = apply_joint_offsets(pregrasp_joint_pose, args.pregrasp_joint_offsets)

        print("\nAdjusted pregrasp joints:")
        for name in names:
            print(f"  {name:16s} {getattr(pregrasp_joint_pose, name): .4f}")

    if args.dry_run:
        print("\nDry run: not moving to pregrasp.")
        return

    print("\n" + "=" * 80)
    print("STAGE 2: Move to predicted pregrasp")
    print("=" * 80)

    execute_pregrasp_motion(
        args=pg_args,
        robot_config=robot_config,
        joint_pose=pregrasp_joint_pose,
    )


def run_act_local_grasp(args: argparse.Namespace) -> None:
    print("\n" + "=" * 80)
    print("STAGE 3: ACT local grasp/lift")
    print("=" * 80)

    if args.skip_act:
        print("Skipping ACT stage.")
        return

    if args.manual_act:
        input(
            "\nManual ACT mode. Run/perform local grasp-lift now. "
            "Press Enter once the cube is lifted and ready for bowl transfer..."
        )
        return

    if not args.act_policy_path:
        raise ValueError("--act-policy-path is required unless --skip-act or --manual-act.")

    camera_cfg = (
        "{ wrist: {"
        f"type: opencv, index_or_path: {args.camera_index_or_path}, "
        f"width: {args.camera_width}, height: {args.camera_height}, fps: {args.camera_fps}"
        "} }"
    )

    cmd = [
        "lerobot-rollout",
        f"--robot.type=so101_follower",
        f"--robot.port={args.robot_port}",
        f"--robot.id={args.robot_id}",
        "--robot.disable_torque_on_disconnect=false",
        "--return_to_initial_position=false",
        f"--robot.cameras={camera_cfg}",
        f"--policy.path={args.act_policy_path}",
    ]

    if args.act_extra_args.strip():
        cmd.extend(shlex.split(args.act_extra_args))

    print("\nRunning ACT command:")
    print(" ".join(shlex.quote(c) for c in cmd))

    if args.dry_run:
        print("\nDry run: not launching ACT rollout.")
        return

    try:
        subprocess.run(
            cmd,
            check=True,
            timeout=None if args.act_timeout_s <= 0 else args.act_timeout_s,
        )
    except subprocess.TimeoutExpired:
        print(
            f"\nACT rollout timed out after {args.act_timeout_s:.1f}s. "
            "Continuing to bowl transfer under assumption cube is lifted."
        )


def predict_bowl_transfer_pose(args: argparse.Namespace) -> tuple[np.ndarray, JointPose]:
    if not args.bowl_transfer_checkpoint.exists():
        raise FileNotFoundError(f"Missing bowl transfer checkpoint: {args.bowl_transfer_checkpoint}")

    bowl_x, bowl_y = args.bowl_xy
    bowl_xyz = np.array([bowl_x, bowl_y, args.bowl_z], dtype=np.float32)

    regressor = BowlTransferRegressor(args.bowl_transfer_checkpoint, device=args.bowl_device)
    pred = regressor.predict(bowl_xyz)

    if len(pred) != 5:
        raise ValueError(f"Expected bowl transfer regressor to predict 5 arm joints, got {np.asarray(pred).shape}")

    # Temporary gripper placeholder. The execution stage replaces this with
    # the current gripper value, so we do not drop the cube during transfer.
    joint_pose = JointPose(
        shoulder_pan=float(pred[0]),
        shoulder_lift=float(pred[1]),
        elbow_flex=float(pred[2]),
        wrist_flex=float(pred[3]),
        wrist_roll=float(pred[4]),
        gripper=0.0,
    )

    print("\nBowl transfer input:")
    print(f"  bowl_xyz = ({bowl_xyz[0]:.4f}, {bowl_xyz[1]:.4f}, {bowl_xyz[2]:.4f})")

    print("\nPredicted over-bowl arm joints:")
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    for name, value in zip(names, pred):
        print(f"  {name:16s} {value: .4f}")
    print("  gripper          <kept from current robot state during transfer>")

    return pred, joint_pose


def execute_bowl_transfer(args: argparse.Namespace, joint_pose: JointPose) -> None:
    print("\n" + "=" * 80)
    print("STAGE 4: Move to bowl and release")
    print("=" * 80)

    if args.dry_run:
        print("Dry run: not executing bowl transfer.")
        return

    robot_config = make_robot_config(args)

    with RobotSession(robot_config) as robot:
        current_joints = read_joint_positions(robot)
        current_gripper = float(current_joints[5])

        safe_transfer_pose = JointPose(
            shoulder_pan=joint_pose.shoulder_pan,
            shoulder_lift=joint_pose.shoulder_lift,
            elbow_flex=joint_pose.elbow_flex,
            wrist_flex=joint_pose.wrist_flex,
            wrist_roll=joint_pose.wrist_roll,
            gripper=current_gripper,
        )

        print(f"Keeping current gripper during transfer: {current_gripper:.4f}")
        print("Moving to predicted over-bowl pose...")
        move_to_joint_pose(
            robot,
            safe_transfer_pose,
            max_step_deg=args.max_step_deg,
            step_time_s=args.step_time_s,
            settle_time_s=args.pose_settle_s,
        )
        print("Reached over-bowl pose.")

        if args.transfer_hold_s > 0:
            time.sleep(args.transfer_hold_s)

        if args.no_release:
            print("Not releasing because --no-release was passed.")
        else:
            if args.release_gripper is None:
                raise ValueError(
                    "Need --release-gripper VALUE to open the gripper. "
                    "Pass --no-release to only move above the bowl."
                )

            release_pose = JointPose(
                shoulder_pan=safe_transfer_pose.shoulder_pan,
                shoulder_lift=safe_transfer_pose.shoulder_lift,
                elbow_flex=safe_transfer_pose.elbow_flex,
                wrist_flex=safe_transfer_pose.wrist_flex,
                wrist_roll=safe_transfer_pose.wrist_roll,
                gripper=float(args.release_gripper),
            )

            print(f"Opening gripper to {args.release_gripper:.4f}...")
            move_to_joint_pose(
                robot,
                release_pose,
                max_step_deg=args.max_step_deg,
                step_time_s=args.step_time_s,
                settle_time_s=args.pose_settle_s,
            )
            print("Released.")

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


def main() -> None:
    args = parse_args()

    print("\nTask 1 pipeline configuration:")
    print(f"  bowl_xy: {args.bowl_xy}")
    print(f"  target_color: {args.target_color}")
    print(f"  pregrasp checkpoint: {args.pregrasp_checkpoint}")
    print(f"  ACT policy path: {args.act_policy_path}")
    print(f"  bowl transfer checkpoint: {args.bowl_transfer_checkpoint}")

    if not args.skip_pregrasp:
        run_localization_and_pregrasp(args)
    else:
        print("\nSkipping pregrasp stage.")

    run_act_local_grasp(args)

    if not args.skip_transfer:
        _, bowl_joint_pose = predict_bowl_transfer_pose(args)
        execute_bowl_transfer(args, bowl_joint_pose)
    else:
        print("\nSkipping bowl transfer stage.")

    print("\nTask 1 pipeline finished.")


if __name__ == "__main__":
    main()
