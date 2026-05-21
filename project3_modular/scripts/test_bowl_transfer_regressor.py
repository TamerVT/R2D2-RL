from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from control.ee_motion import move_to_joint_pose
from control.poses import JointPose, PARK_POSE
from control.robot_session import RobotSession, RobotSessionConfig
from collect_bowl_transfer_dataset import read_joint_positions


DEFAULT_CHECKPOINT = PROJECT_ROOT / "data" / "bowl_transfer_dataset" / "bowl_transfer_regressor.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "bowl_transfer_test_outputs"


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


class BowlTransferMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 5,
        hidden_dim: int = 128,
        num_hidden_layers: int = 3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.ReLU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BowlTransferRegressor:
    def __init__(self, checkpoint_path: Path, device: str = "cpu") -> None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg = ckpt["model_config"]

        self.model = BowlTransferMLP(**cfg).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        norm = ckpt["normalization"]
        self.x_mean = np.asarray(norm["x_mean"], dtype=np.float32)
        self.x_std = np.asarray(norm["x_std"], dtype=np.float32)
        self.y_mean = np.asarray(norm["y_mean"], dtype=np.float32)
        self.y_std = np.asarray(norm["y_std"], dtype=np.float32)
        self.device = torch.device(device)

    def predict(self, bowl_xyz: np.ndarray) -> np.ndarray:
        x = np.asarray(bowl_xyz, dtype=np.float32).reshape(1, 3)
        x_norm = (x - self.x_mean) / self.x_std

        with torch.no_grad():
            pred_norm = self.model(torch.tensor(x_norm, dtype=torch.float32, device=self.device))
        pred = pred_norm.cpu().numpy()[0] * self.y_std + self.y_mean
        return pred.astype(np.float32)


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
            f"Expected two numbers for --bowl_xy, got {value!r}. "
            "Examples: --bowl_xy '(0.10, 0.38)' or --bowl_xy 0.10,0.38"
        )
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Could not parse --bowl_xy {value!r} as two floats."
        ) from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test bowl transfer regressor: "
            "given bowl xyz, predict over-bowl joint pose, optionally execute it, "
            "and optionally open gripper."
        )
    )

    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)

    parser.add_argument("--robot-port", type=str, default="/dev/ttyACM2")
    parser.add_argument("--robot-id", type=str, default="my_awesome_follower_arm")

    parser.add_argument("--camera-name", type=str, default="wrist")
    parser.add_argument("--camera-index-or-path", type=parse_camera_identifier, default=1)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-fourcc", type=str, default="MJPG")
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Do not initialize a camera. Useful because bowl transfer only needs joint positions.",
    )

    parser.add_argument(
        "--bowl_xy",
        type=parse_xy,
        required=True,
        help="Bowl xy in robot frame, e.g. --bowl_xy '(0.10, 0.38)' or --bowl_xy 0.10,0.38",
    )
    parser.add_argument(
        "--bowl-z",
        type=float,
        default=0.0,
        help="Bowl z in robot frame. Defaults to 0.0.",
    )

    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--release", action="store_true", help="Open gripper after reaching predicted pose.")
    parser.add_argument(
        "--release-gripper",
        type=float,
        default=None,
        help=(
            "Gripper joint value used for release. "
            "Required if --release is passed, unless you patch a known safe default."
        ),
    )

    parser.add_argument("--park-after", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hold-at-target-s", type=float, default=1.0)
    parser.add_argument("--hold-after-release-s", type=float, default=0.5)

    parser.add_argument("--max-step-deg", type=float, default=2.0)
    parser.add_argument("--step-time-s", type=float, default=0.04)
    parser.add_argument("--settle-time-s", type=float, default=0.5)

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    return parser.parse_args()


def make_robot_config(args: argparse.Namespace) -> RobotSessionConfig:
    return RobotSessionConfig(
        robot_port=args.robot_port,
        robot_id=args.robot_id,
        camera_name=None if args.no_camera else args.camera_name,
        camera_index_or_path=args.camera_index_or_path,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fps=args.camera_fps,
        camera_fourcc=args.camera_fourcc,
    )


def make_joint_pose(pred: np.ndarray, current_gripper: float) -> JointPose:
    """Build a transfer pose from predicted arm joints while preserving gripper."""
    if len(pred) != 5:
        raise ValueError(f"Expected 5 predicted arm joints, got shape/len {np.asarray(pred).shape}")

    return JointPose(
        shoulder_pan=float(pred[0]),
        shoulder_lift=float(pred[1]),
        elbow_flex=float(pred[2]),
        wrist_flex=float(pred[3]),
        wrist_roll=float(pred[4]),
        gripper=float(current_gripper),
    )


def print_prediction(pred: np.ndarray) -> None:
    print("\nPredicted over-bowl transfer arm joints:")
    for name, value in zip(JOINT_NAMES[:5], pred):
        print(f"  {name:16s} {value: .4f}")
    print("  gripper          <kept from current robot state during transfer>")


def execute_motion(args: argparse.Namespace, robot_config: RobotSessionConfig, pred: np.ndarray) -> None:
    print("\nConnecting to robot for bowl transfer execution...")
    with RobotSession(robot_config) as robot:
        current_joints = read_joint_positions(robot)
        current_gripper = float(current_joints[5])
        joint_pose = make_joint_pose(pred, current_gripper)

        print(f"Keeping current gripper during transfer: {current_gripper:.4f}")
        print("Moving to predicted over-bowl pose...")
        move_to_joint_pose(
            robot,
            joint_pose,
            max_step_deg=args.max_step_deg,
            step_time_s=args.step_time_s,
            settle_time_s=args.settle_time_s,
        )
        print("Reached predicted over-bowl pose.")

        if args.hold_at_target_s > 0:
            time.sleep(args.hold_at_target_s)

        if args.release:
            if args.release_gripper is None:
                raise ValueError(
                    "--release requires --release-gripper VALUE. "
                    "Use a known safe open-gripper joint value from your robot."
                )

            release_pose = JointPose(
                shoulder_pan=joint_pose.shoulder_pan,
                shoulder_lift=joint_pose.shoulder_lift,
                elbow_flex=joint_pose.elbow_flex,
                wrist_flex=joint_pose.wrist_flex,
                wrist_roll=joint_pose.wrist_roll,
                gripper=float(args.release_gripper),
            )

            print(f"Opening gripper to {args.release_gripper:.4f}...")
            move_to_joint_pose(
                robot,
                release_pose,
                max_step_deg=args.max_step_deg,
                step_time_s=args.step_time_s,
                settle_time_s=args.settle_time_s,
            )

            if args.hold_after_release_s > 0:
                time.sleep(args.hold_after_release_s)

        if args.park_after:
            print("Moving to park pose...")
            move_to_joint_pose(
                robot,
                PARK_POSE,
                max_step_deg=args.max_step_deg,
                step_time_s=args.step_time_s,
                settle_time_s=args.settle_time_s,
            )
            print("Reached park pose.")


def save_summary(args: argparse.Namespace, bowl_xyz: np.ndarray, prediction: np.ndarray) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    summary = {
        "checkpoint": str(args.checkpoint),
        "bowl_xyz": bowl_xyz.tolist(),
        "predicted_transfer_joints": prediction.tolist(),
        "execute": bool(args.execute),
        "release": bool(args.release),
        "release_gripper": args.release_gripper,
    }

    path = run_dir / "bowl_transfer_test_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    return path


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    bowl_x, bowl_y = args.bowl_xy
    bowl_xyz = np.array([bowl_x, bowl_y, args.bowl_z], dtype=np.float32)

    regressor = BowlTransferRegressor(args.checkpoint, device=args.device)
    pred = regressor.predict(bowl_xyz)

    print(f"\nRegressor input bowl xyz: ({bowl_xyz[0]:.4f}, {bowl_xyz[1]:.4f}, {bowl_xyz[2]:.4f})")
    print_prediction(pred)

    if args.execute:
        robot_config = make_robot_config(args)
        execute_motion(args, robot_config, pred)
    else:
        print("\nDry run only. Pass --execute to move the robot.")

    summary_path = save_summary(args, bowl_xyz, pred)
    print(f"\nSaved run summary: {summary_path}")


if __name__ == "__main__":
    main()
