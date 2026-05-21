from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from control.robot_session import RobotSession, RobotSessionConfig


DEFAULT_DATASET_JSON = PROJECT_ROOT / "data" / "bowl_transfer_dataset" / "bowl_transfer_examples.json"


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


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
            "Collect bowl transfer regressor data: "
            "given bowl xyz, manually place the follower in a good over-bowl "
            "release/transfer pose, then save current joint positions."
        )
    )

    parser.add_argument("--dataset-json", type=Path, default=DEFAULT_DATASET_JSON)

    parser.add_argument("--robot-port", type=str, default="/dev/ttyACM2")
    parser.add_argument("--robot-id", type=str, default="my_awesome_follower_arm")

    # Camera is not used for labels, but RobotSessionConfig may require it.
    parser.add_argument("--camera-name", type=str, default="wrist")
    parser.add_argument("--camera-index-or-path", type=parse_camera_identifier, default=1)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--camera-fourcc", type=str, default="MJPG")
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Do not initialize a camera. Useful because bowl transfer labels only need joint positions.",
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

    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Optional note, e.g. desk/mat/bowl-edge/fringe.",
    )

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


def _flatten_numeric(value: Any) -> list[float] | None:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        return None
    if arr.size == 6 and np.all(np.isfinite(arr)):
        return [float(x) for x in arr]
    return None


def read_joint_positions(robot: Any) -> list[float]:
    """Best-effort reader for current SO101 follower joint positions.

    This handles a few likely RobotSession / LeRobot wrapper layouts.
    If your local wrapper exposes a different method, patch this function only.
    """

    # Try common observation methods on RobotSession and wrapped robot.
    candidates = [robot]
    for attr in ["robot", "_robot", "follower", "bot"]:
        if hasattr(robot, attr):
            candidates.append(getattr(robot, attr))

    obs = None
    for obj in candidates:
        for meth_name in [
            "get_observation",
            "capture_observation",
            "read_observation",
            "observe",
        ]:
            meth = getattr(obj, meth_name, None)
            if callable(meth):
                try:
                    obs = meth()
                    break
                except TypeError:
                    continue
        if obs is not None:
            break

    if isinstance(obs, dict):
        # Preferred: explicit state vector.
        for key in [
            "observation.state",
            "state",
            "joint_positions",
            "joints",
            "action",
        ]:
            if key in obs:
                maybe = _flatten_numeric(obs[key])
                if maybe is not None:
                    return maybe

        # Fallback: collect named joint entries.
        vals = []
        ok = True
        for name in JOINT_NAMES:
            possible_keys = [
                name,
                f"{name}.pos",
                f"observation.state.{name}.pos",
                f"observation.state.{name}",
            ]
            found = False
            for key in possible_keys:
                if key in obs:
                    vals.append(float(np.asarray(obs[key]).reshape(-1)[0]))
                    found = True
                    break
            if not found:
                ok = False
                break
        if ok:
            return vals

        raise RuntimeError(
            "Could not extract 6 joint positions from observation. "
            f"Available observation keys: {list(obs.keys())}"
        )

    # Try low-level bus access as final fallback.
    for obj in candidates:
        bus = getattr(obj, "bus", None)
        if bus is None:
            continue

        # These names are common in the LeRobot motor bus stack.
        motors = getattr(bus, "motors", None)
        if isinstance(motors, dict):
            names = list(motors.keys())
        else:
            names = JOINT_NAMES

        for read_name in ["read", "sync_read"]:
            read = getattr(bus, read_name, None)
            if callable(read):
                try:
                    values = read("Present_Position", names)
                    maybe = _flatten_numeric(values)
                    if maybe is not None:
                        return maybe
                except Exception:
                    pass

    raise RuntimeError(
        "Could not read current joint positions. "
        "Patch read_joint_positions() for your local RobotSession API."
    )


def load_examples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "examples" in raw:
        return list(raw["examples"])
    raise ValueError(f"Unsupported dataset JSON format: {path}")


def save_examples(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": "Bowl transfer regressor dataset: bowl_xyz -> over-bowl joint pose.",
        "joint_names": JOINT_NAMES,
        "examples": examples,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    robot_config = make_robot_config(args)

    bowl_x, bowl_y = args.bowl_xy
    bowl_xyz = [float(bowl_x), float(bowl_y), float(args.bowl_z)]

    print("\nBowl xyz:")
    print(f"  x={bowl_xyz[0]:.4f}, y={bowl_xyz[1]:.4f}, z={bowl_xyz[2]:.4f}")
    print("\nManually move the follower to a good over-bowl transfer/release pose.")
    print("The pose should be safely above the bowl center and suitable before opening the gripper.")
    input("\nPress Enter when the follower is at the desired pose...")

    with RobotSession(robot_config) as robot:
        time.sleep(0.2)
        joints = read_joint_positions(robot)

    example = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "bowl_xyz": bowl_xyz,
        "release_joint_positions": joints,
        "joint_names": JOINT_NAMES,
        "note": args.note,
    }

    examples = load_examples(args.dataset_json)
    examples.append(example)
    save_examples(args.dataset_json, examples)

    print("\nSaved example:")
    print(json.dumps(example, indent=2))
    print(f"\nDataset: {args.dataset_json}")
    print(f"Total examples: {len(examples)}")


if __name__ == "__main__":
    main()
