from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    RobotProcessorPipeline,
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEE,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)


@dataclass(frozen=True)
class EEPose:
    x: float
    y: float
    z: float
    wx: float
    wy: float
    wz: float
    gripper_pos: float

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


@dataclass
class EEDeltaMotionStack:
    kinematics: RobotKinematics
    joints_to_ee_processor: RobotProcessorPipeline
    delta_to_ee_processor: RobotProcessorPipeline
    ee_to_joints_processor: RobotProcessorPipeline
    step_sizes: dict[str, float]


def make_ee_delta_motion_stack(
    *,
    robot: Any,
    urdf_path: str | Path,
    target_frame_name: str = "gripper_frame_link",
    end_effector_bounds: dict[str, list[float]] | None = None,
    step_sizes: dict[str, float] | None = None,
    max_ee_step_m: float = 0.03,
) -> EEDeltaMotionStack:
    """
    Build LeRobot-native EE-delta motion processors.

    We keep the custom code minimal:
      - LeRobot computes FK
      - LeRobot turns delta EE actions into absolute EE targets
      - LeRobot applies EE safety bounds
      - LeRobot runs IK into joint commands
    """
    urdf_path = Path(urdf_path)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    motor_names = list(robot.bus.motors.keys())

    kinematics = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name=target_frame_name,
        joint_names=motor_names,
    )

    if step_sizes is None:
        # Maximum Cartesian change represented by target_x/y/z = ±1.
        step_sizes = {
            "x": 0.01,
            "y": 0.01,
            "z": 0.01,
        }

    if end_effector_bounds is None:
        # Broad bring-up bounds. We can tighten later if useful.
        end_effector_bounds = {
            "min": [-1.0, -1.0, -1.0],
            "max": [1.0, 1.0, 1.0],
        }

    joints_to_ee_processor = RobotProcessorPipeline(
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=kinematics,
                motor_names=motor_names,
            )
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # This follows the same structure used by LeRobot's EE-delta control pipelines.
    delta_to_ee_processor = RobotProcessorPipeline(
        steps=[
            EEReferenceAndDelta(
                kinematics=kinematics,
                end_effector_step_sizes=step_sizes,
                motor_names=motor_names,
                use_latched_reference=False,
                use_ik_solution=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds=end_effector_bounds,
                max_ee_step_m=max_ee_step_m,
            ),
            GripperVelocityToJoint(
                speed_factor=20.0,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    ee_to_joints_processor = RobotProcessorPipeline(
        steps=[
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=motor_names,
                initial_guess_current_joints=True,
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    return EEDeltaMotionStack(
        kinematics=kinematics,
        joints_to_ee_processor=joints_to_ee_processor,
        delta_to_ee_processor=delta_to_ee_processor,
        ee_to_joints_processor=ee_to_joints_processor,
        step_sizes=step_sizes,
    )


def current_ee_pose(
    *,
    robot: Any,
    motion_stack: EEDeltaMotionStack,
) -> EEPose:
    observation = robot.get_observation()
    ee_obs = motion_stack.joints_to_ee_processor(observation)

    return EEPose(
        x=float(ee_obs["ee.x"]),
        y=float(ee_obs["ee.y"]),
        z=float(ee_obs["ee.z"]),
        wx=float(ee_obs["ee.wx"]),
        wy=float(ee_obs["ee.wy"]),
        wz=float(ee_obs["ee.wz"]),
        gripper_pos=float(ee_obs["ee.gripper_pos"]),
    )


def _make_normalized_delta_action(
    *,
    current_pose: EEPose,
    target_xyz: np.ndarray,
    step_sizes: dict[str, float],
    proportional_gain: float,
) -> tuple[dict[str, float], float]:
    """
    Convert target position error into LeRobot's normalized EE-delta action.
    """
    error_xyz = target_xyz - current_pose.xyz
    distance_m = float(np.linalg.norm(error_xyz))

    dx, dy, dz = error_xyz

    action = {
        "enabled": True,
        "target_x": float(
            np.clip(proportional_gain * dx / step_sizes["x"], -1.0, 1.0)
        ),
        "target_y": float(
            np.clip(proportional_gain * dy / step_sizes["y"], -1.0, 1.0)
        ),
        "target_z": float(
            np.clip(proportional_gain * dz / step_sizes["z"], -1.0, 1.0)
        ),
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper_vel": 0.0,
    }

    return action, distance_m


def step_toward_xyz(
    *,
    robot: Any,
    motion_stack: EEDeltaMotionStack,
    target_xyz: np.ndarray,
    proportional_gain: float = 1.0,
    dry_run: bool = False,
) -> tuple[EEPose, float]:
    """
    Execute one LeRobot-native delta-control step toward target_xyz.
    """
    observation = robot.get_observation()
    current_pose = current_ee_pose(robot=robot, motion_stack=motion_stack)

    delta_action, distance_m = _make_normalized_delta_action(
        current_pose=current_pose,
        target_xyz=target_xyz,
        step_sizes=motion_stack.step_sizes,
        proportional_gain=proportional_gain,
    )

    ee_action = motion_stack.delta_to_ee_processor(
        (delta_action, observation)
    )
    joint_action = motion_stack.ee_to_joints_processor(
        (ee_action, observation)
    )

    if not dry_run:
        robot.send_action(joint_action)

    return current_pose, distance_m


def move_toward_xyz(
    *,
    robot: Any,
    motion_stack: EEDeltaMotionStack,
    target_xyz: np.ndarray,
    tolerance_m: float = 0.01,
    timeout_s: float = 10.0,
    control_hz: float = 10.0,
    proportional_gain: float = 1.0,
    dry_run: bool = False,
    verbose: bool = True,
) -> tuple[EEPose, float, bool]:
    """
    Repeatedly apply small EE-delta actions until close to target or timed out.

    Returns:
        final_pose, final_distance_m, reached
    """
    if tolerance_m <= 0:
        raise ValueError("tolerance_m must be positive.")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive.")
    if control_hz <= 0:
        raise ValueError("control_hz must be positive.")

    dt = 1.0 / control_hz
    start = time.perf_counter()
    step_idx = 0

    while True:
        pose, distance_m = step_toward_xyz(
            robot=robot,
            motion_stack=motion_stack,
            target_xyz=target_xyz,
            proportional_gain=proportional_gain,
            dry_run=dry_run,
        )
        step_idx += 1

        if verbose and (step_idx == 1 or step_idx % int(max(1, control_hz)) == 0):
            print(
                f"  step={step_idx:03d} "
                f"ee=({pose.x:.4f}, {pose.y:.4f}, {pose.z:.4f}) "
                f"dist={distance_m * 100:.2f} cm"
            )

        if distance_m <= tolerance_m:
            return pose, distance_m, True

        elapsed = time.perf_counter() - start
        if elapsed >= timeout_s:
            final_pose = current_ee_pose(robot=robot, motion_stack=motion_stack)
            final_distance_m = float(np.linalg.norm(target_xyz - final_pose.xyz))
            return final_pose, final_distance_m, False

        time.sleep(dt)

def _make_normalized_xy_delta_action(
    *,
    current_pose: EEPose,
    target_xy: np.ndarray,
    step_sizes: dict[str, float],
    proportional_gain: float,
    hold_z: float | None = None,
    z_hold_gain: float = 0.25,
    max_abs_z_command: float = 0.35,
) -> tuple[dict[str, float], float]:
    """
    Convert XY position error into LeRobot's normalized EE-delta action.

    XY motion is the main goal.
    Optionally, apply a weak z correction to keep the end effector roughly
    at a chosen height without rigidly constraining the IK solution.
    """
    current_xy = np.array([current_pose.x, current_pose.y], dtype=np.float64)
    error_xy = target_xy - current_xy
    distance_xy_m = float(np.linalg.norm(error_xy))

    dx, dy = error_xy

    target_z_cmd = 0.0
    if hold_z is not None:
        dz = hold_z - current_pose.z
        target_z_cmd = float(
            np.clip(
                z_hold_gain * dz / step_sizes["z"],
                -max_abs_z_command,
                max_abs_z_command,
            )
        )

    action = {
        "enabled": True,
        "target_x": float(
            np.clip(proportional_gain * dx / step_sizes["x"], -1.0, 1.0)
        ),
        "target_y": float(
            np.clip(proportional_gain * dy / step_sizes["y"], -1.0, 1.0)
        ),
        "target_z": target_z_cmd,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper_vel": 0.0,
    }

    return action, distance_xy_m


def step_toward_xy(
    *,
    robot: Any,
    motion_stack: EEDeltaMotionStack,
    target_xy: np.ndarray,
    proportional_gain: float = 1.0,
    hold_z: float | None = None,
    z_hold_gain: float = 0.25,
    max_abs_z_command: float = 0.35,
    dry_run: bool = False,
) -> tuple[EEPose, float]:
    """
    Execute one LeRobot-native lateral EE-delta step toward target_xy.

    The controller commands x/y motion only and leaves z unconstrained.
    """
    observation = robot.get_observation()
    current_pose = current_ee_pose(robot=robot, motion_stack=motion_stack)

    delta_action, distance_xy_m = _make_normalized_xy_delta_action(
        current_pose=current_pose,
        target_xy=target_xy,
        step_sizes=motion_stack.step_sizes,
        proportional_gain=proportional_gain,
        hold_z=hold_z,
        z_hold_gain=z_hold_gain,
        max_abs_z_command=max_abs_z_command,
    )

    ee_action = motion_stack.delta_to_ee_processor(
        (delta_action, observation)
    )
    joint_action = motion_stack.ee_to_joints_processor(
        (ee_action, observation)
    )

    if not dry_run:
        robot.send_action(joint_action)

    return current_pose, distance_xy_m


def move_toward_xy(
    *,
    robot: Any,
    motion_stack: EEDeltaMotionStack,
    target_xy: np.ndarray,
    tolerance_m: float = 0.02,
    timeout_s: float = 10.0,
    control_hz: float = 10.0,
    proportional_gain: float = 1.0,
    hold_z: float | None = None,
    z_hold_gain: float = 0.25,
    max_abs_z_command: float = 0.35,
    min_safe_z: float | None = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> tuple[EEPose, float, bool]:
    """
    Repeatedly apply lateral EE-delta commands until the XY target is reached
    or timeout occurs.

    Unlike move_toward_xyz(...), this ignores z error entirely.
    Optionally abort if measured z falls below min_safe_z.
    """
    if tolerance_m <= 0:
        raise ValueError("tolerance_m must be positive.")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive.")
    if control_hz <= 0:
        raise ValueError("control_hz must be positive.")

    dt = 1.0 / control_hz
    start = time.perf_counter()
    step_idx = 0

    while True:
        pose, distance_xy_m = step_toward_xy(
            robot=robot,
            motion_stack=motion_stack,
            target_xy=target_xy,
            proportional_gain=proportional_gain,
            hold_z=hold_z,
            z_hold_gain=z_hold_gain,
            max_abs_z_command=max_abs_z_command,
            dry_run=dry_run,
        )
        step_idx += 1

        if min_safe_z is not None and pose.z < min_safe_z:
            if verbose:
                print(
                    f"  aborting XY move: measured z={pose.z:.4f} m "
                    f"fell below min_safe_z={min_safe_z:.4f} m"
                )
            return pose, distance_xy_m, False

        if verbose and (step_idx == 1 or step_idx % int(max(1, control_hz)) == 0):
            print(
                f"  step={step_idx:03d} "
                f"ee=({pose.x:.4f}, {pose.y:.4f}, {pose.z:.4f}) "
                f"xy_dist={distance_xy_m * 100:.2f} cm"
            )

        if distance_xy_m <= tolerance_m:
            return pose, distance_xy_m, True

        elapsed = time.perf_counter() - start
        if elapsed >= timeout_s:
            final_pose = current_ee_pose(robot=robot, motion_stack=motion_stack)
            final_xy = np.array([final_pose.x, final_pose.y], dtype=np.float64)
            final_distance_xy_m = float(np.linalg.norm(target_xy - final_xy))
            return final_pose, final_distance_xy_m, False

        time.sleep(dt)