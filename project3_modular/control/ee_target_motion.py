from __future__ import annotations

import math
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
    ForwardKinematicsJointsToEE,
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

    def position_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def rotvec_array(self) -> np.ndarray:
        return np.array([self.wx, self.wy, self.wz], dtype=np.float64)

    def as_action_dict(self) -> dict[str, float]:
        return {
            "ee.x": float(self.x),
            "ee.y": float(self.y),
            "ee.z": float(self.z),
            "ee.wx": float(self.wx),
            "ee.wy": float(self.wy),
            "ee.wz": float(self.wz),
            "ee.gripper_pos": float(self.gripper_pos),
        }


@dataclass
class EEMotionStack:
    kinematics: RobotKinematics
    joints_to_ee_processor: RobotProcessorPipeline
    ee_to_joints_processor: RobotProcessorPipeline
    motor_names: list[str]


def make_ee_motion_stack(
    *,
    robot: Any,
    urdf_path: str | Path,
    target_frame_name: str = "gripper_frame_link",
    end_effector_bounds: dict[str, list[float]] | None = None,
    safety_max_ee_step_m: float = 0.10,
) -> EEMotionStack:
    """
    Build LeRobot's existing FK and IK processor pipelines.

    - FK: raw joint observation -> ee.x, ee.y, ..., ee.gripper_pos
    - IK: absolute EE action + current robot observation -> joint action
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

    if end_effector_bounds is None:
        # Broad defaults for initial bring-up.
        # We can tighten these once we verify the URDF frame convention.
        end_effector_bounds = {
            "min": [-1.0, -1.0, -1.0],
            "max": [1.0, 1.0, 1.0],
        }

    ee_to_joints_processor = RobotProcessorPipeline(
        steps=[
            EEBoundsAndSafety(
                end_effector_bounds=end_effector_bounds,
                max_ee_step_m=safety_max_ee_step_m,
            ),
            InverseKinematicsEEToJoints(
                kinematics=kinematics,
                motor_names=motor_names,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    return EEMotionStack(
        kinematics=kinematics,
        joints_to_ee_processor=joints_to_ee_processor,
        ee_to_joints_processor=ee_to_joints_processor,
        motor_names=motor_names,
    )


def current_ee_pose(
    *,
    robot: Any,
    motion_stack: EEMotionStack,
) -> EEPose:
    """
    Compute current EE pose from the robot's measured joint state via LeRobot FK.
    """
    observation = robot.get_observation()
    ee_observation = motion_stack.joints_to_ee_processor(observation)

    return EEPose(
        x=float(ee_observation["ee.x"]),
        y=float(ee_observation["ee.y"]),
        z=float(ee_observation["ee.z"]),
        wx=float(ee_observation["ee.wx"]),
        wy=float(ee_observation["ee.wy"]),
        wz=float(ee_observation["ee.wz"]),
        gripper_pos=float(ee_observation["ee.gripper_pos"]),
    )


def ee_pose_to_joint_action(
    *,
    robot: Any,
    motion_stack: EEMotionStack,
    target_pose: EEPose,
) -> dict[str, float]:
    """
    Convert an absolute EE target to a joint-space command using LeRobot IK.
    """
    observation = robot.get_observation()
    ee_action = target_pose.as_action_dict()

    joint_action = motion_stack.ee_to_joints_processor(
        (ee_action, observation)
    )

    return joint_action


def interpolate_ee_positions(
    start: EEPose,
    target: EEPose,
    *,
    max_translation_step_m: float,
) -> list[EEPose]:
    """
    Linearly interpolate only the xyz position.
    Orientation and gripper position are held at the target values.

    This keeps motion smooth while leaving IK/control to LeRobot.
    """
    if max_translation_step_m <= 0:
        raise ValueError("max_translation_step_m must be positive.")

    start_xyz = start.position_array()
    target_xyz = target.position_array()

    distance = float(np.linalg.norm(target_xyz - start_xyz))
    n_steps = max(1, math.ceil(distance / max_translation_step_m))

    poses: list[EEPose] = []

    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        xyz = start_xyz + alpha * (target_xyz - start_xyz)

        poses.append(
            EEPose(
                x=float(xyz[0]),
                y=float(xyz[1]),
                z=float(xyz[2]),
                wx=target.wx,
                wy=target.wy,
                wz=target.wz,
                gripper_pos=target.gripper_pos,
            )
        )

    return poses


def move_to_ee_pose(
    *,
    robot: Any,
    motion_stack: EEMotionStack,
    target_pose: EEPose,
    max_translation_step_m: float = 0.005,
    step_time_s: float = 0.05,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """
    Move smoothly toward an absolute EE pose.

    We generate a sequence of small Cartesian targets ourselves, while LeRobot
    handles EE safety checking and IK conversion at every step.
    """
    start_pose = current_ee_pose(robot=robot, motion_stack=motion_stack)

    trajectory = interpolate_ee_positions(
        start=start_pose,
        target=target_pose,
        max_translation_step_m=max_translation_step_m,
    )

    if verbose:
        print(
            f"Interpolated Cartesian trajectory: {len(trajectory)} steps "
            f"from ({start_pose.x:.4f}, {start_pose.y:.4f}, {start_pose.z:.4f}) "
            f"to ({target_pose.x:.4f}, {target_pose.y:.4f}, {target_pose.z:.4f})"
        )

    for step_idx, pose in enumerate(trajectory, start=1):
        joint_action = ee_pose_to_joint_action(
            robot=robot,
            motion_stack=motion_stack,
            target_pose=pose,
        )

        if not dry_run:
            robot.send_action(joint_action)

        if verbose and (step_idx == 1 or step_idx == len(trajectory) or step_idx % 10 == 0):
            print(
                f"  step {step_idx:3d}/{len(trajectory)} "
                f"ee=({pose.x:.4f}, {pose.y:.4f}, {pose.z:.4f})"
            )

        time.sleep(step_time_s)

class EEMotionError(RuntimeError):
    """Raised when a feedback-controlled EE motion fails safely."""
    pass


def _ee_position_error_m(current: EEPose, target: EEPose) -> float:
    return float(
        np.linalg.norm(
            current.position_array() - target.position_array()
        )
    )


def move_to_ee_pose_feedback_controlled(
    *,
    robot: Any,
    motion_stack: EEMotionStack,
    target_pose: EEPose,
    max_waypoint_step_m: float = 0.01,
    reach_tolerance_m: float = 0.005,
    waypoint_timeout_s: float = 3.0,
    command_period_s: float = 0.10,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """
    Move to an absolute EE pose using small Cartesian waypoints,
    but only advance once the measured FK EE pose is close to each waypoint.

    LeRobot still performs:
      EE action -> safety bounds -> IK -> joint command

    This function adds:
      - staged waypoints
      - measured-state convergence checks
      - per-waypoint timeout
    """
    if max_waypoint_step_m <= 0:
        raise ValueError("max_waypoint_step_m must be positive.")
    if reach_tolerance_m <= 0:
        raise ValueError("reach_tolerance_m must be positive.")
    if waypoint_timeout_s <= 0:
        raise ValueError("waypoint_timeout_s must be positive.")
    if command_period_s <= 0:
        raise ValueError("command_period_s must be positive.")

    start_pose = current_ee_pose(
        robot=robot,
        motion_stack=motion_stack,
    )

    waypoints = interpolate_ee_positions(
        start=start_pose,
        target=target_pose,
        max_translation_step_m=max_waypoint_step_m,
    )

    if verbose:
        print(
            f"Feedback-controlled EE trajectory: {len(waypoints)} waypoints "
            f"from ({start_pose.x:.4f}, {start_pose.y:.4f}, {start_pose.z:.4f}) "
            f"to ({target_pose.x:.4f}, {target_pose.y:.4f}, {target_pose.z:.4f})"
        )

    if dry_run:
        for waypoint_idx, waypoint in enumerate(waypoints, start=1):
            _ = ee_pose_to_joint_action(
                robot=robot,
                motion_stack=motion_stack,
                target_pose=waypoint,
            )
            if verbose:
                print(
                    f"  dry-run waypoint {waypoint_idx:3d}/{len(waypoints)} "
                    f"ee=({waypoint.x:.4f}, {waypoint.y:.4f}, {waypoint.z:.4f})"
                )
        return

    for waypoint_idx, waypoint in enumerate(waypoints, start=1):
        waypoint_start = time.perf_counter()

        while True:
            current_pose = current_ee_pose(
                robot=robot,
                motion_stack=motion_stack,
            )
            error_m = _ee_position_error_m(current_pose, waypoint)

            if error_m <= reach_tolerance_m:
                if verbose:
                    print(
                        f"  reached waypoint {waypoint_idx:3d}/{len(waypoints)} "
                        f"err={error_m * 100:.2f} cm "
                        f"ee=({waypoint.x:.4f}, {waypoint.y:.4f}, {waypoint.z:.4f})"
                    )
                break

            elapsed = time.perf_counter() - waypoint_start
            if elapsed > waypoint_timeout_s:
                raise EEMotionError(
                    f"Timed out while reaching waypoint "
                    f"{waypoint_idx}/{len(waypoints)}. "
                    f"Remaining EE position error: {error_m * 100:.2f} cm."
                )

            joint_action = ee_pose_to_joint_action(
                robot=robot,
                motion_stack=motion_stack,
                target_pose=waypoint,
            )
            robot.send_action(joint_action)
            time.sleep(command_period_s)
