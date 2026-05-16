from __future__ import annotations

import math
import time
from typing import Any

from control.poses import JOINT_KEYS, JointPose


def extract_joint_positions(observation: dict[str, Any]) -> dict[str, float]:
    """Extract follower joint positions from a LeRobot observation dict."""
    positions: dict[str, float] = {}

    for key in JOINT_KEYS:
        if key not in observation:
            raise KeyError(
                f"Missing joint key {key!r} in observation. "
                f"Available keys: {list(observation.keys())}"
            )
        positions[key] = float(observation[key])

    return positions


def move_to_joint_pose(
    robot: Any,
    target_pose: JointPose,
    *,
    max_step_deg: float = 2.0,
    step_time_s: float = 0.04,
    settle_time_s: float = 0.5,
) -> dict[str, Any]:
    """
    Move smoothly from the current robot joint configuration to target_pose.

    The SO101 follower actions are position targets in degrees, so we linearly
    interpolate joint targets and send them through robot.send_action(...).

    Args:
        robot:
            Connected LeRobot robot instance.
        target_pose:
            Desired joint-space pose.
        max_step_deg:
            Maximum change, in degrees, of the most-changing joint per interpolation step.
        step_time_s:
            Sleep duration between target updates.
        settle_time_s:
            Extra wait after the final target is sent.

    Returns:
        Final observation after the motion settles.
    """
    if max_step_deg <= 0:
        raise ValueError("max_step_deg must be positive.")
    if step_time_s <= 0:
        raise ValueError("step_time_s must be positive.")
    if settle_time_s < 0:
        raise ValueError("settle_time_s must be non-negative.")

    start_obs = robot.get_observation()
    start = extract_joint_positions(start_obs)
    target = target_pose.as_action_dict()

    max_delta = max(abs(target[key] - start[key]) for key in JOINT_KEYS)
    n_steps = max(1, math.ceil(max_delta / max_step_deg))

    for step in range(1, n_steps + 1):
        alpha = step / n_steps
        action = {
            key: start[key] + alpha * (target[key] - start[key])
            for key in JOINT_KEYS
        }
        robot.send_action(action)
        time.sleep(step_time_s)

    if settle_time_s > 0:
        time.sleep(settle_time_s)

    return robot.get_observation()