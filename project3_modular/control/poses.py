from __future__ import annotations

from dataclasses import dataclass


JOINT_KEYS = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)


@dataclass(frozen=True)
class JointPose:
    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float
    wrist_flex: float
    wrist_roll: float
    gripper: float

    def as_action_dict(self) -> dict[str, float]:
        return {
            "shoulder_pan.pos": self.shoulder_pan,
            "shoulder_lift.pos": self.shoulder_lift,
            "elbow_flex.pos": self.elbow_flex,
            "wrist_flex.pos": self.wrist_flex,
            "wrist_roll.pos": self.wrist_roll,
            "gripper.pos": self.gripper,
        }


# Current usable center-ish overview pose, taken from:
# observation_20260516_193954.png / associated metadata.
# Fixed observation poses for multi-view cube localization.
# Names refer to the direction the camera view is biased toward.

OBSERVATION_CENTER = JointPose(
    shoulder_pan=2.7253,
    shoulder_lift=-65.2747,
    elbow_flex=14.3297,
    wrist_flex=100.5714,
    wrist_roll=3.1209,
    gripper=1.7206,
)

OBSERVATION_RIGHT = JointPose(
    shoulder_pan=18.1099,
    shoulder_lift=-65.2747,
    elbow_flex=14.3297,
    wrist_flex=96.9670,
    wrist_roll=27.2088,
    gripper=1.3765,
)

OBSERVATION_LEFT = JointPose(
    shoulder_pan=-6.4176,
    shoulder_lift=-60.2637,
    elbow_flex=21.3626,
    wrist_flex=93.4505,
    wrist_roll=-14.2857,
    gripper=1.3765,
)

PARK_POSE = JointPose(
  shoulder_pan=0.3516,
  shoulder_lift=-78.6374,
  elbow_flex=67.0769,
  wrist_flex=80.3516,
  wrist_roll=4.1758,
  gripper=1.3076,
)

OBSERVATION_POSES: dict[str, JointPose] = {
    "center": OBSERVATION_CENTER,
    "left": OBSERVATION_LEFT,
    "right": OBSERVATION_RIGHT,
}