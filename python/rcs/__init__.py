"""Robot control stack python bindings."""

from dataclasses import dataclass

import numpy as np
from rcs._core import __version__, common

from rcs import camera, envs, hand, sim


# TODO: assets must be "installed" with cmake
@dataclass(kw_only=True)
class RobotMetaConfig:

    mjcf_model_path: str
    """Path to the Mujoco XML file that describes the kinematics."""
    # robot_type: common.RobotType
    # """Type of the robot. Checkout all registered types by RobotType.get_all()"""
    dof: int
    "Number of degree of freedom of the robot arm"
    q_home: np.ndarray
    """joint angles in radiant of the robots task independent default home pose, shape: (N,)"""
    joint_limits: np.ndarray
    """hard joint limits of this robot, shape (2, N)"""
    attachment_site: str = "attachment_site"
    """mjcf site to use for IK"""


ROBOTS: dict[common.RobotType, RobotMetaConfig] = {
    common.RobotType.FR3: RobotMetaConfig(
        mjcf_model_path="assets/robots/fr3/fr3.xml",
        dof=7,
        q_home=np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, 0.0]),
        joint_limits=np.array(
            [
                [-2.3093, -1.5133, -2.4937, -2.7478, -2.4800, 0.8521, -2.6895],
                [2.3093, 1.5133, 2.4937, -0.4461, 2.4800, 4.2094, 2.6895],
            ]
        ),
    ),
    common.RobotType.Panda: RobotMetaConfig(
        mjcf_model_path="assets/robots/panda/panda.xml",
        dof=7,
        q_home=np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, 0.0]),
        joint_limits=np.array(
            [
                [
                    -166.0 / 180.0 * np.pi,
                    -101.0 / 180.0 * np.pi,
                    -166.0 / 180.0 * np.pi,
                    -176.0 / 180.0 * np.pi,
                    -166.0 / 180.0 * np.pi,
                    -1.0 / 180.0 * np.pi,
                    -166.0 / 180.0 * np.pi,
                ],
                [
                    166.0 / 180.0 * np.pi,
                    101.0 / 180.0 * np.pi,
                    166.0 / 180.0 * np.pi,
                    -4.0 / 180.0 * np.pi,
                    166.0 / 180.0 * np.pi,
                    215.0 / 180.0 * np.pi,
                    166.0 / 180.0 * np.pi,
                ],
            ]
        ),
    ),
    common.RobotType("XArm7"): RobotMetaConfig(
        mjcf_model_path="assets/robots/xarm7/xarm7.xml",
        dof=7,
        q_home=np.array([0, -45.0 / 180.0 * np.pi, 0, 15.0 / 180.0 * np.pi, 0, -25.0 / 180.0 * np.pi, 0]),
        joint_limits=np.array(
            [
                [-2 * np.pi, -2.094395, -2 * np.pi, -3.92699, -2 * np.pi, -np.pi, -2 * np.pi],
                [2 * np.pi, 2.059488, 2 * np.pi, 0.191986, 2 * np.pi, 1.692969, 2 * np.pi],
            ]
        ),
    ),
    common.RobotType("UR5e"): RobotMetaConfig(
        mjcf_model_path="assets/robots/ur5e/ur5e.xml",
        dof=6,
        q_home=np.array([0.0, -2.02711196, 1.64630026, -1.18999615, -1.57079762, 0.0]),
        joint_limits=np.array(
            [
                [-2 * np.pi, -2 * np.pi, -1 * np.pi, -2 * np.pi, -2 * np.pi, -2 * np.pi],
                [2 * np.pi, 2 * np.pi, 1 * np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi],
            ]
        ),
    ),
    common.RobotType("SO101"): RobotMetaConfig(
        mjcf_model_path="assets/robots/so101/so101.xml",
        dof=5,
        q_home=np.array([-0.01914898, -1.90521916, 1.56476701, 1.04783839, -1.40323926]),
        joint_limits=np.array(
            [
                [
                    -1.9198621771937616,
                    -1.9198621771937634,
                    -1.7453292519943295,
                    -1.6580627969561903,
                    -2.7925268969992407,
                ],
                [1.9198621771937616, 1.9198621771937634, 1.5707963267948966, 1.6580627969561903, 2.7925268969992407],
            ]
        ),
    ),
}


GRIPPER_PATHS = {
    common.GripperType.FrankaHand: "assets/grippers/franka_hand/franka_hand.xml",
    common.GripperType("Robotiq2F85"): "assets/grippers/robotiq_2f85/robotiq_2f85.xml",
}

SCENE_PATHS = {"empty_world": "assets/scenes/empty_world/scene.xml"}

OBJECT_PATHS = {
    "fr3_duo_mount": "assets/objects/fr3_duo_mount/fr3_duo_mount.xml",
    "robotiq_d405_mount": "assets/objects/robotiq_d405_mount/robotiq_d405_mount.xml",
}

CAMERA_PATHS = {
    "d405": "assets/cameras/d405/d405.xml",
    "zed_mini": "assets/cameras/zed_mini/zed_mini.xml",
}

# make submodules available
__all__ = [
    "__doc__",
    "__version__",
    "common",
    "sim",
    "camera",
    "envs",
    "hand",
    "ROBOTS",
    "GRIPPER_PATHS",
    "SCENE_PATHS",
]
