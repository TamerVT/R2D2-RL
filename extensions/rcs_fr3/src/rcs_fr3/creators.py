import logging
import typing

import gymnasium as gym
import numpy as np
import rcs.hand.tilburg_hand
from frankik import FrankaKinematics
from rcs._core.common import Kinematics, Pose
from rcs.camera.hw import HardwareCameraSet
from rcs.envs.base import (
    CameraSetWrapper,
    ControlMode,
    CoverWrapper,
    GripperWrapper,
    HandWrapper,
    HardwareEnv,
    MultiRobotWrapper,
    RelativeActionSpace,
    RelativeTo,
    RobotWrapper,
)
from rcs.envs.creators import RCSHardwareEnvCreator
from rcs.hand.tilburg_hand import TilburgHand
from rcs_fr3 import hw
from rcs_fr3.envs import FR3HW
from rcs_fr3.utils import default_fr3_hw_gripper_cfg, default_fr3_hw_robot_cfg

import rcs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FrankIK(Kinematics):
    def __init__(self, global_solution: bool = False):
        Kinematics.__init__(self)
        self.global_solution = global_solution
        self.kin = FrankaKinematics(robot_type="fr3")

    def forward(self, q0: np.ndarray[tuple[typing.Literal[7]], np.dtype[np.float64]], tcp_offset: Pose) -> Pose:  # type: ignore
        print("forward called")
        return Pose(pose_matrix=self.kin.forward(q0, tcp_offset.pose_matrix()))

    def inverse(  # type: ignore
        self, pose: Pose, q0: np.ndarray[tuple[typing.Literal[7]], np.dtype[np.float64]], tcp_offset: Pose
    ) -> np.ndarray[tuple[typing.Literal[7]], np.dtype[np.float64]] | None:
        return self.kin.inverse(pose.pose_matrix(), q0, tcp_offset.pose_matrix(), global_solution=self.global_solution)


# FYI: this needs to be in global namespace to avoid auto garbage collection issues
# pybind11 3.x would avoid this but with smart_holder but we cannot update due to the subfiles issue yet
FastIK = FrankIK()


class RCSFR3EnvCreator(RCSHardwareEnvCreator):
    def __call__(  # type: ignore
        self,
        ip: str,
        control_mode: ControlMode,
        robot_cfg: hw.FR3Config,
        gripper_cfg: hw.FHConfig | rcs.hand.tilburg_hand.THConfig | None = None,
        camera_set: HardwareCameraSet | None = None,
        max_relative_movement: float | tuple[float, float] | None = None,
        relative_to: RelativeTo = RelativeTo.LAST_STEP,
    ) -> gym.Env:
        """
        Creates a hardware environment for the FR3 robot.

        Args:
            ip (str): IP address of the robot.
            control_mode (ControlMode): Control mode for the robot.
            robot_cfg (hw.FR3Config): Configuration for the FR3 robot.
            gripper_cfg (hw.FHConfig | None): Configuration for the gripper. If None, no gripper is used.
            camera_set (BaseHardwareCameraSet | None): Camera set to be used. If None, no cameras are used.
            max_relative_movement (float | tuple[float, float] | None): Maximum allowed movement. If float, it restricts
                translational movement in meters. If tuple, it restricts both translational (in meters) and rotational
                (in radians) movements. If None, no restriction is applied.
            relative_to (RelativeTo): Specifies whether the movement is relative to a configured origin or the last step.

        Returns:
            gym.Env: The configured hardware environment for the FR3 robot.
        """
        ik = rcs.common.Pin(
            robot_cfg.kinematic_model_path,
            robot_cfg.attachment_site,
            urdf=robot_cfg.kinematic_model_path.endswith(".urdf"),
        )
        # ik = FastIK
        # ik = rcs_robotics_library._core.rl.RoboticsLibraryIK(robot_cfg.kinematic_model_path)
        robot_cfg.ip = ip
        robot = hw.Franka(robot_cfg, ik)

        env: gym.Env = HardwareEnv()
        env = RobotWrapper(env, robot, control_mode)

        env = FR3HW(env)
        if isinstance(gripper_cfg, hw.FHConfig):
            gripper_cfg.ip = ip
            gripper = hw.FrankaHand(gripper_cfg)
            env = GripperWrapper(env, gripper)
        elif isinstance(gripper_cfg, rcs.hand.tilburg_hand.THConfig):
            hand = TilburgHand(gripper_cfg)
            env = HandWrapper(env, hand, binary=True)

        if camera_set is not None:
            camera_set.start()
            camera_set.wait_for_frames()
            logger.info("CameraSet started")
            env = CameraSetWrapper(env, camera_set)

        if relative_to != RelativeTo.NONE:
            env = RelativeActionSpace(env, max_mov=max_relative_movement, relative_to=relative_to)
        return CoverWrapper(env)


class RCSFR3MultiEnvCreator(RCSHardwareEnvCreator):
    def __call__(  # type: ignore
        self,
        name2ip: dict[str, str],
        control_mode: ControlMode,
        robot_cfg: hw.FR3Config,
        gripper_cfg: hw.FHConfig | None = None,
        camera_set: HardwareCameraSet | None = None,
        max_relative_movement: float | tuple[float, float] | None = None,
        relative_to: RelativeTo = RelativeTo.LAST_STEP,
        robot2world: dict[str, rcs.common.Pose] | None = None,
    ) -> gym.Env:

        ik = rcs.common.Pin(
            robot_cfg.kinematic_model_path,
            robot_cfg.attachment_site,
            urdf=robot_cfg.kinematic_model_path.endswith(".urdf"),
        )
        # ik = rcs_robotics_library._core.rl.RoboticsLibraryIK(robot_cfg.kinematic_model_path)

        robots: dict[str, hw.Franka] = {}
        for key, ip in name2ip.items():
            robot_cfg.ip = ip
            robots[key] = hw.Franka(robot_cfg, ik)

        envs: dict[str, gym.Env] = {}
        env: gym.Env
        for key, ip in name2ip.items():
            env = HardwareEnv()
            env = RobotWrapper(env, robots[key], control_mode)
            env = FR3HW(env)
            if gripper_cfg is not None:
                gripper_cfg.ip = ip
                gripper = hw.FrankaHand(gripper_cfg)
                env = GripperWrapper(env, gripper)

            if max_relative_movement is not None:
                env = RelativeActionSpace(env, max_mov=max_relative_movement, relative_to=relative_to)
            envs[key] = env

        env = MultiRobotWrapper(envs, robot2world)
        if camera_set is not None:
            camera_set.start()
            camera_set.wait_for_frames()
            logger.info("CameraSet started")
            env = CameraSetWrapper(env, camera_set)
        return CoverWrapper(env)


class RCSFR3DefaultEnvCreator(RCSHardwareEnvCreator):
    def __call__(  # type: ignore
        self,
        robot_ip: str,
        control_mode: ControlMode = ControlMode.CARTESIAN_TRPY,
        delta_actions: bool = True,
        camera_set: HardwareCameraSet | None = None,
        gripper: bool = True,
    ) -> gym.Env:
        return RCSFR3EnvCreator()(
            ip=robot_ip,
            camera_set=camera_set,
            control_mode=control_mode,
            robot_cfg=default_fr3_hw_robot_cfg(robot_ip),
            gripper_cfg=default_fr3_hw_gripper_cfg(robot_ip) if gripper else None,
            max_relative_movement=(0.2, np.deg2rad(45)) if delta_actions else None,
            relative_to=RelativeTo.LAST_STEP,
        )
