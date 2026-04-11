import logging
from os import PathLike

import gymnasium as gym
import rcs.hand.tilburg_hand
from rcs.camera.hw import HardwareCameraSet
from rcs.envs.base import (
    CameraSetWrapper,
    ControlMode,
    GripperWrapper,
    HandWrapper,
    HardwareEnv,
    MultiRobotWrapper,
    RelativeActionSpace,
    RelativeTo,
    RobotWrapper,
    CoverWrapper,
)
from rcs.envs.creators import RCSHardwareEnvCreator
from rcs.hand.tilburg_hand import TilburgHand
from rcs_panda import hw
from rcs_panda.envs import PandaHW

import rcs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RCSPandaEnvCreator(RCSHardwareEnvCreator):
    def __call__(  # type: ignore
        self,
        ip: str,
        control_mode: ControlMode,
        robot_cfg: hw.PandaConfig,
        collision_guard: str | PathLike | None = None,
        gripper_cfg: hw.FHConfig | rcs.hand.tilburg_hand.THConfig | None = None,
        camera_set: HardwareCameraSet | None = None,
        max_relative_movement: float | tuple[float, float] | None = None,
        relative_to: RelativeTo = RelativeTo.LAST_STEP,
    ) -> gym.Env:
        """
        Creates a hardware environment for the Panda robot.

        Args:
            ip (str): IP address of the robot.
            control_mode (ControlMode): Control mode for the robot.
            robot_cfg (hw.PandaConfig): Configuration for the Panda robot.
            collision_guard (str | PathLike | None): Key to a built-in scene
            robot_cfg (hw.PandaConfig): Configuration for the Panda robot.
            collision_guard (str | PathLike | None): Key to a scene (requires UTN compatible scene package to be present)
                or the path to a mujoco scene for collision guarding. If None, collision guarding is not used.
            gripper_cfg (hw.FHConfig | None): Configuration for the gripper. If None, no gripper is used.
            camera_set (BaseHardwareCameraSet | None): Camera set to be used. If None, no cameras are used.
            max_relative_movement (float | tuple[float, float] | None): Maximum allowed movement. If float, it restricts
                translational movement in meters. If tuple, it restricts both translational (in meters) and rotational
                (in radians) movements. If None, no restriction is applied.
            relative_to (RelativeTo): Specifies whether the movement is relative to a configured origin or the last step.

        Returns:
            gym.Env: The configured hardware environment for the Panda robot.
        """
        ik = rcs.common.Pin(
            robot_cfg.kinematic_model_path,
            robot_cfg.attachment_site,
            urdf=robot_cfg.kinematic_model_path.endswith(".urdf"),
        )
        # ik = rcs_robotics_library._core.rl.RoboticsLibraryIK(robot_cfg.kinematic_model_path)

        robot = hw.Franka(ip, ik)
        robot.set_config(robot_cfg)

        env = HardwareEnv()
        env = RobotWrapper(
            env,
            robot,
            ControlMode.JOINTS if collision_guard is not None else control_mode,
            home_on_reset=True,
        )

        env = PandaHW(env)
        if isinstance(gripper_cfg, hw.FHConfig):
            gripper = hw.FrankaHand(ip, gripper_cfg)
            env = GripperWrapper(env, gripper, binary=True)
        elif isinstance(gripper_cfg, rcs.hand.tilburg_hand.THConfig):
            hand = TilburgHand(gripper_cfg)
            env = HandWrapper(env, hand, binary=True)

        if camera_set is not None:
            camera_set.start()
            camera_set.wait_for_frames()
            logger.info("CameraSet started")
            env = CameraSetWrapper(env, camera_set)

        # if collision_guard is not None:
        #     assert urdf_path is not None
        #     env = CollisionGuard.env_from_xml_paths(
        #         env,
        #         str(rcs.scenes.get(str(collision_guard), collision_guard)),
        #         str(urdf_path),
        #         gripper=True,
        #         check_home_collision=False,
        #         control_mode=control_mode,
        #         tcp_offset=rcs.common.Pose(rcs.common.FrankaHandTCPOffset()),
        #         sim_gui=True,
        #         truncate_on_collision=False,
        #     )
        if max_relative_movement is not None:
            env = RelativeActionSpace(env, max_mov=max_relative_movement, relative_to=relative_to)
        env = CoverWrapper(env)

        return env


class RCSPandaMultiEnvCreator(RCSHardwareEnvCreator):
    def __call__(  # type: ignore
        self,
        ips: list[str],
        control_mode: ControlMode,
        robot_cfg: hw.PandaConfig,
        gripper_cfg: hw.FHConfig | None = None,
        camera_set: HardwareCameraSet | None = None,
        max_relative_movement: float | tuple[float, float] | None = None,
        relative_to: RelativeTo = RelativeTo.LAST_STEP,
    ) -> gym.Env:

        ik = rcs.common.Pin(
            robot_cfg.kinematic_model_path,
            robot_cfg.attachment_site,
            urdf=robot_cfg.kinematic_model_path.endswith(".urdf"),
        )
        # ik = rcs_robotics_library._core.rl.RoboticsLibraryIK(robot_cfg.kinematic_model_path)

        robots: dict[str, hw.Franka] = {}
        for ip in ips:
            robots[ip] = hw.Franka(ip, ik)
            robots[ip].set_config(robot_cfg)

        envs = {}
        for ip in ips:
            env = HardwareEnv()
            env = RobotWrapper(env, robots[ip], control_mode)
            env = PandaHW(env)
            if gripper_cfg is not None:
                gripper = hw.FrankaHand(ip, gripper_cfg)
                env = GripperWrapper(env, gripper, binary=True)

            if max_relative_movement is not None:
                env = RelativeActionSpace(env, max_mov=max_relative_movement, relative_to=relative_to)
            envs[ip] = env

        env = MultiRobotWrapper(envs)
        if camera_set is not None:
            camera_set.start()
            camera_set.wait_for_frames()
            logger.info("CameraSet started")
            env = CameraSetWrapper(env, camera_set)
        env = CoverWrapper(env)
        return env
