import logging

import gymnasium as gym
from rcs.camera.hw import HardwareCameraSet
from rcs.envs.base import (
    CameraSetWrapper,
    ControlMode,
    CoverWrapper,
    GripperWrapper,
    HardwareEnv,
    RelativeActionSpace,
    RelativeTo,
    RobotWrapper,
)
from rcs.envs.creators import RCSHardwareEnvCreator
from rcs_ur5e.hw import RobotiQGripper, RobotiQGripperConfig, UR5e, UR5eConfig

import rcs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RCSUR5eEnvCreator(RCSHardwareEnvCreator):
    def __call__(  # type: ignore
        self,
        robot_cfg: UR5eConfig,
        gripper_cfg: RobotiQGripperConfig | None = None,
        camera_set: HardwareCameraSet | None = None,
        control_mode: ControlMode = ControlMode.CARTESIAN_TRPY,
        max_relative_movement: float | tuple[float, float] | None = None,
        relative_to: RelativeTo = RelativeTo.LAST_STEP,
    ) -> gym.Env:
        ik = rcs.common.Pin(
            robot_cfg.kinematic_model_path,
            robot_cfg.attachment_site,
            urdf=robot_cfg.kinematic_model_path.endswith(".urdf"),
        )
        robot = UR5e(robot_cfg, ik)
        env: gym.Env = HardwareEnv()
        env = RobotWrapper(env, robot, control_mode)

        if gripper_cfg is not None:
            gripper = RobotiQGripper(cfg=gripper_cfg)
            # TODO: binary and other things of the wrappers should also be in the config
            env = GripperWrapper(env, gripper)

        if camera_set is not None:
            camera_set.start()
            camera_set.wait_for_frames()
            logger.info("CameraSet started")
            env = CameraSetWrapper(env, camera_set, include_depth=True)

        if max_relative_movement is not None:
            env = RelativeActionSpace(env, max_mov=max_relative_movement, relative_to=relative_to)
        return CoverWrapper(env)
