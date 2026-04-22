import logging
import typing
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import rcs.hand.tilburg_hand
from frankik import FrankaKinematics
from rcs._core.common import BaseCameraConfig, Gripper, GripperConfig, Kinematics, Pose
from rcs.camera.hw import HardwareCamera, HardwareCameraSet
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
from rcs.envs.scenes import RCSEnvCreator, WrapperConfig
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


@dataclass(kw_only=True)
class HardwareCameraCreatorConfig:
    camera_type_id: str
    camera_cfgs: dict[str, BaseCameraConfig]
    kwargs: dict[str, typing.Any] = field(default_factory=dict)


def _create_realsense_camera(cfg: HardwareCameraCreatorConfig) -> HardwareCamera:
    try:
        from rcs.camera.hw import CalibrationStrategy
        from rcs_realsense.calibration import FR3BaseArucoCalibration
        from rcs_realsense.camera import RealSenseCameraSet
    except ImportError as e:
        raise ImportError("RealSense camera support requires the `rcs_realsense` extension to be installed.") from e

    calibration_strategy = {
        name: typing.cast(CalibrationStrategy, FR3BaseArucoCalibration(name)) for name in cfg.camera_cfgs
    }
    return typing.cast(
        HardwareCamera,
        RealSenseCameraSet(cameras=cfg.camera_cfgs, calibration_strategy=calibration_strategy, **cfg.kwargs),
    )


def _create_digit_camera(cfg: HardwareCameraCreatorConfig) -> HardwareCamera:
    try:
        from rcs.camera.digit_cam import DigitCam
    except ImportError as e:
        raise ImportError("DIGIT camera support requires the `digit_interface` package to be installed.") from e

    return typing.cast(HardwareCamera, DigitCam(cameras=cfg.camera_cfgs))


HARDWARE_CAMERA_CREATORS: dict[str, typing.Callable[[HardwareCameraCreatorConfig], HardwareCamera]] = {
    "realsense": _create_realsense_camera,
    "digit": _create_digit_camera,
}


def _create_hardware_camera_set(
    camera_cfgs: dict[str, HardwareCameraCreatorConfig] | None,
) -> HardwareCameraSet | None:
    if camera_cfgs is None:
        return None
    cameras: list[HardwareCamera] = []
    for cfg in camera_cfgs.values():
        if cfg.camera_type_id not in HARDWARE_CAMERA_CREATORS:
            raise ValueError(f"Unknown hardware camera type id: {cfg.camera_type_id}")
        cameras.append(HARDWARE_CAMERA_CREATORS[cfg.camera_type_id](cfg))
    return HardwareCameraSet(cameras) if cameras else None


def _create_franka_gripper(cfg: GripperConfig) -> Gripper:
    if not isinstance(cfg, hw.FHConfig):
        raise TypeError(f"Expected FHConfig for franka gripper, got {type(cfg).__name__}")
    return hw.FrankaHand(cfg)


def _create_robotiq_gripper(cfg: GripperConfig) -> Gripper:
    try:
        from rcs_robotiq2f85.hw import RobotiQ2F85Gripper, RobotiQ2F85GripperConfig
    except ImportError as e:
        raise ImportError("Robotiq gripper support requires the `rcs_robotiq2f85` extension to be installed.") from e

    if not isinstance(cfg, RobotiQ2F85GripperConfig):
        raise TypeError(f"Expected RobotiQ2F85GripperConfig for robotiq gripper, got {type(cfg).__name__}")
    return RobotiQ2F85Gripper(cfg)


HARDWARE_GRIPPER_CREATORS: dict[str, typing.Callable[[GripperConfig], Gripper]] = {
    rcs.common.GripperType.FrankaHand.id: _create_franka_gripper,
    rcs.common.GripperType("Robotiq2F85").id: _create_robotiq_gripper,
}


@dataclass(kw_only=True)
class FR3HardwareEnvCreatorConfig:
    robot_cfg: hw.FR3Config
    control_mode: ControlMode
    gripper_cfg: GripperConfig | rcs.hand.tilburg_hand.THConfig | None = None
    camera_cfgs: dict[str, HardwareCameraCreatorConfig] | None = None
    max_relative_movement: float | tuple[float, float] | None = None
    relative_to: RelativeTo = RelativeTo.LAST_STEP
    wrapper_cfg: WrapperConfig = field(default_factory=WrapperConfig)


@dataclass(kw_only=True)
class FR3MultiHardwareEnvCreatorConfig:
    robot_cfgs: dict[str, hw.FR3Config]
    control_mode: ControlMode
    gripper_cfgs: dict[str, GripperConfig | rcs.hand.tilburg_hand.THConfig | None] | None = None
    camera_cfgs: dict[str, HardwareCameraCreatorConfig] | None = None
    max_relative_movement: float | tuple[float, float] | None = None
    relative_to: RelativeTo = RelativeTo.LAST_STEP
    robot_to_shared_base_frame: dict[str, rcs.common.Pose] | None = None
    wrapper_cfg: WrapperConfig = field(default_factory=WrapperConfig)


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


class RCSFR3ConfigEnvCreator(RCSEnvCreator[FR3HardwareEnvCreatorConfig]):
    def create_env(self, cfg: FR3HardwareEnvCreatorConfig) -> gym.Env:
        ik = rcs.common.Pin(
            cfg.robot_cfg.kinematic_model_path,
            cfg.robot_cfg.attachment_site,
            urdf=cfg.robot_cfg.kinematic_model_path.endswith(".urdf"),
        )
        robot = hw.Franka(cfg.robot_cfg, ik)

        env: gym.Env = HardwareEnv()
        env = RobotWrapper(env, robot, cfg.control_mode, home_on_reset=cfg.wrapper_cfg.home_on_reset)
        env = FR3HW(env)
        if isinstance(cfg.gripper_cfg, rcs.hand.tilburg_hand.THConfig):
            hand = TilburgHand(cfg.gripper_cfg)
            env = HandWrapper(env, hand, binary=cfg.wrapper_cfg.binary_gripper)
        elif cfg.gripper_cfg is not None:
            gripper_type_id = cfg.gripper_cfg.gripper_type.id
            if gripper_type_id not in HARDWARE_GRIPPER_CREATORS:
                raise ValueError(f"Unknown hardware gripper type id: {gripper_type_id}")
            gripper = HARDWARE_GRIPPER_CREATORS[gripper_type_id](cfg.gripper_cfg)
            env = GripperWrapper(env, gripper, binary=cfg.wrapper_cfg.binary_gripper)

        camera_set = _create_hardware_camera_set(cfg.camera_cfgs)
        if camera_set is not None:
            camera_set.start()
            camera_set.wait_for_frames()
            logger.info("CameraSet started")
            env = CameraSetWrapper(env, camera_set)

        if cfg.relative_to != RelativeTo.NONE:
            env = RelativeActionSpace(env, max_mov=cfg.max_relative_movement, relative_to=cfg.relative_to)
        return CoverWrapper(env)

    def config(self) -> FR3HardwareEnvCreatorConfig:
        raise NotImplementedError("Implement config() in a subclass or pass `cfg=` explicitly.")


class RCSFR3MultiConfigEnvCreator(RCSEnvCreator[FR3MultiHardwareEnvCreatorConfig]):
    def create_env(self, cfg: FR3MultiHardwareEnvCreatorConfig) -> gym.Env:
        envs: dict[str, gym.Env] = {}
        for robot_name, robot_cfg in cfg.robot_cfgs.items():
            envs[robot_name] = RCSFR3ConfigEnvCreator().create_env(
                FR3HardwareEnvCreatorConfig(
                    robot_cfg=robot_cfg,
                    control_mode=cfg.control_mode,
                    gripper_cfg=cfg.gripper_cfgs[robot_name] if cfg.gripper_cfgs is not None else None,
                    camera_cfgs=None,
                    max_relative_movement=cfg.max_relative_movement,
                    relative_to=cfg.relative_to,
                    wrapper_cfg=cfg.wrapper_cfg,
                )
            )

        env: gym.Env = MultiRobotWrapper(envs, cfg.robot_to_shared_base_frame)
        camera_set = _create_hardware_camera_set(cfg.camera_cfgs)
        if camera_set is not None:
            camera_set.start()
            camera_set.wait_for_frames()
            logger.info("CameraSet started")
            env = CameraSetWrapper(env, camera_set)
        return CoverWrapper(env)

    def config(self) -> FR3MultiHardwareEnvCreatorConfig:
        raise NotImplementedError("Implement config() in a subclass or pass `cfg=` explicitly.")
