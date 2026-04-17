import copy
import time
import typing
from abc import ABC
from dataclasses import dataclass, field
from os import PathLike

import gymnasium as gym
import numpy as np
from rcs._core.common import FrankaHandTCPOffset, GripperType, Pose, RobotType
from rcs._core.sim import (
    CameraType,
    SimCameraConfig,
    SimConfig,
    SimGripperConfig,
    SimRobotConfig,
)
from rcs.camera.interface import BaseCameraSet
from rcs.camera.sim import SimCameraSet
from rcs.envs.base import (
    CameraSetWrapper,
    ControlMode,
    CoverWrapper,
    GripperWrapper,
    MultiRobotWrapper,
    RelativeActionSpace,
    RelativeTo,
    RobotWrapper,
    SimEnv,
)
from rcs.envs.sim import GripperWrapperSim, RobotSimWrapper
from rcs.sim.composer import ModelComposer
from rcs.sim.sim import Sim

import rcs
from rcs import CAMERA_PATHS, GRIPPER_PATHS, OBJECT_PATHS, SCENE_PATHS


class BaseSceneConfig:
    pass


class BaseScene(ABC):
    config_type: typing.Type

    def create(self) -> gym.Env:
        raise NotImplementedError

    def load_config(self, key: str) -> BaseSceneConfig:
        # TODO: load type form yaml with type checking
        raise NotImplementedError


@dataclass(kw_only=True)
class WrapperConfig:
    binary_gripper: bool = True
    home_on_reset: bool = True


@dataclass(kw_only=True)
class CameraAdderConfig:
    xml_path: str | None = None
    """path to where the camera xml is, should have world frame, geoms and one camera defined in it, 
    if None, a camera will be added according to the camera_cfgs provided in the scene config and the other parameters in this config"""
    fovy: float = 60.0
    """only used if no xml_path is provided and a camera is added to the model"""
    offset: rcs.common.Pose = field(default_factory=rcs.common.Pose)
    attachment_site: str = "attachment_site"
    """only used when a robot name to attach the camera to is provided"""
    robot_name: str | None = None


@dataclass(kw_only=True)
class SimSceneConfig(BaseSceneConfig):
    robot_cfgs: dict[str, SimRobotConfig]
    sim_cfg: SimConfig
    control_mode: ControlMode
    task: str | None = None
    scene: str = SCENE_PATHS["empty_world"]
    """path or key to load the mujoco scene, e.g. from SCENE_PATHS, will be passed to load_scene()"""
    gripper_cfgs: dict[str, SimGripperConfig] | None = None
    camera_cfgs: dict[str, SimCameraConfig] | None = None
    max_relative_movement: float | tuple[float, float] | None = None
    relative_to: RelativeTo = RelativeTo.LAST_STEP
    robot_to_shared_base_frame: dict[str, rcs.common.Pose] | None = None
    """shared base frame is a common reference frame for all robots in the scene and the origin for all actions and observations, e.g. the middle of franka duo
    thus this transformation defines the offset of each robot's base to this shared base frame."""
    add_gravcomp: bool = False
    wrapper_cfg: WrapperConfig = field(default_factory=WrapperConfig)
    open_gui_on_create: bool = True
    shared_base_frame_to_root_frame: rcs.common.Pose = field(default_factory=rcs.common.Pose)
    """shared base frame is a common reference frame for all robots in the scene and the origin for all actions and observations, e.g. the middle of franka duo
    root_frame defines the origin where the parent of the robot assets are placed"""
    root_frame_to_world: rcs.common.Pose = field(default_factory=rcs.common.Pose)
    """root_frame defines the origin where the parent of the robot assets are placed
    world frame is the mujoco world frame"""
    alternative_combined_robot_mjcf: str | None = None
    """If you dont want to compose your scene with ModelComposer API,
    you can directly provide a combined robot mjcf with correct prefixing and the composer will add it as is without modifications.
    Prefixes need to as follows: robot{robot_name} where robot_name is the key in robot_cfgs, e.g. robot0, robot1, etc.
    root_frame_to_world will be used to place the robot in the world and
    shared_base_frame_to_root_frame will be used to determine the origin of the robot's action space."""
    world_frame_objects: dict[str, tuple[str, rcs.common.Pose]] | None = None
    """dict of object_id to tuple of (object_xml, object2world), will be added to the scene, object2world is the pose of the object in the mujoco world frame"""
    root_frame_objects: dict[str, tuple[str, rcs.common.Pose]] | None = None
    """dict of object_id to tuple of (object_xml, object2root_frame), will be added to the robot object2root_frame is the pose of the object in the root frame of the robot,
    which is defined by shared_base_frame_to_root_frame, object_id must be unique across all objects and robots in the scene"""
    robot_frame_objects: dict[str, dict[str, tuple[str, rcs.common.Pose]]] | None = None
    """dict of robot_name to dict of object_id to tuple of (object_xml, object2robot_frame), objects will be attached to that robot's attachment site,
    object2robot_frame is the pose of the object in the robot attachment-site frame, object_id must be unique across all objects and robots in the scene"""
    camera_adds: dict[str, CameraAdderConfig] | None = None
    """dict of camera_name to CameraAdderConfig, cameras will be added to the scene according to the config, camera_name must be unique across all cameras in the scene"""
    gripper_offsets: dict[str, rcs.common.Pose] | None = None
    """optional offsets for the gripper from the robot's attachment site"""


class SimScene(BaseScene):

    def __init__(
        self,
        config_key: str,
        robot_prefix_template: str = "robot{robot_name}_",
        gripper_prefix_template: str = "gripper{robot_name}_",
    ) -> None:
        super().__init__()
        self.config_key = config_key
        self._cfg = self.load_config(config_key)
        self.robots_names = list(self._cfg.robot_cfgs.keys())
        self.lead_robot_name = next(iter(self.robots_names))
        self.robot_prefix_template = robot_prefix_template
        self.gripper_prefix_template = gripper_prefix_template
        self._prefixed_cfg = self._get_prefixed_cfg()

    @property
    def config(self) -> SimSceneConfig:
        """Returns the configuration for the scene (prefixes applied)."""
        return self._prefixed_cfg

    @property
    def kinematics_cfg(self) -> dict[str, tuple[str, str]]:
        """
        Returns the kinematic configuration for each robot in the scene.
        Returns:
            dict[str, tuple[str, str]]: A dictionary mapping robot names to a tuple of (kinematic_model_path, attachment_site).
        """

        return {
            robot_name: (cfg.kinematic_model_path, cfg.attachment_site)
            for robot_name, cfg in self._cfg.robot_cfgs.items()
        }

    def load_config(self, key: str) -> SimSceneConfig:
        raise NotImplementedError

    def load_scene(self) -> ModelComposer | str | PathLike:
        """Loads the mujoco scene from the given config

        Returns:
            ModelComposer | str | PathLike: path to scene file (mjcf or mjb), or composer object
        """
        composer = ModelComposer(
            model_name="RCS Scene",
            add_gravcomp=self._cfg.add_gravcomp,
        )
        composer.load_base_scene(self._cfg.scene)

        self.add_task_mujoco(self._cfg.task, composer)
        if self._cfg.alternative_combined_robot_mjcf is not None:
            # robot is in one mjcf
            self.add_robot_mujoco(
                composer,
                robot_name=self.lead_robot_name,
                robot_xml=self._cfg.alternative_combined_robot_mjcf,
                robot2world=self._cfg.root_frame_to_world,
            )
        else:
            # robot is composed by composer
            for robot_name in self.robots_names:
                robot_to_shared_frame = (
                    self._cfg.robot_to_shared_base_frame[robot_name]
                    if self._cfg.robot_to_shared_base_frame is not None
                    else rcs.common.Pose()
                )
                robot2world = (
                    robot_to_shared_frame * self._cfg.shared_base_frame_to_root_frame * self._cfg.root_frame_to_world
                )
                self.add_robot_mujoco(
                    composer, robot_name, self._cfg.robot_cfgs[robot_name].kinematic_model_path, robot2world
                )

        if self._cfg.gripper_cfgs is not None:
            # add gripper to each robot
            for robot_name in self.robots_names:
                gripper_xml = GRIPPER_PATHS[self._cfg.gripper_cfgs[robot_name].gripper_type]
                self.add_gripper_mujoco(
                    composer,
                    robot_name,
                    gripper_xml,
                    self._cfg.robot_cfgs[robot_name].attachment_site,
                )

        # add robot-specific objects
        if self._cfg.root_frame_objects is not None:
            for object_id, (object_xml, object2root_frame) in self._cfg.root_frame_objects.items():
                object2world = object2root_frame * self._cfg.root_frame_to_world
                self.add_object_mujoco(composer, object_id, object_xml, object2world)
        # add external objects
        if self._cfg.world_frame_objects is not None:
            for object_id, (object_xml, object2world) in self._cfg.world_frame_objects.items():
                self.add_object_mujoco(composer, object_id, object_xml, object2world)
        # add robot-frame objects
        if self._cfg.robot_frame_objects is not None:
            for robot_name, robot_objects in self._cfg.robot_frame_objects.items():
                attachment_site = self._cfg.robot_cfgs[robot_name].attachment_site
                for object_id, (object_xml, object2robot_frame) in robot_objects.items():
                    self.add_object_robot_frame_mujoco(
                        composer,
                        robot_name=robot_name,
                        object_id=object_id,
                        object_xml=object_xml,
                        object2robot_frame=object2robot_frame,
                        attachment_site=attachment_site,
                    )

        # camera adds
        if self._cfg.camera_adds is not None:
            for camera_name, camera_add_cfg in self._cfg.camera_adds.items():
                camera_pose = (
                    camera_add_cfg.offset * self._cfg.root_frame_to_world
                    if camera_add_cfg.robot_name is None
                    else camera_add_cfg.offset
                )
                if camera_add_cfg.xml_path is not None:
                    composer.add_camera_xml(
                        xml_path=camera_add_cfg.xml_path,
                        name=camera_name,
                        pose=camera_pose,
                        robot_prefix=(
                            self.robot_prefix_template.format(robot_name=camera_add_cfg.robot_name)
                            if camera_add_cfg.robot_name is not None
                            else None
                        ),
                        attachment_site_name=(
                            self._cfg.robot_cfgs[camera_add_cfg.robot_name].attachment_site
                            if camera_add_cfg.robot_name is not None
                            else camera_add_cfg.attachment_site
                        ),
                    )
                    continue

                assert self._cfg.camera_cfgs is not None, "Camera configs must be provided to add cameras."
                assert (
                    camera_name in self._cfg.camera_cfgs
                ), f"Camera config for camera {camera_name} must be provided to add camera {camera_name} to the scene"
                composer.add_camera(
                    resolution=(
                        self._cfg.camera_cfgs[camera_name].resolution_width,
                        self._cfg.camera_cfgs[camera_name].resolution_height,
                    ),
                    fovy=camera_add_cfg.fovy,
                    name=camera_name,
                    pose=camera_pose,
                    robot_prefix=(
                        self.robot_prefix_template.format(robot_name=camera_add_cfg.robot_name)
                        if camera_add_cfg.robot_name is not None
                        else None
                    ),
                    attachment_site_name=(
                        self._cfg.robot_cfgs[camera_add_cfg.robot_name].attachment_site
                        if camera_add_cfg.robot_name is not None
                        else camera_add_cfg.attachment_site
                    ),
                )

        return composer

    def add_task_mujoco(self, key: str | None, composer: ModelComposer):
        """Add task-specific elements to the Mujoco scene."""

    def add_task_env(self, key: str | None, env: gym.Env, simulation: Sim) -> gym.Env:
        """Add task-specific wrappers to the environment."""
        return env

    def add_object_mujoco(
        self, composer: ModelComposer, object_id: str, object_xml: str, object2world: rcs.common.Pose
    ):
        """Add an object to the Mujoco scene."""
        composer.add_object_world_frame(
            object_xml,
            object_prefix=object_id + "_",
            pose=object2world,
        )

    def add_object_robot_frame_mujoco(
        self,
        composer: ModelComposer,
        robot_name: str,
        object_id: str,
        object_xml: str,
        object2robot_frame: rcs.common.Pose,
        attachment_site: str,
    ):
        """Add an object to the Mujoco scene in a robot attachment-site frame."""
        composer.add_object_robot_frame(
            xml_path=object_xml,
            robot_prefix=self.robot_prefix_template.format(robot_name=robot_name),
            object_prefix=object_id + "_",
            attachment_site_name=attachment_site,
            pose=object2robot_frame,
        )

    def add_robot_mujoco(
        self,
        composer: ModelComposer,
        robot_name: str,
        robot_xml: str,
        robot2world: rcs.common.Pose | None = None,
    ):
        if robot2world is None:
            robot2world = rcs.common.Pose()
        robot_prefix = self.robot_prefix_template.format(robot_name=robot_name)
        composer.add_robot(
            robot_xml,
            robot_prefix,
            pose=robot2world,
        )

    def add_robot_env(self, robot_name: str, env: gym.Env, simulation: Sim, ik: rcs.common.Kinematics):
        # rcs wrapper composition
        robot = rcs.sim.SimRobot(sim=simulation, ik=ik, cfg=self.config.robot_cfgs[robot_name])
        env = RobotWrapper(env, robot, self.config.control_mode, home_on_reset=self.config.wrapper_cfg.home_on_reset)
        return RobotSimWrapper(env)

    def add_gripper_mujoco(self, composer: ModelComposer, robot_name: str, gripper_xml: str, attachment_site: str):
        # mujoco scene composition
        assert self._cfg.gripper_cfgs is not None, "Gripper configs must be provided to add grippers."
        gripper_offset = (
            self._cfg.gripper_offsets[robot_name]
            if self._cfg.gripper_offsets is not None and robot_name in self._cfg.gripper_offsets
            else rcs.common.Pose()
        )
        composer.add_gripper(
            xml_path=gripper_xml,
            gripper_prefix=self.gripper_prefix_template.format(robot_name=robot_name),
            robot_prefix=self.robot_prefix_template.format(robot_name=robot_name),
            attachment_site_name=attachment_site,
            pose=gripper_offset,
        )

    def add_gripper_env(self, robot_name: str, simulation: Sim, env: gym.Env):
        # rcs wrapper composition
        assert self._cfg.gripper_cfgs is not None, "Gripper configs must be provided to add grippers."
        gripper = rcs.sim.SimGripper(simulation, self.config.gripper_cfgs[robot_name])
        env = GripperWrapper(env, gripper, binary=self.config.wrapper_cfg.binary_gripper)
        return GripperWrapperSim(env)

    def _get_prefixed_cfg(self) -> SimSceneConfig:
        cfg = copy.deepcopy(self._cfg)
        for robot_name in self.robots_names:
            cfg.robot_cfgs[robot_name].add_prefix(self.robot_prefix_template.format(robot_name=robot_name))
            if cfg.gripper_cfgs is not None:
                cfg.gripper_cfgs[robot_name].add_prefix(self.gripper_prefix_template.format(robot_name=robot_name))
        return cfg

    def create(self) -> gym.Env:

        mjcf = self.load_scene()
        # save the composed scene for debugging
        mjcf.save_mjcf("scene.xml")
        # you can also apply a scene path e.g. the saved one
        # mjcf = "scene.xml"

        simulation = Sim(mjcf, self.config.sim_cfg)

        envs: dict[str, gym.Env] = {}
        env: gym.Env
        for robot_name in self.robots_names:
            env = SimEnv(simulation)
            kinematic_model_path, attachment_site = self.kinematics_cfg[robot_name]
            ik = rcs.common.Pin(
                kinematic_model_path,
                attachment_site,
            )
            # ik = rcs_robotics_library._core.rl.RoboticsLibraryIK(cfg.robot_cfgs[lead_robot_name].kinematic_model_path)

            env = self.add_robot_env(robot_name, env, simulation, ik)
            if self.config.gripper_cfgs is not None:
                env = self.add_gripper_env(robot_name, simulation, env)

            if self.config.relative_to != RelativeTo.NONE:
                env = RelativeActionSpace(
                    env, max_mov=self.config.max_relative_movement, relative_to=self.config.relative_to
                )
            envs[robot_name] = env

        env = MultiRobotWrapper(envs, self.config.robot_to_shared_base_frame)
        if self.config.camera_cfgs is not None:
            camera_set = typing.cast(
                BaseCameraSet,
                SimCameraSet(simulation, self.config.camera_cfgs, physical_units=True, render_on_demand=True),
            )
            env = CameraSetWrapper(env, camera_set, include_depth=True)
        env = self.add_task_env(self.config.task, env, simulation)
        if self.config.open_gui_on_create:
            env.get_wrapper_attr("sim").open_gui()
        return CoverWrapper(env)


class EmptyWorldFR3(SimScene):

    def __init__(self):
        super().__init__("", robot_prefix_template="robot", gripper_prefix_template="gripper")

    def load_config(self, key: str) -> SimSceneConfig:
        robot_cfg = SimRobotConfig(
            robot_type=RobotType.FR3,
            tcp_offset=rcs.common.Pose(pose_matrix=FrankaHandTCPOffset()),
            attachment_site=rcs.ROBOTS[RobotType.FR3].attachment_site,
            kinematic_model_path=rcs.ROBOTS[RobotType.FR3].mjcf_model_path,
            joint_rotational_tolerance=0.05 * (np.pi / 180.0),
            seconds_between_callbacks=0.1,
            trajectory_trace=False,
            arm_collision_geoms=[
                "fr3_link0_collision",
                "fr3_link1_collision",
                "fr3_link2_collision",
                "fr3_link3_collision",
                "fr3_link4_collision",
                "fr3_link5_collision",
                "fr3_link6_collision",
                "fr3_link7_collision",
            ],
            joints=[
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
            ],
            actuators=[
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
            ],
            base="base",
            dof=rcs.ROBOTS[RobotType.FR3].dof,
            joint_limits=rcs.ROBOTS[RobotType.FR3].joint_limits,
            q_home=rcs.ROBOTS[RobotType.FR3].q_home,
        )

        robot_cfgs: dict[str, SimRobotConfig] = {"robot": robot_cfg}
        sim_cfg: SimConfig = SimConfig(async_control=False, realtime=True, frequency=1, max_convergence_steps=500)

        control_mode: ControlMode = ControlMode.CARTESIAN_TQuat
        task: str | None = None
        scene: str = SCENE_PATHS["empty_world"]
        gripper_cfg = SimGripperConfig(
            epsilon_inner=0.005,
            epsilon_outer=0.005,
            seconds_between_callbacks=0.1,
            ignored_collision_geoms=[],
            collision_geoms=["hand_c", "finger_0_left", "finger_0_right"],
            collision_geoms_fingers=["finger_0_left", "finger_0_right"],
            joints=["finger_joint1", "finger_joint2"],
            max_joint_width=0.04,
            min_joint_width=0.0,
            actuator="hand_actuator",
            max_actuator_width=255.0,
            min_actuator_width=0.0,
            gripper_type=GripperType.FrankaHand,
        )
        gripper_cfgs: dict[str, SimGripperConfig] = {"robot": gripper_cfg}
        camera_cfgs: dict[str, SimCameraConfig] | None = {
            "bird_eye": SimCameraConfig(
                identifier="bird_eye",
                type=CameraType.fixed,
                resolution_width=1280,
                resolution_height=720,
                frame_rate=30,
            ),
            "wrist": SimCameraConfig(
                identifier="wrist",
                type=CameraType.fixed,
                resolution_width=1280,
                resolution_height=720,
                frame_rate=30,
            ),
        }
        max_relative_movement: float | tuple[float, float] | None = None
        relative_to: RelativeTo = RelativeTo.LAST_STEP
        robot_to_shared_base_frame: dict[str, rcs.common.Pose] | None = {"robot": rcs.common.Pose()}
        wrapper_cfg: WrapperConfig = WrapperConfig(binary_gripper=True, home_on_reset=True)
        open_gui_on_create = True
        add_gravcomp = True
        shared_base_frame_to_root_frame = rcs.common.Pose()
        root_frame_to_world = rcs.common.Pose()
        alternative_combined_robot_mjcf: str | None = None
        world_frame_objects: dict[str, tuple[str, rcs.common.Pose]] | None = None
        root_frame_objects: dict[str, tuple[str, rcs.common.Pose]] | None = None
        add_camera_adds: dict[str, CameraAdderConfig] | None = {
            "bird_eye": CameraAdderConfig(
                fovy=60.0,
                offset=rcs.common.Pose(
                    translation=[0.271, -0.000, 2.080], quaternion=[0.0060, -0.0060, -0.7067, 0.7074]
                ),
            ),
            "wrist": CameraAdderConfig(
                fovy=60.0,
                offset=rcs.common.Pose(translation=[0, 0, 0], quaternion=[0, 0, -0.3826834, 0.9238795])
                * rcs.common.Pose(translation=[0.062, -0.009, 0.05245], rpy_vector=[0, np.pi, -np.pi / 2]),
                robot_name="robot",
            ),
        }
        gripper_offsets: dict[str, rcs.common.Pose] | None = {
            "robot": rcs.common.Pose(rotation=FrankaHandTCPOffset()[:3, :3], translation=[0, 0, 0])
        }
        return SimSceneConfig(
            robot_cfgs=robot_cfgs,
            sim_cfg=sim_cfg,
            control_mode=control_mode,
            task=task,
            scene=scene,
            gripper_cfgs=gripper_cfgs,
            camera_cfgs=camera_cfgs,
            max_relative_movement=max_relative_movement,
            relative_to=relative_to,
            robot_to_shared_base_frame=robot_to_shared_base_frame,
            wrapper_cfg=wrapper_cfg,
            open_gui_on_create=open_gui_on_create,
            add_gravcomp=add_gravcomp,
            shared_base_frame_to_root_frame=shared_base_frame_to_root_frame,
            root_frame_to_world=root_frame_to_world,
            alternative_combined_robot_mjcf=alternative_combined_robot_mjcf,
            world_frame_objects=world_frame_objects,
            root_frame_objects=root_frame_objects,
            camera_adds=add_camera_adds,
            gripper_offsets=gripper_offsets,
        )


class EmptyWorldFR3Duo(SimScene):

    gripper_mesh_quaternion_offset = [0, 0, 0.7071068, 0.7071068]

    def __init__(self):
        super().__init__(
            "", robot_prefix_template="robot_{robot_name}_", gripper_prefix_template="gripper_{robot_name}_"
        )

    def load_config(self, key: str) -> SimSceneConfig:
        robot_cfg = SimRobotConfig(
            robot_type=RobotType.FR3,
            attachment_site=rcs.ROBOTS[RobotType.FR3].attachment_site,
            kinematic_model_path=rcs.ROBOTS[RobotType.FR3].mjcf_model_path,
            joint_rotational_tolerance=0.05 * (np.pi / 180.0),
            seconds_between_callbacks=0.1,
            trajectory_trace=False,
            arm_collision_geoms=[
                "fr3_link0_collision",
                "fr3_link1_collision",
                "fr3_link2_collision",
                "fr3_link3_collision",
                "fr3_link4_collision",
                "fr3_link5_collision",
                "fr3_link6_collision",
                "fr3_link7_collision",
            ],
            joints=[
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
            ],
            actuators=[
                "fr3_joint1",
                "fr3_joint2",
                "fr3_joint3",
                "fr3_joint4",
                "fr3_joint5",
                "fr3_joint6",
                "fr3_joint7",
            ],
            base="base",
            dof=rcs.ROBOTS[RobotType.FR3].dof,
            joint_limits=rcs.ROBOTS[RobotType.FR3].joint_limits,
            q_home=rcs.ROBOTS[RobotType.FR3].q_home,
        )
        robot_cfg_right = copy.deepcopy(robot_cfg)

        robot_cfgs: dict[str, SimRobotConfig] = {"left": robot_cfg, "right": robot_cfg_right}
        sim_cfg: SimConfig = SimConfig(async_control=False, realtime=True, frequency=1, max_convergence_steps=500)

        control_mode: ControlMode = ControlMode.CARTESIAN_TQuat
        task: str | None = None
        scene: str = SCENE_PATHS["empty_world"]
        gripper_cfg = SimGripperConfig(
            epsilon_inner=0.005,
            epsilon_outer=0.005,
            seconds_between_callbacks=0.1,
            ignored_collision_geoms=[],
            collision_geoms=[],
            collision_geoms_fingers=[],
            joints=["right_driver_joint", "left_driver_joint"],
            max_joint_width=0.005,
            min_joint_width=1.0,
            actuator="fingers_actuator",
            max_actuator_width=0,
            min_actuator_width=255,
            gripper_type=GripperType("Robotiq2F85"),
        )

        gripper_cfg_right = copy.deepcopy(gripper_cfg)
        gripper_cfgs: dict[str, SimGripperConfig] = {"left": gripper_cfg, "right": gripper_cfg_right}

        camera_cfgs: dict[str, SimCameraConfig] | None = {
            "head": SimCameraConfig(
                identifier="head",
                type=CameraType.fixed,
                resolution_width=1280,
                resolution_height=720,
                frame_rate=30,
            ),
            "left_wrist": SimCameraConfig(
                identifier="left_wrist",
                type=CameraType.fixed,
                resolution_width=1280,
                resolution_height=720,
                frame_rate=30,
            ),
            "right_wrist": SimCameraConfig(
                identifier="right_wrist",
                type=CameraType.fixed,
                resolution_width=1280,
                resolution_height=720,
                frame_rate=30,
            ),
        }
        max_relative_movement: float | tuple[float, float] | None = None
        relative_to: RelativeTo = RelativeTo.LAST_STEP
        robot_to_shared_base_frame: dict[str, rcs.common.Pose] | None = {
            "left": Pose(translation=[0, 0.05018, 0.342], quaternion=[-0.436978, 0.0225312, -0.243326, 0.865641]),
            "right": Pose(translation=[0, -0.05018, 0.342], quaternion=[0.436978, 0.0225312, 0.243326, 0.865641]),
        }
        wrapper_cfg: WrapperConfig = WrapperConfig(binary_gripper=True, home_on_reset=True)
        open_gui_on_create = True
        add_gravcomp = True
        shared_base_frame_to_root_frame = rcs.common.Pose()
        root_frame_to_world = rcs.common.Pose()
        alternative_combined_robot_mjcf: str | None = None
        world_frame_objects: dict[str, tuple[str, rcs.common.Pose]] | None = None
        root_frame_objects: dict[str, tuple[str, rcs.common.Pose]] | None = {
            "duo_mount": (OBJECT_PATHS["fr3_duo_mount"], Pose(translation=[0, 0, 0.342], quaternion=[0, 0, 0, 1]))
        }
        robot_frame_objects: dict[str, dict[str, tuple[str, rcs.common.Pose]]] | None = {
            "left": {
                "left_d405_mount": (
                    OBJECT_PATHS["robotiq_d405_mount"],
                    Pose(translation=[0, 0, 0], quaternion=self.gripper_mesh_quaternion_offset),
                )
            },
            "right": {
                "right_d405_mount": (
                    OBJECT_PATHS["robotiq_d405_mount"],
                    Pose(translation=[0, 0, 0], quaternion=self.gripper_mesh_quaternion_offset),
                )
            },
        }
        add_camera_adds: dict[str, CameraAdderConfig] | None = {
            "head": CameraAdderConfig(
                xml_path=CAMERA_PATHS["zed_mini"],
                fovy=60.0,
                offset=rcs.common.Pose(
                    # if duo_mount is spawned at [0, 0, 0.342], these are the offsets
                    translation=[0.0113, -0.0245, 0.695], rpy_vector=[0, np.pi*41/180, 0]
                ),
            ),
            "left_wrist": CameraAdderConfig(
                xml_path=CAMERA_PATHS["d405"],
                fovy=60.0,
                offset=rcs.common.Pose(translation=[0.060, 0, 0.0665], rpy_vector=[-np.pi/2, -np.pi*11/18, 0]), # 20deg offset from normal
                robot_name="left",
            ),
            "right_wrist": CameraAdderConfig(
                xml_path=CAMERA_PATHS["d405"],
                fovy=60.0,
                offset=rcs.common.Pose(translation=[0.060, 0, 0.0665], rpy_vector=[-np.pi/2, -np.pi*11/18, 0]), # 20deg offset from normal
                robot_name="right",
            ),
        }
        gripper_offset = rcs.common.Pose(quaternion=self.gripper_mesh_quaternion_offset, translation=[0, 0, 0])
        return SimSceneConfig(
            robot_cfgs=robot_cfgs,
            sim_cfg=sim_cfg,
            control_mode=control_mode,
            task=task,
            scene=scene,
            gripper_cfgs=gripper_cfgs,
            camera_cfgs=camera_cfgs,
            max_relative_movement=max_relative_movement,
            relative_to=relative_to,
            robot_to_shared_base_frame=robot_to_shared_base_frame,
            wrapper_cfg=wrapper_cfg,
            open_gui_on_create=open_gui_on_create,
            add_gravcomp=add_gravcomp,
            shared_base_frame_to_root_frame=shared_base_frame_to_root_frame,
            root_frame_to_world=root_frame_to_world,
            alternative_combined_robot_mjcf=alternative_combined_robot_mjcf,
            world_frame_objects=world_frame_objects,
            root_frame_objects=root_frame_objects,
            robot_frame_objects=robot_frame_objects,
            camera_adds=add_camera_adds,
            gripper_offsets={"left": gripper_offset, "right": gripper_offset},
        )


class EmptyWorldUR5e(EmptyWorldFR3):

    def load_config(self, key: str) -> SimSceneConfig:
        rt = RobotType("UR5e")
        cfg = super().load_config(key)
        lead_robot_name = next(iter(cfg.robot_cfgs))

        robot_cfg = cfg.robot_cfgs[lead_robot_name]
        robot_cfg.tcp_offset = rcs.common.Pose()
        robot_cfg.attachment_site = rcs.ROBOTS[rt].attachment_site
        robot_cfg.kinematic_model_path = rcs.ROBOTS[rt].mjcf_model_path
        robot_cfg.arm_collision_geoms = []
        robot_cfg.joints = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        robot_cfg.actuators = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
        robot_cfg.dof = rcs.ROBOTS[rt].dof
        robot_cfg.joint_limits = rcs.ROBOTS[rt].joint_limits
        robot_cfg.q_home = rcs.ROBOTS[rt].q_home
        robot_cfg.base = "base"

        gripper_cfg = cfg.gripper_cfgs[lead_robot_name]

        gripper_cfg.actuator = "fingers_actuator"
        gripper_cfg.joints = ["right_driver_joint", "left_driver_joint"]
        gripper_cfg.collision_geoms = []
        gripper_cfg.collision_geoms_fingers = []
        gripper_cfg.max_actuator_width = 0
        gripper_cfg.min_actuator_width = 255
        gripper_cfg.max_joint_width = 0.005
        gripper_cfg.min_joint_width = 1.0
        gripper_cfg.gripper_type = GripperType("Robotiq2F85")

        cfg.camera_cfgs = None
        cfg.camera_adds = None
        cfg.gripper_offsets = None

        return cfg


if __name__ == "__main__":
    scene = EmptyWorldFR3Duo()
    # scene = EmptyWorldFR3()
    # scene = EmptyWorldUR5e()
    env = scene.create()
    obs, info = env.reset()
    print(obs)
    # Duo
    for _ in range(100):
        for _ in range(10):
            # move 1cm in x direction (forward) and close gripper
            act = {
                "left": {"tquat": [0.01, 0, 0, 0, 0, 0, 1], "gripper": [0]},
                "right": {"tquat": [0.01, 0, 0, 0, 0, 0, 1], "gripper": [0]},
            }
            obs, reward, terminated, truncated, info = env.step(act)
            print(obs)
            time.sleep(1.0)
        for _ in range(10):
            # move 1cm in negative x direction (backward) and open gripper
            act = {
                "left": {"tquat": [-0.01, 0, 0, 0, 0, 0, 1], "gripper": [1]},
                "right": {"tquat": [-0.01, 0, 0, 0, 0, 0, 1], "gripper": [1]},
            }
            obs, reward, terminated, truncated, info = env.step(act)
            print(obs)
            time.sleep(1.0)
    # # Single arm
    # for _ in range(100):
    #     for _ in range(10):
    #         # move 1cm in x direction (forward) and close gripper
    #         act = {"robot": {"tquat": [0.01, 0, 0, 0, 0, 0, 1], "gripper": [0]}}
    #         obs, reward, terminated, truncated, info = env.step(act)
    #         print(obs)
    #         time.sleep(1.0)
    #     for _ in range(10):
    #         # move 1cm in negative x direction (backward) and open gripper
    #         act = {"robot": {"tquat": [-0.01, 0, 0, 0, 0, 0, 1], "gripper": [1]}}
    #         obs, reward, terminated, truncated, info = env.step(act)
    #         print(obs)
    #         time.sleep(1.0)
