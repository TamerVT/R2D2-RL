import copy
import typing
from abc import ABC
from dataclasses import dataclass, field
from os import PathLike

import gymnasium as gym
import numpy as np
from rcs._core.common import FrankaHandTCPOffset, RobotType
from rcs._core.sim import SimCameraConfig, SimConfig, SimGripperConfig, SimRobotConfig
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
from rcs import GRIPPER_PATHS, SCENE_PATHS


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
    objects: dict[str, tuple[str, rcs.common.Pose]] | None = None
    """dict of object_id to tuple of (object_xml, object2world), will be added to the scene, object2world is the pose of the object in the mujoco world frame"""
    robot_objects: dict[str, tuple[str, rcs.common.Pose]] | None = None
    """dict of object_id to tuple of (object_xml, object2root_frame), will be added to the robot object2root_frame is the pose of the object in the root frame of the robot,
    which is defined by shared_base_frame_to_root_frame, object_id must be unique across all objects and robots in the scene"""


class SimScene(BaseScene):

    def __init__(
        self,
        config_key: str,
        robot_prefix_template: str = "robot{robot_name}_",
        gripper_prefix_template: str = "gripper{robot_name}_",
    ) -> None:
        super().__init__()
        self.config_key = config_key
        self.cfg = self.load_config(config_key)
        self.robots_names = list(self.cfg.robot_cfgs.keys())
        self.lead_robot_name = next(iter(self.robots_names))
        self.robot_prefix_template = robot_prefix_template
        self.gripper_prefix_template = gripper_prefix_template

    def load_config(self, key: str) -> SimSceneConfig:
        raise NotImplementedError

    def load_scene(self) -> ModelComposer | str | PathLike:
        """Loads the mujoco scene from the given config

        Returns:
            ModelComposer | str | PathLike: path to scene file (mjcf or mjb), or composer object
        """
        composer = ModelComposer(
            model_name="RCS Scene",
            add_gravcomp=self.cfg.add_gravcomp,
        )
        composer.load_base_scene(self.cfg.scene)

        self.add_task_mujoco(self.cfg.task, composer)

        if self.cfg.alternative_combined_robot_mjcf is not None:
            # robot is in one mjcf
            self.add_robot_mujoco(
                composer,
                robot_name=self.lead_robot_name,
                robot_xml=self.cfg.alternative_combined_robot_mjcf,
                robot2world=self.cfg.root_frame_to_world,
            )
        else:
            # robot is composed by composer
            for robot_name in self.robots_names:
                robot_to_shared_frame = (
                    self.cfg.robot_to_shared_base_frame[robot_name]
                    if self.cfg.robot_to_shared_base_frame is not None
                    else rcs.common.Pose()
                )
                robot2world = (
                    robot_to_shared_frame * self.cfg.shared_base_frame_to_root_frame * self.cfg.root_frame_to_world
                )
                self.add_robot_mujoco(
                    composer, robot_name, self.cfg.robot_cfgs[robot_name].kinematic_model_path, robot2world
                )

        if self.cfg.gripper_cfgs is not None:
            # add gripper to each robot
            for robot_name in self.robots_names:
                gripper_xml = GRIPPER_PATHS[self.cfg.gripper_cfgs[robot_name].gripper_type]
                self.add_gripper_mujoco(
                    composer,
                    robot_name,
                    gripper_xml,
                    self.cfg.robot_cfgs[robot_name].attachment_site,
                )

        # add robot-specific objects
        if self.cfg.robot_objects is not None:
            for object_id, (object_xml, object2root_frame) in self.cfg.robot_objects.items():
                object2world = object2root_frame * self.cfg.root_frame_to_world
                self.add_object_mujoco(composer, object_id, object_xml, object2world)
        # add external objects
        if self.cfg.objects is not None:
            for object_id, (object_xml, object2world) in self.cfg.objects.items():
                self.add_object_mujoco(composer, object_id, object_xml, object2world)

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
        composer.add_object_from_xml(
            object_xml,
            prefix=object_id + "_",
            pos=list(object2world.translation()),
            quat=list(object2world.rotation_q()),
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
            pos=list(robot2world.translation()),
            quat=list(robot2world.rotation_q()),
        )

    def add_robot_env(self, robot_name: str, env: gym.Env, simulation: Sim, ik: rcs.common.Kinematics):
        # rcs wrapper composition
        robot = rcs.sim.SimRobot(sim=simulation, ik=ik, cfg=self.prefixed_cfg.robot_cfgs[robot_name])
        env = RobotWrapper(
            env, robot, self.prefixed_cfg.control_mode, home_on_reset=self.prefixed_cfg.wrapper_cfg.home_on_reset
        )
        return RobotSimWrapper(env)

    def add_gripper_mujoco(self, composer: ModelComposer, robot_name: str, gripper_xml: str, attachment_site: str):
        # mujoco scene composition
        assert self.cfg.gripper_cfgs is not None, "Gripper configs must be provided to add grippers."
        composer.add_gripper(
            xml_path=gripper_xml,
            gripper_prefix=self.gripper_prefix_template.format(robot_name=robot_name),
            robot_prefix=self.robot_prefix_template.format(robot_name=robot_name),
            attachment_site_name=attachment_site,
        )

    def add_gripper_env(self, robot_name: str, simulation: Sim, env: gym.Env):
        # rcs wrapper composition
        assert self.cfg.gripper_cfgs is not None, "Gripper configs must be provided to add grippers."
        gripper = rcs.sim.SimGripper(simulation, self.prefixed_cfg.gripper_cfgs[robot_name])
        env = GripperWrapper(env, gripper, binary=self.prefixed_cfg.wrapper_cfg.binary_gripper)
        return GripperWrapperSim(env)

    @property
    def prefixed_cfg(self) -> SimSceneConfig:
        cfg = copy.deepcopy(self.cfg)
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

        cfg = self.prefixed_cfg

        simulation = Sim(mjcf, cfg.sim_cfg)
        ik = rcs.common.Pin(
            self.cfg.robot_cfgs[self.lead_robot_name].kinematic_model_path,
            self.cfg.robot_cfgs[self.lead_robot_name].attachment_site,
        )
        # ik = rcs_robotics_library._core.rl.RoboticsLibraryIK(cfg.robot_cfgs[lead_robot_name].kinematic_model_path)

        envs: dict[str, gym.Env] = {}
        env: gym.Env
        for robot_name in self.robots_names:
            env = SimEnv(simulation)

            env = self.add_robot_env(robot_name, env, simulation, ik)
            if cfg.gripper_cfgs is not None:
                env = self.add_gripper_env(robot_name, simulation, env)

            if cfg.relative_to != RelativeTo.NONE:
                env = RelativeActionSpace(env, max_mov=cfg.max_relative_movement, relative_to=cfg.relative_to)
            envs[robot_name] = env

        env = MultiRobotWrapper(envs, cfg.robot_to_shared_base_frame)
        if cfg.camera_cfgs is not None:
            camera_set = typing.cast(
                BaseCameraSet,
                SimCameraSet(simulation, cfg.camera_cfgs, physical_units=True, render_on_demand=True),
            )
            env = CameraSetWrapper(env, camera_set, include_depth=True)
        env = self.add_task_env(cfg.task, env, simulation)
        if cfg.open_gui_on_create:
            env.get_wrapper_attr("sim").open_gui()
        return CoverWrapper(env)


class EmptyWorldFR3(SimScene):

    def __init__(self):
        super().__init__("", robot_prefix_template="fr3", gripper_prefix_template="fh")

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

        robot_cfgs: dict[str, SimRobotConfig] = {"fr3": robot_cfg}
        sim_cfg: SimConfig = SimConfig(async_control=True, realtime=True, frequency=30, max_convergence_steps=500)

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
        )
        gripper_cfgs: dict[str, SimGripperConfig] = {"fr3": gripper_cfg}
        camera_cfgs: dict[str, SimCameraConfig] | None = None
        max_relative_movement: float | tuple[float, float] | None = None
        relative_to: RelativeTo = RelativeTo.LAST_STEP
        robot_to_shared_base_frame: dict[str, rcs.common.Pose] | None = {"fr3": rcs.common.Pose()}
        wrapper_cfg: WrapperConfig = WrapperConfig(binary_gripper=True, home_on_reset=True)
        open_gui_on_create = True
        add_gravcomp = True
        shared_base_frame_to_root_frame = rcs.common.Pose()
        root_frame_to_world = rcs.common.Pose()
        alternative_combined_robot_mjcf: str | None = None
        objects: dict[str, tuple[str, rcs.common.Pose]] | None = None
        robot_objects: dict[str, tuple[str, rcs.common.Pose]] | None = None
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
            objects=objects,
            robot_objects=robot_objects,
        )


if __name__ == "__main__":
    scene = EmptyWorldFR3()
    env = scene.create()
    obs, info = env.reset()
    print(obs)
    for _ in range(100):
        for _ in range(10):
            # move 1cm in x direction (forward) and close gripper
            act = {"fr3": {"tquat": [0.01, 0, 0, 0, 0, 0, 1], "gripper": [0]}}
            obs, reward, terminated, truncated, info = env.step(act)
        for _ in range(10):
            # move 1cm in negative x direction (backward) and open gripper
            act = {"fr3": {"tquat": [-0.01, 0, 0, 0, 0, 0, 1], "gripper": [1]}}
            obs, reward, terminated, truncated, info = env.step(act)
