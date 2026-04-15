import typing
from abc import ABC
from dataclasses import dataclass, field
from os import PathLike

import gymnasium as gym
import numpy as np
from rcs._core.common import (
    BaseCameraConfig,
    FrankaHandTCPOffset,
    GripperConfig,
    GripperType,
    RobotConfig,
    RobotType,
)
from rcs._core.sim import (
    SimCameraConfig,
    SimConfig,
    SimGripperConfig,
    SimRobot,
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
from rcs import GRIPPER_PATHS, ROBOTS, SCENE_PATHS


class BaseSceneConfig:
    pass


class BaseScene(ABC):
    config_type: typing.Type

    def create(self) -> gym.Env:
        raise NotImplementedError

    def load_config(self, key: str) -> BaseSceneConfig:
        # TODO: load type form yaml with type checking
        pass


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
    robot2world: dict[str, rcs.common.Pose] | None = None
    add_gravcomp: bool = False
    wrapper_cfg: WrapperConfig = field(default_factory=WrapperConfig)
    open_gui_on_create: bool = True
    seperate_robot_mjcf: dict[str, tuple[str, rcs.common.Pose]] | None = None
    """Optional path to mjcf robot files and offset for scene composition, if not provided the kinematic_model_path from the robot configs will be used."""


class SimScene(BaseScene):

    def __init__(self, config_key: str) -> None:
        super().__init__()
        self.config_key = config_key
        self.cfg = self.load_config(config_key)
        self.robots_names = list(self.cfg.robot_cfgs.keys())
        self.lead_robot_name = next(iter(self.robots_names))
        self.robot_prefix_template = "robot{robot_name}_"
        self.gripper_prefix_template = "gripper{robot_name}_"

    def load_config(self, key: str) -> SimSceneConfig:
        raise NotImplementedError

    def load_scene(self, key: str) -> ModelComposer | str | PathLike:
        """Loads the mujoco scene

        Args:
            key (str): Key to identify the scene, e.g. path

        Returns:
            ModelComposer | str | PathLike: path to scene file (mjcf or mjb), or composer object
        """
        composer = ModelComposer(
            model_name=key,
            add_gravcomp=self.cfg.add_gravcomp,
        )
        composer.load_base_scene(key)

        self.add_task_mujoco(self.cfg.task, composer)

        name_path_r2w = (
            self.cfg.seperate_robot_mjcf
            if self.cfg.seperate_robot_mjcf is not None
            else {
                robot_name: (self.cfg.robot_cfgs[robot_name].kinematic_model_path, self.cfg.robot2world[robot_name])
                for robot_name in self.robots_names
            }
        )

        for robot_name, (robot_xml, robot2world) in name_path_r2w.items():
            self.add_robot_mujoco(composer, robot_name, robot_xml, robot2world)
            if self.cfg.gripper_cfgs is not None:
                gripper_xml = GRIPPER_PATHS[self.cfg.gripper_cfgs[robot_name].gripper_type]
                self.add_gripper_mujoco(
                    composer,
                    robot_name,
                    gripper_xml,
                    self.cfg.robot_cfgs[robot_name].attachment_site,
                )

        return composer

    def add_task_mujoco(self, key: str | None, composer: ModelComposer):
        """Add task-specific elements to the Mujoco scene."""
        pass

    def add_task_env(self, key: str | None, env: gym.Env, simulation: Sim) -> gym.Env:
        """Add task-specific wrappers to the environment."""
        return env

    def add_robot_mujoco(
        self, composer: ModelComposer, robot_name: str, robot_xml: str, robot2world: rcs.common.Pose | None = None
    ):
        if robot2world is None:
            robot2world = rcs.common.Pose()
        composer.add_robot(
            robot_xml,
            self.robot_prefix_template.format(robot_name=robot_name),
            pos=list(robot2world.translation()),
            quat=list(robot2world.rotation_q()),
        )

    def add_robot_env(self, robot_name: str, env: gym.Env, simulation: Sim, ik: rcs.common.Kinematics):
        # rcs wrapper composition
        self.cfg.robot_cfgs[robot_name].add_prefix(self.robot_prefix_template.format(robot_name=robot_name))
        robot = rcs.sim.SimRobot(sim=simulation, ik=ik, cfg=self.cfg.robot_cfgs[robot_name])
        env = RobotWrapper(env, robot, self.cfg.control_mode, home_on_reset=self.cfg.wrapper_cfg.home_on_reset)
        env = RobotSimWrapper(env)
        return env

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
        self.cfg.gripper_cfgs[robot_name].add_prefix(self.gripper_prefix_template.format(robot_name=robot_name))
        gripper = rcs.sim.SimGripper(simulation, self.cfg.gripper_cfgs[robot_name])
        env = GripperWrapper(env, gripper, binary=self.cfg.wrapper_cfg.binary_gripper)
        env = GripperWrapperSim(env)
        return env

    def create(self) -> gym.Env:

        mjcf = self.load_scene(self.cfg.scene)
        # save the composed scene for debugging
        # mjcf.save_mjcf(f"scene.xml")
        # you can also apply a scene path e.g. the saved one
        # mjcf = "scene.xml"

        simulation = Sim(mjcf, self.cfg.sim_cfg)
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
            if self.cfg.gripper_cfgs is not None:
                env = self.add_gripper_env(robot_name, simulation, env)

            if self.cfg.relative_to != RelativeTo.NONE:
                env = RelativeActionSpace(env, max_mov=self.cfg.max_relative_movement, relative_to=self.cfg.relative_to)
            envs[robot_name] = env

        env = MultiRobotWrapper(envs, self.cfg.robot2world)
        if self.cfg.camera_cfgs is not None:
            camera_set = typing.cast(
                BaseCameraSet,
                SimCameraSet(simulation, self.cfg.camera_cfgs, physical_units=True, render_on_demand=True),
            )
            env = CameraSetWrapper(env, camera_set, include_depth=True)
        env = self.add_task_env(self.cfg.task, env, simulation)
        if self.cfg.open_gui_on_create:
            env.get_wrapper_attr("sim").open_gui()
        return CoverWrapper(env)


class EmptyWorldFR3(SimScene):

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
        robot2world: dict[str, rcs.common.Pose] | None = {"fr3": rcs.common.Pose()}
        wrapper_cfg: WrapperConfig = WrapperConfig(binary_gripper=True, home_on_reset=True)
        open_gui_on_create = True
        add_gravcomp = True
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
            robot2world=robot2world,
            wrapper_cfg=wrapper_cfg,
            open_gui_on_create=open_gui_on_create,
            add_gravcomp=add_gravcomp,
        )


if __name__ == "__main__":
    scene = EmptyWorldFR3("empty_world_fr3")
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
