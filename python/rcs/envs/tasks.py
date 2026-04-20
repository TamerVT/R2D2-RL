
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import rcs
from rcs._core import sim
from rcs.envs.base import GripperWrapper
from rcs.envs.scenes import BaseTaskConfig, Task, TaskConfig
from rcs.envs.sim import PickCubeSuccessWrapper
from rcs.sim.composer import ModelComposer
from rcs.sim.sim import Sim
import gymnasium as gym


class PickObjSuccessWrapper(gym.Wrapper):
    """
    Wrapper to check if an object is successfully picked up.
    Obj must be lifted 10 cm above its position.
    Computes a reward between 0 and 1 based on:
    - TCP to object distance
    - cube z height
    - whether the arm is standing still once the task is solved.
    """

    def __init__(self, env, robot_name: str, shared2world: rcs.common.Pose, obj_joint_name="box_body"):
        super().__init__(env)
        assert isinstance(self.get_wrapper_attr("robot"), sim.SimRobot), "Robot must be a sim.SimRobot instance."
        # self._robot = cast(sim.SimRobot, self.get_wrapper_attr("robot"))
        self.sim = self.env.get_wrapper_attr("sim")
        self.cube_joint = obj_joint_name

        # self.home_pose = self._robot.get_cartesian_position()
        self._gripper_closing = 0
        self._gripper = self.get_wrapper_attr("gripper")
        self.shared2world = shared2world
        self.robot_name = robot_name



    def step(self, action: dict[str, Any]):  # type: ignore
        obs, reward, _, truncated, info = super().step(action)

        if (
            self._gripper.get_normalized_width() > 0.01
            and self._gripper.get_normalized_width() < 0.99
            and obs["gripper"] == GripperWrapper.BINARY_GRIPPER_CLOSED
        ):
            self._gripper_closing += 1
        else:
            self._gripper_closing = 0

        cube_pose = rcs.common.Pose(translation=self.sim.data.joint(self.cube_joint_name).qpos[:3])



        # cube_pose = self._robot.to_pose_in_robot_coordinates(cube_pose)

        # TODO: bring pose in shared frame
        cube_pose *= self.shared2world.inverse()


        # tcp_to_obj_dist = np.linalg.norm(cube_pose.translation() - self._robot.get_cartesian_position().translation())
        tcp_to_obj_dist = np.linalg.norm(cube_pose.translation() - obs[self.robot_name]["tquat"][:3])


        obj_to_goal_dist = 0.10 - min(cube_pose.translation()[-1], 0.10)
        # obj_to_goal_dist = np.linalg.norm(cube_pose.translation() - self.home_pose.translation())


        # NOTE: 4 depends on the time passing between each step.
        is_grasped = (
            self._gripper_closing >= 4  # gripper is closing since more than 4 steps
            and obs["gripper"] == GripperWrapper.BINARY_GRIPPER_CLOSED  # command is still close
            and tcp_to_obj_dist <= 0.01  # tcp to cube center is max 1cm
        )
        success = obj_to_goal_dist <= 0.022 and info["is_grasped"]
        movement = np.linalg.norm(self.sim.data.qvel)

        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
        static_reward = 1 - np.tanh(5 * movement)
        info["is_grasped"] = is_grasped
        info["success"] = success
        reward = reaching_reward + place_reward * is_grasped + static_reward * success
        reward /= 3  # type: ignore
        return obs, reward, success, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset()
        # self.home_pose = self._robot.get_cartesian_position()
        return obs, info


class RandomSquareObjPos(gym.Wrapper):
    """Wrapper to an object in a simulated environment in a random spot inside a defined square.

    Works only for single robot
    """

    def __init__(self,
                 env: gym.Env,
                 center_pose: rcs.common.Pose,
                 include_rotation: bool = True,
                 obj_joint_name: str = "box_joint",
                 x_width: float = 0.2,
                 y_width: float = 0.2,
    ):
        super().__init__(env)
        self.include_rotation = include_rotation
        self.obj_joint_name = obj_joint_name
        self.center_pose = center_pose
        self.x_width = x_width
        self.y_width = y_width

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)


        # TODO this needs frame transformation
        pos_x = self.center_pose.translation[0]
        pos_y = self.center_pose.translation[1]
        pos_z = self.center_pose.translation[3]

        # place randomly in square
        pos_x = np.random.uniform(pos_x -self.x_width/2, pos_x + self.x_width/2)
        pos_y = np.random.uniform(pos_y -self.y_width/2, pos_y + self.y_width/2)

        if self.include_rotation:
            # 1. Sample a random angle between 0 and 2*pi (360 degrees)
            theta = np.random.uniform(0, 2 * np.pi)
            
            # 2. Convert the angle to a unit quaternion for the Z-axis
            qw = np.cos(theta / 2)
            qz = np.sin(theta / 2)
        else:
            # No rotation (Identity quaternion)
            qw = 1.0
            qz = 0.0

        # qpos array format for a free joint: [x, y, z, qw, qx, qy, qz]
        self.get_wrapper_attr("sim").data.joint(self.obj_joint_name).qpos = [pos_x, pos_y, pos_z, qw, 0, 0, qz]

        return obs, info



@dataclass(kw_only=True)
class PickTaskConfig(BaseTaskConfig):
    shared2world: rcs.common.Pose
    object2root_frame: rcs.common.Pose = field(default_factory=lambda: rcs.common.Pose(translation=np.array([0.5, 0., 0.05]), quaternion=np.array([0, 0, 0, 1])))
    object_xml = rcs.OBJECT_PATHS["green_cube"]
    object_body: str = "box_body"
    object_joint: str = "box_joint"
    include_rotation: bool = True




class PickTask(Task[PickTaskConfig]):
    # TODO: for the reset it should be possible to access the composer and move things!

    @staticmethod
    def add_task_mujoco(cfg: PickTaskConfig, composer: ModelComposer):
        """Add task-specific elements to the Mujoco scene."""
        object2world = cfg.object2root_frame * cfg.root_frame_to_world

        composer.add_object_world_frame(
            cfg.object_xml,
            object_prefix=cfg.object_body + "_",
            pose=object2world,
        )
        # "green_cube": (OBJECT_PATHS["green_cube"], Pose(translation=[0.5, 0, 0.5], quaternion=[0, 0, 0, 1])),

    @staticmethod
    def add_task_env(cfg: PickTaskConfig, env: gym.Env, simulation: Sim) -> gym.Env:
        """Add task-specific wrappers to the environment."""
        object2world = cfg.object2root_frame * cfg.root_frame_to_world
        env = PickCubeSuccessWrapper(env)
        env = RandomSquareObjPos(env, center_pose=object2world, include_rotation=cfg.include_rotation, obj_joint_name=cfg.object_joint)
        return env