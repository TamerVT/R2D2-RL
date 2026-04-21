from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import mujoco as mj
import numpy as np
from rcs.envs.scenes import BaseTaskConfig, SimEnvCreatorConfig, Task
from rcs.sim.composer import ModelComposer
from rcs.sim.sim import Sim

import rcs


class RandomSquareObjsPos(gym.Wrapper):
    """
    Wrapper to position arbitrary number of objects in a simulated environment in random spots inside a defined square.
    """

    def __init__(
        self,
        env: gym.Env,
        center2world: rcs.common.Pose,
        obj_joint_names: list[str] = ["box_joint"],
        obj_position_margin: float = 0.05,
        x_width: float = 0.3,
        y_width: float = 0.3,
        include_rotation: bool = True,
        seed: int = 42,
    ):
        super().__init__(env)
        self.obj_joint_names = obj_joint_names
        self.include_rotation = include_rotation
        self.center2world = center2world
        self.x_width = x_width
        self.y_width = y_width
        self.obj_position_margin = obj_position_margin

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        def get_random_position(x_width, y_width):
            return np.array(
                [
                    np.random.uniform(-x_width / 2, x_width / 2),
                    np.random.uniform(-y_width / 2, y_width / 2),
                    0.0,
                ]
            )

        # place randomly in square with sufficient margin between objects
        obj_positions = [get_random_position(self.x_width, self.y_width) for _ in self.obj_joint_names]
        for i, obj_joint_name in enumerate(self.obj_joint_names):
            # calculate the norm to all other objects
            pos_norms = [get_random_position(self.x_width, self.y_width) for _ in self.obj_joint_names]
        for i, obj_joint_name in enumerate(self.obj_joint_names):
            # calculate the norm to all other objects
            pos_norms = [
                np.linalg.norm(obj_positions[i] - obj_positions[j]) for j in range(len(self.obj_joint_names)) if j != i
            ]
            if len(pos_norms) == 0:
                obj_positions[i] = get_random_position(self.x_width, self.y_width)
            else:
                while min(pos_norms) < self.obj_position_margin:
                    obj_positions[i] = get_random_position(self.x_width, self.y_width)
                    pos_norms = [
                        np.linalg.norm(obj_positions[i] - obj_positions[j])
                        for j in range(len(self.obj_joint_names))
                        if j != i
                    ]

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

            pose_in_center_frame = rcs.common.Pose(translation=obj_positions[i], quaternion=np.array([0, 0, qz, qw]))
            pose_in_world_frame = self.center2world * pose_in_center_frame

            # qpos array format for a free joint: [x, y, z, qw, qx, qy, qz]
            self.get_wrapper_attr("sim").data.joint(obj_joint_name).qpos = np.append(
                pose_in_world_frame.translation(), pose_in_world_frame.rotation_q_wxyz()
            )

        # reset of remaining stack, must happen after our reset!
        return super().reset(seed=seed, options=options)


class ParallelPickSuccessWrapper(gym.Wrapper):
    """
    Wrapper to check whether the parallel pick and place task is successful.
    Given a dict of {goal_obj1: [target_objects], ...}, the wrapper checks if all target objects are placed within the goal_obj1.

    """

    def __init__(
        self,
        env,
        shared2world: rcs.common.Pose,
        goals: dict[str, list[str]],
        bowl_radius: float = 0.06,
        prefix: str = "ParallelPickTask_",
    ):
        super().__init__(env)
        # assert isinstance(self.get_wrapper_attr("robot"), sim.SimRobot), "Robot must be a sim.SimRobot instance."
        # self._robot = cast(sim.SimRobot, self.get_wrapper_attr("robot"))
        self.sim = self.env.get_wrapper_attr("sim")
        self.goals = goals
        self.bowl_radius = bowl_radius

        self.shared2world = shared2world
        self.finished_objects = set()
        self.wrong_objects = set()
        self.unfinished_objects = set([obj_ for obj in goals.values() for obj_ in obj])

        self.prefix = prefix

    def step(self, action: dict[str, Any]):  # type: ignore
        obs, reward, _, truncated, info = super().step(action)
        goal_keys = list(self.goals.keys())
        for goal_obj in goal_keys:
            goal_id = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_BODY, self.prefix + goal_obj + "_body")
            goal_pos = self.sim.data.xpos[goal_id]
            correct_target_objs = self.goals[goal_obj]
            wrong_target_objs = [obj for obj in self.unfinished_objects if obj not in correct_target_objs]
            # Check from list of
            for target_obj in correct_target_objs:
                target_id = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_BODY, self.prefix + target_obj + "_body")
                target_pos = self.sim.data.xpos[target_id]
                if (
                    np.linalg.norm(goal_pos[:2] - target_pos[:2]) <= self.bowl_radius
                    and abs(goal_pos[2] - target_pos[2]) <= 0.2
                ):
                    self.finished_objects.add(target_obj)
                    if target_obj in self.unfinished_objects:
                        self.unfinished_objects.remove(target_obj)
                    if target_obj in self.wrong_objects:
                        self.wrong_objects.remove(target_obj)
            for target_obj in wrong_target_objs:
                target_id = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_BODY, self.prefix + target_obj + "_body")
                target_pos = self.sim.data.xpos[target_id]
                if (
                    np.linalg.norm(goal_pos[:2] - target_pos[:2]) <= self.bowl_radius
                    and abs(goal_pos[2] - target_pos[2]) <= 0.2
                ):
                    self.wrong_objects.add(target_obj)
                    if target_obj in self.unfinished_objects:
                        self.unfinished_objects.remove(target_obj)
        success = float(
            len(self.finished_objects) / len([obj for target_objs in self.goals.values() for obj in target_objs])
        )
        info["success"] = success
        info["correct"] = self.finished_objects
        info["wrong"] = self.wrong_objects
        info["unfinished"] = self.unfinished_objects

        return obs, reward, success, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset()
        # self.home_pose = self._robot.get_cartesian_position()
        return obs, info


@dataclass(kw_only=True)
class ParallelPickTaskConfig(BaseTaskConfig):
    objects_xml = {
        # "blue_tri_cylinder": rcs.OBJECT_PATHS["blue_tri_cylinder"],
        "blue_box": rcs.OBJECT_PATHS["blue_box"],
        "blue_pent_cylinder": rcs.OBJECT_PATHS["blue_pent_cylinder"],
        "blue_hex_cylinder": rcs.OBJECT_PATHS["blue_hex_cylinder"],
        "blue_cylinder": rcs.OBJECT_PATHS["blue_cylinder"],
        # "green_tri_cylinder": rcs.OBJECT_PATHS["green_tri_cylinder"],
        "green_box": rcs.OBJECT_PATHS["green_box"],
        "green_pent_cylinder": rcs.OBJECT_PATHS["green_pent_cylinder"],
        "green_hex_cylinder": rcs.OBJECT_PATHS["green_hex_cylinder"],
        "green_cylinder": rcs.OBJECT_PATHS["green_cylinder"],
    }
    objects_joints = {
        # "blue_tri_cylinder":"blue_tri_cylinder_joint",
        "blue_box": "blue_box_joint",
        "blue_pent_cylinder": "blue_pent_cylinder_joint",
        "blue_hex_cylinder": "blue_hex_cylinder_joint",
        "blue_cylinder": "blue_cylinder_joint",
        # "green_tri_cylinder":"green_tri_cylinder_joint",
        "green_box": "green_box_joint",
        "green_pent_cylinder": "green_pent_cylinder_joint",
        "green_hex_cylinder": "green_hex_cylinder_joint",
        "green_cylinder": "green_cylinder_joint",
    }
    object_center_to_root_frame: rcs.common.Pose = field(
        default_factory=lambda: rcs.common.Pose(
            translation=np.array([0.5, 0.0, 0.05]), quaternion=np.array([0, 0, 0, 1])
        )
    )
    task_id: str = "parallel_pick"
    goals_xml = {"white_bowl": rcs.OBJECT_PATHS["white_bowl"], "black_bowl": rcs.OBJECT_PATHS["black_bowl"]}
    goals_objects = {
        "white_bowl": [
            # "blue_tri_cylinder"
            "blue_box",
            "blue_pent_cylinder",
            "blue_hex_cylinder",
            "blue_cylinder",
        ],
        "black_bowl": [
            # "green_tri_cylinder"
            "green_box",
            "green_pent_cylinder",
            "green_hex_cylinder",
            "green_cylinder",
        ],
    }

    goals_center_to_root_frame: dict[str, rcs.common.Pose] = field(
        default_factory=lambda: {
            "white_bowl": rcs.common.Pose(translation=np.array([0.6, 0.3, 0.0]), quaternion=np.array([0, 0, 0, 1])),
            "black_bowl": rcs.common.Pose(translation=np.array([0.6, -0.3, 0.0]), quaternion=np.array([0, 0, 0, 1])),
        }
    )
    bowl_radius = 0.06

    task_seed = 42
    x_width = 0.3
    y_width = 0.3
    obj_position_margin = 0.05
    prefix = "ParallelPickTask_"
    include_rotation = True


class ParallelPickTask(Task[ParallelPickTaskConfig]):
    # TODO: for the reset it should be possible to access the composer and move things!
    # random.seed(ParallelPickTaskConfig.task_seed)
    @staticmethod
    def add_task_mujoco(cfg: ParallelPickTaskConfig, composer: ModelComposer, env_cfg: SimEnvCreatorConfig):
        """Add task-specific elements to the Mujoco scene."""
        object2world = cfg.object_center_to_root_frame * env_cfg.root_frame_to_world

        for xml in cfg.objects_xml.values():
            composer.add_object_world_frame(
                xml,
                object_prefix=cfg.prefix,
                pose=object2world,
            )

        for xml, center in zip(cfg.goals_xml.values(), cfg.goals_center_to_root_frame.values()):
            goal2world = center * env_cfg.root_frame_to_world
            composer.add_object_world_frame(
                xml,
                object_prefix=cfg.prefix,
                pose=goal2world,
            )

    @staticmethod
    def add_task_env(
        cfg: ParallelPickTaskConfig, env: gym.Env, simulation: Sim, env_cfg: SimEnvCreatorConfig
    ) -> gym.Env:
        """Add task-specific wrappers to the environment."""
        object2world = cfg.object_center_to_root_frame * env_cfg.root_frame_to_world
        shared2world = env_cfg.shared_base_frame_to_root_frame * env_cfg.root_frame_to_world
        obj_joint_names = [cfg.prefix + joint for joint in cfg.objects_joints.values()]

        env = ParallelPickSuccessWrapper(
            env, shared2world, cfg.goals_objects, bowl_radius=cfg.bowl_radius, prefix=cfg.prefix
        )

        # For positioning target objects
        env = RandomSquareObjsPos(
            env,
            x_width=cfg.x_width,
            y_width=cfg.y_width,
            center2world=object2world,
            include_rotation=cfg.include_rotation,
            obj_joint_names=obj_joint_names,
            obj_position_margin=cfg.obj_position_margin,
        )

        # For positioning the bowls
        for k, v in cfg.goals_center_to_root_frame.items():
            goal2world = v * env_cfg.root_frame_to_world
            env = RandomSquareObjsPos(
                env,
                x_width=0.1,
                y_width=0.1,
                center2world=goal2world,
                include_rotation=False,
                obj_joint_names=[cfg.prefix + k + "_joint"],
            )

        return env


rcs.TASKS["parallel_pick"] = ParallelPickTask
