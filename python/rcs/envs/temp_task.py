from dataclasses import dataclass, field

import numpy as np
from numpy.random import random
from rcs.envs.sim import JoinBlocksTaskWrapper, MazeTaskWrapper
from rcs.sim.composer import ModelComposer
import rcs_models
import rcs
from rcs.envs.scenes import BaseTaskConfig, Task
from rcs.sim.sim import Sim
import gymnasium as gym


@dataclass(kw_only=True)
class MazeTaskConfig(BaseTaskConfig):
    shared2world: rcs.common.Pose
    object2root_frame: rcs.common.Pose = field(default_factory=lambda: rcs.common.Pose(translation=np.array([0.5, 0., 0.05]), quaternion=np.array([0, 0, 0, 1])))
    task_dict=rcs.TASKS["balance_board"]
    object_body = "board"
    include_rotation: bool = True
    task_seed = 42 #used to randomise which board is used 
    hard_reset = True



class Maze_Task(Task[MazeTaskConfig]):


    random.seed(MazeTaskConfig.task_seed)
    @staticmethod
    def add_task_mujoco(cfg: MazeTaskConfig, composer: ModelComposer):
        """Add task-specific elements to the Mujoco scene."""
        object2world = cfg.object2root_frame * cfg.root_frame_to_world

        #select the board to use this time
        number_boards = MazeTaskConfig.task_dict["number_board"]
        board_number = random.randint(1, number_boards)

        board_xml_template= MazeTaskConfig.task_dict["objects"][0]["path"]
        board_xml = board_xml_template.format(number=board_number)



        composer.add_object_world_frame(
            board_xml,
            object_prefix=cfg.object_body + "_",
            pose=object2world,
        )


    @staticmethod
    def add_task_env(cfg: MazeTaskConfig, env: gym.Env, simulation: Sim) -> gym.Env:
        """Add task-specific wrappers to the environment."""
        env = MazeTaskWrapper(env)

        return env
    
    @staticmethod
    def hard_reset(cfg: MazeTaskConfig, env: gym.Env, simulation: Sim):

        #TODO add reset to change boards
        pass


@dataclass(kw_only=True)
class JoinBlocksTaskConfig(BaseTaskConfig):
    shared2world: rcs.common.Pose
    object2root_frame: rcs.common.Pose = field(default_factory=lambda: rcs.common.Pose(translation=np.array([0.5, 0., 0.05]), quaternion=np.array([0, 0, 0, 1])))
    task_dict=rcs.TASKS["join_blocks"]
    include_rotation: bool = True
    task_seed = 42 #used to randomise which board is used 
    hard_reset = False



class JoinBlocks_Task(Task[MazeTaskConfig]):


    random.seed(MazeTaskConfig.task_seed)
    @staticmethod
    def add_task_mujoco(cfg: JoinBlocksTaskConfig, composer: ModelComposer):
        """Add task-specific elements to the Mujoco scene."""
        object2world = cfg.object2root_frame * cfg.root_frame_to_world

        for o in JoinBlocksTaskConfig.task_dict["objects"]:
            name = o["name"]
            path = 0["path"]

            composer.add_object_world_frame(
                path,
                object_prefix=name + "_",
                pose=object2world,
            )


    @staticmethod
    def add_task_env(cfg: MazeTaskConfig, env: gym.Env, simulation: Sim) -> gym.Env:
        """Add task-specific wrappers to the environment."""
        
        env = JoinBlocksTaskWrapper(env)

        return env
    
    @staticmethod
    def hard_reset(cfg: MazeTaskConfig, env: gym.Env, simulation: Sim):

        #TODO can be ignored for now
        pass