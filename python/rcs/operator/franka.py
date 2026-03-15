import logging
import threading
from time import sleep

import numpy as np
from rcs._core.common import RPY, Pose, RobotPlatform
from rcs._core.sim import SimConfig
from rcs.camera.hw import HardwareCameraSet
from rcs.envs.base import (
    ControlMode,
    GripperDictType,
    LimitedTQuatRelDictType,
    RelativeActionSpace,
    RelativeTo,
)
from rcs.envs.creators import SimMultiEnvCreator
from rcs.envs.storage_wrapper import StorageWrapper
from rcs.envs.utils import default_digit, default_sim_gripper_cfg, default_sim_robot_cfg
from rcs.utils import SimpleFrameRate
from rcs_fr3.creators import RCSFR3MultiEnvCreator
from rcs_fr3.utils import default_fr3_hw_gripper_cfg, default_fr3_hw_robot_cfg
from rcs_realsense.utils import default_realsense
from rcs.operator.quest import QuestConfig, QuestOperator

logger = logging.getLogger(__name__)




ROBOT2IP = {
    # "left": "192.168.102.1",
    "right": "192.168.101.1",
}


# ROBOT_INSTANCE = RobotPlatform.SIMULATION
ROBOT_INSTANCE = RobotPlatform.HARDWARE

RECORD_FPS = 30
# set camera dict to none disable cameras
# CAMERA_DICT = {
#     "left_wrist": "230422272017",
#     "right_wrist": "230422271040",
#     "side": "243522070385",
#     "bird_eye": "243522070364",
# }
CAMERA_DICT = None
MQ3_ADDR = "10.42.0.1"

# DIGIT_DICT = {
#     "digit_right_left": "D21182",
#     "digit_right_right": "D21193"
# }
DIGIT_DICT = None


DATASET_PATH = "test_data_iris_dual_arm14"
INSTRUCTION = "build a tower with the blocks in front of you"
TELEOP = "quest"

configs = {"quest": QuestConfig(robot_keys=ROBOT2IP.keys(), simulation=ROBOT_INSTANCE == RobotPlatform.SIMULATION, mq3_addr=MQ3_ADDR)}
operators = {
    "quest": QuestOperator,
}


def get_env():
    if ROBOT_INSTANCE == RobotPlatform.HARDWARE:

        cams = []
        if CAMERA_DICT is not None:
            cams.append(default_realsense(CAMERA_DICT))
        if DIGIT_DICT is not None:
            cams.append(default_digit(DIGIT_DICT))

        camera_set = HardwareCameraSet(cams) if cams else None

        env_rel = RCSFR3MultiEnvCreator()(
            name2ip=ROBOT2IP,
            camera_set=camera_set,
            robot_cfg=default_fr3_hw_robot_cfg(async_control=True),
            control_mode=operators[TELEOP].control_mode[0],
            gripper_cfg=default_fr3_hw_gripper_cfg(async_control=True),
            max_relative_movement=(0.5, np.deg2rad(90)),
            relative_to=operators[TELEOP].control_mode[1],
        )
        env_rel = StorageWrapper(
            env_rel, DATASET_PATH, INSTRUCTION, batch_size=32, max_rows_per_group=100, max_rows_per_file=1000
        )
    else:
        # FR3
        robot_cfg = default_sim_robot_cfg("fr3_empty_world")

        sim_cfg = SimConfig()
        sim_cfg.async_control = True
        env_rel = SimMultiEnvCreator()(
            name2id=ROBOT2IP,
            robot_cfg=robot_cfg,
            control_mode=operators[TELEOP].control_mode[0],
            gripper_cfg=default_sim_gripper_cfg(),
            # cameras=default_mujoco_cameraset_cfg(),
            max_relative_movement=0.5,
            relative_to=operators[TELEOP].control_mode[1],
            sim_cfg=sim_cfg,
        )
        sim = env_rel.unwrapped.envs[ROBOT2IP.keys().__iter__().__next__()].sim  # type: ignore
        sim.open_gui()
    return env_rel

def main():
    env_rel = get_env()
    env_rel.reset()
    with env_rel, operators[TELEOP](env_rel, configs[TELEOP]) as op:  # type: ignore
        op.environment_step_loop()


if __name__ == "__main__":
    main()
