import logging
import threading
from time import sleep

import numpy as np
import rcs
from rcs._core.common import RPY, Pose, RobotPlatform
from rcs._core import common
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
from rcs.operator.interface import TeleopLoop

logger = logging.getLogger(__name__)


ROBOT2IP = {
    # "left": "192.168.102.1",
    "right": "192.168.101.1",
}
ROBOT2ID = {
    "left": "0",
    "right": "1",
}


ROBOT_INSTANCE = RobotPlatform.SIMULATION
# ROBOT_INSTANCE = RobotPlatform.HARDWARE

RECORD_FPS = 30
# set camera dict to none disable cameras
# CAMERA_DICT = {
#     "left_wrist": "230422272017",
#     "right_wrist": "230422271040",
#     "side": "243522070385",
#     "bird_eye": "243522070364",
# }
CAMERA_DICT = None
MQ3_ADDR = "192.168.1.219"

# DIGIT_DICT = {
#     "digit_right_left": "D21182",
#     "digit_right_right": "D21193"
# }
DIGIT_DICT = None


DATASET_PATH = "test_data_iris_dual_arm14"
INSTRUCTION = "build a tower with the blocks in front of you"


config = QuestConfig(mq3_addr=MQ3_ADDR, simulation=ROBOT_INSTANCE == RobotPlatform.SIMULATION)


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
            control_mode=QuestOperator.control_mode[0],
            gripper_cfg=default_fr3_hw_gripper_cfg(async_control=True),
            max_relative_movement=(0.5, np.deg2rad(90)),
            relative_to=QuestOperator.control_mode[1],
        )
        env_rel = StorageWrapper(
            env_rel, DATASET_PATH, INSTRUCTION, batch_size=32, max_rows_per_group=100, max_rows_per_file=1000
        )
        operator = QuestOperator(config)
    else:
        # FR3
        rcs.scenes["rcs_icra_scene"] = rcs.Scene(
            mjcf_scene="/home/tobi/coding/rcs_clones/prs/models/scenes/rcs_icra_scene/scene.xml",
            mjcf_robot=rcs.scenes["fr3_simple_pick_up"].mjcf_robot,
            robot_type=common.RobotType.FR3,
        )
        rcs.scenes["pick"] = rcs.Scene(
            mjcf_scene="/home/tobi/coding/rcs_clones/prs/assets/scenes/fr3_simple_pick_up/scene.xml",
            mjcf_robot=rcs.scenes["fr3_simple_pick_up"].mjcf_robot,
            robot_type=common.RobotType.FR3,
        )

        # robot_cfg = default_sim_robot_cfg("fr3_empty_world")
        # robot_cfg = default_sim_robot_cfg("fr3_simple_pick_up")
        robot_cfg = default_sim_robot_cfg("rcs_icra_scene")
        # robot_cfg = default_sim_robot_cfg("pick")

        # resolution = (256, 256)
        # cameras = {
        #     cam: SimCameraConfig(
        #         identifier=cam,
        #         type=CameraType.fixed,
        #         resolution_height=resolution[1],
        #         resolution_width=resolution[0],
        #         frame_rate=0,
        #     )
        #     for cam in ["side", "wrist"]
        # }

        sim_cfg = SimConfig()
        sim_cfg.async_control = True
        env_rel = SimMultiEnvCreator()(
            name2id=ROBOT2ID,
            robot_cfg=robot_cfg,
            control_mode=QuestOperator.control_mode[0],
            gripper_cfg=default_sim_gripper_cfg(),
            # cameras=default_mujoco_cameraset_cfg(),
            max_relative_movement=0.5,
            relative_to=QuestOperator.control_mode[1],
            sim_cfg=sim_cfg,
        )
        # sim = env_rel.unwrapped.envs[ROBOT2IP.keys().__iter__().__next__()].sim  # type: ignore
        sim = env_rel.get_wrapper_attr("sim")
        operator = QuestOperator(config, sim)
        sim.open_gui()
    return env_rel, operator


def main():
    env_rel, operator = get_env()
    env_rel.reset()
    tele = TeleopLoop(env_rel, operator)
    with env_rel, tele:  # type: ignore
        tele.environment_step_loop()


if __name__ == "__main__":
    main()
