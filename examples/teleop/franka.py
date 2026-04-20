import logging
from typing import Any

import numpy as np
from rcs._core import common
from rcs._core.common import Pose, RobotPlatform
from rcs._core.sim import SimConfig
from rcs.camera.hw import HardwareCameraSet
from rcs.envs.base import ControlMode, RelativeTo
from rcs.envs.configs import EmptyWorldFR3Duo
from rcs.envs.creators import SimMultiEnvCreator
from rcs.envs.tasks import PickTaskConfig
from rcs.envs.utils import default_digit, default_sim_gripper_cfg, default_sim_robot_cfg
from rcs.operator.gello import GelloConfig, GelloOperator
from rcs.operator.interface import TeleopLoop
from rcs.operator.quest import QuestConfig, QuestOperator

# from rcs_fr3.creators import RCSFR3MultiEnvCreator
# from rcs_fr3.utils import default_fr3_hw_gripper_cfg, default_fr3_hw_robot_cfg
# from rcs_realsense.utils import default_realsense
from simpub.sim.mj_publisher import MujocoPublisher

import rcs

logger = logging.getLogger(__name__)


ROBOT2IP = {
    "right": "192.168.102.1",
    "left": "192.168.101.1",
}
ROBOT2ID = {
    "left": "1",
    "right": "0",
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
MQ3_ADDR = "192.168.1.20"

# DIGIT_DICT = {
#     "digit_right_left": "D21182",
#     "digit_right_right": "D21193"
# }
DIGIT_DICT = None


DATASET_PATH = "test_data_iris_dual_arm14"
INSTRUCTION = "build a tower with the blocks in front of you"

robot2world = {
    "right": rcs.common.Pose(
        translation=np.array([0, 0, 0]), rpy_vector=np.array([0.89360858, -0.17453293, 0.46425758])
    ),
    "left": rcs.common.Pose(
        translation=np.array([0, 0, 0]), rpy_vector=np.array([-0.89360858, -0.17453293, -0.46425758])
    ),
}

config: QuestConfig | GelloConfig
config = QuestConfig(mq3_addr=MQ3_ADDR, simulation=ROBOT_INSTANCE == RobotPlatform.SIMULATION, switched_left_right=True)
# config = GelloConfig(
#     arms={
#         "right": GelloArmConfig(com_port="/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_E505008B503059384C2E3120FF07332D-if00"),
#         "left": GelloArmConfig(com_port="/dev/serial/by-id/usb-ROBOTIS_OpenRB-150_ABA78B05503059384C2E3120FF062F26-if00"),
#     },
#     simulation=ROBOT_INSTANCE == RobotPlatform.SIMULATION,
# )


def get_env():
    if ROBOT_INSTANCE == RobotPlatform.HARDWARE:

        cams: list[Any] = []
        if CAMERA_DICT is not None:
            cams.append(default_realsense(CAMERA_DICT))
        if DIGIT_DICT is not None:
            cams.append(default_digit(DIGIT_DICT))

        camera_set = HardwareCameraSet(cams) if cams else None

        env_rel = RCSFR3MultiEnvCreator()(
            name2ip=ROBOT2IP,
            camera_set=camera_set,
            robot_cfg=default_fr3_hw_robot_cfg(async_control=True),
            control_mode=config.operator_class.control_mode[0],
            gripper_cfg=default_fr3_hw_gripper_cfg(async_control=True),
            max_relative_movement=(
                0.5 if config.operator_class.control_mode[0] == ControlMode.JOINTS else (0.5, np.deg2rad(90))
            ),
            relative_to=config.operator_class.control_mode[1],
            robot2world=robot2world,
        )
        # env_rel = StorageWrapper(
        #     env_rel, DATASET_PATH, INSTRUCTION, batch_size=32, max_rows_per_group=100, max_rows_per_file=1000
        # )
        operator = GelloOperator(config) if isinstance(config, GelloConfig) else QuestOperator(config)
    else:
        # FR3

        scene = EmptyWorldFR3Duo()
        cfg = scene.config()
        cfg.sim_cfg = SimConfig(async_control=True, realtime=True, frequency=30, max_convergence_steps=500)
        cfg.relative_to = RelativeTo.CONFIGURED_ORIGIN
        if cfg.root_frame_objects is None:
            cfg.root_frame_objects = {}
        # cfg.root_frame_objects["green_cube"] = (rcs.OBJECT_PATHS["green_cube"], Pose(translation=[0.5, 0, 0.5], quaternion=[0, 0, 0, 1]))
        cfg.task_cfg = PickTaskConfig(robot_name="left")

        env_rel = scene.create_env(cfg)
        # env_rel = StorageWrapper(
        #     env_rel, DATASET_PATH, INSTRUCTION, batch_size=32, max_rows_per_group=100, max_rows_per_file=1000
        # )

        sim = env_rel.get_wrapper_attr("sim")
        MujocoPublisher(sim.model, sim.data, MQ3_ADDR, visible_geoms_groups=list(range(1, 3)))
        operator = GelloOperator(config, sim) if isinstance(config, GelloConfig) else QuestOperator(config, sim)
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
