import logging
import threading
from time import sleep

import numpy as np
import rcs
from rcs._core import common
from rcs._core.common import RPY, Pose, RobotPlatform
from rcs._core.sim import CameraType, SimCameraConfig, SimConfig
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
# from rcs_realsense.utils import default_realsense
from simpub.core.simpub_server import SimPublisher
from simpub.parser.simdata import SimObject, SimScene
from simpub.sim.mj_publisher import MujocoPublisher
from simpub.xr_device.meta_quest3 import MetaQuest3

logger = logging.getLogger(__name__)

# download the iris apk from the following repo release: https://github.com/intuitive-robots/IRIS-Meta-Quest3
# in order to use usb connection install adb install adb
# sudo apt install android-tools-adb
# install it on your quest with
# adb install IRIS-Meta-Quest3.apk


INCLUDE_ROTATION = True
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
MQ3_ADDR = "10.42.0.1"
# DIGIT_DICT = {
#     "digit_right_left": "D21182",
#     "digit_right_right": "D21193"
# }
DIGIT_DICT = None

DATASET_PATH = "test_data_iris_dual_arm14"
INSTRUCTION = "build a tower with the blocks in front of you"


class MySimPublisher(SimPublisher):
    def get_update(self):
        return {}


class MySimScene(SimScene):
    def __init__(self):
        super().__init__()
        self.root = SimObject(name="root")


class QuestReader(threading.Thread):

    transform_to_robot = Pose()

    def __init__(self, env: RelativeActionSpace):
        super().__init__()
        self._reader = MetaQuest3("RCSNode")

        self._resource_lock = threading.Lock()
        self._env_lock = threading.Lock()
        self._reset_lock = threading.Lock()
        self._env = env

        self.controller_names = ROBOT2IP.keys() if ROBOT_INSTANCE == RobotPlatform.HARDWARE else ROBOT2ID.keys()
        self._trg_btn = {"left": "index_trigger", "right": "index_trigger"}
        self._grp_btn = {"left": "hand_trigger", "right": "hand_trigger"}
        self._start_btn = "A"
        self._stop_btn = "B"
        self._unsuccessful_btn = "Y"

        self._prev_data = None
        self._exit_requested = False
        self._grp_pos = {key: 1.0 for key in self.controller_names}  # start with opened gripper
        self._last_controller_pose = {key: Pose() for key in self.controller_names}
        self._offset_pose = {key: Pose() for key in self.controller_names}

        for robot in ROBOT2IP if ROBOT_INSTANCE == RobotPlatform.HARDWARE else ROBOT2ID:
            self._env.envs[robot].set_origin_to_current()

        self._step_env = False
        self._set_frame = {key: Pose() for key in self.controller_names}

    def next_action(self) -> tuple[dict[str, Pose], dict[str, float]]:
        with self._resource_lock:
            transforms = {}
            for controller in self.controller_names:
                transform = Pose(
                    translation=(
                        self._last_controller_pose[controller].translation()  # type: ignore
                        - self._offset_pose[controller].translation()
                    ),
                    quaternion=(
                        self._last_controller_pose[controller] * self._offset_pose[controller].inverse()
                    ).rotation_q(),
                )

                set_axes = Pose(quaternion=self._set_frame[controller].rotation_q())

                transform = (
                    QuestReader.transform_to_robot
                    * set_axes.inverse()
                    * transform
                    * set_axes
                    * QuestReader.transform_to_robot.inverse()
                )
                if not INCLUDE_ROTATION:
                    transform = Pose(translation=transform.translation())  # identity rotation
                transforms[controller] = transform
            return transforms, {key: self._grp_pos[key] for key in self.controller_names}

    def run(self):
        rate_limiter = SimpleFrameRate(90, "teleop readout")
        warning_raised = False
        while not self._exit_requested:
            # pos, buttons = self._reader.get_transformations_and_buttons()
            input_data = self._reader.get_controller_data()
            if not warning_raised and input_data is None:
                logger.warning("[Quest Reader] packets empty")
                warning_raised = True
                sleep(0.5)
                continue
            if input_data is None:
                sleep(0.5)
                continue
            if warning_raised:
                logger.warning("[Quest Reader] packets arriving again")
                warning_raised = False

            # start recording
            if input_data[self._start_btn] and (self._prev_data is None or not self._prev_data[self._start_btn]):
                print("start button pressed")
                with self._env_lock:
                    self._env.get_wrapper_attr("start_record")()

            if input_data[self._stop_btn] and (self._prev_data is None or not self._prev_data[self._stop_btn]):
                print("reset successful pressed: resetting env")
                with self._reset_lock:
                    # set successful
                    self._env.get_wrapper_attr("success")()
                    # sleep to allow to let the robot reach the goal
                    sleep(1)
                    # this might also move the robot to the home position
                    self._env.reset()
                    for controller in self.controller_names:
                        self._offset_pose[controller] = Pose()
                        self._last_controller_pose[controller] = Pose()
                        self._grp_pos[controller] = 1
                    continue

            # reset unsuccessful
            if input_data[self._unsuccessful_btn] and (
                self._prev_data is None or not self._prev_data[self._unsuccessful_btn]
            ):
                print("reset unsuccessful pressed: resetting env")
                with self._env_lock:
                    self._env.reset()

            for controller in self.controller_names:
                last_controller_pose = Pose(
                    translation=np.array(input_data[controller]["pos"]),
                    quaternion=np.array(input_data[controller]["rot"]),
                )
                if controller == "left":
                    last_controller_pose = (
                        Pose(translation=np.array([0, 0, 0]), rpy=RPY(roll=0, pitch=0, yaw=np.deg2rad(180)))  # type: ignore
                        * last_controller_pose
                    )

                if input_data[controller][self._trg_btn[controller]] and (
                    self._prev_data is None or not self._prev_data[controller][self._trg_btn[controller]]
                ):
                    # trigger just pressed (first data sample with button pressed)

                    with self._resource_lock:
                        self._offset_pose[controller] = last_controller_pose
                        self._last_controller_pose[controller] = last_controller_pose

                elif not input_data[controller][self._trg_btn[controller]] and (
                    self._prev_data is None or self._prev_data[controller][self._trg_btn[controller]]
                ):
                    # released
                    transform = Pose(
                        translation=(
                            self._last_controller_pose[controller].translation()  # type: ignore
                            - self._offset_pose[controller].translation()
                        ),
                        quaternion=(
                            self._last_controller_pose[controller] * self._offset_pose[controller].inverse()
                        ).rotation_q(),
                    )
                    print(np.linalg.norm(transform.translation()))
                    print(np.rad2deg(transform.total_angle()))
                    with self._resource_lock:
                        self._last_controller_pose[controller] = Pose()
                        self._offset_pose[controller] = Pose()
                    with self._env_lock:
                        self._env.envs[controller].set_origin_to_current()

                elif input_data[controller][self._trg_btn[controller]]:
                    # button is pressed
                    with self._resource_lock:
                        self._last_controller_pose[controller] = last_controller_pose
                else:
                    # trg button is not pressed
                    # TODO: deactivated for now
                    # if data["buttons"][self._home_btn] and (
                    #     self._prev_data is None or not self._prev_data["buttons"][self._home_btn]
                    # ):
                    #     # home button just pressed
                    #     with self._env_lock:
                    #         self._env.unwrapped.robot.move_home()
                    #         self._env.set_origin_to_current()
                    pass

                if input_data[controller][self._grp_btn[controller]] and (
                    self._prev_data is None or not self._prev_data[controller][self._grp_btn[controller]]
                ):
                    # just pressed
                    self._grp_pos[controller] = 0
                if not input_data[controller][self._grp_btn[controller]] and (
                    self._prev_data is None or self._prev_data[controller][self._grp_btn[controller]]
                ):
                    # just released
                    self._grp_pos[controller] = 1

            self._prev_data = input_data
            rate_limiter()

    def stop(self):
        self._exit_requested = True
        self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def stop_env_loop(self):
        self._step_env = False

    def environment_step_loop(self):
        rate_limiter = SimpleFrameRate(RECORD_FPS, "env loop")
        self._step_env = True
        while self._step_env:
            if self._exit_requested:
                self._step_env = False
                break
            with self._reset_lock:
                transforms, grippers = self.next_action()
                actions = {}
                for robot, transform in transforms.items():
                    action = dict(
                        LimitedTQuatRelDictType(tquat=np.concatenate([transform.translation(), transform.rotation_q()]))  # type: ignore
                    )

                    action.update(GripperDictType(gripper=grippers[robot]))
                    actions[robot] = action

                self._env.step(actions)
            rate_limiter()


def main():
    if ROBOT_INSTANCE == RobotPlatform.HARDWARE:

        camera_set = HardwareCameraSet([default_realsense(CAMERA_DICT), default_digit(DIGIT_DICT)]) if CAMERA_DICT is not None else None  # type: ignore
        env_rel = RCSFR3MultiEnvCreator()(
            name2ip=ROBOT2IP,
            camera_set=camera_set,
            robot_cfg=default_fr3_hw_robot_cfg(async_control=True),
            control_mode=ControlMode.CARTESIAN_TQuat,
            gripper_cfg=default_fr3_hw_gripper_cfg(async_control=True),
            max_relative_movement=(0.5, np.deg2rad(90)),
            relative_to=RelativeTo.CONFIGURED_ORIGIN,
        )
        env_rel = StorageWrapper(
            env_rel, DATASET_PATH, INSTRUCTION, batch_size=32, max_rows_per_group=100, max_rows_per_file=1000
        )
        MySimPublisher(MySimScene(), MQ3_ADDR)

        # DIGITAL TWIN: comment out the line above and uncomment the lines below to use a digital twin
        # Note: only supported for one arm at the moment

        # robot_cfg = default_sim_robot_cfg("fr3_empty_world")
        # sim_cfg = SimConfig()
        # sim_cfg.async_control = True
        # twin_env = SimMultiEnvCreator()(
        #     name2id=ROBOT2IP,
        #     robot_cfg=robot_cfg,
        #     control_mode=ControlMode.JOINTS,
        #     gripper_cfg=default_sim_gripper_cfg(),
        #     sim_cfg=sim_cfg,
        # )
        # sim = env_rel.unwrapped.envs[ROBOT2IP.keys().__iter__().__next__()].sim
        # sim.open_gui()
        # MujocoPublisher(sim.model, sim.data, MQ3_ADDR, visible_geoms_groups=list(range(1, 3)))
        # env_rel = DigitalTwin(env_rel, twin_env)

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

        resolution = (256, 256)
        cameras = {
            cam: SimCameraConfig(
                identifier=cam,
                type=CameraType.fixed,
                resolution_height=resolution[1],
                resolution_width=resolution[0],
                frame_rate=0,
            )
            for cam in ["side", "wrist"]
        }

        sim_cfg = SimConfig()
        sim_cfg.async_control = True
        env_rel = SimMultiEnvCreator()(
            name2id=ROBOT2ID,
            robot_cfg=robot_cfg,
            control_mode=ControlMode.CARTESIAN_TQuat,
            gripper_cfg=default_sim_gripper_cfg(),
            # cameras=cameras,
            max_relative_movement=0.5,
            relative_to=RelativeTo.CONFIGURED_ORIGIN,
            sim_cfg=sim_cfg,
        )
        # sim = env_rel.unwrapped.envs[ROBOT2ID.keys().__iter__().__next__()].sim  # type: ignore
        sim = env_rel.get_wrapper_attr("sim")
        sim.open_gui()
        # MySimPublisher(MySimScene(), MQ3_ADDR)
        MujocoPublisher(sim.model, sim.data, MQ3_ADDR, visible_geoms_groups=list(range(1, 3)))

    env_rel.reset()

    with env_rel, QuestReader(env_rel) as action_server:  # type: ignore
        action_server.environment_step_loop()


if __name__ == "__main__":
    main()
