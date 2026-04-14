import logging
from time import sleep

from rcs._core.common import RobotPlatform
from rcs.envs.base import ControlMode, RelativeTo
from rcs.envs.creators import SimEnvCreator
from rcs_xarm7.creators import RCSXArm7EnvCreator
from rcs_xarm7.hw import XArm7Config

import rcs
from rcs import sim

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

"""
The example shows how to create a xArm7 environment with Cartesian control
and a relative action space. The example works both with a real robot and in
simulation.

To test with a real robot, set ROBOT_INSTANCE to RobotPlatform.HARDWARE,
install the rcs_xarm7 extension (`pip install extensions/rcs_xarm7`)
and set the ROBOT_IP variable to the robot's IP address.
"""

ROBOT_IP = "192.168.1.245"
ROBOT_INSTANCE = RobotPlatform.SIMULATION
# ROBOT_INSTANCE = RobotPlatform.HARDWARE


def main():

    if ROBOT_INSTANCE == RobotPlatform.HARDWARE:
        robot_cfg = XArm7Config(ip=ROBOT_IP)
        env_rel = RCSXArm7EnvCreator()(
            robot_cfg=robot_cfg,
            control_mode=ControlMode.CARTESIAN_TQuat,
            relative_to=RelativeTo.LAST_STEP,
            max_relative_movement=0.5,
        )
    else:
        robot_sim_cfg = sim.SimRobotConfig()
        robot_sim_cfg.actuators = [
            "act1",
            "act2",
            "act3",
            "act4",
            "act5",
            "act6",
            "act7",
        ]
        robot_sim_cfg.joints = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        robot_sim_cfg.base = "base"
        robot_sim_cfg.robot_type = rcs.common.RobotType.XArm7
        robot_sim_cfg.attachment_site = "attachment_site"
        robot_sim_cfg.arm_collision_geoms = []
        scene = rcs.scenes["xarm7_empty_world"]
        robot_sim_cfg.kinematic_model_path = rcs.scenes["xarm7_empty_world"].mjcf_robot
        env_rel = SimEnvCreator()(
            robot_cfg=robot_sim_cfg,
            control_mode=ControlMode.CARTESIAN_TQuat,
            gripper_cfg=None,
            # cameras=default_mujoco_cameraset_cfg(),
            max_relative_movement=0.5,
            relative_to=RelativeTo.LAST_STEP,
        )
        sleep(3)  # wait for gui to open
        env_rel.get_wrapper_attr("sim").open_gui()
    obs, info = env_rel.reset()

    for _ in range(100):
        for _ in range(10):
            # move 1cm in x direction (forward) and close gripper
            act = {"tquat": [0.01, 0, 0, 0, 0, 0, 1]}
            obs, reward, terminated, truncated, info = env_rel.step(act)
        for _ in range(10):
            # move 1cm in negative x direction (backward) and open gripper
            act = {"tquat": [-0.01, 0, 0, 0, 0, 0, 1]}
            obs, reward, terminated, truncated, info = env_rel.step(act)


if __name__ == "__main__":
    main()
