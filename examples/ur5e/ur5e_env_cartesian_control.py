import logging
from time import sleep

import gymnasium as gym
import numpy as np
from rcs._core.common import RobotPlatform
from rcs._core.sim import SimConfig
from rcs.camera.sim import SimCameraSet
from rcs.envs.base import (
    CameraSetWrapper,
    ControlMode,
    CoverWrapper,
    GripperWrapper,
    RelativeActionSpace,
    RelativeTo,
    RobotWrapper,
    SimEnv,
)
from rcs.envs.configs import EmptyWorldUR5e
from rcs.envs.sim import GripperWrapperSim, RobotSimWrapper
from rcs_ur5e.creators import RCSUR5eEnvCreator
from rcs_ur5e.hw import UR5eConfig

import rcs
from rcs import sim

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ROBOT_IP = "192.168.1.15"
ROBOT_INSTANCE = RobotPlatform.SIMULATION  # Change to RobotPlatform.HARDWARE for real robot


def main():
    if ROBOT_INSTANCE == RobotPlatform.HARDWARE:
        robot_cfg = UR5eConfig(ip=ROBOT_IP)
        robot_cfg.async_control = False
        env_rel = RCSUR5eEnvCreator()(
            robot_cfg=robot_cfg,
            control_mode=ControlMode.CARTESIAN_TQuat,
            camera_set=None,
            max_relative_movement=0.2,
            relative_to=RelativeTo.LAST_STEP,
        )
    else:
        scene = EmptyWorldUR5e()
        cfg = scene.prefixed_cfg(scene.config())
        ur5e = scene.lead_robot_name(cfg)

        robot_cfg = cfg.robot_cfgs[ur5e]
        gripper_cfg = cfg.gripper_cfgs[ur5e]  # type: ignore[index]
        camera_cfgs = cfg.camera_cfgs
        sim_cfg = SimConfig(
            realtime=False,
            async_control=False,
        )

        mjmodel = scene.create_model(cfg)
        simulation = sim.Sim(mjmodel, sim_cfg)

        kinematic_model_path, attachment_site = scene.kinematics_cfg(cfg)[ur5e]
        ik = rcs.common.Pin(
            kinematic_model_path,
            attachment_site,
        )

        robot = rcs.sim.SimRobot(simulation, ik, robot_cfg)
        env_rel: gym.Env = SimEnv(simulation)
        env_rel = RobotWrapper(env_rel, robot, ControlMode.CARTESIAN_TQuat)

        gripper = sim.SimGripper(simulation, gripper_cfg)
        env_rel = GripperWrapper(env_rel, gripper)

        env_rel = RobotSimWrapper(env_rel)
        env_rel = GripperWrapperSim(env_rel)

        if camera_cfgs is not None:
            camera_set = SimCameraSet(simulation, camera_cfgs, physical_units=True, render_on_demand=True)
            env_rel = CameraSetWrapper(env_rel, camera_set, include_depth=True)  # type: ignore[arg-type]

        env_rel = RelativeActionSpace(
            env_rel,
            max_mov=(0.1, np.deg2rad(5)),
            relative_to=RelativeTo.LAST_STEP,
        )
        env_rel = CoverWrapper(env_rel)
        env_rel.get_wrapper_attr("sim").open_gui()

    obs, info = env_rel.reset()

    for _ in range(100):
        for _ in range(10):
            # move 1cm in x direction (forward) and close gripper
            act = {"tquat": [0.01, 0, 0, 0, 0, 0, 1.0], "gripper": [0]}
            obs, reward, terminated, truncated, info = env_rel.step(act)
            sleep(0.6)
        for _ in range(10):
            # move 1cm in negative x direction (backward) and open gripper
            act = {"tquat": [-0.01, 0, 0, 0, 0, 0, 1.0], "gripper": [1]}
            obs, reward, terminated, truncated, info = env_rel.step(act)
            sleep(0.6)


if __name__ == "__main__":
    main()
