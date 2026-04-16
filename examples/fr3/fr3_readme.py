from time import sleep

import gymnasium as gym
import numpy as np
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
from rcs.envs.scenes import EmptyWorldFR3
from rcs.envs.sim import GripperWrapperSim, RobotSimWrapper

import rcs
from rcs import sim

if __name__ == "__main__":
    # default configs
    scene = EmptyWorldFR3()
    robot_cfg = next(iter(scene.prefixed_cfg.robot_cfgs.values()))
    gripper_cfg = next(iter(scene.prefixed_cfg.gripper_cfgs.values()))
    camera_cfgs = scene.prefixed_cfg.camera_cfgs
    sim_cfg = SimConfig(
        realtime=True,
        async_control=True,
        frequency=1,  # in Hz (1 sec delay)
    )

    s = scene.load_scene()
    s.save_mjcf("scene.xml")
    simulation = sim.Sim(s, sim_cfg)
    ik_robot_cfg = next(iter(scene.cfg.robot_cfgs.values()))
    ik = rcs.common.Pin(
        ik_robot_cfg.kinematic_model_path,
        ik_robot_cfg.attachment_site,
    )

    # base env
    robot = rcs.sim.SimRobot(simulation, ik, robot_cfg)
    env: gym.Env = SimEnv(simulation)
    env = RobotWrapper(env, robot, ControlMode.CARTESIAN_TQuat)

    # gripper
    gripper = sim.SimGripper(simulation, gripper_cfg)
    env = GripperWrapper(env, gripper)

    env = RobotSimWrapper(env)
    env = GripperWrapperSim(env)

    # camera
    camera_set = SimCameraSet(simulation, camera_cfgs, physical_units=True, render_on_demand=True)
    env = CameraSetWrapper(env, camera_set, include_depth=True)  # type: ignore

    # relative actions bounded by 10cm translation and 10 degree rotation
    env = RelativeActionSpace(env, max_mov=(0.1, np.deg2rad(10)), relative_to=RelativeTo.LAST_STEP)
    env = CoverWrapper(env)

    env.get_wrapper_attr("sim").open_gui()
    # wait for gui to open
    sleep(1)
    env.reset()

    # access low level robot api to get current cartesian position
    print(env.get_wrapper_attr("robot").get_cartesian_position())

    for _ in range(10):
        # move 1cm in x direction (forward) and close gripper
        act = {"tquat": [0.01, 0, 0, 0, 0, 0, 1], "gripper": [0]}
        obs, reward, terminated, truncated, info = env.step(act)
        print(obs)
