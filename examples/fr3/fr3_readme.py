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
from rcs.envs.sim import GripperWrapperSim, RobotSimWrapper
from rcs.envs.utils import (
    default_mujoco_cameraset_cfg,
    default_sim_gripper_cfg,
    default_sim_robot_cfg,
)

import rcs
from rcs import sim

if __name__ == "__main__":
    # default configs
    robot_cfg = default_sim_robot_cfg(scene="fr3_empty_world")
    gripper_cfg = default_sim_gripper_cfg()
    cameras = default_mujoco_cameraset_cfg()
    sim_cfg = SimConfig()
    sim_cfg.realtime = True
    sim_cfg.async_control = True
    sim_cfg.frequency = 1  # in Hz (1 sec delay)

    simulation = sim.Sim(mjcf_scene_path, sim_cfg)
    ik = rcs.common.Pin(
        robot_cfg.kinematic_model_path,
        robot_cfg.attachment_site,
        urdf=False,
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
    camera_set = SimCameraSet(simulation, cameras, physical_units=True, render_on_demand=True)
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
