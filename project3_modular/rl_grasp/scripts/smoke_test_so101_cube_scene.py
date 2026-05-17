from __future__ import annotations

import logging
import time

import gymnasium as gym
import numpy as np

import rcs
from rcs import sim
from rcs._core.sim import SimConfig
from rcs.envs.base import (
    ControlMode,
    CoverWrapper,
    GripperWrapper,
    RelativeActionSpace,
    RelativeTo,
    RobotWrapper,
    SimEnv,
)
from rcs.envs.configs import EmptyWorldSO101
from rcs.envs.sim import GripperWrapperSim, RobotSimWrapper


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_so101_cube_env(*, open_gui: bool = True) -> gym.Env:
    scene = EmptyWorldSO101()

    base_cfg = scene.config()

    # Lower the whole SO101 so its physical base sits on the floor plane.
    base_cfg.robot_to_shared_base_frame = {
        "robot": rcs.common.Pose(
            translation=np.array([0.0, 0.0, -0.03])
        )
    }

    base_cfg.root_frame_objects = {
        "green_cube": (
            "project3_modular/rl_grasp/assets/cubes/green_cube_2cm.xml",
            rcs.common.Pose(
                translation=np.array([0.25, 0.00, 0.01]),
                quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            ),
        )
    }

    cfg = scene.prefixed_cfg(base_cfg)
    so101_name = scene.lead_robot_name(cfg)

    robot_cfg = cfg.robot_cfgs[so101_name]
    gripper_cfg = cfg.gripper_cfgs[so101_name]

    sim_cfg = SimConfig(
        realtime=False,
        async_control=False,
    )

    kinematic_model_path, attachment_site = scene.kinematics_cfg(cfg)[so101_name]
    ik = rcs.common.Pin(
        kinematic_model_path,
        attachment_site,
    )

    mjmodel = scene.create_model(cfg)
    simulation = sim.Sim(mjmodel, sim_cfg)

    robot = rcs.sim.SimRobot(simulation, ik, robot_cfg)

    env: gym.Env = SimEnv(simulation)
    env = RobotWrapper(env, robot, ControlMode.JOINTS)

    gripper = sim.SimGripper(simulation, gripper_cfg)
    env = GripperWrapper(env, gripper)

    env = RobotSimWrapper(env)
    env = GripperWrapperSim(env)

    env = RelativeActionSpace(
        env,
        max_mov=np.deg2rad(5),
        relative_to=RelativeTo.LAST_STEP,
    )

    env = CoverWrapper(env)

    if open_gui:
        env.get_wrapper_attr("sim").open_gui()

    return env


def zero_like(x):
    if isinstance(x, dict):
        return {k: zero_like(v) for k, v in x.items()}
    arr = np.asarray(x)
    return np.zeros_like(arr)


def main() -> None:
    env = build_so101_cube_env(open_gui=True)

    obs, info = env.reset()

    print("\n=== SO101 cube scene created ===")
    print("Observation:")
    print(obs)
    print("\nInfo:")
    print(info)
    print("\nAction space:")
    print(env.action_space)

    print("\nKeeping zero action for 200 steps so you can inspect the scene...")
    zero_action = zero_like(env.action_space.sample())

    for step in range(200):
        obs, reward, terminated, truncated, info = env.step(zero_action)

        if step % 25 == 0:
            print(
                f"step={step:03d} "
                f"reward={reward} "
                f"terminated={terminated} "
                f"truncated={truncated}"
            )

        if terminated or truncated:
            print("Environment terminated early.")
            break

        time.sleep(0.03)

    env.close()
    print("\nCube scene smoke test complete.")


if __name__ == "__main__":
    main()
