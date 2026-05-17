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


def build_so101_joint_env(*, open_gui: bool = True) -> gym.Env:
    scene = EmptyWorldSO101()
    cfg = scene.prefixed_cfg(scene.config())
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

    # Small relative joint actions.
    env = RelativeActionSpace(
        env,
        max_mov=np.deg2rad(5),
        relative_to=RelativeTo.LAST_STEP,
    )

    env = CoverWrapper(env)

    if open_gui:
        env.get_wrapper_attr("sim").open_gui()

    return env


def main() -> None:
    env = build_so101_joint_env(open_gui=True)

    obs, info = env.reset()

    print("\n=== Environment created ===")
    print("Observation type:", type(obs))
    print("Observation:")
    print(obs)
    print("\nInfo:")
    print(info)
    print("\nAction space:")
    print(env.action_space)

    print("\nStepping with zero actions for 50 steps...")
    zero_action = env.action_space.sample()

    # Convert sampled structure to all zeros while preserving nested keys/shapes.
    def zero_like(x):
        if isinstance(x, dict):
            return {k: zero_like(v) for k, v in x.items()}
        arr = np.asarray(x)
        return np.zeros_like(arr)

    zero_action = zero_like(zero_action)
    print("Zero action:")
    print(zero_action)

    for step in range(50):
        obs, reward, terminated, truncated, info = env.step(zero_action)

        if step % 10 == 0:
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

    print("\nStepping with random actions for 50 steps...")
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 10 == 0:
            print(
                f"random step={step:03d} "
                f"reward={reward} "
                f"terminated={terminated} "
                f"truncated={truncated}"
            )

        if terminated or truncated:
            print("Environment terminated early.")
            break

        time.sleep(0.03)

    env.close()
    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
