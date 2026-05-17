from __future__ import annotations

import time

import numpy as np

import rcs
from rcs._core.common import Pose
from rcs.envs.base import ControlMode, RelativeTo
from rcs.envs.configs import EmptyWorldSO101
from rcs.envs.tasks import PickTaskConfig


CUSTOM_GREEN_CUBE_XML = (
    "project3_modular/rl_grasp/assets/cubes/green_cube_2cm.xml"
)


def build_so101_pick_env():
    scene = EmptyWorldSO101()
    cfg = scene.config()

    # Lower SO101 so its base sits on the floor plane.
    cfg.robot_to_shared_base_frame = {
        "robot": rcs.common.Pose(
            translation=np.array([0.0, 0.0, -0.03])
        )
    }

    # Joint-delta control is closest to what we will deploy later.
    cfg.control_mode = ControlMode.JOINTS
    cfg.relative_to = RelativeTo.LAST_STEP
    cfg.max_relative_movement = np.deg2rad(5)

    # Use RCS's existing pick task machinery, but with our custom 2 cm cube.
    pick_task_cfg = PickTaskConfig(
        robot_name="robot",
        object_center_to_root_frame=Pose(
            translation=np.array([0.25, 0.00, 0.01]),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
        ),
        object_joint="green_cube_joint",
        include_rotation=True,
    )

    pick_task_cfg.object_xml = CUSTOM_GREEN_CUBE_XML
    cfg.task_cfg = pick_task_cfg

    # Keep spawn randomization local for now.
    # These fields are not exposed by PickTaskConfig itself, so the default
    # RandomSquareObjPos width from RCS remains active initially.
    # We will tighten that in the custom RL wrapper right after this smoke test.

    env = scene.create_env(cfg)
    env.get_wrapper_attr("sim").open_gui()

    return env


def zero_like(x):
    if isinstance(x, dict):
        return {k: zero_like(v) for k, v in x.items()}
    arr = np.asarray(x)
    return np.zeros_like(arr)


def main() -> None:
    env = build_so101_pick_env()

    obs, info = env.reset()

    print("\n=== SO101 RCS PickTask smoke test ===")
    print("Observation:")
    print(obs)
    print("\nInfo:")
    print(info)
    print("\nAction space:")
    print(env.action_space)

    zero_action = zero_like(env.action_space.sample())

    print("\nStepping with zero actions for 200 steps...")
    for step in range(200):
        obs, reward, terminated, truncated, info = env.step(zero_action)

        if step % 25 == 0:
            print(
                f"step={step:03d} "
                f"reward={reward:.4f} "
                f"terminated={terminated} "
                f"truncated={truncated} "
                f"success={info.get('success')} "
                f"is_grasped={info.get('is_grasped')}"
            )

        if terminated or truncated:
            print("Environment terminated.")
            break

        time.sleep(0.03)

    env.close()
    print("\nPickTask smoke test complete.")


if __name__ == "__main__":
    main()
