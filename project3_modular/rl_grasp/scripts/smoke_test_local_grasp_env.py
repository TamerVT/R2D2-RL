from __future__ import annotations

import argparse
import time

import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project3_modular.rl_grasp.envs.so101_local_grasp_env import (
    SO101LocalGraspConfig,
    SO101LocalGraspEnv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    return parser.parse_args()


def print_obs(obs) -> None:
    print("Observation:")
    for key, value in obs.items():
        arr = np.asarray(value)
        print(f"  {key:14s} shape={arr.shape} value={arr}")


def main() -> None:
    args = parse_args()

    config = SO101LocalGraspConfig(
        max_episode_steps=args.steps,
        target_color="green",
    )

    env = SO101LocalGraspEnv(
        config=config,
        open_gui=args.gui,
    )

    obs, info = env.reset()

    print("\n=== SO101LocalGraspEnv smoke test ===")
    print_obs(obs)
    print("\nInfo:")
    print(info)
    print("\nAction space:")
    print(env.action_space)
    print("\nObservation space:")
    print(env.observation_space)

    if args.random_actions:
        print("\nStepping random actions...")
    else:
        print("\nStepping zero actions...")

    for step in range(args.steps):
        if args.random_actions:
            action = env.action_space.sample()
        else:
            action = np.zeros(env.action_space.shape, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        if step % 10 == 0:
            print(
                f"step={step:03d} "
                f"reward={reward:.4f} "
                f"terminated={terminated} "
                f"truncated={truncated} "
                f"success={info.get('success')} "
                f"is_grasped={info.get('is_grasped')}"
            )

        if terminated or truncated:
            print("Episode ended.")
            break

        if args.gui:
            time.sleep(0.03)

    env.close()
    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
