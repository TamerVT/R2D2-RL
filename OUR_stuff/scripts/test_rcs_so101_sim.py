"""Smoke test for the RCS SO101 sim env (lerobot-p3-rcs conda env).

Run from the project3/ root after activating ``lerobot-p3-rcs``::

    MUJOCO_GL=egl python scripts/test_rcs_so101_sim.py

The script creates the default ``EmptyWorldSO101`` scene, resets it once,
steps three random actions, and prints the observation and action space
shapes. It does not require LeRobot hardware paths or the wrist camera.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("MUJOCO_GL", "egl")

    try:
        import rcs
        import rcs.envs.tasks  # noqa: F401  - registers PickTask
        from rcs.envs.configs import EmptyWorldSO101
    except ImportError as exc:
        print(f"[error] RCS not importable: {exc}", file=sys.stderr)
        print(
            "        activate lerobot-p3-rcs and ensure rcs + rcs_so101 are installed.",
            file=sys.stderr,
        )
        return 2

    print(f"rcs version: {rcs.__version__}")
    print(f"ROBOTS:      {list(rcs.ROBOTS.keys())}")
    print(f"TASKS:       {list(rcs.TASKS.keys())}")

    scene = EmptyWorldSO101()
    cfg = scene.config()
    env = scene.create_env(cfg)

    obs, info = env.reset(seed=0)
    print(f"obs keys:      {list(obs.keys())}")
    print(f"action_space:  {env.action_space}")

    for step_i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"step {step_i}: reward={reward}, terminated={terminated}, truncated={truncated}")

    print("smoke test PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
