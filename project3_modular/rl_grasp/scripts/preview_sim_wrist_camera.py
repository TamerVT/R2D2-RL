from pathlib import Path

import cv2
import numpy as np
import mujoco

from project3_modular.rl_grasp.envs.so101_local_grasp_env import SO101LocalGraspEnv


def main() -> None:
    env = SO101LocalGraspEnv(open_gui=False)
    env.reset(seed=0)

    # Match the real local-grasp setup: start with the gripper closed.
    closed_gripper_action = np.array([0, 0, 0, 0, 0, -1], dtype=np.float32)
    for _ in range(10):
        obs, _, _, _, _ = env.step(closed_gripper_action)

    sim = env.env.get_wrapper_attr("sim")
    model = sim.model
    data = sim.data

    camera_names = [model.camera(i).name for i in range(model.ncam)]
    print("Available cameras:", camera_names)

    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera="robotwrist")
    rgb = renderer.render()

    out = Path("/tmp/so101_wrist_preview.png")
    cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"Saved wrist camera preview to {out}")

    env.close()


if __name__ == "__main__":
    main()
