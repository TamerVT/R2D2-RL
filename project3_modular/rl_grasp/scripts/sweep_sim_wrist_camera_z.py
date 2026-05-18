from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np

from project3_modular.rl_grasp.envs.so101_local_grasp_env import SO101LocalGraspEnv


CAMERA_NAME = "robotwrist"
OUT_PATH = Path("/tmp/so101_wrist_z_sweep.png")


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def main() -> None:
    env = SO101LocalGraspEnv(open_gui=False)
    env.reset(seed=0)

    # Match real setup: gripper starts closed.
    closed_gripper_action = np.array([0, 0, 0, 0, 0, -1], dtype=np.float32)
    for _ in range(10):
        env.step(closed_gripper_action)

    sim = env.env.get_wrapper_attr("sim")
    model = sim.model
    data = sim.data

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id < 0:
        raise RuntimeError(f"Camera {CAMERA_NAME!r} not found.")

    base_pos = model.cam_pos[cam_id].copy()

    candidates = [
        ("z -1.00cm", (0.000, 0.000, -0.0100)),
        ("z -0.75cm", (0.000, 0.000, -0.0075)),
        ("z -0.50cm", (0.000, 0.000, -0.0050)),
        ("z -0.25cm", (0.000, 0.000, -0.0025)),
        ("base",      (0.000, 0.000,  0.0000)),
        ("z +0.25cm", (0.000, 0.000,  0.0025)),
        ("z +0.50cm", (0.000, 0.000,  0.0050)),
        ("z +0.75cm", (0.000, 0.000,  0.0075)),
        ("z +1.00cm", (0.000, 0.000,  0.0100)),
    ]

    renderer = mujoco.Renderer(model, height=240, width=320)
    tiles: list[np.ndarray] = []

    for name, delta in candidates:
        model.cam_pos[cam_id] = base_pos + np.asarray(delta, dtype=np.float64)
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=CAMERA_NAME)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tiles.append(label(bgr, name))

    model.cam_pos[cam_id] = base_pos
    mujoco.mj_forward(model, data)

    grid = np.vstack(
        [
            np.hstack(tiles[0:3]),
            np.hstack(tiles[3:6]),
            np.hstack(tiles[6:9]),
        ]
    )

    cv2.imwrite(str(OUT_PATH), grid)
    print(f"Saved z sweep to {OUT_PATH}")

    env.close()


if __name__ == "__main__":
    main()
