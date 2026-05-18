from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from project3_modular.rl_grasp.envs.so101_local_grasp_env import SO101LocalGraspEnv


CAMERA_NAME = "robotwrist"
OUT_PATH = Path("/tmp/so101_wrist_angle_xy_sweep.png")

ANGLES = [-20, -10, 0, 10, 20]


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (7, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def main() -> None:
    env = SO101LocalGraspEnv(open_gui=False)
    env.reset(seed=0)

    # Match the real setup: closed gripper.
    closed_gripper_action = np.array([0, 0, 0, 0, 0, -1], dtype=np.float32)
    for _ in range(10):
        env.step(closed_gripper_action)

    sim = env.env.get_wrapper_attr("sim")
    model = sim.model
    data = sim.data

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id < 0:
        raise RuntimeError(f"Camera {CAMERA_NAME!r} not found.")

    base_quat_wxyz = model.cam_quat[cam_id].copy()
    base_rot = R.from_quat(wxyz_to_xyzw(base_quat_wxyz))

    renderer = mujoco.Renderer(model, height=192, width=256)
    rows: list[np.ndarray] = []

    # Rows: local y, columns: local x
    for y_angle in ANGLES:
        row_tiles: list[np.ndarray] = []
        for x_angle in ANGLES:
            local_rot = R.from_euler(
                "xy",
                [x_angle, y_angle],
                degrees=True,
            )
            new_rot = base_rot * local_rot
            model.cam_quat[cam_id] = xyzw_to_wxyz(new_rot.as_quat())
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=CAMERA_NAME)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            row_tiles.append(label(bgr, f"x {x_angle:+d}°, y {y_angle:+d}°"))

        rows.append(np.hstack(row_tiles))

    model.cam_quat[cam_id] = base_quat_wxyz
    mujoco.mj_forward(model, data)

    grid = np.vstack(rows)
    cv2.imwrite(str(OUT_PATH), grid)
    print(f"Saved x/y angle sweep to {OUT_PATH}")

    env.close()


if __name__ == "__main__":
    main()
