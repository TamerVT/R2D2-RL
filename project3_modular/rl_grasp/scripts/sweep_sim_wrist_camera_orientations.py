from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from project3_modular.rl_grasp.envs.so101_local_grasp_env import SO101LocalGraspEnv


CAMERA_NAME = "robotwrist"
OUT_PATH = Path("/tmp/so101_wrist_orientation_sweep.png")


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def main() -> None:
    env = SO101LocalGraspEnv(open_gui=False)
    env.reset(seed=0)

    sim = env.env.get_wrapper_attr("sim")
    model = sim.model
    data = sim.data

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id < 0:
        raise RuntimeError(f"Camera {CAMERA_NAME!r} not found.")

    base_quat_wxyz = model.cam_quat[cam_id].copy()
    base_rot = R.from_quat(wxyz_to_xyzw(base_quat_wxyz))

    # Local camera-frame rotations to try.
    # MuJoCo cameras look along local -Z.
    candidates = [
        ("base", (0, 0, 0)),
        ("x+90", (90, 0, 0)),
        ("x-90", (-90, 0, 0)),
        ("y+90", (0, 90, 0)),
        ("y-90", (0, -90, 0)),
        ("z+90", (0, 0, 90)),
        ("z-90", (0, 0, -90)),
        ("x180", (180, 0, 0)),
        ("y180", (0, 180, 0)),
    ]

    renderer = mujoco.Renderer(model, height=240, width=320)
    tiles: list[np.ndarray] = []

    for name, euler_xyz_deg in candidates:
        local_rot = R.from_euler("xyz", euler_xyz_deg, degrees=True)
        new_rot = base_rot * local_rot
        model.cam_quat[cam_id] = xyzw_to_wxyz(new_rot.as_quat())
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=CAMERA_NAME)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tiles.append(label(bgr, name))

    # Restore original camera orientation.
    model.cam_quat[cam_id] = base_quat_wxyz
    mujoco.mj_forward(model, data)

    rows = [
        np.hstack(tiles[0:3]),
        np.hstack(tiles[3:6]),
        np.hstack(tiles[6:9]),
    ]
    grid = np.vstack(rows)

    cv2.imwrite(str(OUT_PATH), grid)
    print(f"Saved sweep to {OUT_PATH}")

    env.close()


if __name__ == "__main__":
    main()
