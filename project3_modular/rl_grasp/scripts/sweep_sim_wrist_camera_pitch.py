from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from project3_modular.rl_grasp.envs.so101_local_grasp_env import SO101LocalGraspEnv


CAMERA_NAME = "robotwrist"
OUT_PATH = Path("/tmp/so101_wrist_pitch_sweep.png")


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


def render_sweep(axis: str, angles: list[int], title: str) -> list[np.ndarray]:
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

    renderer = mujoco.Renderer(model, height=240, width=320)
    tiles: list[np.ndarray] = []

    for angle in angles:
        local_rot = R.from_euler(axis, angle, degrees=True)
        new_rot = base_rot * local_rot
        model.cam_quat[cam_id] = xyzw_to_wxyz(new_rot.as_quat())
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera=CAMERA_NAME)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tiles.append(label(bgr, f"{title} {angle:+d}°"))

    env.close()
    return tiles


def main() -> None:
    angles = [-30, -20, -10, 0, 10, 20, 30]

    # We do both local X and local Y sweeps so we do not have to guess
    # which local axis corresponds to "look more down" after the current orientation.
    x_tiles = render_sweep("x", angles, "local x")
    y_tiles = render_sweep("y", angles, "local y")

    blank = np.zeros_like(x_tiles[0])

    # 2 rows for X sweep, 2 rows for Y sweep.
    grid = np.vstack(
        [
            np.hstack(x_tiles[:4]),
            np.hstack(x_tiles[4:] + [blank]),
            np.hstack(y_tiles[:4]),
            np.hstack(y_tiles[4:] + [blank]),
        ]
    )

    cv2.imwrite(str(OUT_PATH), grid)
    print(f"Saved pitch sweep to {OUT_PATH}")


if __name__ == "__main__":
    main()
