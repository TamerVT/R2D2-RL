from __future__ import annotations

from pathlib import Path

import cv2
import mujoco
import numpy as np

from project3_modular.rl_grasp.envs.so101_local_grasp_env import SO101LocalGraspEnv


CAMERA_NAME = "robotwrist"
OUT_DIR = Path("/tmp")

# Offsets in meters: -1 cm, -0.5 cm, baseline, +0.5 cm, +1 cm.
OFFSETS = [-0.010, -0.005, 0.000, 0.005, 0.010]


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


def cm(x: float) -> float:
    return x * 100.0


def build_grid(
    *,
    model,
    data,
    renderer: mujoco.Renderer,
    cam_id: int,
    base_pos: np.ndarray,
    axis_a: int,
    axis_b: int,
    axis_a_name: str,
    axis_b_name: str,
    out_path: Path,
) -> None:
    tiles: list[np.ndarray] = []

    for b in OFFSETS:
        row_tiles: list[np.ndarray] = []
        for a in OFFSETS:
            delta = np.zeros(3, dtype=np.float64)
            delta[axis_a] = a
            delta[axis_b] = b

            model.cam_pos[cam_id] = base_pos + delta
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=CAMERA_NAME)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            text = f"{axis_a_name} {cm(a):+.1f}cm, {axis_b_name} {cm(b):+.1f}cm"
            row_tiles.append(label(bgr, text))

        tiles.append(np.hstack(row_tiles))

    grid = np.vstack(tiles)
    cv2.imwrite(str(out_path), grid)
    print(f"Saved {axis_a_name}-{axis_b_name} sweep to {out_path}")


def main() -> None:
    env = SO101LocalGraspEnv(open_gui=False)
    env.reset(seed=0)

    # Match real local-grasp setup: gripper closed.
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

    # 256×192 per tile -> 1280×960 total per 5×5 grid.
    renderer = mujoco.Renderer(model, height=192, width=256)

    build_grid(
        model=model,
        data=data,
        renderer=renderer,
        cam_id=cam_id,
        base_pos=base_pos,
        axis_a=0,
        axis_b=1,
        axis_a_name="x",
        axis_b_name="y",
        out_path=OUT_DIR / "so101_wrist_large_xy_sweep.png",
    )

    build_grid(
        model=model,
        data=data,
        renderer=renderer,
        cam_id=cam_id,
        base_pos=base_pos,
        axis_a=0,
        axis_b=2,
        axis_a_name="x",
        axis_b_name="z",
        out_path=OUT_DIR / "so101_wrist_large_xz_sweep.png",
    )

    build_grid(
        model=model,
        data=data,
        renderer=renderer,
        cam_id=cam_id,
        base_pos=base_pos,
        axis_a=1,
        axis_b=2,
        axis_a_name="y",
        axis_b_name="z",
        out_path=OUT_DIR / "so101_wrist_large_yz_sweep.png",
    )

    model.cam_pos[cam_id] = base_pos
    mujoco.mj_forward(model, data)
    env.close()


if __name__ == "__main__":
    main()
