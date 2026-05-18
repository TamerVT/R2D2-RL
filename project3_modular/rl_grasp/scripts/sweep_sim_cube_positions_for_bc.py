from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from project3_modular.rl_grasp.envs.so101_local_grasp_env import (
    SO101LocalGraspConfig,
)
from project3_modular.rl_grasp.envs.so101_local_grasp_hil_compat_env import (
    SO101LocalGraspHILCompatConfig,
    SO101LocalGraspHILCompatEnv,
)


OUT_PATH = Path("/tmp/so101_bc_cube_position_sweep.png")

# Sweep around the current nominal cube center (0.18, 0.03).
X_VALUES = [0.14, 0.16, 0.18, 0.20, 0.22]
Y_VALUES = [-0.01, 0.01, 0.03, 0.05, 0.07]


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def render_candidate(x: float, y: float) -> np.ndarray:
    base_cfg = SO101LocalGraspConfig(
        include_wrist_rgb=True,
        cube_center=(x, y, 0.01),
        cube_randomization_xy=(0.0, 0.0),
        # Median real HIL demonstration episode-start pregrasp pose.
        pregrasp_q_home_rad=(
            -0.11353367,
            0.01610939,
            0.36133552,
            1.02187282,
            0.09590584,
        ),
    )

    env = SO101LocalGraspHILCompatEnv(
        SO101LocalGraspHILCompatConfig(base_env=base_cfg),
        open_gui=False,
    )

    obs, _ = env.reset(seed=0)
    chw = obs["observation.images.wrist"]
    rgb = np.transpose(chw, (1, 2, 0))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # Upscale for easier visual comparison.
    bgr = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_NEAREST)

    env.close()
    return label(bgr, f"x={x:.2f}, y={y:.2f}")


def main() -> None:
    rows: list[np.ndarray] = []

    # Rows vary y, columns vary x.
    for y in Y_VALUES:
        row_tiles = [render_candidate(x, y) for x in X_VALUES]
        rows.append(np.hstack(row_tiles))

    grid = np.vstack(rows)
    cv2.imwrite(str(OUT_PATH), grid)
    print(f"Saved cube-position sweep to {OUT_PATH}")


if __name__ == "__main__":
    main()
