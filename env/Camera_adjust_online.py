# -*- coding: utf-8 -*-
"""
Created on Tue May 19 23:30:00 2026

@author: felix
"""
"""Interactive camera calibration / pose tuning for RCS+MuJoCo cameras.

This script:
  1) Builds the SO101 sim using your env_factory.make_so101_sim (which adds cameras via RCS config).
  2) Opens a live window rendering a named MuJoCo camera (default: wrist_cam).
  3) Lets you nudge position + orientation with keyboard.
  4) When you close (ESC / window close), prints the final offset as:
        translation = [x, y, z]
        quaternion  = [w, x, y, z]

Copy these into camera_defs.py -> CameraSpec(..., translation=..., quaternion=...).

Notes
-----
- This adjusts the *compiled* MuJoCo model's camera pose arrays (cam_pos / cam_quat) at runtime.
  It does not persist automatically; it only prints values.
- Requires OpenCV (cv2) for the interactive window.
"""
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Import your existing factory (same folder/module).
from env_factory import make_so101_sim


# ------------------------- Quaternion utilities -------------------------

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, quaternions as (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    s = np.sin(angle_rad / 2.0)
    return np.array([np.cos(angle_rad / 2.0), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / (np.linalg.norm(q) + 1e-12)


# ------------------------- Camera tuning UI -------------------------

@dataclass
class Steps:
    rot_rad: float = np.deg2rad(2.0)
    trans_m: float = 0.005  # 5 mm


HELP = """Keys:
  Rotations (local axes):
    W/S : pitch +X / -X
    A/D : yaw   +Y / -Y
    Q/E : roll  +Z / -Z

  Translations (parent/body frame):
    J/L : x - / x +
    U/O : y + / y -
    I/K : z + / z -

  Step size:
    [   : smaller steps
    ]   : larger steps

  Other:
    P   : print current pos/quat
    R   : reset to initial
    ESC : quit
"""


def _find_camera_id(m, camera_name: str) -> int:
    for i in range(m.ncam):
        if m.camera(i).name == camera_name:
            return i
    raise RuntimeError(f"Camera '{camera_name}' not found. Available: {[m.camera(i).name for i in range(m.ncam)]}")


def calibrate_camera_live(camera_name: str = "wrist_cam", width: int = 640, height: int = 480) -> Tuple[np.ndarray, np.ndarray]:
    """Open a live render window and let the user tune camera pose."""

    try:
        import cv2
    except Exception as e:
        raise ImportError(
            "OpenCV (cv2) is required for the interactive calibration window. "
            "Install it via e.g. `pip install opencv-python`."
        ) from e

    bundle = make_so101_sim(with_cameras=True, headless=True, debug_print=True)
    sim = bundle.sim
    m = sim.model

    import mujoco
    
    data = getattr(sim, "data", None)
    if data is None:
        data = getattr(sim, "mjdata", None)
    if data is None:
        raise AttributeError("Could not find MuJoCo data on sim (expected sim.data or sim.mjdata).")
    
    renderer = mujoco.Renderer(m, height, width)

    cam_id = _find_camera_id(m, camera_name)

    # Copy initial pose
    pos0 = np.array(m.cam_pos[cam_id], dtype=float)
    quat0 = np.array(m.cam_quat[cam_id], dtype=float)

    pos = pos0.copy()
    quat = quat0.copy()

    steps = Steps()

    def apply_pose():
        # MuJoCo model arrays are typically writable; assign via slice for safety.
        m.cam_pos[cam_id] = pos
        m.cam_quat[cam_id] = quat

    apply_pose()

    print("--- Camera calibration ---")
    print(f"Camera: {camera_name}")
    print("Initial translation:", pos0.tolist())
    print("Initial quaternion  :", quat0.tolist())
    print("" + HELP)

    win = "RCS Camera Calibration"

    while True:
        # Render RGB
        
        mujoco.mj_forward(m, data)
        renderer.update_scene(data, camera=cam_id)
        rgb = renderer.render()

        bgr = rgb[..., ::-1].copy()

        # HUD
        cv2.putText(bgr, f"{camera_name}  rot={np.rad2deg(steps.rot_rad):.2f}deg  trans={steps.trans_m*1000:.1f}mm",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(bgr, "ESC quit | P print | R reset | [ ] step", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow(win, bgr)

        key = cv2.waitKey(1) & 0xFF

        # window closed
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

        if key == 27:  # ESC
            break

        changed = False

        # Rotations (incremental)
        if key == ord('a'):
            dq = quat_from_axis_angle(np.array([0, 1, 0]), +steps.rot_rad)
            quat = quat_normalize(quat_mul(dq, quat)); changed = True
        elif key == ord('d'):
            dq = quat_from_axis_angle(np.array([0, 1, 0]), -steps.rot_rad)
            quat = quat_normalize(quat_mul(dq, quat)); changed = True
        elif key == ord('w'):
            dq = quat_from_axis_angle(np.array([1, 0, 0]), +steps.rot_rad)
            quat = quat_normalize(quat_mul(dq, quat)); changed = True
        elif key == ord('s'):
            dq = quat_from_axis_angle(np.array([1, 0, 0]), -steps.rot_rad)
            quat = quat_normalize(quat_mul(dq, quat)); changed = True
        elif key == ord('q'):
            dq = quat_from_axis_angle(np.array([0, 0, 1]), +steps.rot_rad)
            quat = quat_normalize(quat_mul(dq, quat)); changed = True
        elif key == ord('e'):
            dq = quat_from_axis_angle(np.array([0, 0, 1]), -steps.rot_rad)
            quat = quat_normalize(quat_mul(dq, quat)); changed = True

        # Translations
        elif key == ord('j'):
            pos[0] -= steps.trans_m; changed = True
        elif key == ord('l'):
            pos[0] += steps.trans_m; changed = True
        elif key == ord('u'):
            pos[1] += steps.trans_m; changed = True
        elif key == ord('o'):
            pos[1] -= steps.trans_m; changed = True
        elif key == ord('i'):
            pos[2] += steps.trans_m; changed = True
        elif key == ord('k'):
            pos[2] -= steps.trans_m; changed = True

        # Step size
        elif key == ord('['):
            steps.rot_rad = max(np.deg2rad(0.2), steps.rot_rad * 0.8)
            steps.trans_m = max(0.0005, steps.trans_m * 0.8)
        elif key == ord(']'):
            steps.rot_rad = min(np.deg2rad(10.0), steps.rot_rad * 1.25)
            steps.trans_m = min(0.05, steps.trans_m * 1.25)

        # Reset / print
        elif key == ord('r'):
            pos[:] = pos0
            quat[:] = quat0
            changed = True
        elif key == ord('p'):
            print("Current:")
            print("  translation =", pos.tolist())
            print("  quaternion  =", quat.tolist())

        if changed:
            apply_pose()

    cv2.destroyAllWindows()

    print("--- Final camera offset (copy into camera_defs.py) ---")
    print("translation =", pos.tolist())
    print("quaternion  =", quat.tolist())

    return pos, quat


def main(argv: list[str]) -> int:
    camera_name = argv[1] if len(argv) > 1 else "wrist_cam"
    calibrate_camera_live(camera_name=camera_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
