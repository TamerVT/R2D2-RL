"""End-to-end validation for the pixel-to-table projection pipeline.

Loads the HW2 MuJoCo scene (which carries our wrist_cam), places a known
green mocap sphere at a sequence of table-plane targets, renders the wrist
camera, runs ``ColorBlockDetector``, and projects the detected centroid back
to the base frame with ``PixelToTableProjector``. Reports per-trial error in
centimeters and median covariance trace, plus optional debug overlays.

The script reads the wrist-camera pose directly from MuJoCo at runtime and
converts MuJoCo's camera frame ( +x right, +y up, -z forward) into the
pinhole convention used by the projector ( +x right, +y down, +z forward) by
right-multiplying the camera rotation by ``diag(1, -1, -1)``.

Run from ``project3/`` after activating ``lerobot-p3``::

    MUJOCO_GL=egl python scripts/validate_pixel_to_table.py --headless --save-overlays
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUR_STUFF = REPO_ROOT / "OUR_stuff"
sys.path.insert(0, str(OUR_STUFF))

from hybrid_control_rl.config import load_yaml_config
from perception.color_block_detector import ColorBlockDetector
from estimation.pixel_to_table import PixelToTableProjector

DEFAULT_XML = (
    REPO_ROOT
    / "ethz-course-2026"
    / "hw2_robot_control_mdps"
    / "so101_gym"
    / "assets"
    / "so100_pos_ctrl.xml"
)
DEFAULT_OUT = OUR_STUFF / "outputs" / "pixel_to_table_validation"
DEFAULT_CONFIG = OUR_STUFF / "configs" / "hybrid_control_rl" / "calibration.yaml"

# MuJoCo's camera frame uses +y up, -z forward. Right-multiply the rotation
# by this diagonal to obtain the standard image convention (+y down, +z fwd).
MJ_CAM_TO_PINHOLE = np.diag([1.0, -1.0, -1.0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--target-color", type=str, default="green")
    p.add_argument(
        "--qpos",
        type=float,
        nargs="+",
        default=[0.0, -1.56413, 1.57276, 1.57694, 0.0, 0.0],
        help="Joint configuration for the wrist-camera view (default: HW3 student_start).",
    )
    p.add_argument(
        "--z-table",
        type=float,
        default=None,
        help="Override the table z used by the projector. Defaults to the config value.",
    )
    p.add_argument(
        "--target-offsets",
        type=float,
        nargs="*",
        default=[-0.04, -0.02, 0.0, 0.02, 0.04],
        help="Grid of XY offsets (m) around the camera's apparent ground point.",
    )
    p.add_argument("--save-overlays", action="store_true")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def setup_headless_gl_if_needed(headless: bool) -> str | None:
    if headless and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    return os.environ.get("MUJOCO_GL")


def build_intrinsics_from_fovy(width: int, height: int, fovy_deg: float) -> np.ndarray:
    fovy_rad = math.radians(fovy_deg)
    fy = (height / 2.0) / math.tan(fovy_rad / 2.0)
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def camera_pose_pinhole(data, cam_id: int) -> np.ndarray:
    pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64).copy()
    rot_mujoco = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3).copy()
    rot_pinhole = rot_mujoco @ MJ_CAM_TO_PINHOLE
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot_pinhole
    T[:3, 3] = pos
    return T


def project_camera_to_table(T_B_C: np.ndarray, z_table: float) -> np.ndarray | None:
    o = T_B_C[:3, 3]
    forward = T_B_C[:3, 2]  # +z in pinhole frame = optical axis
    if abs(forward[2]) < 1e-6:
        return None
    lam = (z_table - o[2]) / forward[2]
    if lam <= 0:
        return None
    return o + lam * forward


def save_overlay(image: np.ndarray, detection_uv, target_uv, out_path: Path) -> None:
    try:
        import cv2

        overlay = image.copy()
        if detection_uv is not None:
            cv2.circle(overlay, tuple(int(v) for v in detection_uv), 6, (0, 255, 255), 2)
        if target_uv is not None:
            cv2.drawMarker(
                overlay,
                tuple(int(v) for v in target_uv),
                color=(255, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
            )
        import imageio.v2 as imageio

        imageio.imwrite(out_path, overlay)
    except ImportError:
        pass


def main() -> int:
    args = parse_args()
    setup_headless_gl_if_needed(args.headless)

    import mujoco

    xml_path = args.xml.resolve()
    if not xml_path.exists():
        print(f"[error] MJCF not found at {xml_path}", file=sys.stderr)
        return 2

    config = load_yaml_config(args.config)
    detector = ColorBlockDetector(config)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    if cam_id < 0:
        print("[error] camera 'wrist_cam' not found in MJCF.", file=sys.stderr)
        return 3
    fovy = float(model.cam_fovy[cam_id])

    # The HW2 arm renders RGB axis capsules and a translucent red ee_site
    # sphere near the gripper. They are visual-only (contype=0) but dominate
    # the wrist-camera view at ~8 cm distance, drowning out the 1 cm mocap
    # target on the table. Hide them at runtime — no effect on dynamics.
    for geom_name in ("ee_x_axis", "ee_y_axis", "ee_z_axis"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if gid >= 0:
            model.geom_rgba[gid, 3] = 0.0
    for site_name in ("ee_site", "left_cam_focus"):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid >= 0:
            model.site_rgba[sid, 3] = 0.0

    # Enlarge the mocap green sphere so it's robustly detected in a wrist-cam
    # view at ~30 cm. The MJCF default is radius 0.01; bump to 0.02 (~25 px).
    mocap_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target")
    if mocap_geom_id >= 0:
        geom_start = model.body_geomadr[mocap_geom_id]
        geom_num = model.body_geomnum[mocap_geom_id]
        for gi in range(geom_start, geom_start + geom_num):
            model.geom_size[gi, 0] = 0.02

    qpos_input = np.asarray(args.qpos, dtype=np.float64)
    n_q = min(qpos_input.shape[0], data.qpos.shape[0])
    data.qpos[:n_q] = qpos_input[:n_q]
    mujoco.mj_forward(model, data)

    K = build_intrinsics_from_fovy(args.width, args.height, fovy)
    z_table = args.z_table if args.z_table is not None else float(
        config["workspace"]["z_object_center"]
    )

    sim_proj_config = {
        "camera": {"K": K.tolist(), "width": args.width, "height": args.height},
        "transforms": {
            "T_E_C": {
                "translation": [0.0, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
        "workspace": {
            "z_object_center": z_table,
            "bounds_xy": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
        },
        "uncertainty": {
            "pixel_sigma_base": 3.0,
            "calibration_sigma_xy": 0.0,
        },
    }
    projector = PixelToTableProjector(sim_proj_config)

    T_B_C = camera_pose_pinhole(data, cam_id)
    cam_to_table = project_camera_to_table(T_B_C, z_table)
    if cam_to_table is None:
        print("[error] camera optical axis is parallel to the table; pick another qpos.", file=sys.stderr)
        return 4

    print("Wrist-cam pixel-to-table validation")
    print(f"  xml:            {xml_path}")
    print(f"  qpos:           {list(np.round(qpos_input, 4))}")
    print(f"  fovy:           {fovy:.2f} deg")
    print(f"  K:")
    for row in K:
        print(f"    {row}")
    print(f"  z_table:        {z_table:.4f}")
    print(f"  cam pos (B):    {np.round(T_B_C[:3, 3], 4)}")
    print(f"  cam axis (B):   {np.round(T_B_C[:3, 2], 4)}  (optical, pinhole)")
    print(f"  cam->table xy:  {np.round(cam_to_table[:2], 4)}")
    print()

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    errors_m: list[float] = []
    cov_traces: list[float] = []

    print(
        "  trial | target_xy             | det_uv         | est_xy                 | err_cm  | cov_trace"
    )
    print(
        "  ----- | --------------------- | -------------- | ---------------------- | ------- | ---------"
    )

    trial_idx = 0
    for dx in args.target_offsets:
        for dy in args.target_offsets:
            trial_idx += 1
            target_xy = cam_to_table[:2] + np.array([dx, dy])
            data.mocap_pos[0] = np.array([target_xy[0], target_xy[1], z_table])
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera="wrist_cam")
            image = renderer.render()

            detections = detector.detect(image, target_color=args.target_color)
            if not detections:
                print(
                    f"   {trial_idx:>3} | ({target_xy[0]:+.3f}, {target_xy[1]:+.3f})   | no detection   | "
                    f"-                      | -       | -"
                )
                continue

            det = detections[0]
            uv = det.centroid_uv
            est = projector.project_from_T_BE(uv, T_B_C, covariance_uv=det.covariance_uv)
            if not est.valid:
                print(
                    f"   {trial_idx:>3} | ({target_xy[0]:+.3f}, {target_xy[1]:+.3f})   | "
                    f"({uv[0]:6.1f},{uv[1]:6.1f}) | invalid: {est.reason}"
                )
                continue

            err_m = float(np.linalg.norm(est.xy_base - target_xy))
            errors_m.append(err_m)
            cov_traces.append(float(np.trace(est.covariance_xy)))

            print(
                f"   {trial_idx:>3} | ({target_xy[0]:+.3f}, {target_xy[1]:+.3f})   | "
                f"({uv[0]:6.1f},{uv[1]:6.1f}) | ({est.xy_base[0]:+.3f}, {est.xy_base[1]:+.3f})    | "
                f"{err_m * 100.0:6.2f}  | {np.trace(est.covariance_xy):.2e}"
            )

            if args.save_overlays:
                target_h = K @ np.array([target_xy[0], target_xy[1], z_table]) - K @ T_B_C[:3, 3]
                save_overlay(
                    image=image,
                    detection_uv=uv,
                    target_uv=None,
                    out_path=output_dir / f"trial_{trial_idx:03d}.png",
                )

    renderer.close()

    if not errors_m:
        print("\n[warn] no valid detections — try a different qpos or target offsets.")
        return 5

    errors_cm = np.array(errors_m) * 100.0
    print()
    print("Summary")
    print(f"  trials with valid estimate: {len(errors_m)}")
    print(f"  mean error:                 {errors_cm.mean():.2f} cm")
    print(f"  median error:               {np.median(errors_cm):.2f} cm")
    print(f"  max error:                  {errors_cm.max():.2f} cm")
    print(f"  median cov trace:           {np.median(cov_traces):.2e} m^2")
    if args.save_overlays:
        print(f"  overlays saved to:          {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
