"""Project 3 — MuJoCo wrist-camera feed smoke test.

Loads the SO-100/SO-101 MJCF used by the ETH hw2 reference, steps the simulation
for a few hundred frames, renders RGB from a named wrist/end-effector camera,
prints diagnostics, and saves a short PNG/MP4 sequence in headless mode.

Default camera: `wrist_cam` (fixed mount on the Fixed_Jaw body).
Fallback camera: `left_wrist` (target-mode camera that already exists in the
homework MJCF).

Run:
    conda activate lerobot-p3
    python scripts/test_wrist_camera_feed.py
    # or headlessly without a display:
    MUJOCO_GL=egl python scripts/test_wrist_camera_feed.py --headless
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
HW2_ROOT = REPO_ROOT / "ethz-course-2026" / "hw2_robot_control_mdps"
DEFAULT_XML = HW2_ROOT / "so101_gym" / "assets" / "so100_pos_ctrl.xml"
DEFAULT_OUT = REPO_ROOT / "outputs" / "wrist_cam_demo"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--xml", type=Path, default=DEFAULT_XML, help="Path to MJCF model.")
    p.add_argument(
        "--camera",
        type=str,
        default="wrist_cam",
        help="Named camera to render (falls back to left_wrist if missing).",
    )
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--frames", type=int, default=120, help="Number of frames to render.")
    p.add_argument(
        "--ctrl-decimation",
        type=int,
        default=10,
        help="Sim steps between rendered frames.",
    )
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--save-pngs",
        action="store_true",
        help="Save every rendered frame as PNG (default: only first/mid/last).",
    )
    p.add_argument("--save-video", action="store_true", help="Save MP4 if imageio-ffmpeg available, else GIF.")
    p.add_argument("--headless", action="store_true", help="Force offscreen rendering (sets MUJOCO_GL=egl if unset).")
    p.add_argument(
        "--viewer",
        action="store_true",
        help="Open the interactive MuJoCo viewer instead of/in addition to saving frames.",
    )
    return p.parse_args()


def setup_headless_gl_if_needed(headless: bool) -> str | None:
    """Choose an offscreen GL backend on Linux/WSL where no display is available.

    Returns the value of MUJOCO_GL that ends up in env (or None if unset).
    """
    if headless and "MUJOCO_GL" not in os.environ:
        # EGL is the most portable headless backend for MuJoCo on Linux.
        os.environ["MUJOCO_GL"] = "egl"
    return os.environ.get("MUJOCO_GL")


def resolve_camera(model, requested: str) -> tuple[str, int]:
    """Resolve a camera name, falling back to left_wrist or the first camera."""
    import mujoco

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, requested)
    if cam_id >= 0:
        return requested, cam_id
    # fallback to left_wrist
    fb = "left_wrist"
    fb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, fb)
    if fb_id >= 0:
        print(f"[warn] camera '{requested}' not found; falling back to '{fb}'.", file=sys.stderr)
        return fb, fb_id
    if model.ncam == 0:
        raise RuntimeError("MJCF defines no cameras.")
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, 0)
    print(f"[warn] camera '{requested}' not found; using first camera '{name}'.", file=sys.stderr)
    return name, 0


def drive_joints(data, t: float) -> None:
    """Send a gentle sinusoidal control signal so the camera view actually changes."""
    nu = data.ctrl.shape[0]
    if nu == 0:
        return
    # small joint-space sweep around the home pose
    home = np.array([0.0, -1.4, 1.0, 0.0, 0.0, 0.2])[:nu]
    sweep = 0.25 * np.sin(2.0 * np.pi * 0.25 * t + np.arange(nu) * 0.5)
    sweep[-1] = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.5 * t)  # gripper open/close
    data.ctrl[:] = home + sweep


def maybe_save_video(frames: list[np.ndarray], out_dir: Path, fps: int) -> Path | None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[warn] imageio not available; skipping video.", file=sys.stderr)
        return None
    mp4_path = out_dir / "wrist_cam_demo.mp4"
    try:
        imageio.mimsave(mp4_path, frames, fps=fps, codec="libx264", quality=8)
        return mp4_path
    except Exception as e:
        print(f"[warn] MP4 save failed ({e}); falling back to GIF.", file=sys.stderr)
        gif_path = out_dir / "wrist_cam_demo.gif"
        imageio.mimsave(gif_path, frames, fps=fps)
        return gif_path


def run(args: argparse.Namespace) -> int:
    mujoco_gl = setup_headless_gl_if_needed(args.headless)

    import mujoco

    xml_path = args.xml.resolve()
    if not xml_path.exists():
        print(f"[error] MJCF not found at {xml_path}", file=sys.stderr)
        return 2
    print(f"[info] loading MJCF: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    cam_name, cam_id = resolve_camera(model, args.camera)
    print(f"[info] using camera: name='{cam_name}'  id={cam_id}  ncam={model.ncam}")
    print(f"[info] MUJOCO_GL={mujoco_gl}  width={args.width} height={args.height}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare the offscreen renderer.
    try:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    except Exception as e:
        print(
            "[error] Could not create MuJoCo Renderer. "
            "On headless systems set MUJOCO_GL=egl (or =osmesa) before running. "
            f"Underlying error: {e}",
            file=sys.stderr,
        )
        return 3

    # Warm up the simulator at the home pose used by the homework env.
    home = np.array([0.0, -1.57, 1.0, 1.0, 0.0, 0.02239])
    n_home = min(home.shape[0], data.qpos.shape[0])
    data.qpos[:n_home] = home[:n_home]
    mujoco.mj_forward(model, data)

    frames: list[np.ndarray] = []
    sim_dt = float(model.opt.timestep)
    saved_pngs: list[Path] = []

    try:
        import imageio.v2 as imageio
    except ImportError:
        imageio = None

    for i in range(args.frames):
        t = i * args.ctrl_decimation * sim_dt
        drive_joints(data, t)
        for _ in range(args.ctrl_decimation):
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=cam_name)
        frame = renderer.render()  # (H, W, 3) uint8 RGB
        frames.append(frame.copy())

        first_mid_last = i in {0, args.frames // 2, args.frames - 1}
        if (args.save_pngs or first_mid_last) and imageio is not None:
            png = args.output_dir / f"frame_{i:04d}.png"
            imageio.imwrite(png, frame)
            saved_pngs.append(png)

    # Diagnostics on the last frame.
    last = frames[-1]
    print(f"[info] frames rendered: {len(frames)}")
    print(f"[info] frame shape: {last.shape}  dtype: {last.dtype}")
    print(f"[info] pixel min/max: {int(last.min())}/{int(last.max())}  mean: {float(last.mean()):.1f}")

    video_path = None
    if args.save_video:
        fps = max(1, int(round(1.0 / (args.ctrl_decimation * sim_dt))))
        video_path = maybe_save_video(frames, args.output_dir, fps=fps)

    print(f"[info] output dir: {args.output_dir}")
    if saved_pngs:
        print(f"[info] saved PNGs ({len(saved_pngs)}):")
        for p in saved_pngs:
            print(f"        {p}")
    if video_path is not None:
        print(f"[info] saved video: {video_path}")

    renderer.close()

    if args.viewer:
        # Best-effort interactive viewer (requires a display). External camera.
        try:
            import mujoco.viewer

            print("[info] launching interactive viewer — close the window to exit.")
            mujoco.viewer.launch(model, data)
        except Exception as e:
            print(f"[warn] viewer not available: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(run(parse_args()))
