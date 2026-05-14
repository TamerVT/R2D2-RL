"""Render a single screenshot of the Project 3 RCS scene.

Builds an SO101 sim env with a colored cube on the floor and a wrist camera,
resets it, pulls the wrist-camera RGB frame from the observation, and writes
both the wrist view and an external 'world' view PNG to ``outputs/``.

Run from ``project3/`` after activating ``lerobot-p3-rcs``::

    MUJOCO_GL=egl python scripts/render_project3_screenshot.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
R2D2_RL = REPO_ROOT / "r2d2_rl"
sys.path.insert(0, str(R2D2_RL))

DEFAULT_OUT = R2D2_RL / "outputs" / "project3_screenshot"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--cube-xy", type=float, nargs=2, default=[0.21, -0.03])
    p.add_argument("--cube-z", type=float, default=0.02)
    p.add_argument("--cube-color", type=str, default="green")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument(
        "--external-view",
        action="store_true",
        help="Also render an external MuJoCo viewer angle for context.",
    )
    p.add_argument(
        "--lift-dz",
        type=float,
        default=0.18,
        help="After reset, step the env to lift the gripper by this many metres.",
    )
    return p.parse_args()


def _setup_egl() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    shader_cache = Path("/tmp") / "mesa_shader_cache"
    shader_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MESA_SHADER_CACHE_DIR", str(shader_cache))


def _save_rgb(rgb, path: Path) -> bool:
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("[warn] imageio not available; cannot save PNG.", file=sys.stderr)
        return False
    imageio.imwrite(path, rgb)
    return True


def _coerce_rgb(frame) -> "np.ndarray | None":  # noqa: F821 (np is imported lazily)
    """Extract a HxWx3 uint8 array from an RCS CameraFrame / dict / ndarray."""
    import numpy as np

    if frame is None:
        return None
    if isinstance(frame, np.ndarray):
        return _to_uint8(frame)
    for attr in ("rgb", "image", "color", "data"):
        candidate = getattr(frame, attr, None)
        if isinstance(candidate, np.ndarray):
            return _to_uint8(candidate)
    if isinstance(frame, dict):
        for key in ("rgb", "image", "color", "data"):
            if key in frame and isinstance(frame[key], np.ndarray):
                return _to_uint8(frame[key])
    return None


def _to_uint8(arr):
    import numpy as np

    if arr.dtype == np.uint8:
        return arr
    arr = np.clip(arr, 0.0, 1.0) if arr.dtype.kind == "f" else arr
    return (arr * 255).astype(np.uint8) if arr.dtype.kind == "f" else arr.astype(np.uint8)


def _print_obs_summary(obs) -> None:
    import numpy as np

    def describe(value, depth: int = 0):
        prefix = "  " * depth
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (dict,)):
                    print(f"{prefix}{k}:")
                    describe(v, depth + 1)
                elif isinstance(v, np.ndarray):
                    print(f"{prefix}{k}: ndarray shape={v.shape} dtype={v.dtype}")
                else:
                    print(f"{prefix}{k}: {type(v).__name__}")
        else:
            print(f"{prefix}{type(value).__name__}")

    print("Observation summary:")
    describe(obs, depth=1)


def _close_env(env) -> None:
    close = getattr(env, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:  # pragma: no cover - cleanup should not hide output paths.
        print(f"[warn] env.close() failed: {exc}", file=sys.stderr)


def _lift_gripper(env, lift_dz: float):
    """Move the SO101 gripper upward before rendering the wrist camera."""
    import numpy as np

    if lift_dz <= 0:
        return env.reset(seed=0)

    obs, info = env.reset(seed=0)
    action = {}
    for robot_key, robot_space in env.action_space.spaces.items():
        robot_action = {}
        if "tquat" in robot_space.spaces:
            robot_action["tquat"] = np.array([0.0, 0.0, lift_dz, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if "gripper" in robot_space.spaces:
            robot_action["gripper"] = np.array([1.0], dtype=np.float32)
        action[robot_key] = robot_action

    obs, reward, terminated, truncated, info = env.step(action)
    print(
        f"[info] lifted gripper by {lift_dz:.3f} m: "
        f"reward={reward}, terminated={terminated}, truncated={truncated}"
    )
    return obs, info


def main() -> int:
    args = parse_args()
    _setup_egl()

    try:
        from envs.project3_so101_env import CubeSpec, Project3SO101Config, Project3SO101Env
    except ImportError as exc:
        print(f"[error] cannot import envs.project3_so101_env: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    p3_cfg = Project3SO101Config(
        cubes=[CubeSpec(color=args.cube_color, xy=tuple(args.cube_xy), z=args.cube_z)],
        wrist_camera_resolution=(args.width, args.height),
        headless=True,
    )
    scene = Project3SO101Env(p3_cfg)
    env = scene.create_env(scene.config())

    obs, info = _lift_gripper(env, args.lift_dz)
    _print_obs_summary(obs)

    def _wrist_rgb_from(obs_dict) -> "np.ndarray | None":  # noqa: F821
        frames = obs_dict.get("frames") if isinstance(obs_dict, dict) else None
        if not isinstance(frames, dict):
            return None
        wrist = frames.get("wrist")
        if not isinstance(wrist, dict):
            return _coerce_rgb(wrist)
        rgb_entry = wrist.get("rgb")
        if isinstance(rgb_entry, dict):
            return _coerce_rgb(rgb_entry.get("data"))
        return _coerce_rgb(rgb_entry)

    wrist_rgb = _wrist_rgb_from(obs)
    if wrist_rgb is None:
        obs, _, _, _, _ = env.step(env.action_space.sample())
        wrist_rgb = _wrist_rgb_from(obs)

    if wrist_rgb is None:
        print("[error] could not extract wrist RGB from observation.", file=sys.stderr)
        print("        obs keys:", list(obs.keys()) if isinstance(obs, dict) else type(obs))
        _close_env(env)
        return 3

    wrist_path = args.output_dir / "wrist_cam.png"
    saved = _save_rgb(wrist_rgb, wrist_path)
    print(f"[info] wrist RGB shape: {wrist_rgb.shape}, dtype: {wrist_rgb.dtype}")
    print(f"[info] saved wrist screenshot: {wrist_path if saved else '(failed)'}")

    if args.external_view:
        import mujoco
        sim = env.get_wrapper_attr("sim")
        model = sim.model
        data = sim.data
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        renderer.update_scene(data)
        external_rgb = renderer.render()
        renderer.close()
        external_path = args.output_dir / "external_view.png"
        if _save_rgb(external_rgb, external_path):
            print(f"[info] saved external view:    {external_path}")

    _close_env(env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
