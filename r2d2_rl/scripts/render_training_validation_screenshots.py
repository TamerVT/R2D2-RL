"""Render screenshots for validating Project 3 sim-to-real training geometry.

Outputs wrist-camera and external PNGs for:

1. LeRobot-compatible local grasp training resets across multiple seeds.
2. Hybrid pick/release phase boundaries using the current classical pipeline.

Run from ``project3/`` with ``lerobot-p3-rcs`` active:

    MUJOCO_GL=egl python r2d2_rl/scripts/render_training_validation_screenshots.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
R2D2_RL = REPO_ROOT / "r2d2_rl"
sys.path.insert(0, str(R2D2_RL))

DEFAULT_OUT = R2D2_RL / "outputs" / "training_validation_screenshots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--cube-color", default="green")
    parser.add_argument("--cube-xy", type=float, nargs=2, default=[0.21, -0.03])
    parser.add_argument("--cube-z", type=float, default=0.01)
    parser.add_argument("--bowl-xyz", type=float, nargs=3, default=[0.30, 0.10, 0.05])
    parser.add_argument("--initial-lift-dz", type=float, default=0.0)
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=REPO_ROOT / "p3_local_grasp_hil_multicolor_colorcond_v1.zip",
        help="Only recorded in metadata; the renderer does not train.",
    )
    parser.add_argument(
        "--pregrasp-regressor-checkpoint",
        type=Path,
        default=REPO_ROOT / "p3_required_sim_training_artifacts.zip",
    )
    parser.add_argument("--use-pregrasp-regressor", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def setup_headless_rendering() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    shader_cache = Path("/tmp") / "mesa_shader_cache"
    shader_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MESA_SHADER_CACHE_DIR", str(shader_cache))


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.asarray(rgb, dtype=np.uint8))


def save_external_view(env: Any, path: Path, width: int, height: int) -> None:
    import mujoco

    sim = env.get_wrapper_attr("sim")
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.19, 0.02, 0.07], dtype=np.float64)
    camera.distance = 0.55
    camera.azimuth = 135.0
    camera.elevation = -32.0

    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    renderer.update_scene(sim.data, camera=camera)
    rgb = renderer.render()
    renderer.close()
    save_rgb(rgb, path)


def save_lerobot_wrist(obs: dict[str, np.ndarray], path: Path) -> None:
    from rl.lerobot_compat import IMAGE_KEY

    chw = np.asarray(obs[IMAGE_KEY], dtype=np.uint8)
    hwc = np.transpose(chw, (1, 2, 0))
    save_rgb(hwc, path)


def hybrid_wrist_rgb(obs: dict[str, Any], camera_name: str = "robotwrist") -> np.ndarray | None:
    frames = obs.get("frames") if isinstance(obs, dict) else None
    if not isinstance(frames, dict):
        return None
    wrist = frames.get(camera_name)
    if not isinstance(wrist, dict):
        return None
    rgb = wrist.get("rgb")
    if isinstance(rgb, dict) and isinstance(rgb.get("data"), np.ndarray):
        return np.asarray(rgb["data"], dtype=np.uint8)
    return None


def cube_xyz_from_joint(env: Any, joint_name: str) -> list[float]:
    sim = env.get_wrapper_attr("sim")
    return np.asarray(sim.data.joint(joint_name).qpos[:3], dtype=np.float64).round(5).tolist()


def robot_xyz_from_obs(obs: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(obs, dict):
        return None
    robot = obs.get("robot")
    if not isinstance(robot, dict):
        return None
    if "xyzrpy" in robot:
        return np.asarray(robot["xyzrpy"], dtype=np.float64).reshape(-1)[:3].round(5).tolist()
    if "tquat" in robot:
        return np.asarray(robot["tquat"], dtype=np.float64).reshape(-1)[:3].round(5).tolist()
    return None


def render_training_resets(args: argparse.Namespace) -> list[dict[str, Any]]:
    from rl.lerobot_align_grasp_env import LeRobotAlignGraspEnv, LeRobotAlignGraspEnvConfig
    from rl.lerobot_compat import IMAGE_KEY, STATE_KEY

    records: list[dict[str, Any]] = []
    for seed in args.training_seeds:
        cfg = LeRobotAlignGraspEnvConfig(
            cube_color=args.cube_color,
            use_pregrasp_regressor=args.use_pregrasp_regressor,
            pregrasp_regressor_checkpoint=args.pregrasp_regressor_checkpoint,
        )
        env = LeRobotAlignGraspEnv(cfg)
        try:
            obs, info = env.reset(seed=int(seed))
            prefix = f"training_reset_seed{seed}"
            save_lerobot_wrist(obs, args.output_dir / f"{prefix}_wrist.png")
            save_external_view(env.env, args.output_dir / f"{prefix}_external.png", args.width, args.height)
            records.append(
                {
                    "kind": "lerobot_training_reset",
                    "seed": int(seed),
                    "cube_color": args.cube_color,
                    "cube_xyz": cube_xyz_from_joint(env.env, env._object_joint_name),
                    "lerobot_state_positions": np.asarray(obs[STATE_KEY][0:6], dtype=np.float32)
                    .round(5)
                    .tolist(),
                    "image_shape": list(np.asarray(obs[IMAGE_KEY]).shape),
                    "state_shape": list(np.asarray(obs[STATE_KEY]).shape),
                    "camera_name": cfg.wrist_camera_name,
                    "cube_center": list(cfg.cube_center),
                    "cube_randomization_xy": list(cfg.cube_randomization_xy),
                    "pregrasp_info": {
                        key: np.asarray(value).round(5).tolist()
                        for key, value in info.items()
                        if key.startswith("pregrasp_")
                    },
                    "files": [
                        str(args.output_dir / f"{prefix}_wrist.png"),
                        str(args.output_dir / f"{prefix}_external.png"),
                    ],
                }
            )
        finally:
            env.close()
    return records


def save_hybrid_snapshot(
    env: Any,
    obs: dict[str, Any] | None,
    args: argparse.Namespace,
    name: str,
    camera_name: str,
) -> dict[str, Any]:
    wrist = hybrid_wrist_rgb(obs or {}, camera_name=camera_name)
    files = []
    if wrist is not None:
        wrist_path = args.output_dir / f"hybrid_{name}_wrist.png"
        save_rgb(wrist, wrist_path)
        files.append(str(wrist_path))
    external_path = args.output_dir / f"hybrid_{name}_external.png"
    save_external_view(env, external_path, args.width, args.height)
    files.append(str(external_path))
    return {
        "kind": "hybrid_snapshot",
        "name": name,
        "camera_name": camera_name,
        "robot_xyz": robot_xyz_from_obs(obs),
        "files": files,
    }


def render_hybrid_pick_release(args: argparse.Namespace) -> list[dict[str, Any]]:
    from control.waypoint_controller import RcsWaypointController
    from envs.project3_so101_env import CubeSpec, Project3SO101Config, Project3SO101Env
    from estimation.block_belief import BlockBelief
    from hybrid_control_rl.config import load_yaml_config
    from planning.hybrid_waypoint_planner import HybridWaypointPlanner
    from runtime.rcs_sim_adapters import RcsWristBlockObserver, ScriptedAlignGraspPolicy

    config = load_yaml_config(R2D2_RL / "configs" / "hybrid_control_rl" / "eval1.yaml")
    p3_cfg = Project3SO101Config(
        cubes=[CubeSpec(color=args.cube_color, xy=tuple(args.cube_xy), z=args.cube_z)],
        wrist_camera_resolution=(args.width, args.height),
        headless=True,
    )
    wrist_camera_name = p3_cfg.wrist_camera_name
    scene = Project3SO101Env(p3_cfg)
    env = scene.create_env(scene.config())
    controller = RcsWaypointController(
        env,
        use_pregrasp_regressor=args.use_pregrasp_regressor,
        pregrasp_regressor_checkpoint=args.pregrasp_regressor_checkpoint
        if args.use_pregrasp_regressor
        else None,
    )

    records: list[dict[str, Any]] = []
    try:
        controller.reset(seed=0)
        if args.initial_lift_dz > 0:
            controller.step_delta(np.array([0.0, 0.0, args.initial_lift_dz]), gripper=1.0)
        records.append(save_hybrid_snapshot(env, controller.last_obs, args, "initial_observe", wrist_camera_name))

        observer = RcsWristBlockObserver(controller, config, camera_name=wrist_camera_name)
        belief = observer.observe(args.cube_color)
        if belief is None or not belief.initialized:
            belief = BlockBelief(
                color=args.cube_color,
                mean_xy=np.asarray(args.cube_xy, dtype=np.float64),
                covariance_xy=np.eye(2, dtype=np.float64) * 1e-6,
                last_seen_time=float(controller.step_count),
                confidence=1.0,
                initialized=True,
            )

        planner = HybridWaypointPlanner(config)
        pregrasp = planner.plan_pregrasp(belief)
        controller.execute(pregrasp.waypoints)
        records.append(save_hybrid_snapshot(env, controller.last_obs, args, "pregrasp", wrist_camera_name))

        policy = ScriptedAlignGraspPolicy(controller, config)
        policy.run("align_grasp", args.cube_color, belief)
        records.append(save_hybrid_snapshot(env, controller.last_obs, args, "after_grasp_close", wrist_camera_name))

        post = planner.plan_post_grasp(belief, np.asarray(args.bowl_xyz, dtype=np.float64))
        controller.execute(post.lift.waypoints)
        records.append(save_hybrid_snapshot(env, controller.last_obs, args, "after_lift", wrist_camera_name))

        controller.execute(post.transport.waypoints)
        records.append(save_hybrid_snapshot(env, controller.last_obs, args, "before_release", wrist_camera_name))

        controller.execute(post.release.waypoints)
        records.append(save_hybrid_snapshot(env, controller.last_obs, args, "after_release", wrist_camera_name))

        for record in records:
            record["cube_xyz"] = cube_xyz_from_joint(env, "cube_green_0_box_joint")
        cube_final = np.asarray(cube_xyz_from_joint(env, "cube_green_0_box_joint"), dtype=np.float64)
        bowl = np.asarray(args.bowl_xyz, dtype=np.float64)
        release_xy_error = float(np.linalg.norm(cube_final[:2] - bowl[:2]))
        records.append(
            {
                "kind": "hybrid_summary",
                "belief_xy": np.asarray(belief.mean_xy, dtype=np.float64).round(5).tolist(),
                "bowl_xyz": list(args.bowl_xyz),
                "cube_final_xyz": cube_final.round(5).tolist(),
                "release_xy_error": release_xy_error,
                "visual_release_valid": bool(release_xy_error < 0.05),
                "note": (
                    "Release screenshots are diagnostic only. If visual_release_valid is false, "
                    "the cube was not transported to the bowl in this scripted validation run."
                ),
            }
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return records


def main() -> int:
    args = parse_args()
    setup_headless_rendering()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = {
        "config": {
            "training_seeds": args.training_seeds,
            "cube_color": args.cube_color,
            "normalization_stats": str(args.normalization_stats),
            "pregrasp_regressor_checkpoint": str(args.pregrasp_regressor_checkpoint),
            "use_pregrasp_regressor": bool(args.use_pregrasp_regressor),
        },
        "training": render_training_resets(args),
        "hybrid_pick_release": render_hybrid_pick_release(args),
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(records, indent=2, default=str))
    print(f"wrote screenshots to {args.output_dir}")
    print(f"wrote metadata to {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
