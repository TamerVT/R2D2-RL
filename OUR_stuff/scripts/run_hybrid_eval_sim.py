"""Run the hybrid Project 3 pipeline end-to-end in the RCS SO101 sim.

This is a plumbing/integration runner: it uses RCS for the robot sim and wrist
camera, HSV perception for target localization, the pure-numeric waypoint
planner, the hybrid executor state machine, and a scripted ``align_grasp``
adapter until the learned local policy is trained.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUR_STUFF = REPO_ROOT / "OUR_stuff"
sys.path.insert(0, str(OUR_STUFF))

DEFAULT_OUT = OUR_STUFF / "outputs" / "hybrid_eval_sim"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=OUR_STUFF / "configs" / "hybrid_control_rl" / "eval1.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cube-color", default="green")
    parser.add_argument("--cube-xy", type=float, nargs=2, default=[0.21, -0.03])
    parser.add_argument("--cube-z", type=float, default=0.02)
    parser.add_argument("--bowl-xyz", type=float, nargs=3, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--initial-lift-dz", type=float, default=0.18)
    parser.add_argument(
        "--enable-watchdog",
        action="store_true",
        help=(
            "Enable wrist-color visibility checks during transport. "
            "Disabled by default for scripted sim smoke."
        ),
    )
    parser.add_argument("--strict-controller", action="store_true")
    parser.add_argument("--save-images", action="store_true")
    return parser.parse_args()


def setup_headless_rendering() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    shader_cache = Path("/tmp") / "mesa_shader_cache"
    shader_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MESA_SHADER_CACHE_DIR", str(shader_cache))


def save_rgb(rgb: np.ndarray, path: Path) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, rgb)


def save_external_view(env, path: Path, width: int, height: int) -> None:
    import mujoco

    sim = env.get_wrapper_attr("sim")
    renderer = mujoco.Renderer(sim.model, height=height, width=width)
    renderer.update_scene(sim.data)
    rgb = renderer.render()
    renderer.close()
    save_rgb(rgb, path)


def main() -> int:
    args = parse_args()
    setup_headless_rendering()

    from control.waypoint_controller import RcsWaypointController
    from envs.project3_so101_env import CubeSpec, Project3SO101Config, Project3SO101Env
    from hybrid_control_rl.config import load_yaml_config
    from planning.hybrid_waypoint_planner import HybridWaypointPlanner
    from runtime.hybrid_task_executor import HybridTaskExecutor, TaskGoal
    from runtime.rcs_sim_adapters import (
        RcsColorVisibilityChecker,
        RcsWristBlockObserver,
        ScriptedAlignGraspPolicy,
        wrist_rgb_from_obs,
    )

    config = load_yaml_config(args.config)
    goal_cfg = (config.get("eval") or {}).get("goals", [{}])[0]
    target_color = goal_cfg.get("target_color", args.cube_color)
    if target_color == "any":
        target_color = args.cube_color
    bowl_xyz = np.asarray(
        args.bowl_xyz if args.bowl_xyz is not None else goal_cfg.get("bowl_xyz_base", [0.30, 0.10, 0.05]),
        dtype=np.float64,
    )

    p3_cfg = Project3SO101Config(
        cubes=[CubeSpec(color=args.cube_color, xy=tuple(args.cube_xy), z=args.cube_z)],
        wrist_camera_resolution=(args.width, args.height),
        headless=True,
    )
    scene = Project3SO101Env(p3_cfg)
    env = scene.create_env(scene.config())
    controller = RcsWaypointController(env, strict_position=args.strict_controller)

    try:
        controller.reset(seed=args.seed)
        if args.initial_lift_dz > 0:
            controller.step_delta(np.array([0.0, 0.0, args.initial_lift_dz]), gripper=1.0)

        if args.save_images and controller.last_obs is not None:
            initial_rgb = wrist_rgb_from_obs(controller.last_obs)
            if initial_rgb is not None:
                save_rgb(initial_rgb, args.output_dir / "initial_wrist.png")
            save_external_view(env, args.output_dir / "initial_external.png", args.width, args.height)

        planner = HybridWaypointPlanner(config)
        observer = RcsWristBlockObserver(controller, config)
        policy = ScriptedAlignGraspPolicy(controller, config)
        visibility = RcsColorVisibilityChecker(controller, config) if args.enable_watchdog else None
        executor = HybridTaskExecutor(
            config=config,
            planner=planner,
            observer=observer,
            controller=controller,
            local_policy=policy,
            visibility_checker=visibility,
        )

        result = executor.run_goal(TaskGoal(target_color=target_color, bowl_xyz_base=bowl_xyz))

        if args.save_images and controller.last_obs is not None:
            final_rgb = wrist_rgb_from_obs(controller.last_obs)
            if final_rgb is not None:
                save_rgb(final_rgb, args.output_dir / "final_wrist.png")
            save_external_view(env, args.output_dir / "final_external.png", args.width, args.height)

        print("Hybrid sim result:")
        print(f"  success:        {result.success}")
        print(f"  final_state:    {result.final_state.value}")
        print(f"  attempts:       {result.attempts}")
        if result.failure_reason:
            print(f"  failure_reason: {result.failure_reason}")
        if observer.last_detection is not None and observer.last_estimate is not None:
            print(
                "  last_detection: "
                f"uv={observer.last_detection.centroid_uv.round(2).tolist()} "
                f"area={observer.last_detection.area_px:.1f} "
                f"confidence={observer.last_detection.confidence:.3f}"
            )
            print(
                "  last_estimate:  "
                f"xy={observer.last_estimate.xy_base.round(4).tolist()} "
                f"valid={observer.last_estimate.valid}"
            )
        print("  trace:")
        for event in result.trace:
            print(f"    [{event.attempt}] {event.state.value}: {event.detail}")
        if args.save_images:
            print(f"  images:         {args.output_dir}")
        return 0 if result.success else 1
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    sys.exit(main())
