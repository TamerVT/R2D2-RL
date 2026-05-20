"""Run a multi-cube, multi-goal eval (Eval 2 / Eval 3) in the RCS SO101 sim.

The scene YAML is read from ``--config`` (default Eval 3); cubes are spawned
from the ``scene.cubes`` section, and the goal sequence comes from
``eval.goals``. Each goal is executed by ``HybridTaskExecutor.run_goal`` in
order via ``run_sequence``; per-goal traces + final-state screenshots are
written under ``--output-dir``.

Examples::

    # Eval 2 (clutter, single goal) with the scripted local policy:
    python r2d2_rl/scripts/run_eval_sequence.py \
        --config r2d2_rl/configs/hybrid_control_rl/eval2.yaml \
        --output-dir r2d2_rl/outputs/eval2_smoke --save-images

    # Eval 3 (3 goals) with a trained SB3 align_grasp policy:
    python r2d2_rl/scripts/run_eval_sequence.py \
        --config r2d2_rl/configs/hybrid_control_rl/eval3.yaml \
        --sb3-align-grasp-checkpoint r2d2_rl/outputs/hil_bc_sac_<ts>/final_model.zip \
        --output-dir r2d2_rl/outputs/eval3_run --save-images
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
R2D2_RL = REPO_ROOT / "r2d2_rl"
sys.path.insert(0, str(R2D2_RL))

DEFAULT_OUT = R2D2_RL / "outputs" / "eval_sequence"
DEFAULT_PREGRASP_ARTIFACTS = REPO_ROOT / "p3_required_sim_training_artifacts.zip"
DEFAULT_REGRESSOR = R2D2_RL / "outputs" / "pregrasp_regressor" / "best_pregrasp_mlp.pt"


@dataclass
class GoalResult:
    index: int
    target_color: str
    bowl_xyz: list[float]
    success: bool
    final_state: str
    attempts: int
    failure_reason: str | None
    trace: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=R2D2_RL / "configs" / "hybrid_control_rl" / "eval3.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--use-pregrasp-regressor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Flo's xy->joint pregrasp regressor for each approach phase.",
    )
    parser.add_argument(
        "--pregrasp-regressor-checkpoint",
        type=Path,
        default=DEFAULT_REGRESSOR,
        help="Pregrasp regressor checkpoint (.pt or artifacts .zip).",
    )
    parser.add_argument(
        "--sb3-align-grasp-checkpoint",
        type=Path,
        default=None,
        help="Optional Flo/SB3 SAC checkpoint zip for the learned align_grasp phase.",
    )
    parser.add_argument("--sb3-align-grasp-device", default="cpu")
    parser.add_argument("--sb3-align-grasp-max-steps", type=int, default=80)
    parser.add_argument(
        "--enable-watchdog",
        action="store_true",
        help="Enable wrist-color visibility checks during transport.",
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


def _cube_specs_from_config(config: dict) -> list:
    from envs.project3_so101_env import CubeSpec

    scene_cfg = config.get("scene") or {}
    raw = scene_cfg.get("cubes") or []
    if not raw:
        # Fall back to a single cube based on the first goal's color, placed
        # at the default workspace center.
        goals = (config.get("eval") or {}).get("goals") or []
        color = goals[0].get("target_color", "green") if goals else "green"
        if color == "any":
            color = "green"
        return [CubeSpec(color=color, xy=(0.21, -0.03), z=0.01)]

    specs = []
    for entry in raw:
        xy = entry.get("xy") or entry.get("xy_base") or [0.21, -0.03]
        specs.append(
            CubeSpec(
                color=str(entry["color"]),
                xy=(float(xy[0]), float(xy[1])),
                z=float(entry.get("z", 0.01)),
                yaw=float(entry.get("yaw", 0.0)),
            )
        )
    return specs


def _goals_from_config(config: dict, fallback_color: str = "green"):
    from runtime.hybrid_task_executor import TaskGoal

    raw_goals = (config.get("eval") or {}).get("goals") or []
    if not raw_goals:
        return [TaskGoal(target_color=fallback_color, bowl_xyz_base=np.array([0.30, 0.10, 0.05]))]

    goals = []
    for entry in raw_goals:
        color = entry.get("target_color", fallback_color)
        if color == "any":
            color = fallback_color
        bowl = entry.get("bowl_xyz_base", [0.30, 0.10, 0.05])
        goals.append(
            TaskGoal(
                target_color=str(color),
                bowl_xyz_base=np.asarray(bowl, dtype=np.float64).reshape(3),
            )
        )
    return goals


def _trace_to_dicts(trace) -> list[dict]:
    out = []
    for event in trace:
        out.append(
            {
                "attempt": int(getattr(event, "attempt", 0)),
                "state": str(getattr(event, "state", "")).split(".")[-1].lower(),
                "detail": str(getattr(event, "detail", "")),
            }
        )
    return out


def main() -> int:
    args = parse_args()
    setup_headless_rendering()

    from control.waypoint_controller import RcsWaypointController
    from envs.project3_so101_env import Project3SO101Config, Project3SO101Env
    from hybrid_control_rl.config import load_yaml_config
    from planning.hybrid_waypoint_planner import HybridWaypointPlanner
    from runtime.hybrid_task_executor import HybridTaskExecutor
    from runtime.rcs_sim_adapters import (
        RcsColorVisibilityChecker,
        RcsWristBlockObserver,
        ScriptedAlignGraspPolicy,
        wrist_rgb_from_obs,
    )

    config = load_yaml_config(args.config)
    cube_specs = _cube_specs_from_config(config)
    goals = _goals_from_config(config)
    if not goals:
        raise SystemExit("No goals declared in config.eval.goals.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    p3_cfg = Project3SO101Config(
        cubes=cube_specs,
        wrist_camera_resolution=(args.width, args.height),
        headless=True,
    )
    wrist_camera_name = p3_cfg.wrist_camera_name
    scene = Project3SO101Env(p3_cfg)
    env = scene.create_env(scene.config())
    controller = RcsWaypointController(
        env,
        strict_position=args.strict_controller,
        use_pregrasp_regressor=args.use_pregrasp_regressor,
        pregrasp_regressor_checkpoint=args.pregrasp_regressor_checkpoint
        if args.use_pregrasp_regressor
        else None,
    )

    try:
        controller.reset(seed=args.seed)

        if args.save_images and controller.last_obs is not None:
            initial_rgb = wrist_rgb_from_obs(controller.last_obs, camera_name=wrist_camera_name)
            if initial_rgb is not None:
                save_rgb(initial_rgb, args.output_dir / "initial_wrist.png")
            save_external_view(env, args.output_dir / "initial_external.png", args.width, args.height)

        planner = HybridWaypointPlanner(config)
        observer = RcsWristBlockObserver(controller, config, camera_name=wrist_camera_name)
        policy_name = "scripted"
        if args.sb3_align_grasp_checkpoint is not None:
            if not args.sb3_align_grasp_checkpoint.exists():
                raise FileNotFoundError(args.sb3_align_grasp_checkpoint)
            from runtime.sb3_visual_align_grasp_policy import SB3VisualAlignGraspPolicy

            policy = SB3VisualAlignGraspPolicy(
                controller=controller,
                config=config,
                checkpoint_path=args.sb3_align_grasp_checkpoint,
                device=args.sb3_align_grasp_device,
                max_steps=args.sb3_align_grasp_max_steps,
                camera_name=wrist_camera_name,
            )
            policy_name = "sb3_visual"
        else:
            policy = ScriptedAlignGraspPolicy(controller, config)

        visibility = (
            RcsColorVisibilityChecker(controller, config, camera_name=wrist_camera_name)
            if args.enable_watchdog
            else None
        )
        executor = HybridTaskExecutor(
            config=config,
            planner=planner,
            observer=observer,
            controller=controller,
            local_policy=policy,
            visibility_checker=visibility,
        )

        # Run goals one-by-one so per-goal images + traces can be captured.
        per_goal: list[GoalResult] = []
        n_successes = 0
        for idx, goal in enumerate(goals):
            color = goal.target_color
            result = executor.run_goal(goal)
            per_goal.append(
                GoalResult(
                    index=idx,
                    target_color=color,
                    bowl_xyz=goal.bowl_xyz_base.round(4).tolist(),
                    success=bool(result.success),
                    final_state=result.final_state.value,
                    attempts=int(result.attempts),
                    failure_reason=result.failure_reason,
                    trace=_trace_to_dicts(result.trace),
                )
            )
            if result.success:
                n_successes += 1

            if args.save_images and controller.last_obs is not None:
                tag = f"goal{idx:02d}_{color}"
                rgb = wrist_rgb_from_obs(controller.last_obs, camera_name=wrist_camera_name)
                if rgb is not None:
                    save_rgb(rgb, args.output_dir / f"{tag}_wrist.png")
                save_external_view(env, args.output_dir / f"{tag}_external.png", args.width, args.height)

            if not result.success:
                # `run_sequence` short-circuits on the first failure; we mirror
                # that semantics here so failure traces match Project 3 grading.
                break

        all_pass = n_successes == len(goals)
        summary = {
            "config": str(args.config),
            "policy": policy_name,
            "wrist_camera": wrist_camera_name,
            "num_goals": len(goals),
            "num_successes": n_successes,
            "all_pass": all_pass,
            "per_goal": [asdict(g) for g in per_goal],
        }
        with (args.output_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)

        print(f"Eval sequence result ({Path(args.config).name}):")
        print(f"  policy:        {policy_name}")
        print(f"  wrist_camera:  {wrist_camera_name}")
        print(f"  cubes in scene:")
        for spec in cube_specs:
            print(f"    {spec.color:8s}  xy={spec.xy}  z={spec.z}")
        print(f"  goals:         {len(goals)}")
        for entry in per_goal:
            status = "OK" if entry.success else f"FAIL ({entry.failure_reason})"
            print(
                f"    [{entry.index}] target={entry.target_color:7s}  "
                f"bowl={entry.bowl_xyz}  final={entry.final_state}  -> {status}"
            )
        print(f"  successes:     {n_successes}/{len(goals)}")
        if args.save_images:
            print(f"  images:        {args.output_dir}")
        print(f"  summary:       {args.output_dir / 'summary.json'}")
        return 0 if all_pass else 1
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    sys.exit(main())
