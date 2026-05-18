from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from stable_baselines3 import SAC

from project3_modular.rl_grasp.envs.so101_local_grasp_hil_compat_env import (
    SO101LocalGraspHILCompatEnv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        default="project3_modular/rl_grasp/outputs/visual_hil_sac_sim_rollout",
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic SAC actions instead of deterministic mean actions.",
    )
    return parser.parse_args()


def wrist_obs_to_bgr(obs: dict[str, np.ndarray]) -> np.ndarray:
    chw = obs["observation.images.wrist"]
    rgb = np.transpose(chw, (1, 2, 0))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = SO101LocalGraspHILCompatEnv(open_gui=False)
    model = SAC.load(
        args.checkpoint,
        env=env,
        device=args.device,
    )

    deterministic = not args.stochastic

    print(f"Loaded SAC checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Deterministic actions: {deterministic}")

    summaries: list[dict] = []

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)

        total_reward = 0.0
        success = False
        ever_grasped = False
        max_lift = 0.0
        min_xy_dist = float("inf")
        min_xyz_dist = float("inf")
        frames: list[np.ndarray] = []

        last_info = info

        for step in range(args.max_steps):
            if args.save_video:
                frames.append(wrist_obs_to_bgr(obs))

            action, _ = model.predict(
                obs,
                deterministic=deterministic,
            )

            obs, reward, terminated, truncated, info = env.step(action)
            last_info = info

            total_reward += float(reward)

            local_success = bool(info.get("local_success", False))
            sim_is_grasped = bool(info.get("sim_is_grasped", False))
            cube_lift = float(info.get("cube_lift", 0.0))
            xy_dist = float(info.get("xy_dist", np.nan))
            xyz_dist = float(info.get("xyz_dist", np.nan))

            success = success or local_success
            ever_grasped = ever_grasped or sim_is_grasped
            max_lift = max(max_lift, cube_lift)

            if not np.isnan(xy_dist):
                min_xy_dist = min(min_xy_dist, xy_dist)
            if not np.isnan(xyz_dist):
                min_xyz_dist = min(min_xyz_dist, xyz_dist)

            if terminated or truncated:
                break

        if args.save_video and frames:
            video_path = output_dir / f"episode_{ep:03d}.mp4"
            height, width = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (width, height),
            )
            for frame in frames:
                writer.write(frame)
            writer.release()

        summary = {
            "episode": ep,
            "steps": step + 1,
            "total_reward": total_reward,
            "success": success,
            "ever_grasped": ever_grasped,
            "max_cube_lift": max_lift,
            "min_xy_dist": None if min_xy_dist == float("inf") else min_xy_dist,
            "min_xyz_dist": None if min_xyz_dist == float("inf") else min_xyz_dist,
            "final_cube_lift": float(last_info.get("cube_lift", 0.0)),
            "final_sim_is_grasped": bool(last_info.get("sim_is_grasped", False)),
        }
        summaries.append(summary)

        print(
            f"Episode {ep:02d} | "
            f"steps={summary['steps']:03d} | "
            f"reward={summary['total_reward']:.3f} | "
            f"success={summary['success']} | "
            f"ever_grasped={summary['ever_grasped']} | "
            f"max_lift={summary['max_cube_lift']:.4f} | "
            f"min_xy={summary['min_xy_dist']:.4f} | "
            f"min_xyz={summary['min_xyz_dist']:.4f}"
        )

    env.close()

    successes = sum(int(s["success"]) for s in summaries)
    grasped = sum(int(s["ever_grasped"]) for s in summaries)

    summary_path = output_dir / "rollout_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2))

    print()
    print(f"Successes: {successes}/{args.episodes}")
    print(f"Ever grasped: {grasped}/{args.episodes}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
