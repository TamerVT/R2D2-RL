from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from project3_modular.rl_grasp.envs.so101_local_grasp_hil_compat_env import (
    SO101LocalGraspHILCompatEnv,
)
from project3_modular.rl_grasp.scripts.train_real_hil_bc_policy import (
    RealHILBCPolicy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to BC checkpoint, e.g. .../checkpoints/best.pt",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        default="project3_modular/rl_grasp/outputs/real_hil_bc_sim_rollout_v1",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save 128x128 wrist-camera rollout videos.",
    )
    return parser.parse_args()


def stats_to_tensors(stats: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "state_mean": torch.tensor(stats["state_mean"], dtype=torch.float32, device=device),
        "state_std": torch.tensor(stats["state_std"], dtype=torch.float32, device=device),
        "action_mean": torch.tensor(stats["action_mean"], dtype=torch.float32, device=device),
        "action_std": torch.tensor(stats["action_std"], dtype=torch.float32, device=device),
    }


@torch.no_grad()
def predict_action(
    model: RealHILBCPolicy,
    obs: dict[str, np.ndarray],
    stats: dict[str, torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    image = torch.from_numpy(obs["observation.images.wrist"]).float() / 255.0
    image = image.unsqueeze(0).to(device)

    state = torch.from_numpy(obs["observation.state"]).float().unsqueeze(0).to(device)
    norm_state = (state - stats["state_mean"]) / stats["state_std"]

    pred_norm_action = model(image, norm_state)
    pred_action = (
        pred_norm_action * stats["action_std"]
        + stats["action_mean"]
    )

    return pred_action.squeeze(0).cpu().numpy().astype(np.float32)


def wrist_obs_to_bgr(obs: dict[str, np.ndarray]) -> np.ndarray:
    chw = obs["observation.images.wrist"]
    rgb = np.transpose(chw, (1, 2, 0))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = RealHILBCPolicy().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats = stats_to_tensors(checkpoint["normalization_stats"], device=device)

    env = SO101LocalGraspHILCompatEnv(open_gui=False)

    rollout_summaries: list[dict] = []

    print(f"Loaded BC checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)

        total_reward = 0.0
        final_info = info
        success = False
        frames: list[np.ndarray] = []

        predicted_gripper_targets: list[float] = []
        converted_base_gripper_actions: list[float] = []

        for step in range(args.max_steps):
            if args.save_video:
                frames.append(wrist_obs_to_bgr(obs))

            action = predict_action(model, obs, stats, device)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            predicted_gripper_targets.append(float(action[5]))

            obs, reward, terminated, truncated, info = env.step(action)

            base_action = info.get("hil_compat_base_action")
            if base_action is not None:
                converted_base_gripper_actions.append(float(base_action[5]))

            total_reward += float(reward)
            final_info = info

            local_success = bool(info.get("local_success", False))
            success = success or local_success

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
            "final_cube_lift": float(final_info.get("cube_lift", np.nan)),
            "final_xy_dist": float(final_info.get("xy_dist", np.nan)),
            "final_xyz_dist": float(final_info.get("xyz_dist", np.nan)),
            "final_sim_is_grasped": bool(final_info.get("sim_is_grasped", False)),
        }
        rollout_summaries.append(summary)

        if predicted_gripper_targets:
            grip_min = min(predicted_gripper_targets)
            grip_max = max(predicted_gripper_targets)
        else:
            grip_min = grip_max = float("nan")

        if converted_base_gripper_actions:
            base_grip_min = min(converted_base_gripper_actions)
            base_grip_max = max(converted_base_gripper_actions)
        else:
            base_grip_min = base_grip_max = float("nan")

        print(
            f"Episode {ep:02d} | "
            f"steps={summary['steps']:03d} | "
            f"reward={summary['total_reward']:.3f} | "
            f"success={summary['success']} | "
            f"grasped={summary['final_sim_is_grasped']} | "
            f"lift={summary['final_cube_lift']:.4f} | "
            f"pred_gripper=[{grip_min:.3f}, {grip_max:.3f}] | "
            f"base_gripper=[{base_grip_min:.3f}, {base_grip_max:.3f}]"
        )

    env.close()

    with (output_dir / "rollout_summary.json").open("w") as f:
        json.dump(rollout_summaries, f, indent=2)

    successes = sum(int(s["success"]) for s in rollout_summaries)
    print()
    print(f"Successes: {successes}/{args.episodes}")
    print(f"Saved summary to: {output_dir / 'rollout_summary.json'}")


if __name__ == "__main__":
    main()
