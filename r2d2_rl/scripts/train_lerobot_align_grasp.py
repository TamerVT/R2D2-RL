"""Train a visual SAC align-grasp policy on the LeRobot-compatible RCS env.

Run from ``project3`` with ``lerobot-p3-rcs`` active:

    MUJOCO_GL=egl python r2d2_rl/scripts/train_lerobot_align_grasp.py --smoke-only

The learned policy consumes the same keys expected on real SO-101 rollouts:
``observation.images.wrist`` and ``observation.state``. The actor outputs a
scaled [-1, 1] action internally; the env receives a 6D absolute SO-101
follower target in calibrated LeRobot-like units.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
R2D2_RL = REPO_ROOT / "r2d2_rl"
sys.path.insert(0, str(R2D2_RL))


def _hidden_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part) for part in value.split(",") if part.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("hidden sizes must be comma-separated positive ints")
    return sizes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--warmup-steps", type=int, default=1_000)
    p.add_argument("--update-every", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-capacity", type=int, default=50_000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=R2D2_RL / "outputs" / "lerobot_align_grasp_visual_sac",
    )
    p.add_argument("--normalization-stats", type=Path)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cube-color", type=str, default="green")
    p.add_argument("--use-pregrasp-regressor", action="store_true")
    p.add_argument(
        "--pregrasp-regressor-checkpoint",
        type=Path,
        default=R2D2_RL / "outputs" / "pregrasp_regressor" / "best_pregrasp_mlp.pt",
    )
    p.add_argument("--hidden-sizes", type=_hidden_sizes, default=(512, 256))
    p.add_argument("--image-feature-dim", type=int, default=256)
    p.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run a tiny reset/step/update/save pass to verify simulator and trainer wiring.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    shader_cache = Path("/tmp") / "mesa_shader_cache"
    shader_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MESA_SHADER_CACHE_DIR", str(shader_cache))

    import torch

    from rl.lerobot_align_grasp_env import LeRobotAlignGraspEnv, LeRobotAlignGraspEnvConfig
    from rl.lerobot_compat import IMAGE_KEY, STATE_KEY, load_state_normalization, scaled_to_lerobot_action
    from rl.visual_replay_buffer import VisualReplayBuffer
    from rl.visual_sac import VisualSACAgent, VisualSACConfig

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.smoke_only:
        args.total_steps = min(args.total_steps, 6)
        args.warmup_steps = min(args.warmup_steps, 2)
        args.batch_size = min(args.batch_size, 2)
        args.buffer_capacity = min(args.buffer_capacity, 128)
        args.log_every = 1
        args.checkpoint_every = 10_000
        args.hidden_sizes = (64, 64)
        args.image_feature_dim = min(args.image_feature_dim, 64)

    state_mean, state_std = load_state_normalization(args.normalization_stats)

    env_cfg = LeRobotAlignGraspEnvConfig(
        cube_color=args.cube_color,
        use_pregrasp_regressor=args.use_pregrasp_regressor,
        pregrasp_regressor_checkpoint=args.pregrasp_regressor_checkpoint,
    )
    env = LeRobotAlignGraspEnv(env_cfg)

    obs, info = env.reset(seed=args.seed)
    image_shape = tuple(int(v) for v in obs[IMAGE_KEY].shape)
    state_dim = int(obs[STATE_KEY].shape[0])
    act_dim = int(env.action_space.shape[0])
    print(f"[info] image_shape={image_shape} state_dim={state_dim} act_dim={act_dim}")
    print(f"[info] first reset keys={list(obs.keys())} camera_available={info.get('camera_available')}")

    agent = VisualSACAgent(
        act_dim=act_dim,
        config=VisualSACConfig(
            hidden_sizes=args.hidden_sizes,
            image_feature_dim=args.image_feature_dim,
            state_mean=tuple(float(v) for v in state_mean),
            state_std=tuple(float(v) for v in state_std),
            device=device,
        ),
    )
    buffer = VisualReplayBuffer(args.buffer_capacity, image_shape, state_dim, act_dim)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config["checkpoint_dir"] = str(run_config["checkpoint_dir"])
    run_config["normalization_stats"] = (
        None if run_config["normalization_stats"] is None else str(run_config["normalization_stats"])
    )
    run_config["pregrasp_regressor_checkpoint"] = str(run_config["pregrasp_regressor_checkpoint"])
    run_config["hidden_sizes"] = list(run_config["hidden_sizes"])
    run_config["env_config"] = asdict(env_cfg)
    (args.checkpoint_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))

    episode_return = 0.0
    episode_length = 0
    episodes_done = 0
    last_metrics: dict[str, float] = {}
    start = time.time()

    for step in range(1, args.total_steps + 1):
        if step <= args.warmup_steps:
            scaled_action = np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            scaled_action = agent.act(obs, deterministic=False)
        env_action = scaled_to_lerobot_action(scaled_action)

        next_obs, reward, terminated, truncated, info = env.step(env_action)
        done_for_bootstrap = bool(terminated)
        buffer.add(obs, scaled_action, reward, next_obs, done_for_bootstrap)

        episode_return += float(reward)
        episode_length += 1
        if terminated or truncated:
            episodes_done += 1
            print(
                f"[ep {episodes_done:4d}] step={step:6d} return={episode_return:+.3f} "
                f"length={episode_length} success={float(info.get('success', 0.0)):.0f}"
            )
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
        else:
            obs = next_obs

        if step > args.warmup_steps and len(buffer) >= args.batch_size and step % args.update_every == 0:
            batch = buffer.sample(args.batch_size, device=device)
            last_metrics = agent.update(batch)

        if step % args.log_every == 0:
            elapsed = time.time() - start
            metrics = " ".join(f"{key}={value:+.3f}" for key, value in last_metrics.items())
            print(
                f"[step {step:6d}] {step / max(elapsed, 1e-6):.1f} env/s "
                f"buffer={len(buffer)} reward={float(reward):+.3f} "
                f"xy={float(info.get('xy_dist', np.nan)):.4f} lift={float(info.get('cube_lift', 0.0)):.4f} "
                f"{metrics}"
            )

        if step % args.checkpoint_every == 0:
            ckpt_path = args.checkpoint_dir / f"visual_sac_step_{step:06d}.pt"
            agent.save(ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

    if args.smoke_only and not last_metrics:
        env.close()
        raise RuntimeError("--smoke-only finished without running a visual SAC update.")

    final_path = args.checkpoint_dir / "visual_sac_final.pt"
    agent.save(final_path)
    print(f"[done] final checkpoint: {final_path}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
