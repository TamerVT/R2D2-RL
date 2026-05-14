"""SAC trainer for the local ``align_grasp`` policy.

Run from the project root with ``lerobot-p3-rcs`` active::

    MUJOCO_GL=egl python r2d2_rl/scripts/train_align_grasp.py \
        --total-steps 100000 \
        --checkpoint-dir r2d2_rl/outputs/align_grasp_sac

The script is intentionally short: env → replay buffer → SAC agent →
periodic update + logging + checkpointing. The model and obs shapes are
read from the env, and an inference-friendly final checkpoint is saved at
the end (or whenever ``--checkpoint-every`` triggers).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
R2D2_RL = REPO_ROOT / "r2d2_rl"
sys.path.insert(0, str(R2D2_RL))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--warmup-steps", type=int, default=1_000)
    p.add_argument("--update-every", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--buffer-capacity", type=int, default=200_000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--checkpoint-every", type=int, default=10_000)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=R2D2_RL / "outputs" / "align_grasp_sac",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run 5 env steps + 1 update + 1 save to verify wiring, then exit.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    import torch

    from rl.align_grasp_env import AlignGraspEnv, AlignGraspEnvConfig
    from rl.replay_buffer import ReplayBuffer
    from rl.sac import SACAgent, SACConfig

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = AlignGraspEnv(AlignGraspEnvConfig())
    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    print(f"[info] obs_dim={obs_dim}  act_dim={act_dim}")

    agent = SACAgent(obs_dim, act_dim, SACConfig(device=args.device))
    buffer = ReplayBuffer(args.buffer_capacity, obs_dim, act_dim)

    total_steps = 5 if args.smoke_only else args.total_steps
    warmup = min(args.warmup_steps, total_steps)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset(seed=args.seed)
    start_time = time.time()
    episode_return = 0.0
    episode_length = 0
    episodes_done = 0
    last_metrics: dict[str, float] = {}

    for step in range(1, total_steps + 1):
        if step < warmup:
            action = env.action_space.sample().astype(np.float32)
        else:
            action = agent.act(obs, deterministic=False).astype(np.float32)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated)
        buffer.add(obs, action, reward, next_obs, done)
        episode_return += float(reward)
        episode_length += 1

        if terminated or truncated:
            episodes_done += 1
            if step % args.log_every < 5 or args.smoke_only:
                print(
                    f"[ep {episodes_done:4d}] step={step:6d}  return={episode_return:+.2f}  "
                    f"length={episode_length}  success={float(info.get('success', 0.0)):.0f}"
                )
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
        else:
            obs = next_obs

        if step >= warmup and len(buffer) >= args.batch_size and step % args.update_every == 0:
            batch = buffer.sample(args.batch_size, device=args.device)
            last_metrics = agent.update(batch)

        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            metrics_str = "  ".join(f"{k}={v:+.3f}" for k, v in last_metrics.items())
            print(
                f"[step {step:6d}] {step / max(elapsed, 1e-6):.1f} env/s  buffer={len(buffer)}  {metrics_str}"
            )

        if step % args.checkpoint_every == 0 or args.smoke_only:
            ckpt_path = args.checkpoint_dir / f"sac_step_{step:06d}.pt"
            agent.save(ckpt_path)
            print(f"[ckpt] saved {ckpt_path}")

    final_path = args.checkpoint_dir / "sac_final.pt"
    agent.save(final_path)
    print(f"[done] final checkpoint: {final_path}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
