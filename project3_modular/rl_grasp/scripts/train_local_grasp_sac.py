from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from project3_modular.rl_grasp.envs.so101_local_grasp_env import (
    SO101LocalGraspConfig,
    SO101LocalGraspEnv,
)


class SuccessLoggingCallback(BaseCallback):
    """
    Logs episodic success and grasp statistics from info dicts.

    RCS PickTask writes:
      info["success"]
      info["is_grasped"]
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.successes: list[float] = []
        self.grasps: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" not in info:
                continue

            success = float(bool(info.get("local_success", info.get("success", False))))
            is_grasped = float(bool(info.get("sim_is_grasped", False)))

            self.successes.append(success)
            self.grasps.append(is_grasped)

            recent_success = np.mean(self.successes[-50:])
            recent_grasp = np.mean(self.grasps[-50:])

            self.logger.record("rollout/success_rate_50ep", float(recent_success))
            self.logger.record("rollout/grasp_rate_50ep", float(recent_grasp))

        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a state-based SAC policy for local SO101 grasping."
    )

    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)

    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument("--joint-delta-deg", type=float, default=5.0)
    parser.add_argument("--cube-randomization-width", type=float, default=0.06)

    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)

    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "project3_modular" / "rl_grasp" / "outputs",
    )
    parser.add_argument(
        "--success-lift-delta-m",
        type=float,
        default=0.005,
        help="Required cube lift above reset height for success.",
    )

    return parser.parse_args()


def make_env_config(args: argparse.Namespace) -> SO101LocalGraspConfig:
    return SO101LocalGraspConfig(
        cube_center=(0.18, 0.03, 0.01),
        cube_randomization_xy=(
            args.cube_randomization_width,
            args.cube_randomization_width,
        ),
        max_episode_steps=args.episode_steps,
        joint_delta_deg=args.joint_delta_deg,
        target_color="green",
        success_lift_delta_m=args.success_lift_delta_m,
    )


def make_monitored_env(
    config: SO101LocalGraspConfig,
    *,
    seed: int,
) -> Monitor:
    env = SO101LocalGraspEnv(
        config=config,
        open_gui=False,
    )
    env.reset(seed=seed)

    return Monitor(
        env,
        info_keywords=(
            "success",
            "local_success",
            "sim_is_grasped",
            "cube_z",
            "cube_lift",
            "xy_dist",
            "xyz_dist",
        ),
    )


def save_run_config(
    run_dir: Path,
    args: argparse.Namespace,
    env_config: SO101LocalGraspConfig,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "args": vars(args),
        "env_config": {
            "cube_xml": env_config.cube_xml,
            "cube_center": list(env_config.cube_center),
            "cube_randomization_xy": list(env_config.cube_randomization_xy),
            "max_episode_steps": env_config.max_episode_steps,
            "joint_delta_deg": env_config.joint_delta_deg,
            "robot_z_offset": env_config.robot_z_offset,
            "target_color": env_config.target_color,
            "action_scale": env_config.action_scale,
            "pregrasp_q_home_rad": env_config.pregrasp_q_home_rad,
        },
    }

    # pathlib Paths are not JSON serializable by default
    payload["args"]["output_root"] = str(args.output_root)

    path = run_dir / "run_config.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / f"sac_state_local_grasp_{timestamp}"
    model_dir = run_dir / "models"
    best_dir = run_dir / "best_model"
    eval_log_dir = run_dir / "eval"
    tb_dir = run_dir / "tb"
    checkpoint_dir = run_dir / "checkpoints"

    for path in [model_dir, best_dir, eval_log_dir, tb_dir, checkpoint_dir]:
        path.mkdir(parents=True, exist_ok=True)

    env_config = make_env_config(args)
    save_run_config(run_dir, args, env_config)

    train_env = make_monitored_env(
        env_config,
        seed=args.seed,
    )
    eval_env = make_monitored_env(
        env_config,
        seed=args.seed + 1,
    )

    model = SAC(
        policy="MultiInputPolicy",
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        tensorboard_log=str(tb_dir),
        policy_kwargs={
            "net_arch": [256, 256],
        },
        verbose=1,
        seed=args.seed,
        device="auto",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_dir),
        log_path=str(eval_log_dir),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix="sac_local_grasp",
        save_replay_buffer=True,
        save_vecnormalize=False,
    )

    success_callback = SuccessLoggingCallback()

    callbacks = CallbackList(
        [
            eval_callback,
            checkpoint_callback,
            success_callback,
        ]
    )

    print("\n=== Training SAC local grasp policy ===")
    print(f"Run dir:              {run_dir}")
    print(f"Total timesteps:      {args.total_timesteps}")
    print(f"Episode length:       {args.episode_steps}")
    print(f"Cube randomization:   ±{args.cube_randomization_width / 2:.3f} m")
    print(f"TensorBoard dir:      {tb_dir}")
    print()

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name="SAC_state_local_grasp",
        progress_bar=True,
    )

    final_model_path = model_dir / "final_model"
    model.save(str(final_model_path))
    model.save_replay_buffer(str(model_dir / "final_replay_buffer"))

    train_env.close()
    eval_env.close()

    print("\nTraining complete.")
    print(f"Final model:       {final_model_path}.zip")
    print(f"Best model dir:    {best_dir}")
    print(f"Run directory:     {run_dir}")


if __name__ == "__main__":
    main()
