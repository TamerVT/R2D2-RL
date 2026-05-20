"""BC-pretrain + SAC finetune trainer for SO-101 local grasp (RCS sim).

Two-stage training:

1. **BC pretraining** of the SAC actor on the LeRobot multicolor HIL dataset
   (``p3_local_grasp_hil_multicolor_colorcond_v1``). Supervised MSE between the
   actor's deterministic output (in scaled ``[-1, 1]`` action space) and the
   demonstrated actions in the same scaled space.
2. **SAC RL finetune** in the RCS sim (``LeRobotAlignGraspEnv``).

The env and the dataset share the same observation/action interface
(``observation.images.wrist`` CHW uint8 [3, 128, 128] + ``observation.state``
float32 [24], 6D absolute follower-joint action in calibrated LeRobot units),
so the actor trained on real demos transfers directly to sim.

Run::

    MUJOCO_GL=egl python r2d2_rl/scripts/train_visual_hil_compat_sac.py \
        --output-dir r2d2_rl/outputs/hil_bc_sac_v1 \
        --normalization-stats r2d2_rl/outputs/real_hil_bc_warmstart/normalization_stats.json \
        --bc-dataset-root r2d2_rl/outputs/hil_dataset/p3_local_grasp_hil_multicolor_colorcond_v1 \
        --bc-pretrain-epochs 80 --bc-zero-motor-currents \
        --total-timesteps 50000
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter


REPO_ROOT = Path(__file__).resolve().parents[2]
R2D2_RL = REPO_ROOT / "r2d2_rl"
if str(R2D2_RL) not in sys.path:
    sys.path.insert(0, str(R2D2_RL))

# These imports rely on r2d2_rl being on sys.path (handled above).
from rl.lerobot_align_grasp_env import LeRobotAlignGraspEnv, LeRobotAlignGraspEnvConfig  # noqa: E402
from scripts.train_real_hil_bc_policy import (  # noqa: E402
    RealHILBCDataset,
    make_episode_split,
)


def _make_align_grasp_env(env_cfg: LeRobotAlignGraspEnvConfig) -> LeRobotAlignGraspEnv:
    """Module-level env factory.

    Must be importable + picklable so ``SubprocVecEnv`` (spawn start method)
    can recreate the env inside each worker process. Each worker builds its
    own independent RCS sim + EGL context.
    """
    return LeRobotAlignGraspEnv(env_cfg)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class TrainConfig:
    output_dir: str
    normalization_stats: str

    # Env
    cube_color: str = "green"
    use_pregrasp_regressor: bool = True
    pregrasp_regressor_checkpoint: str | None = None

    total_timesteps: int = 50_000
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Number of parallel RCS environments (SubprocVecEnv). Env stepping is the
    # wall-clock bottleneck (~4 fps single-env); parallelising it across CPU
    # cores is the biggest speedup that does NOT change env dynamics. n_envs=1
    # keeps the original single-process path.
    n_envs: int = 8

    # SAC
    learning_rate: float = 3e-4
    buffer_size: int = 10_000
    learning_starts: int = 1_000
    batch_size: int = 128
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"

    # BC pretraining of the SAC actor on the real-demo dataset.
    bc_dataset_root: str | None = None
    bc_pretrain_epochs: int = 0
    bc_batch_size: int = 128
    bc_lr: float = 3e-4
    bc_val_episode_fraction: float = 0.2
    bc_zero_motor_currents: bool = True

    checkpoint_freq: int = 10_000

    # Periodic deterministic-policy evaluation (GraspEvalCallback). Lets a
    # human watching the log catch a misbehaving run (e.g. hovering) early.
    eval_freq: int = 5_000
    n_eval_episodes: int = 8

    # Resume an existing SB3 SAC checkpoint. When set, BC pretraining is
    # skipped and total_timesteps is interpreted as *additional* RL timesteps.
    resume_from: str | None = None


# ---------------------------------------------------------------------
# CNN feature extractor
# ---------------------------------------------------------------------


class WristImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(image))


class VisualHILFeaturesExtractor(BaseFeaturesExtractor):
    """Extract wrist RGB (CNN, 256D) + normalized 24D state -> 280D vector."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        state_mean: list[float],
        state_std: list[float],
        image_feature_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim=image_feature_dim + 24)
        self.image_encoder = WristImageEncoder(out_dim=image_feature_dim)
        mean = torch.tensor(state_mean, dtype=torch.float32)
        std = torch.clamp(torch.tensor(state_std, dtype=torch.float32), min=1e-6)
        self.register_buffer("state_mean", mean)
        self.register_buffer("state_std", std)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        image = observations["observation.images.wrist"]
        state = observations["observation.state"]
        image_feat = self.image_encoder(image)
        norm_state = (state - self.state_mean) / self.state_std
        return torch.cat([image_feat, norm_state], dim=-1)


# ---------------------------------------------------------------------
# SB3 callbacks
# ---------------------------------------------------------------------


class LocalGraspInfoCallback(BaseCallback):
    """Forward env-side scalars (cube_lift, xy_dist, success, ...) to TB."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos")
        if not infos:
            return True
        info = infos[0] if isinstance(infos, list) else infos
        for key in ("cube_lift", "xy_dist", "xyz_dist"):
            if key in info:
                self.logger.record(f"env/{key}", float(info[key]))
        if "local_success" in info:
            self.logger.record("env/local_success", float(bool(info["local_success"])))
        if "sim_is_grasped" in info:
            self.logger.record("env/sim_is_grasped", float(bool(info["sim_is_grasped"])))
        return True


class GraspEvalCallback(BaseCallback):
    """Periodically roll out the *deterministic* policy and report behaviour.

    Every ``eval_freq`` steps this runs ``n_eval_episodes`` greedy episodes on
    a dedicated eval env, tallies grasp / lift / success rates, and prints a
    one-line verdict so a human watching the log can decide whether to
    intervene (kill a run that is hovering, etc.). Results are also written to
    ``<output_dir>/eval_log.jsonl`` (one JSON object per evaluation) and to
    TensorBoard under ``eval/``.
    """

    def __init__(
        self,
        eval_env,
        *,
        eval_freq: int,
        n_eval_episodes: int,
        log_path: Path,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self._eval_env = eval_env
        self._eval_freq = max(1, int(eval_freq))
        self._n_eval_episodes = max(1, int(n_eval_episodes))
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_eval_bucket = 0

    @staticmethod
    def _verdict(
        grasp_rate: float,
        success_rate: float,
        mean_xy_dist: float,
        mean_peak_lift: float,
        success_threshold: float,
    ) -> str:
        if success_rate >= 0.5:
            return "GRASPING (good -- policy lifts the cube)"
        if success_rate > 0.0:
            return "PARTIAL (some successes -- keep training)"
        if grasp_rate > 0.0:
            pct = 100.0 * mean_peak_lift / max(1e-9, success_threshold)
            return (
                f"GRASPS-NOT-LIFTS (peak lift {mean_peak_lift * 1000:.1f}mm "
                f"= {pct:.0f}% of {success_threshold * 1000:.0f}mm threshold)"
            )
        if mean_xy_dist < 0.02:
            return "HOVERING (aligns but never grasps -- INTERVENE: reward loophole?)"
        return "EXPLORING (not yet aligning -- early training, keep going)"

    def _run_eval(self) -> dict:
        n = self._n_eval_episodes
        threshold = float(self._eval_env.cfg.success_lift_delta_m)
        n_grasp = n_lift = n_success = 0
        returns: list[float] = []
        final_xy: list[float] = []
        peak_lifts: list[float] = []
        for ep in range(n):
            obs, _info = self._eval_env.reset(seed=10_000 + ep)
            ep_ret = 0.0
            ep_grasp = ep_lift = ep_success = False
            ep_peak_lift = 0.0
            terms: dict = {}
            for _step in range(self._eval_env.cfg.max_episode_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self._eval_env.step(action)
                ep_ret += float(reward)
                terms = info.get("reward_terms", {})
                if float(terms.get("valid_cube_grasp", 0.0)) > 0.5:
                    ep_grasp = True
                cube_lift = float(terms.get("cube_lift", 0.0))
                ep_peak_lift = max(ep_peak_lift, cube_lift)
                if cube_lift >= threshold:
                    ep_lift = True
                if bool(info.get("local_success", False)):
                    ep_success = True
                if terminated or truncated:
                    break
            n_grasp += int(ep_grasp)
            n_lift += int(ep_lift)
            n_success += int(ep_success)
            returns.append(ep_ret)
            final_xy.append(float(terms.get("xy_dist", float("nan"))))
            peak_lifts.append(ep_peak_lift)
        grasp_rate = n_grasp / n
        lift_rate = n_lift / n
        success_rate = n_success / n
        mean_xy = float(np.nanmean(final_xy)) if final_xy else float("nan")
        mean_peak_lift = float(np.mean(peak_lifts)) if peak_lifts else 0.0
        return {
            "step": int(self.num_timesteps),
            "n_episodes": n,
            "grasp_rate": grasp_rate,
            "lift_rate": lift_rate,
            "success_rate": success_rate,
            "mean_return": float(np.mean(returns)),
            "mean_final_xy_dist": mean_xy,
            "mean_peak_lift": mean_peak_lift,
            "success_threshold": threshold,
            "verdict": self._verdict(
                grasp_rate, success_rate, mean_xy, mean_peak_lift, threshold
            ),
        }

    def _emit(self, result: dict) -> None:
        with self._log_path.open("a") as f:
            f.write(json.dumps(result) + "\n")
        self.logger.record("eval/grasp_rate", result["grasp_rate"])
        self.logger.record("eval/lift_rate", result["lift_rate"])
        self.logger.record("eval/success_rate", result["success_rate"])
        self.logger.record("eval/mean_return", result["mean_return"])
        self.logger.record("eval/mean_final_xy_dist", result["mean_final_xy_dist"])
        self.logger.record("eval/mean_peak_lift", result["mean_peak_lift"])
        print(
            f"[eval @ step {result['step']:6d}] "
            f"grasp={result['grasp_rate']:.0%}  lift={result['lift_rate']:.0%}  "
            f"success={result['success_rate']:.0%}  "
            f"mean_return={result['mean_return']:+.1f}  "
            f"peak_lift={result['mean_peak_lift'] * 1000:.1f}mm  "
            f"xy_dist={result['mean_final_xy_dist']:.4f}  ->  {result['verdict']}",
            flush=True,
        )

    def _on_training_start(self) -> None:
        # Anchor the eval bucket to the current step count so a fresh run's
        # first eval lands at eval_freq and a resumed run's lands one bucket
        # after where it left off.
        self._last_eval_bucket = int(self.num_timesteps) // self._eval_freq

    def _on_step(self) -> bool:
        # Bucket comparison (not exact modulo): with a VecEnv num_timesteps
        # advances by n_envs each call and would skip an exact multiple.
        bucket = int(self.num_timesteps) // self._eval_freq
        if bucket > self._last_eval_bucket:
            self._last_eval_bucket = bucket
            self._emit(self._run_eval())
        return True

    def _on_training_end(self) -> None:
        # Final evaluation so the last checkpoint is always characterized
        # (skipped if the last step already triggered one).
        bucket = int(self.num_timesteps) // self._eval_freq
        if bucket > self._last_eval_bucket:
            self._last_eval_bucket = bucket
            self._emit(self._run_eval())


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def load_normalization_stats(path: str | Path) -> dict[str, list[float]]:
    stats = json.loads(Path(path).read_text())
    for key in ("state_mean", "state_std"):
        if key not in stats:
            raise RuntimeError(f"Normalization stats missing key {key!r}")
    return stats


def env_actions_to_scaled_actions(
    actions: torch.Tensor, action_low: torch.Tensor, action_high: torch.Tensor
) -> torch.Tensor:
    actions = torch.clamp(actions, action_low, action_high)
    return torch.clamp(
        2.0 * (actions - action_low) / (action_high - action_low) - 1.0, -1.0, 1.0
    )


def scaled_actions_to_env_actions(
    scaled_actions: torch.Tensor, action_low: torch.Tensor, action_high: torch.Tensor
) -> torch.Tensor:
    scaled_actions = torch.clamp(scaled_actions, -1.0, 1.0)
    return action_low + 0.5 * (scaled_actions + 1.0) * (action_high - action_low)


# ---------------------------------------------------------------------
# Actor BC pretraining
# ---------------------------------------------------------------------


@torch.no_grad()
def evaluate_actor_bc(
    *,
    actor: nn.Module,
    loader: DataLoader,
    device: torch.device,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
) -> dict[str, float]:
    actor.eval()
    total_scaled_mse = 0.0
    total_raw_mae = 0.0
    total_count = 0
    gripper_true: list[np.ndarray] = []
    gripper_pred: list[np.ndarray] = []

    for batch in loader:
        image_float = batch["image"].to(device)
        image_uint8 = torch.clamp(torch.round(image_float * 255.0), 0, 255).to(torch.uint8)
        state = batch["state"].to(device)
        action = batch["action"].to(device)
        obs = {"observation.images.wrist": image_uint8, "observation.state": state}

        pred_scaled = actor(obs, deterministic=True)
        target_scaled = env_actions_to_scaled_actions(action, action_low, action_high)
        pred_env = scaled_actions_to_env_actions(pred_scaled, action_low, action_high)

        batch_size = action.shape[0]
        total_scaled_mse += float(F.mse_loss(pred_scaled, target_scaled, reduction="sum").item())
        total_raw_mae += float(F.l1_loss(pred_env, action, reduction="sum").item())
        total_count += batch_size
        gripper_true.append(action[:, 5].detach().cpu().numpy())
        gripper_pred.append(pred_env[:, 5].detach().cpu().numpy())

    denom = max(1, total_count * 6)
    y_true = np.concatenate(gripper_true)
    y_pred = np.concatenate(gripper_pred)
    grip_corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "scaled_mse": total_scaled_mse / denom,
        "raw_mae": total_raw_mae / denom,
        "gripper_corr": grip_corr,
        "gripper_true_p95": float(np.percentile(y_true, 95)),
        "gripper_pred_p95": float(np.percentile(y_pred, 95)),
    }


def bc_pretrain_sac_actor(*, model: SAC, cfg: TrainConfig, output_dir: Path) -> None:
    if cfg.bc_dataset_root is None or cfg.bc_pretrain_epochs <= 0:
        print("Skipping real-demo actor BC pretraining.")
        return

    print()
    print("=== Real-demo BC pretraining of SAC actor ===")

    dataset = RealHILBCDataset(
        Path(cfg.bc_dataset_root),
        image_size=128,
        zero_motor_currents=cfg.bc_zero_motor_currents,
    )
    train_indices, val_indices = make_episode_split(
        dataset.episode_indices,
        val_fraction=cfg.bc_val_episode_fraction,
        seed=cfg.seed,
    )

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=cfg.bc_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(model.device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=cfg.bc_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(model.device.type == "cuda"),
    )

    actor = model.policy.actor
    actor.train()
    optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.bc_lr)
    action_low = torch.tensor(model.action_space.low, dtype=torch.float32, device=model.device)
    action_high = torch.tensor(model.action_space.high, dtype=torch.float32, device=model.device)
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard_bc_pretrain"))
    best_val = float("inf")

    for epoch in range(1, cfg.bc_pretrain_epochs + 1):
        actor.train()
        running_loss = 0.0
        running_count = 0
        for batch in train_loader:
            image_float = batch["image"].to(model.device)
            image_uint8 = torch.clamp(
                torch.round(image_float * 255.0), 0, 255
            ).to(torch.uint8)
            state = batch["state"].to(model.device)
            action = batch["action"].to(model.device)
            obs = {"observation.images.wrist": image_uint8, "observation.state": state}
            pred_scaled = actor(obs, deterministic=True)
            target_scaled = env_actions_to_scaled_actions(action, action_low, action_high)
            loss = F.mse_loss(pred_scaled, target_scaled)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_size = action.shape[0]
            running_loss += float(loss.item()) * batch_size
            running_count += batch_size

        train_loss = running_loss / max(1, running_count)
        val_metrics = evaluate_actor_bc(
            actor=actor,
            loader=val_loader,
            device=model.device,
            action_low=action_low,
            action_high=action_high,
        )
        writer.add_scalar("bc_actor/train_scaled_mse", train_loss, epoch)
        writer.add_scalar("bc_actor/val_scaled_mse", val_metrics["scaled_mse"], epoch)
        writer.add_scalar("bc_actor/val_raw_mae", val_metrics["raw_mae"], epoch)
        writer.add_scalar("bc_actor/val_gripper_corr", val_metrics["gripper_corr"], epoch)
        writer.add_scalar("bc_actor/val_gripper_true_p95", val_metrics["gripper_true_p95"], epoch)
        writer.add_scalar("bc_actor/val_gripper_pred_p95", val_metrics["gripper_pred_p95"], epoch)
        print(
            f"BC epoch {epoch:03d}/{cfg.bc_pretrain_epochs} | "
            f"train_scaled_mse={train_loss:.6f} | "
            f"val_scaled_mse={val_metrics['scaled_mse']:.6f} | "
            f"val_raw_mae={val_metrics['raw_mae']:.4f} | "
            f"grip_corr={val_metrics['gripper_corr']:.3f}"
        )
        if val_metrics["scaled_mse"] < best_val:
            best_val = val_metrics["scaled_mse"]
            model.save(str(output_dir / "sac_actor_bc_pretrained_best"))

    writer.close()
    model.save(str(output_dir / "sac_actor_bc_pretrained_last"))
    print(f"Best actor BC validation scaled MSE: {best_val:.6f}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--normalization-stats", required=True)
    parser.add_argument("--cube-color", default="green")
    parser.add_argument(
        "--use-pregrasp-regressor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Bring the sim arm to the regressor-predicted pregrasp pose at each reset (default on; pass --no-use-pregrasp-regressor to disable).",
    )
    parser.add_argument("--pregrasp-regressor-checkpoint", default=None)
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Parallel RCS envs (SubprocVecEnv). 1 = single-process. Speeds up "
        "rollout collection without changing env dynamics.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--ent-coef", default="auto")
    parser.add_argument("--bc-dataset-root")
    parser.add_argument("--bc-pretrain-epochs", type=int, default=0)
    parser.add_argument("--bc-batch-size", type=int, default=128)
    parser.add_argument("--bc-lr", type=float, default=3e-4)
    parser.add_argument("--bc-val-episode-fraction", type=float, default=0.2)
    parser.add_argument("--bc-zero-motor-currents", action="store_true")
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="Run a deterministic-policy evaluation every N steps (GraspEvalCallback).",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=8,
        help="Episodes per periodic evaluation.",
    )
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    return TrainConfig(
        output_dir=args.output_dir,
        normalization_stats=args.normalization_stats,
        cube_color=args.cube_color,
        use_pregrasp_regressor=args.use_pregrasp_regressor,
        pregrasp_regressor_checkpoint=args.pregrasp_regressor_checkpoint,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=args.ent_coef,
        bc_dataset_root=args.bc_dataset_root,
        bc_pretrain_epochs=args.bc_pretrain_epochs,
        bc_batch_size=args.bc_batch_size,
        bc_lr=args.bc_lr,
        bc_val_episode_fraction=args.bc_val_episode_fraction,
        bc_zero_motor_currents=args.bc_zero_motor_currents,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        resume_from=args.resume_from,
    )


def main() -> None:
    cfg = parse_args()
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    stats = load_normalization_stats(cfg.normalization_stats)

    env_cfg = LeRobotAlignGraspEnvConfig(
        cube_color=cfg.cube_color,
        use_pregrasp_regressor=cfg.use_pregrasp_regressor,
    )
    if cfg.pregrasp_regressor_checkpoint is not None:
        env_cfg.pregrasp_regressor_checkpoint = cfg.pregrasp_regressor_checkpoint

    n_envs = max(1, int(cfg.n_envs))
    if n_envs > 1:
        # Parallel RCS envs across CPU cores. 'spawn' avoids inheriting any
        # GL/EGL context from the parent (fork would corrupt MuJoCo rendering).
        env = SubprocVecEnv(
            [partial(_make_align_grasp_env, env_cfg) for _ in range(n_envs)],
            start_method="spawn",
        )
        env = VecMonitor(env, filename=str(output_dir / "monitor.csv"))
    else:
        env = Monitor(LeRobotAlignGraspEnv(env_cfg), filename=str(output_dir / "monitor.csv"))

    policy_kwargs = {
        "features_extractor_class": VisualHILFeaturesExtractor,
        "features_extractor_kwargs": {
            "state_mean": stats["state_mean"],
            "state_std": stats["state_std"],
            "image_feature_dim": 256,
        },
        "net_arch": {"pi": [512, 256], "qf": [512, 256]},
        "activation_fn": nn.ReLU,
        "normalize_images": True,
    }

    # Keep the learner schedule explicit.  Forcing gradient_steps=-1 with a
    # VecEnv preserves update-per-transition parity, but it turns each rollout
    # chunk into many SAC updates and can erase the wall-clock speedup from
    # async/vectorized simulation.
    gradient_steps = cfg.gradient_steps

    if cfg.resume_from is None:
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts,
            batch_size=cfg.batch_size,
            tau=cfg.tau,
            gamma=cfg.gamma,
            train_freq=cfg.train_freq,
            gradient_steps=gradient_steps,
            ent_coef=cfg.ent_coef,
            verbose=1,
            tensorboard_log=str(output_dir / "tensorboard_rl"),
            policy_kwargs=policy_kwargs,
            seed=cfg.seed,
            device=cfg.device,
        )
    else:
        print(f"Resuming SAC from checkpoint: {cfg.resume_from}")
        model = SAC.load(
            cfg.resume_from,
            env=env,
            device=cfg.device,
            tensorboard_log=str(output_dir / "tensorboard_rl"),
        )
        model.gradient_steps = gradient_steps

    print()
    print("=== Visual HIL-compatible SAC setup ===")
    print("Output dir:       ", output_dir)
    print("Device:           ", model.device)
    print("Timesteps:        ", cfg.total_timesteps)
    print("Parallel envs:    ", n_envs)
    print("Gradient steps:   ", gradient_steps)
    print("Replay buffer:    ", cfg.buffer_size)
    print("BC pretrain eps:  ", cfg.bc_pretrain_epochs)
    print("Resume from:      ", cfg.resume_from)

    if cfg.resume_from is None:
        bc_pretrain_sac_actor(model=model, cfg=cfg, output_dir=output_dir)
    else:
        print("Skipping BC pretraining because this is an RL resume run.")

    # CheckpointCallback counts _on_step calls, which advance by n_envs total
    # timesteps each; divide so checkpoints land every checkpoint_freq *total*
    # steps regardless of n_envs.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, cfg.checkpoint_freq // n_envs),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="visual_hil_sac",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Dedicated eval env (separate sim) so periodic evaluation never disturbs
    # the training rollout state.
    eval_env = LeRobotAlignGraspEnv(env_cfg)
    grasp_eval_callback = GraspEvalCallback(
        eval_env,
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        log_path=output_dir / "eval_log.jsonl",
    )
    callback = CallbackList(
        [LocalGraspInfoCallback(), grasp_eval_callback, checkpoint_callback]
    )

    print()
    print("=== SAC RL training ===")
    print(f"Eval every {cfg.eval_freq} steps over {cfg.n_eval_episodes} episodes "
          f"-> {output_dir / 'eval_log.jsonl'}")
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callback,
        tb_log_name="SAC_visual_hil_compat",
        reset_num_timesteps=(cfg.resume_from is None),
        progress_bar=True,
    )
    model.save(str(output_dir / "final_model"))
    env.close()
    eval_env.close()
    print()
    print("Training finished. Saved final model to:", output_dir / "final_model.zip")


if __name__ == "__main__":
    main()
