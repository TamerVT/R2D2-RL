#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/home/frainer/Desktop/FS26/RobotLearning/R2D2-RL")

ENV_CONFIG_PATH = ROOT / "project3_modular/rl_grasp/configs/env_local_grasp_so101_hil_record.json"

MERGED_DATASET_ROOT = Path(
    "/home/frainer/.cache/huggingface/lerobot/frainer/"
    "p3_local_grasp_hil_multicolor_colorcond_v1"
)
STATS_PATH = MERGED_DATASET_ROOT / "meta/stats.json"

OUTPUT_CONFIG_PATH = (
    ROOT / "project3_modular/rl_grasp/configs/"
    "train_local_grasp_so101_hil_sac_colorcond_v1.json"
)

OUTPUT_DIR = (
    ROOT / "project3_modular/rl_grasp/outputs/"
    "hilserl_local_grasp_sac_colorcond_v1"
)


def flatten_nested_numbers(x: Any) -> list[float]:
    if isinstance(x, list):
        out: list[float] = []
        for item in x:
            out.extend(flatten_nested_numbers(item))
        return out
    return [float(x)]


def main() -> None:
    env_record_cfg = json.loads(ENV_CONFIG_PATH.read_text())
    stats = json.loads(STATS_PATH.read_text())

    env_cfg = env_record_cfg["env"]

    # Training configs require the environment ChoiceRegistry type.
    env_cfg["type"] = "gym_manipulator"

    # Our runtime color-conditioning patch appends 6 one-hot target-color values:
    # original 18D state -> 24D state.
    env_cfg["features"]["observation.state"]["shape"] = [24]

    # Leader-mode SAC predicts direct 6D SO101 joint-position actions.
    env_cfg["features"]["action"]["shape"] = [6]

    image_mean = flatten_nested_numbers(stats["observation.images.wrist"]["mean"])
    image_std = flatten_nested_numbers(stats["observation.images.wrist"]["std"])

    if len(image_mean) != 3 or len(image_std) != 3:
        raise ValueError(
            f"Expected RGB mean/std of length 3, got "
            f"mean={image_mean}, std={image_std}"
        )

    train_cfg = {
        "output_dir": str(OUTPUT_DIR),
        "job_name": "hilserl_local_grasp_sac_colorcond_v1",
        "resume": False,
        "seed": 1000,
        "num_workers": 4,
        "batch_size": 256,
        "steps": 100000,
        "eval_freq": 20000,
        "log_freq": 100,
        "save_checkpoint": True,
        "save_freq": 5000,
        "use_policy_training_preset": True,
        "optimizer": None,
        "scheduler": None,
        "wandb": {
            "enable": False,
            "project": "p3_local_grasp_hilserl",
            "disable_artifact": True,
        },
        "dataset": {
            "repo_id": "frainer/p3_local_grasp_hil_multicolor_colorcond_v1",
            "root": str(MERGED_DATASET_ROOT),
            "use_imagenet_stats": False,
        },
        "env": env_cfg,
        "policy": {
            "type": "sac",
            "n_obs_steps": 1,
            "normalization_mapping": {
                "VISUAL": "MEAN_STD",
                "STATE": "MIN_MAX",
                "ENV": "MIN_MAX",
                "ACTION": "MIN_MAX",
            },
            "input_features": {
                "observation.images.wrist": {
                    "type": "VISUAL",
                    "shape": [3, 128, 128],
                },
                "observation.state": {
                    "type": "STATE",
                    "shape": [24],
                },
            },
            "output_features": {
                "action": {
                    "type": "ACTION",
                    "shape": [6],
                },
            },
            "device": "cuda",
            "use_amp": False,
            "dataset_stats": {
                "observation.images.wrist": {
                    "mean": image_mean,
                    "std": image_std,
                },
                "observation.state": {
                    "min": stats["observation.state"]["min"],
                    "max": stats["observation.state"]["max"],
                },
                "action": {
                    "min": stats["action"]["min"],
                    "max": stats["action"]["max"],
                },
            },
            "repo_id": "frainer/p3_local_grasp_hil_sac_colorcond_v1",
            "storage_device": "cpu",
            "vision_encoder_name": "helper2424/resnet10",
            "freeze_vision_encoder": True,
            "image_encoder_hidden_dim": 32,
            "shared_encoder": True,

            # Important: in leader mode, the 6D action is fully continuous:
            # 5 joints + gripper absolute position. The stock gamepad/EE examples
            # use discrete gripper actions, but our leader-action dataset does not.
            "num_discrete_actions": None,

            "online_steps": 100000,
            "online_buffer_capacity": 100000,
            "offline_buffer_capacity": 100000,
            "online_step_before_learning": 100,
            "policy_update_freq": 1,
            "discount": 0.97,
            "temperature_init": 0.01,
            "num_critics": 2,
            "num_subsample_critics": None,
            "critic_lr": 0.0003,
            "actor_lr": 0.0003,
            "temperature_lr": 0.0003,
            "critic_target_update_weight": 0.005,
            "utd_ratio": 2,
            "state_encoder_hidden_dim": 256,
            "latent_dim": 256,
            "target_entropy": None,
            "use_backup_entropy": True,
            "grad_clip_norm": 40.0,
            "critic_network_kwargs": {
                "hidden_dims": [256, 256],
                "activate_final": True,
                "final_activation": None,
            },
            "actor_network_kwargs": {
                "hidden_dims": [256, 256],
                "activate_final": True,
            },
            "policy_kwargs": {
                "use_tanh_squash": True,
                "std_min": 1e-5,
                "std_max": 5,
                "init_final": 0.05,
            },
            "actor_learner_config": {
                "learner_host": "127.0.0.1",
                "learner_port": 50051,
                "policy_parameters_push_frequency": 4,
            },
            "concurrency": {
                "actor": "threads",
                "learner": "threads",
            },
        },
    }

    OUTPUT_CONFIG_PATH.write_text(json.dumps(train_cfg, indent=2))
    print(f"Wrote:\n  {OUTPUT_CONFIG_PATH}")
    print(f"Output directory:\n  {OUTPUT_DIR}")
    print("Key shapes:")
    print("  observation.state -> [24]")
    print("  action            -> [6]")
    print("Dataset:")
    print("  frainer/p3_local_grasp_hil_multicolor_colorcond_v1")


if __name__ == "__main__":
    main()
