# `r2d2_rl/` — Project 3 workspace

All project-specific Python lives under `r2d2_rl/` so that the RCS source
tree (kept under `../external/robot-control-stack/`) doesn't shadow our
project modules.

## Layout

```
r2d2_rl/
├── README.md                  This file.
│
├── hybrid_control_rl/         YAML config loader (extends + deep_merge).
├── perception/                HSV color block detector.
├── estimation/                Pixel-to-table projection, block belief tracker.
├── planning/                  Hybrid waypoint planner (pregrasp/lift/transport/release).
├── control/                   RCS waypoint controller adapter
│                              + optional pregrasp joint regressor loader.
├── runtime/                   Hybrid task executor + RCS sim adapters
│                              + learned-policy adapters, including SB3/Flo.
├── envs/                      RCS Project 3 SO101 env + colored cubes.
│
├── rl/                        SAC trainers/envs for the align_grasp policy.
│   ├── align_grasp_env.py     Training env wrapping Project3SO101Env.
│   ├── sac.py                 SAC agent (actor + critic + auto-temperature).
│   ├── replay_buffer.py       FIFO replay buffer.
│   ├── lerobot_align_grasp_env.py  RCS sim with LeRobot wrist/state/action ABI.
│   ├── visual_sac.py          Legacy visual SAC for LeRobot-compatible observations.
│   ├── visual_replay_buffer.py Replay buffer for wrist image + state.
│   └── __init__.py
│
├── RL_envs/                   Generic RL building blocks.
│   ├── networks.py            MLP / GaussianActor / SquashedGaussianActor / DoubleQNet.
│   ├── so100_mdp_utils.py     reset_robot / process_action / compute_reward / get_obs.
│   ├── rotation_utils.py      Quaternion utilities (quat_mul, rot_mat_to_quat, ...).
│   ├── so100_rl_env.py        HW4 SO100 RL env (EE tracking; reference only).
│   ├── cartpole_wrapper.py    Toy env (not used in Project 3).
│   └── grid_world.py          Toy env (not used in Project 3).
│
├── calibration/               LeRobot SO-follower / SO-leader JSON calibration.
├── configs/hybrid_control_rl/ YAML configs (base + per-eval overrides).
├── scripts/                   Runtime entry points.
│   ├── run_hybrid_eval_sim.py        End-to-end Eval 1 in RCS sim.
│   ├── render_project3_screenshot.py Wrist + external view PNG.
│   ├── validate_pixel_to_table.py    Closed-loop projection accuracy.
│   ├── test_rcs_so101_sim.py         RCS env smoke test.
│   ├── test_wrist_camera_feed.py     Legacy MuJoCo wrist-cam demo.
│   ├── train_align_grasp.py          Legacy custom SAC trainer.
│   ├── train_lerobot_align_grasp.py  Legacy custom visual SAC trainer.
│   ├── train_real_hil_bc_policy.py   Canonical BC warm-start on real HIL demos.
│   ├── train_visual_hil_compat_sac.py Canonical BC-pretrain + SB3 SAC trainer.
│   ├── Cam_calibration.py            WIP camera intrinsic calibration.
│   ├── Cam_workflow.py               Color-detection prototype.
│   ├── pid_control.py                HW2-style quintic-PID demo.
│   └── RL_Preprocess.py              Placeholder stub.
│
├── tests/                     Unit tests.
├── docs/                      Design docs (audit, revised spec, RCS overlap).
└── outputs/                   Curated artifacts + RL checkpoints.
```

## Imports

The package boundary is `r2d2_rl/` itself. Scripts and tests add
`r2d2_rl/` to `sys.path` so legacy script imports stay flat:

```python
from estimation.pixel_to_table import PixelToTableProjector
from perception.color_block_detector import ColorBlockDetector
from rl.sac import SACAgent
from RL_envs.networks import SquashedGaussianActor, DoubleQNet
```

Package-style imports also work from the repository root, e.g.
`from r2d2_rl.rl.sac import SACAgent`.

- Scripts: `Path(__file__).resolve().parents[2]` is the repo root, and
  `r2d2_rl/` is inserted into `sys.path` at the top of each script.
- Tests: `r2d2_rl/tests/__init__.py` inserts the repo root and `r2d2_rl/`
  into `sys.path`, and exposes `BASE_CONFIG_PATH` for config-loading tests.
- Repo-root `../conftest.py` does the same for any pytest-based runs.

## Running things

From the repo root (`project3/`), with the `lerobot-p3-rcs` conda env active:

```bash
# Full pipeline end-to-end (Eval 1):
MUJOCO_GL=egl python r2d2_rl/scripts/run_hybrid_eval_sim.py --save-images

# Full pipeline with Flo's SB3 visual align_grasp checkpoint:
MUJOCO_GL=egl python r2d2_rl/scripts/run_hybrid_eval_sim.py \
    --sb3-align-grasp-checkpoint r2d2_rl/outputs/hil_bc_sac_v1/final_model.zip \
    --save-images

# Wrist + external screenshot:
MUJOCO_GL=egl python r2d2_rl/scripts/render_project3_screenshot.py --external-view

# Pixel-to-table validation (legacy lerobot-p3 env):
conda activate lerobot-p3
MUJOCO_GL=egl python r2d2_rl/scripts/validate_pixel_to_table.py --headless

# Canonical sim-to-real training path:
MUJOCO_GL=egl python r2d2_rl/scripts/train_real_hil_bc_policy.py \
    --dataset-root r2d2_rl/outputs/hil_dataset/p3_local_grasp_hil_multicolor_colorcond_v1 \
    --output-dir r2d2_rl/outputs/real_hil_bc_warmstart \
    --epochs 80 --zero-motor-currents

MUJOCO_GL=egl python r2d2_rl/scripts/train_visual_hil_compat_sac.py \
    --output-dir r2d2_rl/outputs/hil_bc_sac_v1 \
    --normalization-stats r2d2_rl/outputs/real_hil_bc_warmstart/normalization_stats.json \
    --bc-dataset-root r2d2_rl/outputs/hil_dataset/p3_local_grasp_hil_multicolor_colorcond_v1 \
    --bc-pretrain-epochs 80 --bc-zero-motor-currents \
    --total-timesteps 100000

# Tests (works in both envs):
python -m unittest discover -s r2d2_rl/tests -p 'test_*.py'
```

The visual LeRobot-compatible policy boundary is:

- `observation.images.wrist`: uint8 RGB, channel-first `[3, 128, 128]`
- `observation.state`: 24D float32 `[positions, velocities, currents, target_color_onehot]`
- action: 6D absolute SO-101 follower target in calibrated real-like units

The LeRobot-compatible env uses its own 2 cm cube MJCFs for
`blue/green/purple/orange/yellow/red` so this training surface can match the
HIL target-color encoding without patching RCS assets.

The hybrid sim uses the same patched SO101 `robotwrist` camera and 2 cm cube
assets. Its approach phase also uses Flo's pregrasp xy->joint regressor by
default, so the wrist frame handed to `align_grasp` has the same top-down
gripper/cube geometry as the BC+SAC training resets. Pass
`--no-use-pregrasp-regressor` only for legacy Cartesian waypoint debugging.

`runtime/sb3_visual_align_grasp_policy.py` loads Flo's SB3 `final_model.zip`
and exposes the hybrid executor's `run("align_grasp", color, belief) -> bool`
interface. This keeps HSV perception, pixel-to-table projection, Kalman
belief, classical approach/lift/transport/release, and swaps only the local
visual grasp phase.

The older `train_align_grasp.py`, `train_lerobot_align_grasp.py`, and
`runtime/lerobot_visual_align_grasp_policy.py` remain as reference paths for
the custom SAC stack, but they are not the canonical sim-to-real workflow.
