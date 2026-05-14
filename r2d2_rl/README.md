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
├── control/                   RCS waypoint controller adapter.
├── runtime/                   Hybrid task executor + RCS sim adapters
│                              + learned-policy adapter.
├── envs/                      RCS Project 3 SO101 env + colored cubes.
│
├── rl/                        SAC trainer for the align_grasp policy.
│   ├── align_grasp_env.py     Training env wrapping Project3SO101Env.
│   ├── sac.py                 SAC agent (actor + critic + auto-temperature).
│   ├── replay_buffer.py       FIFO replay buffer.
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
│   ├── train_align_grasp.py          SAC trainer for the align_grasp policy.
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

# Wrist + external screenshot:
MUJOCO_GL=egl python r2d2_rl/scripts/render_project3_screenshot.py --external-view

# Pixel-to-table validation (legacy lerobot-p3 env):
conda activate lerobot-p3
MUJOCO_GL=egl python r2d2_rl/scripts/validate_pixel_to_table.py --headless

# Train the align_grasp SAC policy:
MUJOCO_GL=egl python r2d2_rl/scripts/train_align_grasp.py \
    --total-steps 100000 \
    --checkpoint-dir r2d2_rl/outputs/align_grasp_sac

# Tests (works in both envs):
python -m unittest discover -s r2d2_rl/tests -p 'test_*.py'
```
