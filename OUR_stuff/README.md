# `OUR_stuff/` — Project 3 workspace

This folder follows the layout from Felix's `master` branch: all team-specific
code lives under `OUR_stuff/` so the RCS source tree can sit alongside it
(here in `../external/robot-control-stack/`, in Felix's checkout cloned at
the repo root) without filename collisions.

## Layout

```
OUR_stuff/
├── README.md                  This file.
├── COMPARISON.md              Side-by-side: Felix's contributions vs ours.
│
├── hybrid_control_rl/         YAML config loader (extends + deep_merge).
├── perception/                HSV color block detector.
├── estimation/                Pixel-to-table projection, block belief tracker.
├── planning/                  Hybrid waypoint planner (pregrasp/lift/transport/release).
├── control/                   RCS waypoint controller adapter.
├── runtime/                   Hybrid task executor + RCS sim adapters.
├── envs/                      RCS Project 3 SO101 env + colored cubes.
│
├── RL_envs/                   ETH HW4 RL building blocks (from Felix).
│   ├── networks.py            MLP / GaussianActor / SquashedGaussianActor / DoubleQNet.
│   ├── so100_rl_env.py        HW4 SO100 RL env (EE tracking; reference only).
│   ├── so100_mdp_utils.py     reset_robot / process_action / compute_reward / get_obs.
│   ├── rotation_utils.py      Quaternion utilities (quat_mul, rot_mat_to_quat, ...).
│   ├── cartpole_wrapper.py    Toy env (HW4 ex2/DQN — not used in Project 3).
│   └── grid_world.py          Toy env (HW4 ex1 — not used in Project 3).
│
├── calibration/               LeRobot SO-follower / SO-leader JSON calibration (from Felix).
├── configs/hybrid_control_rl/ YAML configs (base + per-eval overrides).
├── scripts/                   Runtime entry points.
│   ├── run_hybrid_eval_sim.py        End-to-end Eval 1 in RCS sim.
│   ├── render_project3_screenshot.py Wrist + external view PNG.
│   ├── validate_pixel_to_table.py    Closed-loop projection accuracy.
│   ├── test_rcs_so101_sim.py         RCS env smoke test.
│   ├── test_wrist_camera_feed.py     Legacy MuJoCo wrist-cam demo.
│   ├── Cam_calibration.py            WIP camera intrinsic calibration.
│   ├── Cam_workflow.py               Color-detection prototype.
│   ├── pid_control.py                HW2-style quintic-PID demo (from Felix).
│   └── RL_Preprocess.py              Placeholder stub (from Felix).
│
├── tests/                     47 unit tests, all green.
├── docs/                      Design docs (audit, revised spec, RCS overlap).
└── outputs/                   Curated artifacts (wrist demo, screenshots, validation PNGs).
```

## Imports

The package boundary is `OUR_stuff/` itself. Scripts and tests add
`OUR_stuff/` to `sys.path` so imports stay flat:

```python
from estimation.pixel_to_table import PixelToTableProjector
from perception.color_block_detector import ColorBlockDetector
from RL_envs.networks import SquashedGaussianActor   # Felix's HW4 networks
```

- Scripts: `Path(__file__).resolve().parents[2]` is the repo root, and
  `OUR_stuff/` is inserted into `sys.path` at the top of each script.
- Tests: `OUR_stuff/tests/__init__.py` inserts `OUR_stuff/` into `sys.path`
  on package import, and exposes `BASE_CONFIG_PATH` for config-loading tests.
- Repo-root `../conftest.py` does the same for any pytest-based runs.

## Running things

From the repo root (`project3/`), with the `lerobot-p3-rcs` conda env active:

```bash
# Full pipeline end-to-end (Eval 1):
MUJOCO_GL=egl python OUR_stuff/scripts/run_hybrid_eval_sim.py --save-images

# Wrist + external screenshot:
MUJOCO_GL=egl python OUR_stuff/scripts/render_project3_screenshot.py --external-view

# Pixel-to-table validation (legacy lerobot-p3 env):
conda activate lerobot-p3
MUJOCO_GL=egl python OUR_stuff/scripts/validate_pixel_to_table.py --headless

# Tests (works in both envs):
python -m unittest discover -s OUR_stuff/tests -t OUR_stuff -p 'test_*.py'
```
