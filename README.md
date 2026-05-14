# R2D2-RL — Project 3: Visual Pick-and-Place with SO-101

ETH Robot Learning Project 3. Hybrid **classical control + local RL** pipeline
for visual pick-and-place on the SO-101 arm, built on top of
[RobotControlStack (RCS)](https://github.com/RobotControlStack/robot-control-stack)
for sim and hardware abstraction.

The three evaluations:

- **Eval 1** — single block → bowl (BC or RL allowed)
- **Eval 2** — two-color cluster, pick the target color → bowl (RL required)
- **Eval 3** — four colors, three sequential pick-and-place goals (RL required)

The policy operates on RGB observations from the SO-101 wrist camera; bowl
targets are given as robot-frame `(x, y, z)` coordinates.

## Pipeline at a glance

```
  Wrist RGB ─► ColorBlockDetector ─► PixelToTableProjector ─► BlockBeliefTracker
                                                                      │
                                                                      ▼
                            ┌────────────────────────────────────────┴────┐
                            │                                              │
                            ▼                                              ▼
                  HybridWaypointPlanner ◄──── HybridTaskExecutor ───► LocalPolicy
                            │              (state machine)            (align_grasp)
                            ▼
                  RcsWaypointController ─► Project3SO101Env ─► RCS / MuJoCo
```

Pipeline phases (executor state machine):

```
  observe ─► approach ─► align_grasp ─► lift ─► transport ─► release
                              │                    │
                              │                    │ if target lost N frames
                              │                    ▼
                              │              recovery (retreat + re-observe)
                              ▼
                       (RL: contact-rich)   classical waypoints
```

Design decisions baked in:

- **Only `align_grasp` is RL.** Lift, transport, and release are classical
  waypoints — contact dynamics only matter at the grasp itself.
- **RCS is the sim/control substrate.** Our code reuses `rcs.common.Pin` (IK),
  `rcs.envs.configs.EmptyWorldSO101` (sim scene), `rcs.camera.sim.SimCameraSet`
  (wrist camera), and `rcs.envs.base.RelativeActionSpace` (action wrapper).
- **Lost-cube recovery is a vision watchdog**, not a learned "hold" policy.
  If the target color disappears for `recovery.max_lost_frames` consecutive
  frames during transport, the executor retreats to a safe pose and
  re-localizes, up to `recovery.max_attempts` times.

## Repository layout

We follow the `OUR_stuff/` workspace pattern Felix introduced on the
`origin/master` branch: all team-specific Python lives under `OUR_stuff/`
so RCS source (kept in `external/robot-control-stack/`) doesn't shadow
project module names. See [`OUR_stuff/COMPARISON.md`](OUR_stuff/COMPARISON.md)
for a per-feature breakdown of which teammate contributed what.

```
project3/
├── README.md                          You are here.
├── SIM_WRIST_CAMERA_README.md         Detailed wrist-camera setup notes.
├── conftest.py                        Sys-path shim so pytest finds OUR_stuff/.
│
├── OUR_stuff/                         All Project 3 code (Felix's layout).
│   ├── README.md                      Workspace overview + run commands.
│   ├── COMPARISON.md                  Per-feature ours vs Felix vs Insalatone.
│   ├── hybrid_control_rl/             YAML config loader (extends + deep_merge).
│   ├── perception/                    HSV color block detector.
│   ├── estimation/                    Pixel-to-table + per-color Kalman.
│   ├── planning/                      Belief-gated waypoint planner.
│   ├── control/                       RCS waypoint controller adapter.
│   ├── runtime/                       Hybrid task executor + RCS adapters.
│   ├── envs/                          RCS SO-101 scene + colored cubes.
│   ├── RL_envs/                       HW4 RL utilities (from Felix; reserved
│   │                                    for the upcoming align_grasp trainer).
│   ├── calibration/                   LeRobot SO-follower/leader JSONs (from Felix).
│   ├── configs/hybrid_control_rl/     YAML configs (base + per-eval overrides).
│   ├── scripts/                       Entry points (run_hybrid_eval_sim.py, …).
│   ├── tests/                         47 unit tests.
│   ├── docs/                          Design docs (audit, revised spec, RCS overlap).
│   └── outputs/                       Curated artifacts.
│
├── external/robot-control-stack/      RCS clone (gitignored, installed via pip).
├── ethz-course-2026/                  ETH coursework (gitignored).
└── legacy/                            Parked earlier work (gitignored).
```

## Environments

Two conda envs are maintained because RCS pins different MuJoCo:

| Env | Python | MuJoCo | Purpose |
|---|---|---|---|
| `lerobot-p3` | 3.12 | 3.8.0 | Legacy wrist-cam demo, pure-numeric tests |
| `lerobot-p3-rcs` | 3.11 | 3.2.6 | **Active runtime** — RCS, sim env, full pipeline |

Activate the RCS env before doing anything sim-related:

```bash
source /home/explo22/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-p3-rcs
```

## Quick start

End-to-end sim run (single Eval 1 goal):

```bash
MUJOCO_GL=egl python OUR_stuff/scripts/run_hybrid_eval_sim.py --save-images
# -> OUR_stuff/outputs/hybrid_eval_sim/{initial,final}_{wrist,external}.png
```

Render a Project 3 sim screenshot (cube + wrist camera + external view):

```bash
MUJOCO_GL=egl python OUR_stuff/scripts/render_project3_screenshot.py --external-view
# -> OUR_stuff/outputs/project3_screenshot/{wrist_cam,external_view}.png
```

Validate pixel-to-table projection against a known mocap target (legacy env):

```bash
conda activate lerobot-p3
MUJOCO_GL=egl python OUR_stuff/scripts/validate_pixel_to_table.py --headless
# -> median error ~0.23 cm, max ~1.7 cm (out of 25 trial offsets)
```

Run the test suite (works in both envs):

```bash
python -m unittest discover -s OUR_stuff/tests -t OUR_stuff -p 'test_*.py'
# 47 tests, ~150 ms
```

## What is implemented

| Component | Status | Where |
|---|---|---|
| HSV color block detector (R/G/B/Y, hue-wrap red, covariance estimate) | done | `OUR_stuff/perception/color_block_detector.py` |
| Pixel-to-table ray-plane projection (intrinsics, distortion, T_E_C / T_B_C entry points, FD covariance) | done | `OUR_stuff/estimation/pixel_to_table.py` |
| Per-color static Kalman belief tracker (predict, update, contact-aware Q) | done | `OUR_stuff/estimation/block_belief.py` |
| Hybrid waypoint planner (pregrasp, lift, transport, release, recovery) | done | `OUR_stuff/planning/hybrid_waypoint_planner.py` |
| RCS waypoint controller (proportional-step, gripper-preserving) | done | `OUR_stuff/control/waypoint_controller.py` |
| Hybrid task executor / state machine (Eval 1 single, Eval 3 sequence) | done | `OUR_stuff/runtime/hybrid_task_executor.py` |
| Project 3 RCS env (SO-101 + colored cubes + wrist camera) | done | `OUR_stuff/envs/project3_so101_env.py` |
| YAML config system (base + extends + per-eval overrides) | done | `OUR_stuff/hybrid_control_rl/config.py`, `OUR_stuff/configs/hybrid_control_rl/` |
| End-to-end sim runner | done | `OUR_stuff/scripts/run_hybrid_eval_sim.py` |
| Screenshot tooling | done | `OUR_stuff/scripts/render_project3_screenshot.py` |
| Pixel-to-table closed-loop validator | done | `OUR_stuff/scripts/validate_pixel_to_table.py` |
| Unit tests (47, all green) | done | `OUR_stuff/tests/` |
| RL building blocks (`build_mlp`, `GaussianActor`, `SquashedGaussianActor`, `DoubleQNet`) | reserved | `OUR_stuff/RL_envs/networks.py` (from Felix) — will back the SAC `align_grasp` trainer |
| LeRobot SO-follower / SO-leader calibration JSON | reserved | `OUR_stuff/calibration/` (from Felix) — for real-hardware connect |

## What is NOT yet implemented

| Component | Notes |
|---|---|
| **Trained `align_grasp` RL policy** | Currently `ScriptedAlignGraspPolicy` — moves to estimated XY and closes the gripper. Required for Eval 2 / Eval 3 per the project rubric. Next major slice. |
| RL training env + trainer (SAC/PPO on top of `Project3SO101Env`) | Needed to train the above. |
| Multi-cube clutter scene for Eval 2 | The env supports a list of cubes; no Eval 2 specific runner script yet. |
| Eval 3 multi-goal sequence runner script | `executor.run_sequence()` exists; no top-level driver script. |
| Camera intrinsic + extrinsic calibration scripts | `Cam_calibration.py` is a WIP. Needed for real hardware deployment only — sim derives intrinsics from RCS camera fovy. |
| LeRobot `so101_follower` path shim for `rcs_so101.hw` | Needed for real hardware (sim path is unaffected). |
| Structured rollout logging (`logs/hybrid_rollouts/<timestamp>/`) | Spec section 15. Currently the eval prints to stdout only. |
| Debug overlays (mask + centroid + projected coord per frame) | Useful for debugging real-camera failure modes. |

## Documentation

- **[`OUR_stuff/README.md`](OUR_stuff/README.md)** — workspace layout and run commands.
- **[`OUR_stuff/COMPARISON.md`](OUR_stuff/COMPARISON.md)** — per-feature breakdown: ours vs Felix vs Insalatone.
- **[`SIM_WRIST_CAMERA_README.md`](SIM_WRIST_CAMERA_README.md)** — wrist-camera setup guide.
- **[`OUR_stuff/docs/HYBRID_CONTROL_RL_TRAJECTORY_SPEC_REVISED.md`](OUR_stuff/docs/HYBRID_CONTROL_RL_TRAJECTORY_SPEC_REVISED.md)** — implementation plan adapted to the actual codebase.
- **[`OUR_stuff/docs/CODEBASE_AUDIT_hybrid_control_rl.md`](OUR_stuff/docs/CODEBASE_AUDIT_hybrid_control_rl.md)** — audit of what existed before this project started.
- **[`OUR_stuff/docs/RCS_OVERLAP_AUDIT.md`](OUR_stuff/docs/RCS_OVERLAP_AUDIT.md)** — which RCS modules replace which of our utilities and which stay ours.
