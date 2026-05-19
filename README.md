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

All project-specific Python lives under `r2d2_rl/` so that RCS source
(kept in `external/robot-control-stack/`) doesn't shadow project module
names.

```
project3/
├── README.md                          You are here.
├── SIM_WRIST_CAMERA_README.md         Detailed wrist-camera setup notes.
├── conftest.py                        Sys-path shim so pytest finds r2d2_rl/.
│
├── r2d2_rl/                           All Project 3 code.
│   ├── README.md                      Workspace overview + run commands.
│   ├── hybrid_control_rl/             YAML config loader (extends + deep_merge).
│   ├── perception/                    HSV color block detector.
│   ├── estimation/                    Pixel-to-table + per-color Kalman.
│   ├── planning/                      Belief-gated waypoint planner.
│   ├── control/                       RCS waypoint controller adapter.
│   ├── runtime/                       Hybrid task executor + RCS adapters
│   │                                  + learned-policy adapter.
│   ├── envs/                          RCS SO-101 scene + colored cubes.
│   ├── rl/                            SAC trainer, replay buffer,
│   │                                  align_grasp training env.
│   ├── RL_envs/                       Generic RL building blocks
│   │                                  (MLP / GaussianActor / DoubleQNet).
│   ├── calibration/                   LeRobot SO-follower / SO-leader JSONs.
│   ├── configs/hybrid_control_rl/     YAML configs (base + per-eval overrides).
│   ├── scripts/                       Entry points (run_hybrid_eval_sim.py,
│   │                                  train_align_grasp.py, …).
│   ├── tests/                         Unit tests.
│   ├── docs/                          Design docs (audit, revised spec, RCS overlap).
│   └── outputs/                       Curated artifacts + RL checkpoints.
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
MUJOCO_GL=egl python r2d2_rl/scripts/run_hybrid_eval_sim.py --save-images
# -> r2d2_rl/outputs/hybrid_eval_sim/{initial,final}_{wrist,external}.png
```

Render a Project 3 sim screenshot (cube + wrist camera + external view):

```bash
MUJOCO_GL=egl python r2d2_rl/scripts/render_project3_screenshot.py --external-view
# -> r2d2_rl/outputs/project3_screenshot/{wrist_cam,external_view}.png
```

Validate pixel-to-table projection against a known mocap target (legacy env):

```bash
conda activate lerobot-p3
MUJOCO_GL=egl python r2d2_rl/scripts/validate_pixel_to_table.py --headless
# -> median error ~0.23 cm, max ~1.7 cm (out of 25 trial offsets)
```

Run the test suite (works in both envs):

```bash
python -m unittest discover -s r2d2_rl/tests -t r2d2_rl -p 'test_*.py'
# 47 tests, ~150 ms
```

## What is implemented

| Component | Status | Where |
|---|---|---|
| HSV color block detector (R/G/B/Y, hue-wrap red, covariance estimate) | done | `r2d2_rl/perception/color_block_detector.py` |
| Pixel-to-table ray-plane projection (intrinsics, distortion, T_E_C / T_B_C entry points, FD covariance) | done | `r2d2_rl/estimation/pixel_to_table.py` |
| Per-color static Kalman belief tracker (predict, update, contact-aware Q) | done | `r2d2_rl/estimation/block_belief.py` |
| Hybrid waypoint planner (pregrasp, lift, transport, release, recovery) | done | `r2d2_rl/planning/hybrid_waypoint_planner.py` |
| RCS waypoint controller (proportional-step, gripper-preserving) | done | `r2d2_rl/control/waypoint_controller.py` |
| Hybrid task executor / state machine (Eval 1 single, Eval 3 sequence) | done | `r2d2_rl/runtime/hybrid_task_executor.py` |
| Project 3 RCS env (SO-101 + colored cubes + wrist camera) | done | `r2d2_rl/envs/project3_so101_env.py` |
| YAML config system (base + extends + per-eval overrides) | done | `r2d2_rl/hybrid_control_rl/config.py`, `r2d2_rl/configs/hybrid_control_rl/` |
| End-to-end sim runner | done | `r2d2_rl/scripts/run_hybrid_eval_sim.py` |
| Screenshot tooling | done | `r2d2_rl/scripts/render_project3_screenshot.py` |
| Pixel-to-table closed-loop validator | done | `r2d2_rl/scripts/validate_pixel_to_table.py` |
| Unit tests (47, all green) | done | `r2d2_rl/tests/` |
| RL building blocks (`build_mlp`, `GaussianActor`, `SquashedGaussianActor`, `DoubleQNet`) | done | `r2d2_rl/RL_envs/networks.py` |
| SAC agent (actor + critic + auto-temperature + replay buffer) | done | `r2d2_rl/rl/sac.py`, `r2d2_rl/rl/replay_buffer.py` |
| `align_grasp` RL training env (resets near pregrasp, shaped reward, flat obs) | done | `r2d2_rl/rl/align_grasp_env.py` |
| SAC training script | done | `r2d2_rl/scripts/train_align_grasp.py` |
| Learned policy adapter (loads checkpoint, plugs into hybrid executor) | done | `r2d2_rl/runtime/learned_align_grasp_policy.py` |
| LeRobot SO-follower / SO-leader calibration JSON | reserved | `r2d2_rl/calibration/` — for real-hardware connect |

## What is NOT yet implemented

| Component | Notes |
|---|---|
| **Trained `align_grasp` checkpoint** | The training plumbing is in place; an actual SAC run still has to be executed (`python r2d2_rl/scripts/train_align_grasp.py --total-steps 100000`). Until then the executor uses the `ScriptedAlignGraspPolicy` placeholder. |
| Multi-cube clutter scene for Eval 2 | The env supports a list of cubes; no Eval 2 specific runner script yet. |
| Eval 3 multi-goal sequence runner script | `executor.run_sequence()` exists; no top-level driver script. |
| Camera intrinsic + extrinsic calibration scripts | `Cam_calibration.py` is a WIP. Needed for real hardware deployment only — sim derives intrinsics from RCS camera fovy. |
| LeRobot `so101_follower` path shim for `rcs_so101.hw` | Needed for real hardware (sim path is unaffected). |
| Structured rollout logging (`logs/hybrid_rollouts/<timestamp>/`) | Currently the eval prints to stdout only. |
| Debug overlays (mask + centroid + projected coord per frame) | Useful for debugging real-camera failure modes. |

## Documentation

- **[`r2d2_rl/README.md`](r2d2_rl/README.md)** — workspace layout and run commands.
- **[`SIM_WRIST_CAMERA_README.md`](SIM_WRIST_CAMERA_README.md)** — wrist-camera setup guide.
- **[`r2d2_rl/docs/HYBRID_CONTROL_RL_TRAJECTORY_SPEC_REVISED.md`](r2d2_rl/docs/HYBRID_CONTROL_RL_TRAJECTORY_SPEC_REVISED.md)** — implementation plan adapted to the actual codebase.
- **[`r2d2_rl/docs/CODEBASE_AUDIT_hybrid_control_rl.md`](r2d2_rl/docs/CODEBASE_AUDIT_hybrid_control_rl.md)** — audit of what existed before this project started.
- **[`r2d2_rl/docs/RCS_OVERLAP_AUDIT.md`](r2d2_rl/docs/RCS_OVERLAP_AUDIT.md)** — which RCS modules replace which of our utilities and which stay ours.
