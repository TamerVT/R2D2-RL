# Teammate contributions — what came from where

This repo started with three parallel implementations of Project 3 on
different branches. After consolidation, `main` runs the **RCS-based hybrid
pipeline** (Tamer + Claude), with **Felix's `OUR_stuff/` layout pattern** and
his **HW4 RL utility files** integrated as building blocks for the
forthcoming `align_grasp` policy. The third branch (Insalatone's
direct-MuJoCo controller) is intentionally not merged.

## What came from each contributor

### Felix Lupp (`origin/master`)

Felix's branch put the full RCS source at the repo root and added an
`OUR_stuff/` workspace folder with HW4 RL exercise files and a handful of
scripts. What we **kept and integrated** from Felix:

| File / pattern | Where it lives now | Why we kept it |
|---|---|---|
| The `OUR_stuff/` layout itself | `OUR_stuff/` | Cleanly separates project code from the RCS source we vendor under `external/robot-control-stack/`. Felix's idea. |
| `RL_envs/networks.py` | `OUR_stuff/RL_envs/networks.py` | `build_mlp`, `GaussianActor`, `SquashedGaussianActor`, `DoubleQNet` — exactly what we need for SAC/PPO when we train the `align_grasp` policy. We will import from this file when that lands. |
| `RL_envs/rotation_utils.py` | `OUR_stuff/RL_envs/rotation_utils.py` | Quaternion utilities (`quat_mul`, `quat_conjugate`, `rot_mat_to_quat`) wrapping `mujoco.mju_*`. Useful for grasp orientation math. |
| `RL_envs/so100_mdp_utils.py` | `OUR_stuff/RL_envs/so100_mdp_utils.py` | `process_action` (normalized→joint range), `reset_robot`, `compute_reward`, `get_obs`. Reference patterns when we build a Project-3-specific RL training env. |
| `RL_envs/so100_rl_env.py` | `OUR_stuff/RL_envs/so100_rl_env.py` | The HW4 SO100 RL env. EE tracking, not pick-and-place — kept for reference only. |
| `RL_envs/cartpole_wrapper.py`, `grid_world.py` | `OUR_stuff/RL_envs/` | Toy envs from HW4 ex1/ex2. Not used in Project 3; kept because Felix shipped them and they're harmless. |
| `scripts/pid_control.py` | `OUR_stuff/scripts/pid_control.py` | HW2-style quintic-spline + PID demo (relies on `exercises.ex1.ik_track` and `exercises.ex2.pid_control`). Not used at runtime, but a useful reference for joint-space tracking. |
| `scripts/RL_Preprocess.py` | `OUR_stuff/scripts/RL_Preprocess.py` | Empty placeholder Felix left for future RL preprocessing. Kept so it's clear what slot is reserved. |
| `calibration/robots/so_follower/*.json` | `OUR_stuff/calibration/...` | LeRobot SO-follower calibration JSONs Felix captured on his arm. Required if we ever connect to that specific hardware. |
| `calibration/teleoperators/so_leader/*.json` | `OUR_stuff/calibration/...` | Same for the SO-leader teleoperator. |

What we **did not** take from Felix:

| Item | Reason |
|---|---|
| Full RCS source at repo root | Duplicate. We already have RCS under `external/robot-control-stack/` and install it into `lerobot-p3-rcs`. Either layout works; we picked the external/ form because it doesn't shadow our project's package namespace. |
| `camera_calib.npz` | The recorded calibration converged to `fx=4487, fy=10835` with distortion in the tens — a failed calibration run. Sim uses RCS's fovy-derived intrinsics; real hardware needs a clean re-run. |
| `cube_detector` (1-line stub file) | No content. |
| Felix's copies of `Cam_calibration.py`, `Cam_workflow.py`, `test_wrist_camera_feed.py`, `SIM_WRIST_CAMERA_README.md` | We already had identical/equivalent versions. |

### Insalatone (`origin/main`, pre-restructure, not merged)

Insalatone pushed two files (`controller.py` + `so101_env.py`) directly at
the repo root. **This path is not used.** It is a parallel, RCS-free
implementation:

- Direct MuJoCo `mj_jacSite`-based IK (vs. our RCS `rcs.common.Pin` IK).
- Hard-codes the HW4 MJCF path `hw4_reinforcement_learning/assets/mujoco/so100_pos_ctrl.xml` (vs. our RCS-supplied SO101 MJCF).
- Re-implements color detection inside the env's `BlockPerception` class (vs. our shared `perception/color_block_detector.py`).
- Reimplements pinhole pixel→camera→world inside the env (vs. our shared `estimation/pixel_to_table.py`).

The team decided to commit to the RCS path, so Insalatone's two files are
not merged into `main`. They remain accessible on the old `origin/main`
history before this branch was rebuilt.

### Tamer + Claude (the RCS pipeline)

Everything in `OUR_stuff/` outside `RL_envs/`, `calibration/`, and the two
named scripts is the RCS-based hybrid pipeline:

```
perception/   estimation/   planning/   control/   runtime/   envs/
hybrid_control_rl/   configs/   tests/   docs/
scripts/run_hybrid_eval_sim.py
scripts/render_project3_screenshot.py
scripts/test_rcs_so101_sim.py
scripts/validate_pixel_to_table.py
```

This is the canonical pipeline (see `../README.md` for the overview).

## Where Felix's modules slot in to the RCS pipeline

Today, Felix's modules are **available but not yet imported** by the pipeline.
The connection points:

| Felix module | Will be used by | When |
|---|---|---|
| `RL_envs/networks.py` (`SquashedGaussianActor`, `DoubleQNet`) | The SAC trainer for the `align_grasp` policy. | When the RL training env is added (currently `ScriptedAlignGraspPolicy` is the placeholder). |
| `RL_envs/so100_mdp_utils.py` (`process_action`, `compute_reward` shape) | The Project-3-specific RL training env wrapping `Project3SO101Env`. | Same. |
| `RL_envs/rotation_utils.py` | `planning/hybrid_waypoint_planner.py` if we ever expose grasp-orientation deltas (not needed for the current axis-aligned grasp). | Optional. |
| `calibration/robots/so_follower/*.json` | `rcs_so101.hw` when we add the LeRobot `so101_follower` shim and connect to physical hardware. | After hardware setup. |

So Felix's networks + MDP utils are the natural building blocks for the
**next big slice** (training the `align_grasp` policy). The integration is
deliberate: rather than re-implementing SAC actor/critic networks, we
import his.

## Feature-by-feature: ours vs Felix's

Where the same need is solved differently in Felix's branch versus our
pipeline, here's the side-by-side:

| Capability | Felix's branch (`OUR_stuff/...` as shipped to `master`) | Our RCS pipeline (the active code) | Better / worse |
|---|---|---|---|
| Robot env | None — Felix kept HW4's `so100_rl_env.py` as reference; no Project 3 env. | `envs/project3_so101_env.py` built on `rcs.envs.configs.EmptyWorldSO101` with wrist cam, colored cubes. | **Ours is more capable** — it's a Project 3 scene; Felix's is a placeholder. |
| Action space | HW4 normalized joint actions in `[-1, 1]^6` via `so100_mdp_utils.process_action`. | RCS dict `{tquat, gripper}` with `RelativeActionSpace` capping delta size. | Ours integrates with RCS hardware path; Felix's is RL-trainer ready. Different concerns. |
| Perception | Not in Felix's branch (he's pre-perception). | `perception/color_block_detector.py` with HSV + contour + hue-wrap red + covariance. | Only one implementation. |
| Pixel-to-world | Not in Felix's branch. | `estimation/pixel_to_table.py` — calibrated pinhole ray-plane with FD covariance; accepts RCS extrinsics directly. | Only one implementation. |
| Belief tracking | Not in Felix's branch. | `estimation/block_belief.py` — per-color stationary Kalman with contact-aware Q. | Only one implementation. |
| IK | None in Felix's branch (HW4 file is left as homework stub). | RCS `rcs.common.Pin.forward/inverse` (Pinocchio-backed). | Ours uses a validated solver; Felix's slot is empty. |
| Waypoint controller | None. | `control/waypoint_controller.py` — proportional cartesian step over `env.step({tquat, gripper})`. | Only one. |
| Task executor / state machine | None. | `runtime/hybrid_task_executor.py` — observe → approach → align_grasp → lift → transport → release with visibility watchdog + recovery. | Only one. |
| RL networks | `RL_envs/networks.py` — `build_mlp`, `GaussianActor`, `SquashedGaussianActor`, `DoubleQNet` (PPO + SAC ready). | None yet — we have a `ScriptedAlignGraspPolicy` placeholder. | **Felix's is the substrate for our next slice.** We will import from his file. |
| Configuration | None (Felix uses plain Python). | `hybrid_control_rl/config.py` + `configs/hybrid_control_rl/*.yaml` — YAML with `extends` + `deep_merge`. | Ours is more structured; Felix's is more flexible for one-off scripts. |
| Sim camera | Reused HW2 wrist-camera demo (`scripts/test_wrist_camera_feed.py`). | RCS `rcs.camera.sim.SimCameraSet` wired into `envs/project3_so101_env.py`, returning RGB + intrinsics + extrinsics in the obs. | Ours gives intrinsics/extrinsics automatically; Felix's is render-only. |
| Tests | Felix has no tests. | 47 unit tests across perception, estimation, planning, controller, executor, adapters. | **Ours is the only test surface.** |
| Documentation | `SIM_WRIST_CAMERA_README.md` (same as ours; both branches inherit from earlier work). | `README.md`, `OUR_stuff/README.md`, `OUR_stuff/COMPARISON.md`, design docs under `OUR_stuff/docs/`. | Ours is more extensive. |
| Calibration data | LeRobot SO-follower / SO-leader JSON files for Felix's hardware. `camera_calib.npz` (broken). | None for real hardware yet — sim derives intrinsics from RCS camera fovy. | **Felix's JSONs are the real-hardware enablers when we connect.** |

## Net assessment

- The **canonical runtime** is the RCS pipeline.
- **Felix's layout pattern** (`OUR_stuff/`) is now adopted as the package boundary.
- **Felix's `RL_envs/networks.py` + `so100_mdp_utils.py`** are reserved as the foundation for the SAC training env we still need to build.
- **Felix's calibration JSONs** are reserved for when we wire `rcs_so101.hw` to real hardware.
- Insalatone's direct-MuJoCo path is not integrated; it stands as the alternative we explicitly rejected after the team committed to RCS.
