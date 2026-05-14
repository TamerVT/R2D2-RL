# Codebase Audit: Hybrid Control + Local RL

Audit date: 2026-05-14. Scope: `project3`, vendored ETH homework 2/3/4 code, local calibration files, and a shallow local clone of `RobotControlStack/robot-control-stack` at `external/robot-control-stack` for integration planning.

## Summary

The repository already has useful MuJoCo SO100/SO101 simulation plumbing, wrist-camera smoke tests, homework imitation-learning scenes with cubes/bowls, and homework RL skeletons. It does not yet have a unified hybrid runtime, color-block perception, calibrated pixel-to-table projection, belief tracking, or a local manipulation RL environment. Robot Control Stack (RCS) adds a cleaner Gymnasium wrapper model, SO101 MJCF/IK, relative action wrappers, and a generic pick task, but it should be treated as an optional external dependency until its Python/MuJoCo/LeRobot compatibility and AGPL license implications are resolved.

## Audit Table

| Area | Existing files/classes/functions | Status | Problems | Recommended change |
|---|---|---:|---|---|
| Camera access | `scripts/test_wrist_camera_feed.py`; `hw3.sim_env.BaseSO100SimEnv.render_rgb`; `record_teleop_demos.py`; RCS `SimCameraSet`/`CameraSetWrapper` | partial | No unified camera API for sim and real; RCS SO101 sim disables cameras by default. | Add a thin camera adapter that can wrap MuJoCo renderer, LeRobot/OpenCV camera, or RCS camera set. |
| Calibration | `scripts/Cam_calibration.py`; `scripts/Cam_workflow.py`; `SIM_WRIST_CAMERA_README.md`; motor JSON under `calibration/` | partial | `Cam_calibration.py` calls undefined `save_calibration_npz`; no YAML intrinsics/extrinsics; motor calibration only. | Create `configs/hybrid_control_rl/calibration.yaml`; repair/wrap camera calibration scripts after config design. |
| FK/IK | MuJoCo `site_xpos`/`site_xmat`; `hw2/exercises/ex1.py::ik_track`; RCS `Pin` and `SO101IK` | partial | HW2 IK is an unfinished exercise; RCS IK needs separate build/install and may not match local LeRobot API. | Reuse MuJoCo FK in sim; use RCS IK only behind optional adapter; implement minimal DLS IK if needed. |
| Low-level control | `hw3.sim_env.set_targets`, `set_gripper`, mocap setters; HW2 PID/quintic scripts; RCS `RobotWrapper`, `RelativeActionSpace`, gripper wrappers | partial | HW2 PID/quintic are TODO; no safety-checked waypoint controller. | Add `control/waypoint_controller.py` as adapter over HW3 sim first, then LeRobot/RCS. |
| Perception/color detection | `scripts/Cam_workflow.py` YOLO cup prototype | partial | Detects COCO cup, not Project 3 colored blocks; hard-coded camera pose. | Implement HSV/Lab `perception/color_block_detector.py` with config thresholds. |
| Coordinate transforms | `Cam_workflow` ray-plane helpers; HW4 `get_obs` base-frame conversion | partial | No explicit frame convention or reusable transforms. | Add calibrated `PixelToTableProjector` with named frames `B`, `E`, `C`, `I`. |
| State estimation | HW3 sim exposes cube states; no real estimator | missing | No belief/covariance tracking or per-color memory. | Add `estimation/block_belief.py` static Kalman tracker. |
| Trajectory/waypoint planning | HW2 quintic exercise; HW3 mocap/target setters | missing | No task-level waypoint planner, receding-horizon update, or belief gating. | Add `planning/hybrid_waypoint_planner.py`; keep first version conservative and position-only. |
| RL environment | `hw4/envs/so100_rl_env.py`; `hw3/hw3/sim_env.py`; RCS `PickTask` | partial | HW4 is EE tracking, not manipulation; HW3 scene is not Gymnasium; RCS pick task is generic and not Project 3 bowl/color sequence. | Build local manipulation env by wrapping HW3 scene first; evaluate RCS env as a second backend. |
| RL algorithm | HW4 PPO/SAC agents, replay/rollout buffers, train scripts | partial | PPO/SAC contain TODO placeholders; flat vector-only observations. | Complete or replace with stable baseline only after local env is defined; SAC is better fit for local continuous control. |
| Teleop/demo data | `hw3/scripts/record_teleop_demos.py`, `dagger_eval.py`, `compute_actions.py`; legacy real-data protocol | exists | Simulation teleop is zarr-based; real LeRobot pipeline is separate. | Reuse HW3 zarr demos for BC warm start; keep real LeRobot dataset path separate. |
| Eval scripts | `hw3/scripts/eval.py`; `student_eval/run_eval.py`; HW4 eval scripts | exists | No hybrid eval entry point; no Eval 3 relocalization loop. | Add `scripts/run_hybrid_eval_sim.py` after perception/projection/controller are testable. |
| Config system | `pyproject.toml`; argparse in scripts; homework config dicts | partial | No unified YAML config for calibration/perception/planning/eval goals. | Use simple YAML files under `configs/hybrid_control_rl/`; avoid introducing a heavy framework. |
| Logging/checkpointing | `outputs/`; HW4 TensorBoard/checkpoints; HW3 zarr datasets | partial | No structured rollout logs with detections/beliefs/phase traces. | Add `logs/hybrid_rollouts/<timestamp>/` writer once state machine exists. |

## Robot Control Stack Findings

- RCS provides `EmptyWorldSO101`, SO101 MJCF assets, `ControlMode` wrappers, `RelativeActionSpace`, and SO101 cartesian/joint examples.
- RCS has a generic `PickTask` with randomized object position and shaped reward, useful as a model for a local manipulation environment.
- RCS installation prefers Python 3.11 and pins `mujoco==3.2.6`; this conflicts with the current `lerobot-p3` Python 3.12/MuJoCo 3.8.0 setup. Downgrading MuJoCo would break `scripts/test_wrist_camera_feed.py` and the HW2/HW3/HW4 code using the 3.8 renderer API.
- RCS uses pybind11 + Pinocchio 3.7.0 + OMPL native extensions; building requires `cmake`, `ninja`, and system X11/GLFW headers.
- RCS `rcs_so101.hw` imports `lerobot.robots.so101_follower.so101_follower`; this checkout exposes SO101 through `lerobot.robots.so_follower`. The hardware module will not import without adapters or upstream changes.
- RCS is AGPL-3.0. Vendoring the source forces the entire derivative work to be AGPL-3.0.

### Decision (2026-05-14, revised same day)

RCS is the project's primary sim/control backend. The hw2/hw3/hw4 envs are now reference material only. AGPL-3.0 is accepted on the deliverable. The plan:

1. Build a separate conda env `lerobot-p3-rcs` on Python 3.11 + MuJoCo 3.2.6. Install RCS + `rcs_so101` from `external/robot-control-stack`.
2. Keep `lerobot-p3` (Python 3.12 + MuJoCo 3.8.0) for the existing wrist-camera demo and for env-agnostic numeric tests.
3. For sim: use `EmptyWorldSO101` + a Project 3 task wrapper built on the `PickTask` pattern. Add the wrist camera through `rcs.camera.sim` since RCS disables SO101 cameras by default.
4. For hardware: add a shim package `lerobot.robots.so101_follower` that re-exports from `lerobot.robots.so_follower` to make `rcs_so101.hw` importable.
5. Prefer RCS imports over reimplementing wherever they cover the same need (IK, gym wrappers, RelativeActionSpace, RPC, camera abstraction). Code we already wrote that is pure-numeric (color detector, pixel-to-table projector, block belief tracker, upcoming waypoint planner) stays — it is env-agnostic.
