# Revised Hybrid Control + Local RL Trajectory Plan

This plan adapts `hybrid_control_rl_trajectory_spec.md` to the current repository. As of the 2026-05-14 RCS decision, RCS (`external/robot-control-stack`) is the primary sim/control backend; the hw2/hw3/hw4 envs are reference material only.

## 0. Simplified Phase Structure (2026-05-14)

The original spec proposed three RL phases: `align_grasp`, `lift_recover`, `release`. We are collapsing this:

- **RL phase (the only one)**: `align_grasp` — final XY/Z/yaw correction + gripper closure, where contact dynamics matter.
- **Classical**: approach to pregrasp, lift, transport to bowl, release (open gripper + retreat upward).
- **Recovery loop**: during transport the wrist camera periodically checks that the target color is still visible. If it disappears for `recovery.max_lost_frames` consecutive checks (likely cube slipped out of the gripper, or was never grasped), the executor retreats to `recovery.return_xyz_base`, re-localizes, and restarts up to `recovery.max_attempts` times.

Holding/recovery does not need a learned policy: gripper-closed transport with a vision watchdog and classical return-to-safe-pose is sufficient and simpler to validate.

## 1. Existing Modules to Reuse

- Wrist-camera smoke test: `scripts/test_wrist_camera_feed.py`.
- Project/task documentation: `README.md`, `SIM_WRIST_CAMERA_README.md`, assignment PDF.
- HW3 MuJoCo pick-place scenes and state access as reference material only: `ethz-course-2026/hw3_imitation_learning/hw3/sim_env.py`.
- HW3 teleop/zarr/demo tooling as reference material: `record_teleop_demos.py`, `compute_actions.py`, `dagger_eval.py`.
- HW3 eval scripts and autograder-compatible policy names as compatibility references: `scripts/eval.py`, `student_eval/run_eval.py`, `ObstaclePolicy`, `MultiTaskPolicy`.
- HW4 RL scaffolding as algorithm reference only: replay buffers, network modules, PPO/SAC train/eval layout.
- RCS runtime backend: `EmptyWorldSO101`, `RelativeActionSpace`, `PickTask`, SO101 MJCF/IK, `CameraSetWrapper`, and `SimCameraSet`.

## 2. Existing Modules to Modify

- Repair or supersede `scripts/Cam_calibration.py`; it has useful UI logic but an undefined save function and no YAML output.
- Keep `scripts/Cam_workflow.py` as a prototype only; replace hard-coded YOLO cup detection with configured color-block detection.
- Extend HW3 sim wrappers through adapters instead of editing autograder-sensitive classes directly.
- Complete HW4 SAC/PPO only if we choose to train local RL in-house; otherwise wrap a known working RL library later.

## 3. New Modules Required

Minimal first-pass layout:

```text
configs/hybrid_control_rl/
perception/color_block_detector.py
estimation/pixel_to_table.py
estimation/block_belief.py
planning/hybrid_waypoint_planner.py
control/waypoint_controller.py
rl/local_manipulation_policy.py
envs/project3_so101_env.py
runtime/hybrid_task_executor.py
scripts/validate_pixel_to_table.py
scripts/run_hybrid_eval_sim.py
```

RCS-specific integration should live behind an adapter, for example `envs/rcs_so101_backend.py`, not mixed into HW3 code.

## 4. Backward-Compatible Interfaces

- Do not rename `ObstaclePolicy` or `MultiTaskPolicy`; HW3 eval imports them.
- Do not change `student_eval/run_eval.py` behavior.
- Preserve HW3 zarr dataset keys and action/state key conventions.
- Preserve existing wrist camera name `wrist_cam` in the HW2/HW3 MJCF path.
- Keep LeRobot SO101 calibration IDs and JSON files separate from hybrid YAML camera calibration.

## 5. Minimal Stubs First

- `ColorBlockDetector`: HSV thresholds, contour centroid, confidence, covariance.
- `PixelToTableProjector`: calibrated pinhole ray to fixed table height using MuJoCo FK in sim.
- `BlockBeliefTracker`: stationary 2D Kalman update per color.
- `HybridWaypointPlanner`: pregrasp, lift, transport, release waypoints with fixed orientation.
- `WaypointController`: thin RCS `env.step({"tquat": ..., "gripper": ...})` adapter with safety checks.
- `LocalManipulationPolicy`: mock `align_grasp` runner first, then SAC/BC-backed policy.

## 6. Scripts and Configs Needed

Create these before full training:

```text
configs/hybrid_control_rl/base.yaml
configs/hybrid_control_rl/calibration.yaml
configs/hybrid_control_rl/eval1.yaml
configs/hybrid_control_rl/eval2.yaml
configs/hybrid_control_rl/eval3.yaml
scripts/validate_pixel_to_table.py
scripts/run_hybrid_eval_sim.py
scripts/train_local_rl.py
```

RCS should be evaluated in a separate environment first because its README recommends Python 3.11 and pins MuJoCo 3.2.6. Do not install it into `lerobot-p3` until dependency conflicts are resolved.

## 7. Implementation Order

1. Add YAML config loader and calibration/perception/planning config files.
2. Implement color detector and synthetic detector tests.
3. Implement pixel-to-table projection using pose/FK inputs, then validation script.
4. Implement belief tracker and tests.
5. Implement waypoint planner and RCS waypoint controller adapter.
6. Implement hybrid task executor with mocked local RL policy.
7. Build an RCS Project 3 Gymnasium env for final alignment/grasp.
8. Complete or replace SAC implementation; train a local policy in sim.
9. Add structured rollout logging and debug overlays.
10. Add Eval 2 and Eval 3 goal-conditioned loops with mandatory relocalization.
11. Port Project 3 cubes/bowls and wrist camera into an RCS task wrapper.

## RCS Integration Strategy

Use RCS as the default project runtime. The HW3 scene still contains useful Project 3-style cube/bowl examples, but it is reference material only. The practical plan is:

1. Keep the shallow clone under ignored `external/robot-control-stack` for inspection.
2. Use the `lerobot-p3-rcs` conda env for runtime work.
3. Prototype a `Project3SO101RcsEnv` by composing `EmptyWorldSO101` plus a custom task based on RCS `PickTask`.
4. Use RCS cartesian `tquat`, binary gripper commands, `CameraSetWrapper`, and `RelativeActionSpace`.
5. Keep pure-numeric modules importable in both `lerobot-p3` and `lerobot-p3-rcs`.

## Assumptions

- Base frame `B` is the MuJoCo robot base frame unless a real-robot calibration file overrides it.
- Wrist camera frame `C` is a fixed transform from end-effector frame `E`; current MJCF target-body camera is a sim convenience and should not be treated as real extrinsics.
- Object localization starts as 2D table XY plus known object-center height.
- Eval 2/3 learned behavior must include final alignment/grasp. Lift, transport, release, and lost-cube recovery are classical.
