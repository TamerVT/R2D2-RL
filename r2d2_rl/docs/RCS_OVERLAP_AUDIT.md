# RCS-vs-Our-Code Overlap Audit

Audit date: 2026-05-14. Performed after RCS + `rcs_so101` were installed in the `lerobot-p3-rcs` conda env (Python 3.11, MuJoCo 3.2.6).

The goal: identify which of our modules are subsumed by RCS (delete or replace), and which are complementary (keep, but possibly refactor to use RCS primitives).

## Summary

RCS provides the **sim/control substrate**: SO101 sim env, IK/FK, Gymnasium wrappers, action spaces, camera abstraction, RPC, MuJoCo wrappers, pose math, an object/task registry, and a generic `PickTask` template. It does **not** provide domain-specific perception, multi-frame state estimation, or a high-level multi-phase task executor — those stay in our code.

## RCS Capability Inventory (verified by import)

| Area | RCS module / class | Notes |
|---|---|---|
| SO101 sim scene | `rcs.envs.configs.EmptyWorldSO101` | Returns `SimEnvCreatorConfig`; camera disabled by default |
| SO101 IK / FK | `rcs.common.Pin(mjcf, attachment_site)` | `.forward(q) -> Pose`, `.inverse(pose, q0) -> q` |
| Pose math | `rcs.common.Pose` | translation, quaternion, composition, inverse |
| Robot registry | `rcs.ROBOTS[RobotType('SO101')]` | mjcf path, dof, joint_limits, q_home, attachment_site |
| Object assets | `rcs.OBJECT_PATHS` | `green_cube`, etc. — re-usable MJCF for cubes |
| Task registry | `rcs.TASKS` | `pick` task auto-registers on import of `rcs.envs.tasks` |
| Gym env base | `rcs.envs.base.SimEnv` | Wraps a `rcs.sim.Sim` instance |
| Robot wrapper | `rcs.envs.base.RobotWrapper` | Exposes robot actions in chosen `ControlMode` |
| Gripper wrapper | `rcs.envs.base.GripperWrapper` | Binary open/close API |
| Relative actions | `rcs.envs.base.RelativeActionSpace(max_mov=, relative_to=)` | Local Δxyz / Δquat / Δjoint deltas, exactly the RL action shape we want |
| Sim wrappers | `rcs.envs.sim.RobotSimWrapper`, `GripperWrapperSim` | Collision flag handling, IK success info |
| Camera (sim) | `rcs.camera.sim.SimCameraSet`, `SimCameraConfig` | On-demand offscreen rendering; integrates via `CameraSetWrapper` |
| Pick task scaffold | `rcs.envs.tasks.PickTask`, `RandomSquareObjPos`, `PickObjSuccessWrapper` | Reach + place + static reward; randomized cube reset |
| Hardware control | `rcs_so101.hw.SO101`, `SO101Gripper` | Requires LeRobot `so101_follower` path (needs shim later) |
| RPC | `rcs.rpc.client`, `rcs.rpc.server` | For remote model inference / data collection |

## Overlap with Our Modules

| Our module | Replace with RCS? | Reason / action |
|---|---|---|
| `hybrid_control_rl/config.py` (YAML loader, deep_merge, extends) | No | RCS uses dataclass configs (`SimEnvCreatorConfig`), not YAML. Keep our YAML loader for our hybrid pipeline config; bridge to RCS configs in the env wrapper. |
| `perception/color_block_detector.py` (HSV + contour) | No | RCS has no domain-specific perception. Keep. |
| `estimation/pixel_to_table.py` (ray-plane intersection + cov) | No | RCS provides FK via `rcs.common.Pin`, but no pixel-to-plane projection. Keep this module; it now accepts either 4x4 matrices or pose-like objects with `.pose_matrix()`, so `rcs.common.Pin.forward(q)` can be used directly. |
| `estimation/block_belief.py` (per-color static Kalman) | No | RCS doesn't track objects over time. Keep. |
| `tests/test_color_detector_synthetic.py` | No | Pure-numeric, env-independent. Keep. |
| `tests/test_pixel_to_table_projection.py` | No | Pure-numeric, env-independent. Keep. |
| `tests/test_block_belief_tracker.py` | No | Pure-numeric, env-independent. Keep. |
| `scripts/validate_pixel_to_table.py` (uses HW2 MJCF + `mujoco.Renderer`) | Optional | Works today. A future RCS-native rewrite would use `EmptyWorldSO101` + `SimCameraSet`, but no functional gain. Defer. |
| `planning/hybrid_waypoint_planner.py` (forthcoming) | No | RCS gives action primitives (`RelativeActionSpace`); the multi-phase sequencer is our addition. |
| `runtime/hybrid_task_executor.py` (forthcoming) | No | Same. |
| Project 3 env (forthcoming) | **Build on RCS** | New file `envs/project3_so101_env.py` that composes `EmptyWorldSO101` + colored cubes from `rcs.OBJECT_PATHS` + bin + wrist camera + `RelativeActionSpace`. Replaces what we'd otherwise wrap around hw3. |
| Waypoint controller (forthcoming) | **Build on RCS** | Thin adapter that calls `env.step({'tquat': delta, 'gripper': g})` instead of writing a joint-space controller. |

## Concrete Changes Already Applied

- Conda env `lerobot-p3-rcs` created (Python 3.11 + MuJoCo 3.2.6 + Pinocchio 3.7 + glfw via conda-forge).
- `rcs` core and `rcs_so101` extension built and installed from `external/robot-control-stack/`.
- `scripts/test_rcs_so101_sim.py` — verifies `EmptyWorldSO101` resets and steps.
- `envs/project3_so101_env.py` — composes `EmptyWorldSO101` + wrist camera + colored cubes from `rcs.OBJECT_PATHS` (green) and in-tree red/blue/yellow MJCFs.
- `scripts/render_project3_screenshot.py` — saves wrist-cam + external view PNGs.
- `runtime/rcs_sim_adapters.py` — RCS-backed `RcsWristBlockObserver`, `RcsColorVisibilityChecker`, and `ScriptedAlignGraspPolicy`. The pixel-to-table adapter delegates to `PixelToTableProjector.project_from_T_BC` instead of reimplementing the geometry.
- `control/waypoint_controller.py` — proportional-step controller (replaced earlier dominant-axis stepping); `gripper=None` preserves the current gripper state read from the observation.
- `scripts/run_hybrid_eval_sim.py` — end-to-end runner; current sim succeeds (`success=True`, scripted align/grasp).

## RCS Coverage Verified (no RCS replacement exists for these)

- Pixel-to-table projection: searched `rcs/`, `rcs.envs.*`, `rcs.sim.*`, `rcs.camera.*` — no ray-plane utility. `rcs.common.Pin` provides FK/IK; pixel back-projection is ours to keep.
- Waypoint controller / cartesian-delta loop: RCS has `RelativeActionSpace` (per-step delta wrapper) and `rcs.ompl.mj_ompl.MjOMPL` (full collision-aware joint planner). Neither does "step toward a base-frame waypoint until reached" out of the box. `RelativeActionSpace` is already underneath our env. `MjOMPL` is overkill for short unobstructed cartesian moves and brings OMPL planning time; keep our straight-line controller until obstacle-avoidance becomes a real need.
- Object/color perception: no detector in RCS. Keep ours.
- Belief tracking: no Kalman filter in RCS. Keep ours.

## Decisions

## Decisions

1. **Keep** our pure-numeric perception/estimation modules; they are complementary, not redundant.
2. **Build new** code (env wrapper, planner, executor) on top of RCS primitives rather than reinventing them.
3. `PixelToTableProjector` accepts `rcs.common.Pin` directly because it converts pose-like objects returned by `.forward(q)` through `.pose_matrix()`.
4. **Defer** hardware integration (`rcs_so101.hw`) until we add a LeRobot path shim for `so101_follower`.
