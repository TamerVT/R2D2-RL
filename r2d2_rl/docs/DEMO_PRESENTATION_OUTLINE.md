# Project 3 demo presentation outline

Roughly 8-10 slides; 15 min talk including demo. Pacing: 1 min title, 2 min
problem + pipeline, 4 min results, 4 min demo, 1 min wrap.

## Slide 1 - Title

Project 3: Visual pick-and-place with SO-101.
Team R2D2-RL.

## Slide 2 - Task

Three evals on SO-101 + wrist camera + colored cubes:

1. **Eval 1** -- single cube pick & place.
2. **Eval 2** -- clutter: pick the requested color from 4 cubes.
3. **Eval 3** -- multi-goal sequence: red -> blue -> green into 3 bowls.

Constraints: real-time on the arm; sim must transfer to real (sim-to-real).

## Slide 3 - Pipeline (diagram)

```
       wrist RGB
          |
   +-------------+   color, belief.xy     +----------+
   | HSV detect  |---------------------->| Belief   |
   |  +  pixel-> |                       | (per-color
   |  table      |                       |  Kalman) |
   +-------------+                       +----------+
                                              |
                                              v
                  +--------+    +----------+    +------------+    +----------+
       observe -->| approach|-->| align    |-->| lift /     |--> | release  |
                  | (regress|   | _grasp   |   | transport  |    | (way-    |
                  | -or)    |   | (RL SAC) |   | (waypoint) |    |  point)  |
                  +--------+    +----------+    +------------+    +----------+
                                                       |
                                                       v (vision watchdog)
                                                   re-localize
```

Hybrid: classical for everything except the contact-rich grasp.

## Slide 4 - The learned piece (RL setup)

- **Env**: `LeRobotAlignGraspEnv` (RCS, control mode JOINTS, 5° max delta,
  pregrasp regressor at reset, +-6 cm cube randomization).
- **Observation** (matches real arm HIL dataset): wrist RGB CHW
  [3,128,128] uint8 + 24-D state (joints + velocities + zero placeholders +
  color one-hot).
- **Action**: 6-D absolute joint+gripper targets in calibrated LeRobot units.
- **Reward** (Flo's two-phase):
    - pre-grasp: exp(-20*xyz_dist) + exp(-40*xy_dist) + (exp(-40*xy_dist) *
      exp(-60*z_dist)) + clip(progress) + close-when-near + 2*valid_grasp -
      action_penalty.
    - post-grasp: 1.5*valid_grasp + 6*lift + 3*tcp_lift_while_grasped +
      10*success - action_penalty.

## Slide 5 - Training (BC -> SAC)

- **Stage 1 BC**: 70-episode multicolor HIL demo dataset, supervised
  `(image, state) -> 6-D action`. ~80 epochs, GPU. Produces
  `normalization_stats.json` + BC actor.
- **Stage 2 BC-pretrain + SAC**: SB3 SAC, actor MSE-supervised on the same
  demos for 80 epochs, then 100k SAC RL steps in RCS sim. ~3 h on RTX 3060.
- Checkpoint format: SB3 SAC zip (loaded via
  `SB3VisualAlignGraspPolicy` in the hybrid executor).

## Slide 6 - Results

(Fill from `summary.json` after training run.)

- Sim Eval 1: M/M successes.
- Sim Eval 2: M/M successes.
- Sim Eval 3: N/N successes (where N = 3 goals).
- BC val: scaled_mse ~0.0016, gripper_corr ~0.95.
- SAC ep_rew_mean: <starting -33, final value>.

## Slide 7 - Sim-to-real considerations

- Wrist camera placement matches the real arm (in `gripper_body`); we tested
  this on `wrist_in_wrist_body/...` renders.
- Same 24-D state / 6-D action conventions in sim and real; LeRobot
  calibration JSON drives both.
- Pregrasp regressor brings the arm to the demonstrated starting pose at
  every reset.
- BC is the load-bearing piece for sim-to-real (sees only real demos); SAC
  refines in sim.

## Slide 8 - What we'd improve with more time

- Domain randomization on lighting + cube HSV jitter.
- Wider cube spawn distribution (already at +-6 cm; could go to +-10 cm).
- Camera FOV tuning -- some randomized cube corners still fall just outside
  the 70 deg wrist FOV.
- Real-arm bring-up (camera recalibration, HSV tuning under venue lights).

## Slide 9 - Live demo

Have ready in three terminal tabs:

```bash
# Tab A
bash r2d2_rl/scripts/verify_all_evals.sh   # baseline scripted runs

# Tab B
MUJOCO_GL=egl python r2d2_rl/scripts/run_eval_sequence.py \
    --config r2d2_rl/configs/hybrid_control_rl/eval3.yaml \
    --output-dir r2d2_rl/outputs/eval3_demo \
    --sb3-align-grasp-checkpoint r2d2_rl/outputs/hil_bc_sac_v1/final_model.zip \
    --enable-watchdog --save-images

# Tab C  (open in browser)
tensorboard --logdir r2d2_rl/outputs/
```

Have `r2d2_rl/outputs/eval3_smoke/initial_external.png` and per-goal
externals printed/visible as fallback screenshots if live sim hiccups.

## Slide 10 - Q&A

Anticipated questions:

- "Why SAC instead of PPO?" -- continuous action, off-policy sample
  efficiency, automatic entropy. Plus BC-pretrain compatibility.
- "Why is the gripper horizontal in the external view?" -- sim XML kinematics
  vs LeRobot calibration zero offsets differ slightly. The wrist camera view
  still matches the real arm, which is what the policy sees.
- "How does the policy generalize to unseen cube positions?" -- BC dataset
  spans the full workspace; sim training uses +-6 cm uniform randomization on
  top of that.
- "What if the cube falls during transport?" -- vision watchdog: if HSV
  detection of target color drops below threshold for N frames during
  transport, controller returns to a safe pose and re-localizes (up to 3
  attempts).
