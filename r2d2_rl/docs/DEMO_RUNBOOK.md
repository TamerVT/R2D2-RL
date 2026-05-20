# Project 3 demo runbook

End-to-end commands for Eval 1 / 2 / 3 in the RCS SO-101 sim. Every command
assumes:

- Working directory: `project3/`
- Conda env active: `conda activate lerobot-p3-rcs`
- `MUJOCO_GL=egl` exported (or prefixed inline as below)

## 0. Pre-flight (one-time, already done; here for reference)

```bash
# Unzip pregrasp regressor (already in r2d2_rl/outputs/pregrasp_regressor/best_pregrasp_mlp.pt)
unzip -j -o p3_required_sim_training_artifacts.zip \
      'project3_modular/outputs/pregrasp_regressor/best_pregrasp_mlp.pt' \
      -d r2d2_rl/outputs/pregrasp_regressor/

# Unzip the multicolor HIL dataset
mkdir -p r2d2_rl/outputs/hil_dataset
unzip -o p3_local_grasp_hil_multicolor_colorcond_v1.zip \
      -d r2d2_rl/outputs/hil_dataset/

# RCS SO-101 XML is patched in place (see external/robot-control-stack/...).
# All 62 unit tests should pass before any demo:
MUJOCO_GL=egl python -m unittest discover -s r2d2_rl/tests -p 'test_*.py'
```

## 1. Train the policy

Two-stage: standalone BC, then BC-pretrain + SAC RL finetune.

```bash
# Stage 1: BC (~5-15 min on GPU)
MUJOCO_GL=egl python r2d2_rl/scripts/train_real_hil_bc_policy.py \
    --dataset-root r2d2_rl/outputs/hil_dataset/p3_local_grasp_hil_multicolor_colorcond_v1 \
    --output-dir   r2d2_rl/outputs/real_hil_bc_warmstart \
    --epochs 80 --zero-motor-currents

# Stage 2: BC pretrain + SAC RL (~2.5-3.5 h on GPU)
MUJOCO_GL=egl python r2d2_rl/scripts/train_visual_hil_compat_sac.py \
    --output-dir            r2d2_rl/outputs/hil_bc_sac_v1 \
    --normalization-stats   r2d2_rl/outputs/real_hil_bc_warmstart/normalization_stats.json \
    --bc-dataset-root       r2d2_rl/outputs/hil_dataset/p3_local_grasp_hil_multicolor_colorcond_v1 \
    --bc-pretrain-epochs 80 --bc-zero-motor-currents \
    --total-timesteps 100000
```

Produces `r2d2_rl/outputs/hil_bc_sac_v1/final_model.zip` -- the demo checkpoint.

While training, watch:

```bash
tensorboard --logdir r2d2_rl/outputs/   # http://localhost:6006
# Key metrics: rollout/ep_rew_mean, env/local_success, env/cube_lift
```

## 2. Eval 1 -- single cube pick-and-place

```bash
MUJOCO_GL=egl python r2d2_rl/scripts/run_eval_sequence.py \
    --config       r2d2_rl/configs/hybrid_control_rl/eval1.yaml \
    --output-dir   r2d2_rl/outputs/eval1_run \
    --sb3-align-grasp-checkpoint r2d2_rl/outputs/hil_bc_sac_v1/final_model.zip \
    --enable-watchdog --save-images
```

Pass criterion: `successes: 1/1` in stdout.

## 3. Eval 2 -- clutter (4 cubes, target=red)

```bash
MUJOCO_GL=egl python r2d2_rl/scripts/run_eval_sequence.py \
    --config       r2d2_rl/configs/hybrid_control_rl/eval2.yaml \
    --output-dir   r2d2_rl/outputs/eval2_run \
    --sb3-align-grasp-checkpoint r2d2_rl/outputs/hil_bc_sac_v1/final_model.zip \
    --enable-watchdog --save-images
```

Pass criterion: `successes: 1/1`. The HSV detector + per-color belief tracker
isolate the requested target from the three distractor cubes.

## 4. Eval 3 -- multi-goal sequence (red, blue, green)

```bash
MUJOCO_GL=egl python r2d2_rl/scripts/run_eval_sequence.py \
    --config       r2d2_rl/configs/hybrid_control_rl/eval3.yaml \
    --output-dir   r2d2_rl/outputs/eval3_run \
    --sb3-align-grasp-checkpoint r2d2_rl/outputs/hil_bc_sac_v1/final_model.zip \
    --enable-watchdog --save-images
```

Pass criterion: `successes: 3/3`. The executor's `run_sequence` re-localizes
the next target after each release.

## 5. If the trained SAC underperforms

Drop the `--sb3-align-grasp-checkpoint` flag; the runner falls back to the
classical `ScriptedAlignGraspPolicy` (move-to-belief-then-close-gripper).
Useful as a backup demo path if RL doesn't converge in time.

```bash
MUJOCO_GL=egl python r2d2_rl/scripts/run_eval_sequence.py \
    --config       r2d2_rl/configs/hybrid_control_rl/eval3.yaml \
    --output-dir   r2d2_rl/outputs/eval3_scripted \
    --enable-watchdog --save-images
```

In our smoke tests the scripted path passes 1/1 (Eval 2) and 3/3 (Eval 3).

## 6. Outputs of each run

`<output-dir>/summary.json` -- structured per-goal trace (state machine
events, attempts, final state, failure reason).

`<output-dir>/initial_wrist.png` + `initial_external.png` -- scene at episode
start.

`<output-dir>/goalNN_<color>_wrist.png` + `_external.png` -- scene after
each goal in the sequence.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `Unknown target color 'X'` | Add HSV ranges for `X` to `r2d2_rl/configs/hybrid_control_rl/base.yaml` under `perception.colors`. |
| `Could not find RCS RandomSquareObjPos wrapper` (training only) | RCS install regression; reinstall RCS in `lerobot-p3-rcs`. |
| Wrist camera frame is uniform gray (no cube) | Camera placement issue. Check that `external/robot-control-stack/assets/robots/so101/so101.xml` still has the Project-3 wrist camera inside `gripper_body`. |
| SAC checkpoint loads but the run hangs | `--sb3-align-grasp-max-steps` (default 80) too high for cube position. Lower to 40-50 and re-run. |
| Cube not visible from wrist at edge of randomization | Expected at extreme corners of the ±6 cm window. The BC policy was trained to handle off-center starts; the SAC finetune should improve robustness. |

## 8. Hardware bring-up (Eval 1+ on real arm) -- post-demo or contingency

Not in scope of the sim demo. Outline only:

1. Wire LeRobot's `so101_follower` to `rcs_so101.hw` (LeRobot HIL-SERL patch
   already present in `external/`; needs applying + verifying).
2. Re-run camera intrinsic calibration on the real arm. The shipped
   `camera_calib.npz` is broken (fx=4487, distortion in tens).
3. Validate HSV thresholds under the venue lights; bump saturation/value
   bands if needed.
4. Run a manual single-cube pick on real, then increment to the full Eval 1.
