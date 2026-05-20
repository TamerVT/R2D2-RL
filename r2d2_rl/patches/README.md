# RCS patches

`external/robot-control-stack/` is a clone of RobotControlStack and is
**git-ignored** by this repo — so any local modifications to it are not
tracked. The Project 3 sim depends on a small set of such modifications,
captured here as a patch so a fresh checkout can reproduce the environment.

## `rcs_so101_project3.patch`

Modifies two RCS asset files:

- `assets/robots/so101/so101.xml`
  - Adds the **Project 3 wrist camera** (`<camera name="wrist">`) inside
    `gripper_body` — `LeRobotAlignGraspEnv` / `Project3SO101Env` reference it
    as `robotwrist` at runtime. Without this the env fails ("fixed camera id
    outside valid range").
  - Darkens the SO-101 materials to near-black (sim-to-real visual match).
  - Adjusts the `gripper_body` quaternion for the wrist-camera framing.
- `assets/scenes/empty_world/scene.xml`
  - Replaces the checker groundplane with a flat gray table (`rgba
    0.722 0.678 0.663` = `#B8ADA9`).

## Applying it (one-time, after cloning RCS)

```bash
# from project3/
cd external/robot-control-stack
git apply --check ../../r2d2_rl/patches/rcs_so101_project3.patch   # verify
git apply         ../../r2d2_rl/patches/rcs_so101_project3.patch   # apply
```

If `--check` fails, the RCS clone is at a different revision than the patch
was generated against — regenerate with
`git -C external/robot-control-stack diff > r2d2_rl/patches/rcs_so101_project3.patch`
after re-applying the changes by hand, or rebase the patch.

## Regenerating

The patch is `git diff` of the modified RCS working tree:

```bash
git -C external/robot-control-stack diff > r2d2_rl/patches/rcs_so101_project3.patch
```
