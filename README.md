# R2D2-RL — Project 3: Visual Pick-and-Place with SO-101

ETH Robot Learning, Project 3. Team work-in-progress repo for the
hybrid **modular control + reinforcement learning** approach to three
visual pick-and-place evaluations on the SO-101 arm.

- **Eval 1** — single block → bowl (BC or RL allowed)
- **Eval 2** — two-color cluster, pick the target color → bowl (RL required)
- **Eval 3** — four colors, three sequential pick-and-place goals (RL required)

The policy must operate on RGB observations from the wrist camera;
bowl targets are given as robot-frame `(x, y, z)` coordinates.

## Where to start

Read [`SIM_WRIST_CAMERA_README.md`](SIM_WRIST_CAMERA_README.md) — the
full setup, run, and design guide. It covers:

- environment setup (`conda` env `lerobot-p3` with `mujoco 3.8.0` etc.)
- the one-line wrist-camera demo command and expected output
- how the `wrist_cam` MuJoCo camera is configured and how to edit it
- the plan for real-hardware calibration (intrinsics, hand-eye, fixed mount)
- limitations and the suggested next steps (scene MJCF, perception, RL)

## Quick run

```bash
source /home/explo22/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-p3
MUJOCO_GL=egl python scripts/test_wrist_camera_feed.py --headless --save-video
# -> outputs/wrist_cam_demo/{frame_*.png, wrist_cam_demo.mp4}
```

## Layout

```
R2D2-RL/
├── README.md                      # you are here
├── SIM_WRIST_CAMERA_README.md     # detailed guide
├── scripts/
│   └── test_wrist_camera_feed.py  # MuJoCo wrist-camera smoke test
├── ethz-course-2026/              # vendored ETH course reference repo
│   └── hw2_robot_control_mdps/    # SO arm gym + MJCF (so_arm100.xml has our wrist_cam)
└── outputs/                       # demo artifacts (frames + MP4)
```

## Status

- [x] MuJoCo sim environment running
- [x] Named wrist camera (`wrist_cam`) + offscreen RGB feed working in headless mode
- [ ] Scene MJCF (table, blocks, bowls) — next
- [ ] Gymnasium wrapper exposing `wrist_image` in obs dict — next
- [ ] Perception module (RGB → block xyz) — after scene
- [ ] Classical IK + waypoint controller — after scene
- [ ] RL for contact-rich subphases — for Eval 2/3
