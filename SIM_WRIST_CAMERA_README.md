# Project 3 — MuJoCo Wrist-Camera Setup Guide

Authoritative reference for what's in this repo (R2D2-RL) right now, why it's
here, and how to extend it. Read this before adding sim/perception code.

---

## 1. Project context

Project 3 evaluates the SO-101 arm on three visual pick-and-place tasks
(see `../Project 3_ Reinforcement Learning – Final Details (1).pdf`):

- **Eval 1** — single block → bowl (BC or RL allowed)
- **Eval 2** — two-colored cluster → target color → bowl (RL **required**)
- **Eval 3** — four colors, three sequential goals (RL **required**)

Hard constraints:

- Policy must operate on **visual observations** of the blocks
- Bowl target locations are given as **robot-frame `(x, y, z)`**, easy to swap
  at evaluation time
- The wrist camera is an RGB camera

### Our approach (as of now)

Hybrid **modular control + RL**:

1. Capture an RGB frame from the **wrist camera** at a standardized pose
2. Run a perception module → estimated block `(x, y, z)` in robot frame
3. Use **classical IK + waypoint motion** to move near the block / bowl
4. Use a **learned/RL policy** only for the contact-rich subphases (final
   alignment, grasp, lift, release)
5. For Eval 3, re-localize before each step in the sequence

Working entirely in **MuJoCo simulation** first; sim-to-real and real-robot
teleop demos come later.

---

## 2. Repo layout

```
R2D2-RL/
├── SIM_WRIST_CAMERA_README.md      # this file
├── ethz-course-2026/               # vendored ETH course reference repo
│   └── hw2_robot_control_mdps/     # SO arm gym + MJCF — the sim we build on
│       ├── env/so100_tracking_env.py
│       ├── scripts/*.py            # interactive viewer, IK, PID, RL train
│       └── so101_gym/assets/
│           ├── so100_pos_ctrl.xml          # entry-point MJCF (worldbody + lights)
│           └── trs_so_arm100/
│               └── so_arm100.xml           # arm body tree + cameras  ← we edited
├── scripts/
│   └── test_wrist_camera_feed.py   # our wrist-camera smoke test
└── outputs/                        # demo artifacts
    ├── wrist_cam_demo/             # frames + MP4 from wrist_cam
    └── left_wrist_demo/            # frames from the original left_wrist camera
```

External reference (cloned via Git):
<https://github.com/mees-robot-learning-course/ethz-course-2026/tree/main/hw2_robot_control_mdps>

Also relevant (not yet integrated):
<https://github.com/RobotControlStack/robot-control-stack> — SO101 in MuJoCo

---

## 3. Environment setup

An existing conda env `lerobot-p3` already has every dependency
(`mujoco 3.8.0`, `numpy 2.2.6`, `gymnasium 1.3.0`, `imageio 2.37.3`, `opencv 4.13`).

Activate it:

```bash
source /home/explo22/miniforge3/etc/profile.d/conda.sh
conda activate lerobot-p3
```

From scratch on another machine:

```bash
mamba create -y -n lerobot-p3 python=3.12
mamba activate lerobot-p3
pip install -r ethz-course-2026/hw2_robot_control_mdps/requirements.txt
pip install -e ethz-course-2026/hw2_robot_control_mdps
pip install imageio imageio-ffmpeg opencv-python
```

**Linux/WSL2 OpenGL note.** MuJoCo's interactive viewer needs a display; WSL2
without WSLg has none. For offscreen rendering set `MUJOCO_GL=egl` before
importing `mujoco` (the demo script does this automatically with `--headless`).
If EGL is unavailable, install OSMesa (`sudo apt install libosmesa6`) and use
`MUJOCO_GL=osmesa`.

---

## 4. Running the wrist-camera demo

From the repo root:

```bash
conda activate lerobot-p3
MUJOCO_GL=egl python scripts/test_wrist_camera_feed.py --headless --save-video
```

Expected stdout tail:

```
[info] using camera: name='wrist_cam'  id=1  ncam=2
[info] MUJOCO_GL=egl  width=640 height=480
[info] frames rendered: 120
[info] frame shape: (480, 640, 3)  dtype: uint8
[info] pixel min/max: 0/255  mean: ~138
[info] saved video: outputs/wrist_cam_demo/wrist_cam_demo.mp4
```

Artifacts land in `outputs/wrist_cam_demo/`:

- `frame_0000.png`, `frame_<mid>.png`, `frame_<last>.png` — sample stills
  (use `--save-pngs` to dump every frame)
- `wrist_cam_demo.mp4` — MP4, or `.gif` if ffmpeg is missing

The first frame (mid of demo) currently shows the gripper jaws, the
end-effector axis triad, and the ground plane — a tight first-person wrist
view. There is **no scene yet** (no table, blocks, or bowls); that comes next.

### Useful flags

| flag                  | default                       | purpose |
|-----------------------|-------------------------------|---------|
| `--camera <name>`     | `wrist_cam`                   | MJCF camera name; falls back to `left_wrist` if missing |
| `--frames N`          | `120`                         | how many frames to render |
| `--ctrl-decimation K` | `10`                          | sim sub-steps between rendered frames |
| `--width / --height`  | `640 / 480`                   | render resolution |
| `--save-pngs`         | off                           | dump every frame, not just first/mid/last |
| `--save-video`        | off                           | encode an MP4 (or GIF fallback) |
| `--headless`          | off                           | force `MUJOCO_GL=egl` for offscreen render |
| `--viewer`            | off                           | open the interactive MuJoCo viewer at the end (needs display) |
| `--xml <path>`        | hw2 `so100_pos_ctrl.xml`      | override the MJCF |
| `--output-dir <path>` | `outputs/wrist_cam_demo` | override artifacts dir |

---

## 5. How the wrist camera is configured

### 5.1 Where it lives in the MJCF

The arm body tree is `ethz-course-2026/hw2_robot_control_mdps/so101_gym/assets/trs_so_arm100/so_arm100.xml`.
Inside the `<body name="Fixed_Jaw">` element (the gripper / end-effector link),
the model defines **two cameras**:

```xml
<!-- pre-existing: over-the-shoulder view, auto-aims at the gripper -->
<camera name="left_wrist"
        pos="0  0  0.55"
        fovy="20"
        mode="targetbody"
        target="vx300s_left/camera_focus" />

<!-- added for Project 3: tight first-person wrist view -->
<camera name="wrist_cam"
        pos="0.02 -0.02 0.08"
        fovy="55"
        mode="targetbody"
        target="vx300s_left/camera_focus" />
```

`vx300s_left/camera_focus` is a tiny body wedged between the fingertips; both
cameras use it as their look-at target, which lets us specify only the camera
**position** and have MuJoCo compute orientation each step.

### 5.2 What each field means

- **`pos="x y z"`** — meters, in the parent body's local frame. `Fixed_Jaw` is
  rotated `euler="0 1.57079 0"` relative to the wrist link, so in this frame:
  - `+X_FixedJaw` ≈ "back of the gripper" (toward the wrist motor)
  - `-Y_FixedJaw` ≈ along the fingertips (grasp approach direction)
  - `+Z_FixedJaw` ≈ "above the gripper" in the arm's home pose
  
  So `(0.02, -0.02, 0.08)` reads as: 2 cm back, 2 cm toward the fingers,
  8 cm above the wrist.
- **`fovy="55"`** — vertical field of view in degrees. Horizontal FOV follows
  from the render aspect ratio. Real SO101 wrist cameras typically run
  60–80° vertical; 55° is a conservative starting point.
- **`mode="targetbody" + target="..."`** — auto-aim at the target body each
  frame. Set `pos` only; orientation is solved for you. **Convenient for sim,
  not physically realistic for a rigid mount** — the view "follows" the
  gripper as the joint angles change.
- **`mode="fixed"` + `quat="w x y z"` / `euler` / `xyaxes`** — the camera is
  rigidly attached to the parent body, like a real bolted-on camera. Camera
  looks down its local `−Z`. This is the mode to use once we calibrate
  against the physical SO101.

### 5.3 Why two cameras?

- `left_wrist` was already in the hw2 MJCF and is referenced by the course
  exercises. We didn't touch it.
- `wrist_cam` is the one we'll use for Project 3 perception. The pose was
  hand-tuned to roughly match an SO101 wrist-mounted RGB camera.

Both render fine and the demo script accepts either via `--camera`.

---

## 6. How to edit the camera

Open the MJCF and change the `wrist_cam` element. No build step — re-running
`test_wrist_camera_feed.py` reloads it.

```
ethz-course-2026/hw2_robot_control_mdps/so101_gym/assets/trs_so_arm100/so_arm100.xml
```

Quick iteration loop:

```bash
# edit the wrist_cam line, then:
MUJOCO_GL=egl python scripts/test_wrist_camera_feed.py --headless --frames 1
# look at outputs/wrist_cam_demo/frame_0000.png
```

Common tweaks:

- **Move it closer to the gripper** → decrease `z` in `pos` (try `0.05`).
- **Wider field of view** → increase `fovy` (e.g. `70`).
- **Different aspect ratio / resolution** → pass `--width / --height` to the
  script *and* add `resolution="W H"` to the `<camera>` element if you want
  it baked into the model.
- **Tilt slightly forward** → switch to `mode="fixed"` and add a `quat` or
  `xyaxes`. Cameras look down their local `−Z`; the easiest spec is
  `xyaxes="x_right_x x_right_y x_right_z y_up_x y_up_y y_up_z"`.

---

## 7. Configuring for real-hardware calibration

`mode="targetbody"` is a sim convenience. The real wrist camera is rigidly
bolted, so once we mount it we need to:

1. **Pick the mount link.** Almost certainly `Wrist_Pitch_Roll` or
   `Fixed_Jaw`. Move the `<camera>` element into that body.
2. **Measure pose `(x, y, z, quat)`** relative to that link's origin:
   - *CAD method:* open the SO101 STEP/STL of the wrist link in CAD, identify
     the camera mount face, record translation + rotation from link origin to
     camera optical center.
   - *Hand–eye calibration:* place a checkerboard at known robot-frame
     positions, capture from the real wrist camera, solve PnP with
     `cv2.solvePnP`, back out camera-in-wrist using forward kinematics.
3. **Measure intrinsics → set `fovy`.** Run `cv2.calibrateCamera` to get
   `fx, fy, image_height`, then:
   ```python
   fovy_deg = 2 * math.degrees(math.atan2(image_height / 2, fy))
   ```
4. **Switch to fixed mode** and pin resolution to the real camera's:

   ```xml
   <camera name="wrist_cam"
           pos="0.012 -0.020 0.045"          <!-- meters, from calibration -->
           quat="0.7071 0.7071 0 0"          <!-- from calibration -->
           fovy="58"                          <!-- from intrinsics -->
           resolution="640 480" />
   ```

5. **Sanity-check sim vs real.** Render one sim image and capture one real
   image at the same joint config with the same workspace contents. The
   visual gap between them is what perception has to be robust to.

Store the measured pose/intrinsics in a single source of truth — suggested
`configs/wrist_cam.yaml` (not created yet) — and have both the MJCF
generator and the real-robot code read from it.

---

## 8. Known limitations (today)

- **No scene yet.** The MJCF has only the robot arm and a mocap target
  sphere. No light-gray table, blocks, or bowls. Building the Eval 1 scene
  is the next step.
- **`targetbody` camera mode** — view auto-tracks the gripper. Fine for sim
  prototyping, but does **not** mirror a real rigidly-mounted camera. Switch
  to `mode="fixed"` after calibration.
- **No Gymnasium wrapper exposing `wrist_image` yet.** `SO100TrackEnv`
  returns proprioception only. A minimal subclass that adds
  `wrist_image: HxWx3 uint8` to the obs dict is the obvious next addition.
- **No live viewer demonstrated under WSL2.** Use `--viewer` only on a
  machine with WSLg or native X.
- **so_arm100 mesh set, not SO101-specific meshes.** Kinematics are
  compatible (same `so100_follower` / `so101_follower` aliases in LeRobot),
  but visual fidelity is generic.
- **No RL started.** Intentional — perception + scene + control loop first.

---

## 9. Suggested next steps

1. **Build an Eval 1 scene MJCF.** Add to the worldbody:
   - light-gray table plane (`#B8ADA9` ≈ rgba 0.722 0.678 0.663 1)
   - 1+ colored block bodies with `freejoint`, parameterized colors
   - 1+ bowl bodies (mesh or cylindrical primitive) at robot-frame `(x,y,z)`
2. **Wrap `SO100TrackEnv`** so observations are
   `{"wrist_image": HxWx3 uint8, "qpos": (6,), "target_xyz": (3,)}`.
3. **Perception stub** — wrist RGB + target color → estimated block
   `(x, y, z)`. Train/eval against MuJoCo ground truth before any real data.
4. **Classical waypoint controller** for go-to-pregrasp and bowl-drop.
   Borrow the IK + PID utilities already in `hw2/scripts/`.
5. **Local RL** only on contact-rich phases (Eval 2/3 requirement).

---

## 10. Quick reference (one-liners)

```bash
# activate env
source /home/explo22/miniforge3/etc/profile.d/conda.sh && conda activate lerobot-p3

# run wrist-cam demo, save MP4
MUJOCO_GL=egl python scripts/test_wrist_camera_feed.py --headless --save-video

# compare against the original left_wrist camera
MUJOCO_GL=egl python scripts/test_wrist_camera_feed.py --headless --camera left_wrist \
    --output-dir outputs/left_wrist_demo

# open the interactive viewer (needs display, e.g. WSLg)
python scripts/test_wrist_camera_feed.py --viewer --frames 1
```
