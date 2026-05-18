from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import pyquaternion as pyq


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SCENE_XML = (
    REPO_ROOT
    / "project3_modular"
    / "rl_grasp"
    / "assets"
    / "teleop"
    / "so101_local_grasp_teleop_scene.xml"
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "project3_modular"
    / "rl_grasp"
    / "demos"
    / "raw_local_grasp_hw3style"
)

CAMERA_NAMES: tuple[str, ...] = ("angle", "side", "top")

ARM_JOINT_NAMES: tuple[str, ...] = ("1", "2", "3", "4", "5")
GRIPPER_JOINT_NAME = "6"
ALL_JOINT_NAMES: tuple[str, ...] = ARM_JOINT_NAMES + (GRIPPER_JOINT_NAME,)

CUBE_JOINT_NAME = "green_cube_joint"
MOCAP_BODY_NAME = "mocap_target"
GRIPPER_SITE_NAME = "gripper"

# Matches the grasp-local initial configuration we saw in the RCS smoke tests.
DEFAULT_ARM_QPOS = np.array(
    [-0.01914895, -1.90082799, 1.56454447, 1.04777133, -1.40323939],
    dtype=np.float64,
)

DEFAULT_GRIPPER_OPEN_QPOS = 1.50


# ---------------------------------------------------------------------
# Keymap
# ---------------------------------------------------------------------

DEFAULT_KEYMAP: dict[int, str] = {
    65362: "move_up",       # up arrow
    65364: "move_down",     # down arrow
    65361: "move_left",     # left arrow
    65363: "move_right",    # right arrow
    ord("w"): "move_forward",
    ord("s"): "move_backward",
    ord("d"): "rot_x_pos",
    ord("a"): "rot_x_neg",
    ord("e"): "rot_y_pos",
    ord("q"): "rot_y_neg",
    ord("x"): "rot_z_pos",
    ord("y"): "rot_z_neg",
    ord("f"): "gripper_open",
    ord("g"): "gripper_close",
    ord("r"): "reset",
    32: "record",           # space
    13: "end_episode",      # enter
    10: "end_episode",
    27: "escape",           # esc
}


def load_keymap(path: Path | None) -> dict[int, str]:
    if path is None:
        return DEFAULT_KEYMAP

    if not path.exists():
        raise FileNotFoundError(f"Keymap not found: {path}")

    data = json.loads(path.read_text())
    return {int(entry["raw"]): action for action, entry in data.items()}


def label_for(action_name: str, keymap: dict[int, str]) -> str:
    for raw, action in keymap.items():
        if action != action_name:
            continue

        if raw == 32:
            return "SPACE"
        if raw in (10, 13):
            return "ENTER"
        if raw == 27:
            return "ESC"
        if raw in (65361, 65362, 65363, 65364):
            arrows = {
                65361: "←",
                65362: "↑",
                65363: "→",
                65364: "↓",
            }
            return arrows[raw]
        if 32 <= raw <= 126:
            return chr(raw)

        return f"raw:{raw}"

    return "?"


# ---------------------------------------------------------------------
# Quaternion / key handling
# ---------------------------------------------------------------------

def rotate_quaternion(
    quat_wxyz: np.ndarray,
    axis_xyz: list[float],
    angle_deg: float,
) -> np.ndarray:
    axis = np.asarray(axis_xyz, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat_wxyz) * pyq.Quaternion(
        axis=axis,
        angle=np.deg2rad(angle_deg),
    )
    return q.normalised.elements


def handle_teleop_key(
    action_name: str,
    data: mujoco.MjData,
    model: mujoco.MjModel,
    mocap_id: int,
    gripper_actuator_id: int,
    translation_step_m: float,
    rotation_step_deg: float,
    gripper_step: float,
) -> None:
    if action_name == "move_up":
        data.mocap_pos[mocap_id, 2] += translation_step_m
    elif action_name == "move_down":
        data.mocap_pos[mocap_id, 2] -= translation_step_m
    elif action_name == "move_left":
        data.mocap_pos[mocap_id, 0] -= translation_step_m
    elif action_name == "move_right":
        data.mocap_pos[mocap_id, 0] += translation_step_m
    elif action_name == "move_forward":
        data.mocap_pos[mocap_id, 1] += translation_step_m
    elif action_name == "move_backward":
        data.mocap_pos[mocap_id, 1] -= translation_step_m
    elif action_name == "rot_x_pos":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [1, 0, 0], rotation_step_deg
        )
    elif action_name == "rot_x_neg":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [1, 0, 0], -rotation_step_deg
        )
    elif action_name == "rot_y_pos":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [0, 1, 0], rotation_step_deg
        )
    elif action_name == "rot_y_neg":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [0, 1, 0], -rotation_step_deg
        )
    elif action_name == "rot_z_pos":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [0, 0, 1], rotation_step_deg
        )
    elif action_name == "rot_z_neg":
        data.mocap_quat[mocap_id] = rotate_quaternion(
            data.mocap_quat[mocap_id], [0, 0, 1], -rotation_step_deg
        )
    elif action_name == "gripper_open":
        data.ctrl[gripper_actuator_id] += gripper_step
    elif action_name == "gripper_close":
        data.ctrl[gripper_actuator_id] -= gripper_step

    lo = model.actuator_ctrlrange[:, 0]
    hi = model.actuator_ctrlrange[:, 1]
    data.ctrl[:] = np.clip(data.ctrl, lo, hi)


# ---------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------

class LocalGraspTeleopRecorder:
    def __init__(
        self,
        *,
        xml_path: Path,
        output_dir: Path,
        keymap_path: Path | None,
        control_hz: float,
        render_w: int,
        render_h: int,
        cube_randomization_width_m: float,
        translation_step_m: float,
        rotation_step_deg: float,
        gripper_step: float,
        success_lift_delta_m: float,
        seed: int | None,
    ) -> None:
        self.xml_path = xml_path
        self.output_dir = output_dir
        self.control_hz = float(control_hz)
        self.dt_ctrl = 1.0 / self.control_hz
        self.cube_randomization_width_m = float(cube_randomization_width_m)
        self.translation_step_m = float(translation_step_m)
        self.rotation_step_deg = float(rotation_step_deg)
        self.gripper_step = float(gripper_step)
        self.success_lift_delta_m = float(success_lift_delta_m)
        self.rng = np.random.default_rng(seed)

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        self.sim_dt = float(self.model.opt.timestep)
        self.substeps = max(1, int(round(self.dt_ctrl / self.sim_dt)))

        self.keymap = load_keymap(keymap_path)

        self.mocap_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            MOCAP_BODY_NAME,
        )
        if self.mocap_body_id == -1:
            raise ValueError(f"Mocap body '{MOCAP_BODY_NAME}' not found.")

        self.mocap_id = int(self.model.body_mocapid[self.mocap_body_id])
        if self.mocap_id < 0:
            raise ValueError(f"Body '{MOCAP_BODY_NAME}' is not a mocap body.")

        self.gripper_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            GRIPPER_SITE_NAME,
        )
        if self.gripper_site_id == -1:
            raise ValueError(f"Site '{GRIPPER_SITE_NAME}' not found.")

        self.joint_qpos_indices = self._find_joint_qpos_indices(ALL_JOINT_NAMES)
        self.arm_qpos_indices = self.joint_qpos_indices[:5]
        self.gripper_qpos_index = int(self.joint_qpos_indices[5])

        self.joint_actuator_ids = self._find_joint_actuators(ALL_JOINT_NAMES)
        self.gripper_actuator_id = int(self.joint_actuator_ids[5])

        cube_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            CUBE_JOINT_NAME,
        )
        if cube_joint_id == -1:
            raise ValueError(f"Cube joint '{CUBE_JOINT_NAME}' not found.")
        cube_qpos_start = int(self.model.jnt_qposadr[cube_joint_id])
        self.cube_qpos_slice = np.arange(cube_qpos_start, cube_qpos_start + 7)

        for cam in CAMERA_NAMES:
            cam_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                cam,
            )
            if cam_id == -1:
                raise ValueError(f"Camera '{cam}' not found.")

        self.renderer = mujoco.Renderer(
            self.model,
            height=render_h,
            width=render_w,
        )

        self.recording = False
        self.running = True
        self.episodes_done = 0

        self._episode_obs: list[dict[str, np.ndarray]] = []
        self._episode_gripper_ctrl: list[np.ndarray] = []
        self._episode_cube_lift: list[float] = []

        self.cube_start_z = 0.01
        self.latest_lift_m = 0.0
        self.latest_success = False

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_config(
            keymap_path=keymap_path,
            control_hz=control_hz,
            render_w=render_w,
            render_h=render_h,
            seed=seed,
        )

        self.reset_episode()

    # ------------------------------------------------------------------
    # Model plumbing
    # ------------------------------------------------------------------

    def _find_joint_qpos_indices(self, joint_names: tuple[str, ...]) -> np.ndarray:
        indices = []
        for name in joint_names:
            jid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            if jid == -1:
                raise ValueError(f"Joint '{name}' not found.")
            indices.append(int(self.model.jnt_qposadr[jid]))
        return np.asarray(indices, dtype=np.int32)

    def _find_joint_actuators(self, joint_names: tuple[str, ...]) -> np.ndarray:
        actuator_ids = []

        for joint_name in joint_names:
            jid = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if jid == -1:
                raise ValueError(f"Joint '{joint_name}' not found.")

            matches = []
            for aid in range(self.model.nu):
                transmission_joint_id = int(self.model.actuator_trnid[aid, 0])
                if transmission_joint_id == jid:
                    matches.append(aid)

            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one actuator for joint '{joint_name}', "
                    f"found {matches}."
                )

            actuator_ids.append(matches[0])

        return np.asarray(actuator_ids, dtype=np.int32)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _get_joint_state(self) -> np.ndarray:
        return self.data.qpos[self.joint_qpos_indices].copy()

    def _get_ee_state(self) -> np.ndarray:
        pos = self.data.site_xpos[self.gripper_site_id].copy()
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.gripper_site_id])
        return np.concatenate([pos, quat])

    def _get_cube_state(self) -> np.ndarray:
        return self.data.qpos[self.cube_qpos_slice].copy()

    def _get_obs(self) -> dict[str, np.ndarray]:
        joints = self._get_joint_state()
        ee = self._get_ee_state()
        cube = self._get_cube_state()

        return {
            "state_joints": joints.astype(np.float32),
            "state_ee": ee.astype(np.float32),
            "state_cube": cube.astype(np.float32),
            "state_gripper": np.array([joints[-1]], dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Reset / stepping
    # ------------------------------------------------------------------

    def reset_episode(self) -> None:
        mujoco.mj_resetData(self.model, self.data)

        # Arm/gripper start.
        self.data.qpos[self.arm_qpos_indices] = DEFAULT_ARM_QPOS
        self.data.qpos[self.gripper_qpos_index] = DEFAULT_GRIPPER_OPEN_QPOS

        # Cube local randomization: center (0.18, 0.03), uniform ± width/2.
        cube = self.data.qpos[self.cube_qpos_slice]
        cube[:] = np.array([0.18, 0.03, 0.01, 1.0, 0.0, 0.0, 0.0])

        half = self.cube_randomization_width_m / 2.0
        cube[0] += self.rng.uniform(-half, half)
        cube[1] += self.rng.uniform(-half, half)

        mujoco.mj_forward(self.model, self.data)

        # Set actuator targets to current joint positions.
        q = self._get_joint_state()
        for act_id, q_i in zip(self.joint_actuator_ids, q, strict=True):
            self.data.ctrl[act_id] = q_i

        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(self.data.ctrl, lo, hi)

        # Match mocap target to current gripper pose.
        ee = self._get_ee_state()
        self.data.mocap_pos[self.mocap_id] = ee[:3]
        self.data.mocap_quat[self.mocap_id] = ee[3:]

        mujoco.mj_forward(self.model, self.data)

        self.cube_start_z = float(self._get_cube_state()[2])
        self.latest_lift_m = 0.0
        self.latest_success = False

    def step_sim(self) -> None:
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        cube_z = float(self._get_cube_state()[2])
        self.latest_lift_m = cube_z - self.cube_start_z
        self.latest_success = self.latest_lift_m >= self.success_lift_delta_m

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(self) -> None:
        self._episode_obs.append(self._get_obs())
        self._episode_gripper_ctrl.append(
            np.array([self.data.ctrl[self.gripper_actuator_id]], dtype=np.float32)
        )
        self._episode_cube_lift.append(float(self.latest_lift_m))

    def discard_episode(self) -> None:
        self._episode_obs.clear()
        self._episode_gripper_ctrl.clear()
        self._episode_cube_lift.clear()

    def save_episode(self) -> None:
        if not self._episode_obs:
            print("No recorded steps to save.")
            return

        episode_dir = self.output_dir / f"episode_{self.episodes_done:04d}"
        episode_dir.mkdir(parents=True, exist_ok=False)

        stacked = {
            key: np.stack([obs[key] for obs in self._episode_obs], axis=0)
            for key in self._episode_obs[0].keys()
        }
        action_gripper = np.stack(self._episode_gripper_ctrl, axis=0)
        cube_lift = np.asarray(self._episode_cube_lift, dtype=np.float32)

        np.savez_compressed(
            episode_dir / "trajectory.npz",
            **stacked,
            action_gripper=action_gripper,
            cube_lift_m=cube_lift,
        )

        summary = {
            "episode_index": self.episodes_done,
            "num_steps": int(len(self._episode_obs)),
            "max_cube_lift_m": float(cube_lift.max(initial=0.0)),
            "final_cube_lift_m": float(cube_lift[-1]),
            "reached_lift_success": bool(
                cube_lift.max(initial=0.0) >= self.success_lift_delta_m
            ),
        }

        (episode_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

        print(
            f"Saved episode {self.episodes_done:04d}: "
            f"{summary['num_steps']} steps, "
            f"max lift={summary['max_cube_lift_m'] * 1000:.1f} mm, "
            f"success={summary['reached_lift_success']}"
        )

        self.episodes_done += 1
        self.discard_episode()

    # ------------------------------------------------------------------
    # Rendering / UI
    # ------------------------------------------------------------------

    def render_camera_bgr(self, camera_name: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera_name)
        rgb = self.renderer.render()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def compose_views(self) -> np.ndarray:
        images = {cam: self.render_camera_bgr(cam) for cam in CAMERA_NAMES}

        views = []
        for cam in CAMERA_NAMES:
            img = images[cam].copy()
            cv2.putText(
                img,
                cam,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            views.append(img)

        top_row = np.concatenate(views[:2], axis=1)
        bottom = views[2]
        pad_w = top_row.shape[1] - bottom.shape[1]
        if pad_w > 0:
            padding = np.zeros((bottom.shape[0], pad_w, 3), dtype=bottom.dtype)
            bottom_row = np.concatenate([bottom, padding], axis=1)
        else:
            bottom_row = bottom

        return np.concatenate([top_row, bottom_row], axis=0)

    def overlay_status(self, img_bgr: np.ndarray) -> np.ndarray:
        img = img_bgr.copy()

        status = (
            f"{'REC' if self.recording else 'IDLE'} | "
            f"ep {self.episodes_done} | "
            f"buffer {len(self._episode_obs)} | "
            f"lift {self.latest_lift_m * 1000:.1f} mm | "
            f"success {self.latest_success}"
        )
        cv2.putText(
            img,
            status,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (0, 0, 255) if self.recording else (255, 255, 255),
            2,
        )

        hints = (
            f"{label_for('record', self.keymap)} rec | "
            f"{label_for('end_episode', self.keymap)} save/reset | "
            f"{label_for('reset', self.keymap)} discard/reset | "
            f"{label_for('escape', self.keymap)} quit"
        )
        cv2.putText(
            img,
            hints,
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        return img

    # ------------------------------------------------------------------
    # Config metadata
    # ------------------------------------------------------------------

    def _write_run_config(
        self,
        *,
        keymap_path: Path | None,
        control_hz: float,
        render_w: int,
        render_h: int,
        seed: int | None,
    ) -> None:
        cfg = {
            "xml_path": str(self.xml_path),
            "keymap_path": None if keymap_path is None else str(keymap_path),
            "control_hz": control_hz,
            "render_w": render_w,
            "render_h": render_h,
            "cube_randomization_width_m": self.cube_randomization_width_m,
            "translation_step_m": self.translation_step_m,
            "rotation_step_deg": self.rotation_step_deg,
            "gripper_step": self.gripper_step,
            "success_lift_delta_m": self.success_lift_delta_m,
            "seed": seed,
            "arm_qpos_start": DEFAULT_ARM_QPOS.tolist(),
            "gripper_open_qpos_start": DEFAULT_GRIPPER_OPEN_QPOS,
        }

        (self.output_dir / "run_config.json").write_text(
            json.dumps(cfg, indent=2) + "\n"
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        cv2.namedWindow("SO101 Local Grasp Teleop", cv2.WINDOW_AUTOSIZE)

        last = time.perf_counter()

        try:
            while self.running:
                k_raw = cv2.waitKeyEx(1)
                if k_raw != -1:
                    action = self.keymap.get(k_raw)
                    self.handle_action(action)

                now = time.perf_counter()
                dt = now - last
                if dt < self.dt_ctrl:
                    time.sleep(self.dt_ctrl - dt)
                last = time.perf_counter()

                if self.recording:
                    self.record_step()

                self.step_sim()

                img = self.overlay_status(self.compose_views())
                cv2.imshow("SO101 Local Grasp Teleop", img)

        finally:
            cv2.destroyAllWindows()
            print(f"Done. Saved {self.episodes_done} episode(s).")
            print(f"Output directory: {self.output_dir}")

    def handle_action(self, action: str | None) -> None:
        if action is None:
            return

        if action == "escape":
            self.running = False
            return

        if action == "record":
            self.recording = not self.recording
            print("RECORDING ON" if self.recording else "RECORDING OFF")
            return

        if action == "end_episode":
            if self.recording:
                self.save_episode()
                self.recording = False
            else:
                print("Not recording; nothing saved.")
            self.reset_episode()
            return

        if action == "reset":
            if self.recording or self._episode_obs:
                print("Current episode discarded.")
            self.recording = False
            self.discard_episode()
            self.reset_episode()
            return

        handle_teleop_key(
            action_name=action,
            data=self.data,
            model=self.model,
            mocap_id=self.mocap_id,
            gripper_actuator_id=self.gripper_actuator_id,
            translation_step_m=self.translation_step_m,
            rotation_step_deg=self.rotation_step_deg,
            gripper_step=self.gripper_step,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HW3-style local grasp teleop recorder for SO101."
    )

    parser.add_argument("--xml", type=Path, default=DEFAULT_SCENE_XML)
    parser.add_argument("--keymap-json", type=Path, default=None)

    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--render-w", type=int, default=640)
    parser.add_argument("--render-h", type=int, default=480)

    parser.add_argument("--cube-randomization-width-m", type=float, default=0.04)
    parser.add_argument("--translation-step-m", type=float, default=0.01)
    parser.add_argument("--rotation-step-deg", type=float, default=10.0)
    parser.add_argument("--gripper-step", type=float, default=0.10)
    parser.add_argument("--success-lift-delta-m", type=float, default=0.005)

    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.xml.exists():
        raise FileNotFoundError(
            f"Teleop scene XML not found: {args.xml}\n"
            "Run prepare_local_grasp_teleop_scene.py first."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"demos_{timestamp}"

    recorder = LocalGraspTeleopRecorder(
        xml_path=args.xml,
        output_dir=output_dir,
        keymap_path=args.keymap_json,
        control_hz=args.control_hz,
        render_w=args.render_w,
        render_h=args.render_h,
        cube_randomization_width_m=args.cube_randomization_width_m,
        translation_step_m=args.translation_step_m,
        rotation_step_deg=args.rotation_step_deg,
        gripper_step=args.gripper_step,
        success_lift_delta_m=args.success_lift_delta_m,
        seed=args.seed,
    )
    recorder.run()


if __name__ == "__main__":
    main()
