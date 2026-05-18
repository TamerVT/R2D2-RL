from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from project3_modular.rl_grasp.teleop_utils import (
    CAMERA_NAMES,
    CUBE_DIM,
    CUBE_JOINT_NAME,
    JOINT_NAMES,
    ZarrEpisodeWriter,
    compose_camera_views,
    handle_teleop_key,
    load_keymap,
)


MOCAP_INDEX = 0
EE_SITE_NAME = "ee_site"

DEFAULT_SCENE_XML = (
    REPO_ROOT
    / "project3_modular"
    / "rl_grasp"
    / "assets"
    / "teleop"
    / "so101_local_grasp_teleop.xml"
)

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "project3_modular"
    / "rl_grasp"
    / "demos"
    / "raw_local_grasp"
    / "teleop"
)


class SO101LocalGraspTeleopRecorder:
    def __init__(
        self,
        *,
        xml_path: Path,
        out_zarr: Path,
        control_hz: float = 10.0,
        render_w: int = 640,
        render_h: int = 480,
        window_name: str = "SO101 Local Grasp Teleop",
        keymap_path: Path | None = None,
        cube_randomization_width_m: float = 0.04,
        success_lift_delta_m: float = 0.005,
        seed: int | None = None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        if self.model.nmocap != 1:
            raise ValueError(
                f"Expected exactly 1 mocap body, got nmocap={self.model.nmocap}."
            )

        self.ee_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            EE_SITE_NAME,
        )
        if self.ee_site_id == -1:
            raise ValueError(f"Site '{EE_SITE_NAME}' not found.")

        for cam in CAMERA_NAMES:
            cam_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                cam,
            )
            if cam_id == -1:
                raise ValueError(f"Camera '{cam}' not found.")

        self.qpos_idx = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_JOINT,
                        name,
                    )
                ]
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )

        self.act_id = {
            name: mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name,
            )
            for name in JOINT_NAMES
        }
        if any(v == -1 for v in self.act_id.values()):
            missing = [k for k, v in self.act_id.items() if v == -1]
            raise ValueError(f"Missing actuators: {missing}")

        cube_jnt_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            CUBE_JOINT_NAME,
        )
        if cube_jnt_id == -1:
            raise ValueError(f"Joint '{CUBE_JOINT_NAME}' not found.")

        cube_qpos_start = self.model.jnt_qposadr[cube_jnt_id]
        self.cube_qpos_idx = np.arange(
            cube_qpos_start,
            cube_qpos_start + CUBE_DIM,
        )

        out_zarr.parent.mkdir(parents=True, exist_ok=True)
        self.writer = ZarrEpisodeWriter(
            out_zarr,
            joint_dim=len(JOINT_NAMES),
            ee_dim=7,
            cube_dim=CUBE_DIM,
            gripper_dim=1,
            obstacle_dim=3,
            flush_every=12,
        )
        self.writer.set_attrs(
            xml=str(xml_path),
            joint_names=list(JOINT_NAMES),
            state_joints_spec="qpos(joints)",
            state_ee_spec="ee_pos(3) + ee_quat_wxyz(4)",
            state_cube_spec="cube_pos(3) + cube_quat_wxyz(4)",
            state_gripper_spec="gripper_joint_qpos(1)",
            action_gripper_spec="gripper_ctrl(1)",
            control_hz=float(control_hz),
            cameras_display=list(CAMERA_NAMES),
            cube_randomization_width_m=float(cube_randomization_width_m),
            success_lift_delta_m=float(success_lift_delta_m),
        )

        self.renderer = mujoco.Renderer(
            self.model,
            height=render_h,
            width=render_w,
        )
        self.window_name = window_name

        self.control_hz = float(control_hz)
        self.dt_ctrl = 1.0 / self.control_hz
        self.sim_dt = float(self.model.opt.timestep)
        self.substeps = max(1, int(round(self.dt_ctrl / self.sim_dt)))

        self.cube_randomization_width_m = float(cube_randomization_width_m)
        self.success_lift_delta_m = float(success_lift_delta_m)

        self.episodes_done = 0
        self.recording = False
        self.running = True

        self.latest_cube_lift_m = 0.0
        self.latest_success = False
        self._cube_start_z = 0.01

        self._key_to_action = load_keymap(keymap_path)
        print(f"Loaded key mapping from {keymap_path or 'default'}")

        self._reset_episode()

    # ------------------------------------------------------------------
    # Reset and state
    # ------------------------------------------------------------------

    def _reset_to_keyframe(self, key_name: str = "student_start") -> None:
        key_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_KEY,
            key_name,
        )
        if key_id == -1:
            raise ValueError(f"Keyframe '{key_name}' not found.")

        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        mujoco.mj_forward(self.model, self.data)

    def _get_q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_idx].copy()

    def _get_cube_state(self) -> np.ndarray:
        return self.data.qpos[self.cube_qpos_idx].copy()

    def _get_ee_state(self) -> np.ndarray:
        pos = self.data.site_xpos[self.ee_site_id].copy()
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.ee_site_id])
        return np.concatenate([pos, quat])

    def _clip_ctrl(self) -> None:
        lo = self.model.actuator_ctrlrange[:, 0]
        hi = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:] = np.clip(self.data.ctrl, lo, hi)

    def _init_pose_and_targets(self) -> None:
        mujoco.mj_forward(self.model, self.data)

        self.data.mocap_pos[MOCAP_INDEX] = self.data.site_xpos[
            self.ee_site_id
        ].copy()

        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.ee_site_id])
        self.data.mocap_quat[MOCAP_INDEX] = quat

        q = self._get_q()
        for i, name in enumerate(JOINT_NAMES):
            self.data.ctrl[self.act_id[name]] = q[i]
        self._clip_ctrl()

        mujoco.mj_forward(self.model, self.data)

    def _reset_episode(self) -> None:
        self._reset_to_keyframe("student_start")

        half = self.cube_randomization_width_m / 2.0
        self.data.qpos[self.cube_qpos_idx[0]] += self.rng.uniform(-half, half)
        self.data.qpos[self.cube_qpos_idx[1]] += self.rng.uniform(-half, half)

        mujoco.mj_forward(self.model, self.data)
        self._init_pose_and_targets()

        self._cube_start_z = float(self._get_cube_state()[2])
        self.latest_cube_lift_m = 0.0
        self.latest_success = False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def _record_step(self) -> None:
        state_joints = self._get_q()
        state_ee = self._get_ee_state()
        state_cube = self._get_cube_state()

        state_gripper = np.array(
            [state_joints[-1]],
            dtype=np.float32,
        )
        action_gripper = np.array(
            [self.data.ctrl[self.act_id[JOINT_NAMES[-1]]]],
            dtype=np.float32,
        )
        dummy_obstacle = np.zeros(3, dtype=np.float32)

        self.writer.append(
            state_joints,
            state_ee,
            state_cube,
            state_gripper,
            action_gripper,
            dummy_obstacle,
        )

    def _finalize_on_exit(self) -> None:
        if self.recording:
            self.writer.end_episode()
            self.episodes_done += 1
            print(f"Episode {self.episodes_done} saved on exit.")
            self.recording = False

    # ------------------------------------------------------------------
    # Keyboard interaction
    # ------------------------------------------------------------------

    def _handle_key(self, k_raw: int, _k_ascii: int) -> None:
        action = self._key_to_action.get(k_raw)

        if action == "escape":
            if self.recording:
                self.writer.end_episode()
                self.episodes_done += 1
                print(f"Episode {self.episodes_done} saved on exit.")
                self.recording = False
            self.running = False
            return

        if action == "record":
            self.recording = not self.recording
            print("RECORDING ON" if self.recording else "RECORDING OFF")
            return

        if action == "end_episode":
            if self.recording:
                self.writer.end_episode()
                self.episodes_done += 1
                print(f"Episode {self.episodes_done} saved.")
                self.recording = False
            self._reset_episode()
            return

        if action == "reset":
            if self.recording:
                self.writer.discard_episode()
                self.recording = False
                print("Episode DISCARDED.")
            self._reset_episode()
            return

        if action is None:
            return

        handle_teleop_key(
            action,
            self.data,
            self.model,
            MOCAP_INDEX,
            self.act_id[JOINT_NAMES[-1]],
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_bgr(self, camera_name: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera_name)
        return cv2.cvtColor(
            self.renderer.render(),
            cv2.COLOR_RGB2BGR,
        )

    def _compose_views(self) -> np.ndarray:
        images = {cam: self._render_bgr(cam) for cam in CAMERA_NAMES}
        return compose_camera_views(images, CAMERA_NAMES)

    def _update_lift_status(self) -> None:
        cube_z = float(self._get_cube_state()[2])
        self.latest_cube_lift_m = cube_z - self._cube_start_z
        self.latest_success = (
            self.latest_cube_lift_m >= self.success_lift_delta_m
        )

    def _overlay_status(self, img_bgr: np.ndarray) -> np.ndarray:
        img = img_bgr.copy()

        status = (
            f"{'REC' if self.recording else 'IDLE'} | "
            f"ep {self.episodes_done} | "
            f"lift {self.latest_cube_lift_m * 1000:.1f} mm | "
            f"success {self.latest_success}"
        )
        cv2.putText(
            img,
            status,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
        )

        hint = "SPACE rec | ENTER save/reset | R discard/reset | ESC quit"
        cv2.putText(
            img,
            hint,
            (10, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        return img

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        last = time.perf_counter()
        try:
            while self.running:
                k_raw = cv2.waitKeyEx(1)
                if k_raw != -1:
                    k_ascii = k_raw & 0xFF
                    self._handle_key(k_raw, k_ascii)

                now = time.perf_counter()
                dt = now - last
                if dt < self.dt_ctrl:
                    time.sleep(self.dt_ctrl - dt)
                last = time.perf_counter()

                if self.recording:
                    self._record_step()

                for _ in range(self.substeps):
                    mujoco.mj_step(self.model, self.data)

                self._update_lift_status()

                img = self._overlay_status(self._compose_views())
                cv2.imshow(self.window_name, img)

        finally:
            self._finalize_on_exit()
            self.writer.flush()
            cv2.destroyAllWindows()
            print(
                f"Flushed buffers. "
                f"{self.episodes_done} episode(s) saved. Done."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record local SO101 grasp demonstrations."
    )
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_SCENE_XML,
    )
    parser.add_argument(
        "--keymap-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--cube-randomization-width-m",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--success-lift-delta-m",
        type=float,
        default=0.005,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    if not args.xml.exists():
        raise FileNotFoundError(
            f"Teleop XML not found: {args.xml}\n"
            "Run prepare_local_grasp_teleop_assets.py first."
        )

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = DEFAULT_OUTPUT_ROOT / ts
    out = run_dir / "so101_local_grasp_teleop.zarr"

    SO101LocalGraspTeleopRecorder(
        xml_path=args.xml,
        out_zarr=out,
        control_hz=args.control_hz,
        render_w=640,
        render_h=480,
        keymap_path=args.keymap_json,
        cube_randomization_width_m=args.cube_randomization_width_m,
        success_lift_delta_m=args.success_lift_delta_m,
        seed=args.seed,
    ).run()


if __name__ == "__main__":
    main()
