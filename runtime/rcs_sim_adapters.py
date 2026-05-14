"""RCS-backed adapters for the hybrid Project 3 runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from control.waypoint_controller import RcsWaypointController
from estimation.block_belief import BlockBelief, BlockBeliefTracker
from estimation.pixel_to_table import PixelToTableProjector, TablePointEstimate
from perception.color_block_detector import ColorBlockDetector, Detection2D
from planning.hybrid_waypoint_planner import Waypoint


def wrist_frame_from_obs(obs: dict[str, Any], camera_name: str = "wrist") -> dict[str, Any] | None:
    frames = obs.get("frames") if isinstance(obs, dict) else None
    if not isinstance(frames, dict):
        return None
    camera = frames.get(camera_name)
    if not isinstance(camera, dict):
        return None
    rgb = camera.get("rgb")
    return rgb if isinstance(rgb, dict) else None


def wrist_rgb_from_obs(obs: dict[str, Any], camera_name: str = "wrist") -> np.ndarray | None:
    frame = wrist_frame_from_obs(obs, camera_name=camera_name)
    if frame is None:
        return None
    data = frame.get("data")
    if isinstance(data, np.ndarray):
        return data
    return None


def project_detection_to_table(
    detection: Detection2D,
    frame: dict[str, Any],
    config: dict[str, Any],
) -> TablePointEstimate:
    """Project an RCS wrist-camera detection to base-frame table coordinates.

    Builds a :class:`PixelToTableProjector` directly from the RCS-supplied
    ``frame["intrinsics"]`` (3x4) and ``frame["extrinsics"]`` (world-to-cam),
    so the geometric math lives in one place — the projector. The classmethod
    keeps a small in-process cache keyed on intrinsics + image shape so we
    don't rebuild the projector on every detection.
    """
    K_full = np.asarray(frame["intrinsics"], dtype=np.float64)
    K = K_full[:3, :3]
    T_C_B = np.asarray(frame["extrinsics"], dtype=np.float64).reshape(4, 4)
    T_B_C = np.linalg.inv(T_C_B)
    height, width = _rgb_shape(frame)

    projector = _projector_for_rcs(K, width, height, config)
    return projector.project_from_T_BC(
        uv=detection.centroid_uv,
        T_B_C=T_B_C,
        covariance_uv=detection.covariance_uv,
    )


_RCS_PROJECTOR_CACHE: dict[tuple, PixelToTableProjector] = {}


def _projector_for_rcs(
    K: np.ndarray,
    width: int,
    height: int,
    config: dict[str, Any],
) -> PixelToTableProjector:
    workspace = config.get("workspace") or {}
    uncertainty = config.get("uncertainty") or {}
    cache_key = (
        K.tobytes(),
        int(width),
        int(height),
        float(workspace.get("z_object_center", 0.02)),
        tuple(workspace.get("bounds_xy", {}).get("x", [-np.inf, np.inf])),
        tuple(workspace.get("bounds_xy", {}).get("y", [-np.inf, np.inf])),
        float(uncertainty.get("calibration_sigma_xy", 0.0)),
    )
    cached = _RCS_PROJECTOR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    projector_config = {
        "camera": {"K": K.tolist(), "width": int(width), "height": int(height)},
        "workspace": workspace,
        "uncertainty": uncertainty,
        # transforms.T_E_C is omitted; only project_from_T_BC is used.
    }
    projector = PixelToTableProjector(projector_config)
    _RCS_PROJECTOR_CACHE[cache_key] = projector
    return projector


def _rgb_shape(frame: dict[str, Any]) -> tuple[int, int]:
    data = frame.get("data")
    if isinstance(data, np.ndarray) and data.ndim >= 2:
        return int(data.shape[0]), int(data.shape[1])
    return 0, 0


@dataclass
class RcsWristBlockObserver:
    """Observe a target color through the RCS wrist camera and update belief."""

    controller: RcsWaypointController
    config: dict[str, Any]
    camera_name: str = "wrist"
    measurements_per_observe: int = 5

    def __post_init__(self) -> None:
        self.detector = ColorBlockDetector(self.config)
        self.tracker = BlockBeliefTracker(self.config)
        self.last_detection: Detection2D | None = None
        self.last_estimate: TablePointEstimate | None = None
        if self.measurements_per_observe < 1:
            raise ValueError("measurements_per_observe must be >= 1.")

    def observe(self, target_color: str) -> BlockBelief | None:
        latest_belief: BlockBelief | None = None
        saw_detection = False

        for sample_idx in range(self.measurements_per_observe):
            if self.controller.last_obs is None:
                return latest_belief
            frame = wrist_frame_from_obs(self.controller.last_obs, self.camera_name)
            if frame is None:
                return latest_belief
            rgb = frame.get("data")
            if not isinstance(rgb, np.ndarray):
                return latest_belief
            detections = self.detector.detect(rgb, target_color=target_color)
            if detections:
                saw_detection = True
                detection = detections[0]
                estimate = project_detection_to_table(detection, frame, self.config)
                self.last_detection = detection
                self.last_estimate = estimate
                if estimate.valid:
                    latest_belief = self.tracker.update(
                        target_color,
                        estimate.xy_base,
                        estimate.covariance_xy,
                        timestamp=float(self.controller.step_count),
                        confidence=detection.confidence,
                    )

            if sample_idx < self.measurements_per_observe - 1:
                # ``gripper=None`` tells the controller to preserve the current
                # gripper state. Forcing 1.0 (open) would drop a held cube if
                # the watchdog re-observes mid-transport.
                self.controller.step_delta(np.zeros(3), gripper=None)

        if not saw_detection:
            self.last_detection = None
            self.last_estimate = None
        return latest_belief


@dataclass
class RcsColorVisibilityChecker:
    """Visibility watchdog using the current RCS wrist-camera frame."""

    controller: RcsWaypointController
    config: dict[str, Any]
    camera_name: str = "wrist"

    def __post_init__(self) -> None:
        self.detector = ColorBlockDetector(self.config)

    def is_visible(self, target_color: str) -> bool:
        if self.controller.last_obs is None:
            return False
        rgb = wrist_rgb_from_obs(self.controller.last_obs, self.camera_name)
        if rgb is None:
            return False
        return bool(self.detector.detect(rgb, target_color=target_color))


@dataclass
class ScriptedAlignGraspPolicy:
    """Temporary local align/grasp policy for end-to-end sim plumbing.

    This is not the final learned policy. It gives the executor a concrete
    policy adapter today by moving to the estimated grasp point and closing the
    gripper. A trained RL policy should later implement the same ``run`` method.
    """

    controller: RcsWaypointController
    config: dict[str, Any]

    def __post_init__(self) -> None:
        planning = self.config.get("planning") or {}
        self.z_grasp = float(planning.get("z_grasp", 0.025))
        self.rpy_base = np.asarray(planning.get("grasp_orientation_rpy", [np.pi, 0.0, 0.0]), dtype=np.float64)
        control = self.config.get("control") or {}
        self.timeout_s = float(control.get("waypoint_timeout_s", 5.0))

    def run(self, phase: str, target_color: str, belief: BlockBelief) -> bool:
        if phase != "align_grasp":
            return False
        xy = np.asarray(belief.mean_xy, dtype=np.float64).reshape(2)
        align = Waypoint(
            name="scripted_align",
            xyz_base=np.array([xy[0], xy[1], self.z_grasp], dtype=np.float64),
            rpy_base=self.rpy_base,
            gripper=1.0,
            timeout_s=self.timeout_s,
            metadata={"target_color": target_color, "policy": "scripted"},
        )
        close = Waypoint(
            name="scripted_grasp_close",
            xyz_base=align.xyz_base.copy(),
            rpy_base=self.rpy_base,
            gripper=0.0,
            timeout_s=self.timeout_s,
            metadata={"target_color": target_color, "policy": "scripted"},
        )
        return self.controller.execute((align, close))


