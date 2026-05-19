"""Camera definitions for RCS scenes.

This module keeps camera-related config in one place so env_factory.py stays small.

RCS splits cameras into:
  * camera_cfgs: runtime properties (resolution, fps, type)
  * camera_adds: placement (fixed in root frame or mounted on robot attachment_site)

See RCS scene configuration guide for this mental model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from rcs._core.sim import SimCameraConfig, CameraType


# --- Compatibility imports (RCS versions may expose these in different modules) ---
try:
    from rcs.envs.scenes import CameraAdderConfig, WrapperConfig
except Exception:  # pragma: no cover
    # Fallback paths used by some versions
    from rcs.envs.base import CameraAdderConfig, WrapperConfig  # type: ignore

import rcs
try:
    from rcs.camera.sim import SimCameraConfig
except Exception:  # pragma: no cover
    # Older versions sometimes place this under rcs._core.sim or similar
    from rcs._core.sim import SimCameraConfig  # type: ignore


@dataclass(frozen=True)

class CameraSpec:
    name: str

    # Runtime
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    fovy: float = 60.0

    # Placement
    robot_name: Optional[str] = "robot"
    translation: Tuple[float, float, float] = (0.0, 0.0, 0.06)
    quaternion: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)



def default_camera_specs() -> list[CameraSpec]:
    """Default cameras for your SO101 scene.

    Start with one wrist camera. Add overhead cams later if needed.
    """
    return [
        CameraSpec(
            name="wrist_cam",
            resolution=(640, 480),
            fps=30,
            fovy=60.0,
            robot_name="robot",
            translation=np.array([-0.05730948863999999, 0.039625134080000006, 0.050967311359999996], dtype=float),
            quaternion=np.array([-0.2822836851883647, 0.6834130661436099, -0.29910590147044996, -0.6031568301894753], dtype=float),
        ),
    ]


def apply_cameras(cfg: Any,
                 specs: Iterable[CameraSpec],
                 *,
                 include_depth: bool = False) -> None:
    """Apply cameras to an RCS scene config.

    This populates cfg.camera_cfgs and cfg.camera_adds.

    If include_depth=True, configures wrapper to include depth.
    """

    # Runtime properties
    camera_cfgs: Dict[str, SimCameraConfig] = {}

    # Placement
    camera_adds: Dict[str, CameraAdderConfig] = {}

    for s in specs:
        w, h = s.resolution  # assuming you store (width, height)
        
        camera_cfgs[s.name] = SimCameraConfig(
            identifier=s.identifier if hasattr(s, "identifier") and s.identifier is not None else s.name,
            frame_rate=int(s.fps),
            resolution_width=int(w),
            resolution_height=int(h),
            type=CameraType.fixed,   # or make this configurable per CameraSpec
        )


        camera_adds[s.name] = CameraAdderConfig(
            fovy=s.fovy,
            robot_name=s.robot_name,
            offset=rcs.common.Pose(
                translation=np.asarray(s.translation, dtype=float),
                quaternion=np.asarray(s.quaternion, dtype=float),
            ),
        )

    cfg.camera_cfgs = camera_cfgs
    cfg.camera_adds = camera_adds

    # Ensure wrapper config exists and set depth flag
    if getattr(cfg, "wrapper_cfg", None) is None:
        cfg.wrapper_cfg = WrapperConfig()

    cfg.wrapper_cfg.include_depth = bool(include_depth)
    
    
    

