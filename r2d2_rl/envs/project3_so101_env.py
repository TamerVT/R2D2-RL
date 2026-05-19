"""Project 3 RCS-based SO101 sim environment.

Extends ``rcs.envs.configs.EmptyWorldSO101`` with:

- a wrist camera attached to the SO101 gripper attachment site;
- one or more colored cubes placed on the floor in front of the arm;
- ``headless=True`` by default so the env can be created in WSL2 / CI without
  a display.

The cube assets reuse ``rcs.OBJECT_PATHS["green_cube"]`` for the green block.
Red / blue / yellow blocks are loaded from in-tree MJCFs under
``envs/assets/cubes/`` because RCS does not ship those colors out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rcs._core.sim import CameraType, SimCameraConfig
from rcs.envs.configs import EmptyWorldSO101
from rcs.envs.scenes import CameraAdderConfig, SimEnvCreatorConfig

import rcs

CUBE_ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "cubes"

# Map color name -> object MJCF path. Falls back to the in-tree red/blue/yellow
# MJCFs we add alongside RCS's green_cube.
DEFAULT_CUBE_PATHS: dict[str, str] = {
    "green": rcs.OBJECT_PATHS["green_cube"],
    "red": str(CUBE_ASSETS_DIR / "red_cube.xml"),
    "blue": str(CUBE_ASSETS_DIR / "blue_cube.xml"),
    "yellow": str(CUBE_ASSETS_DIR / "yellow_cube.xml"),
}


@dataclass(kw_only=True)
class CubeSpec:
    """One cube to drop into the scene at a fixed table-plane position."""

    color: str
    xy: tuple[float, float]
    z: float = 0.02
    yaw: float = 0.0


@dataclass(kw_only=True)
class Project3SO101Config:
    """Project 3 specific overrides on top of :class:`EmptyWorldSO101`.

    Defaults reproduce a simple Eval-1 scene: a single green cube placed in
    front of the SO101 with a wrist camera enabled.
    """

    cubes: list[CubeSpec] = field(
        default_factory=lambda: [CubeSpec(color="green", xy=(0.21, -0.03), z=0.02)]
    )
    wrist_camera_resolution: tuple[int, int] = (640, 480)
    wrist_camera_fovy: float = 55.0
    wrist_camera_offset: rcs.common.Pose | None = None
    headless: bool = True


class Project3SO101Env(EmptyWorldSO101):
    """SO101 scene with wrist camera + colored cubes for Project 3."""

    def __init__(self, p3_cfg: Project3SO101Config | None = None) -> None:
        super().__init__()
        self.p3_cfg = p3_cfg or Project3SO101Config()

    def config(self) -> SimEnvCreatorConfig:  # type: ignore[override]
        cfg = super().config()

        width, height = self.p3_cfg.wrist_camera_resolution
        cfg.camera_cfgs = {
            "wrist": SimCameraConfig(
                identifier="wrist",
                type=CameraType.fixed,
                resolution_width=width,
                resolution_height=height,
                frame_rate=30,
            ),
        }

        wrist_offset = self.p3_cfg.wrist_camera_offset
        if wrist_offset is None:
            # Sit a few cm above and forward of the SO101 gripper attachment
            # site. The yaw bias compensates for the attachment-site frame so
            # the default cube at (0.21, -0.03) appears near the wrist-view
            # center after the scripted pre-render lift.
            wrist_offset = rcs.common.Pose(
                translation=np.array([0.04, 0.0, 0.06]),
                rpy_vector=np.array([np.pi / 2, 0.0, -0.8]),
            )
        cfg.camera_adds = {
            "wrist": CameraAdderConfig(
                fovy=self.p3_cfg.wrist_camera_fovy,
                offset=wrist_offset,
                robot_name="robot",
            ),
        }

        cube_objects: dict[str, tuple[str, rcs.common.Pose]] = {}
        for idx, cube in enumerate(self.p3_cfg.cubes):
            xml_path = DEFAULT_CUBE_PATHS.get(cube.color)
            if xml_path is None:
                raise ValueError(f"No MJCF registered for cube color '{cube.color}'.")
            half_yaw = cube.yaw / 2.0
            cube_objects[f"cube_{cube.color}_{idx}"] = (
                xml_path,
                rcs.common.Pose(
                    translation=np.array([cube.xy[0], cube.xy[1], cube.z]),
                    quaternion=np.array([0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)]),
                ),
            )
        cfg.world_frame_objects = cube_objects

        cfg.headless = self.p3_cfg.headless
        return cfg


def make_env(p3_cfg: Project3SO101Config | None = None) -> Any:
    """Convenience: build and return a ready-to-step Gym env."""
    scene = Project3SO101Env(p3_cfg)
    return scene.create_env(scene.config())
