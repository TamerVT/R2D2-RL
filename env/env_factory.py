"""Environment factory for SO101 simulation.

Keeps scene/env creation in one place and delegates camera setup to camera_defs.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rcs
from rcs.envs.configs import EmptyWorldSO101
from rcs.envs.base import ControlMode, RelativeTo

# Local camera config helper (separate file).
from camera_config import apply_cameras, default_camera_specs

PROJECT_ROOT = Path.cwd()
SCENE_XML = PROJECT_ROOT / "assets" / "scene_linker.xml"


@dataclass
class SimBundle:
    env: Any          # RCS gym env
    sim: Any          # MuJoCo sim wrapper
    robot: Any        # robot handle
    camera_names: list[str]


def make_so101_sim(*,
                   with_cameras: bool = True,
                   headless: bool = False,
                   debug_print: bool = True) -> SimBundle:
    """Create SO101 environment and return handles.

    Parameters
    ----------
    with_cameras:
        If True, adds cameras through RCS config (camera_cfgs + camera_adds).
    headless:
        If True, disables GUI.
    debug_print:
        If True, prints bodies/cameras found after compilation.
    """

    scene = EmptyWorldSO101()
    cfg = scene.config()

    # Use project-owned scene linker XML (your composed scene assets).
    cfg.scene = str(SCENE_XML)

    # Control config
    cfg.control_mode = ControlMode.JOINTS
    cfg.relative_to = RelativeTo.LAST_STEP

    # Optional cameras via RCS config (preferred over MJCF edits).
    if with_cameras:
        apply_cameras(cfg, default_camera_specs(), include_depth=False)

    # Optional headless mode
    cfg.headless = headless

    env = scene.create_env(cfg)

    sim = env.get_wrapper_attr("sim")

    # Robot handle
    robots = env.get_wrapper_attr("robot")
    robot = robots["robot"] if isinstance(robots, dict) else robots

    # Names for debugging
    cam_names = [sim.model.camera(i).name for i in range(sim.model.ncam)]

    if debug_print:
        m = sim.model
        print("Bodies:", [m.body(i).name for i in range(m.nbody)])
        print("Sites :", [m.site(i).name for i in range(m.nsite)])
        print("Cams  :", cam_names)

    return SimBundle(env=env, sim=sim, robot=robot, camera_names=cam_names)


def render_camera(sim, camera_name: str, width: int = 640, height: int = 480):
    """Convenience wrapper to render a camera to an RGB numpy array."""
    # RCS sim wrapper typically forwards to MuJoCo renderer.
    return sim.render(width=width, height=height, camera_name=camera_name)


if __name__ == "__main__":
    bundle = make_so101_sim(with_cameras=True, headless=False, debug_print=True)
    # Example: render one frame if the camera exists
    if "wrist_cam" in bundle.camera_names:
        img = render_camera(bundle.sim, "wrist_cam", 640, 480)
        print("Rendered wrist_cam frame:", img.shape, img.dtype)
