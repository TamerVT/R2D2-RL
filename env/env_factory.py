"""Environment factory for SO101 simulation.

Keeps scene/env creation in one place and delegates camera setup to camera_defs.py.
"""

from __future__ import annotations


import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rcs
from rcs.envs.configs import EmptyWorldSO101
from rcs.envs.base import ControlMode, RelativeTo

PROJECT_ROOT = Path.cwd()
SCENE_XML = PROJECT_ROOT / "assets" / "scene_linker.xml"


@dataclass
class SimBundle:
    env: Any          # RCS gym env
    sim: Any          # MuJoCo sim wrapper
    robot: Any        # robot handle
    camera_names: list[str]
    
class CustomScene(EmptyWorldSO101):
    def create_model(self, cfg):
        # 2. Bypass the hardcoded rcs XML generation and load your custom wrapper
        return str(SCENE_XML)

def make_so101_sim(*,
                   with_cameras: bool = True,
                   headless: bool = False,
                   debug_print: bool = True,
                   background_noise: bool = False,
                   noise_low: float = 0.3,
                   noise_high: float = 0.7,
                   seed: int | None = None) -> SimBundle:

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
    
    cfg.scene = str(SCENE_XML) # loads the linker xml file
    if isinstance(cfg.robot_cfgs, list) and len(cfg.robot_cfgs) > 0:
        # If it's a list, modify the first robot config entry
        cfg.robot_cfgs[0].xml_path = "assets/so101.xml"  # Replace with your actual custom file path
        print(f"Redirected robot_cfgs[0] path to: {cfg.robot_cfgs[0].xml_path}")
    elif isinstance(cfg.robot_cfgs, dict):
        # If it's a dictionary, modify the first entry
        first_key = list(cfg.robot_cfgs.keys())[0]
        robot_config = cfg.robot_cfgs[first_key]
        
        # 2. Assign your custom file path to the correct attribute
        robot_config.kinematic_model_path = "assets/so101.xml"
    # Control config
    cfg.control_mode = ControlMode.JOINTS
    cfg.relative_to = RelativeTo.LAST_STEP

    # Optional headless mode
    cfg.headless = headless

    env = scene.create_env(cfg)
    sim = env.get_wrapper_attr("sim")
    _randomize_background(sim, noise_low, noise_high, seed)
    
    m = sim.model
    
    add_lights(sim, n_lights=2, debug=debug_print)

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

# some helper functions

def _randomize_background(sim, low=0.3, high=0.7, seed=None):
    import numpy as np

    m = sim.model
    rng = np.random.default_rng(seed)

    if not hasattr(m, "tex_rgb"):
        return

    # overwrite entire texture buffer
    m.tex_rgb[:] = rng.uniform(low, high, size=m.tex_rgb.shape)




def make_renderer(sim, width=640, height=480):
    import mujoco
    return mujoco.Renderer(sim.model, height, width)


def render_camera(renderer, sim, camera_name):
    import mujoco

    m = sim.model

    # get MuJoCo data handle (robust)
    d = getattr(sim, "data", None)
    if d is None:
        d = getattr(sim, "mjdata", None)
    if d is None:
        raise RuntimeError("Could not find sim data")

    cam_id = mujoco.mj_name2id(
        m,
        mujoco.mjtObj.mjOBJ_CAMERA,
        camera_name
    )

    # IMPORTANT: same as calibration script
    mujoco.mj_forward(m, d)

    renderer.update_scene(d, camera=cam_id)

    img = renderer.render()
    return np.asarray(img)



def add_lights(sim,
               n_lights: int = 2,
               hue: tuple = (1.0, 0.95, 0.85),
               intensity: float = 1.2,
               randomize: bool = True,
               seed: int | None = None,
               debug: bool = False):

    import numpy as np

    m = sim.model
    rng = np.random.default_rng(seed)

    if m.nlight == 0:
        print("[add_lights] WARNING: Model has 0 lights. Cannot add at runtime.")
        return

    n_used = min(n_lights, m.nlight)

    for i in range(n_used):

        # --- position ---
        if randomize:
            pos = np.array([
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
                rng.uniform(0.8, 1.5)
            ])
        else:
            pos = np.array([0.0, 0.0, 1.2])

        m.light_pos[i] = pos

        # --- direction (toward table center) ---
        target = np.array([0.0, 0.0, 0.0])
        direction = target - pos
        direction /= (np.linalg.norm(direction) + 1e-8)

        m.light_dir[i] = direction

        # --- color ---
        base = np.array(hue)

        if randomize:
            jitter = rng.uniform(0.85, 1.15, size=3)
            color = np.clip(base * jitter * intensity, 0, 1)
        else:
            color = np.clip(base * intensity, 0, 1)

        # --- apply lighting ---
        m.light_diffuse[i] = color
        m.light_specular[i] = 0.2 * color

        # IMPORTANT: this replaces your missing global ambient
        m.light_ambient[i] = 0.3 * color

        # --- activate ---
        m.light_active[i] = 1

        if hasattr(m, "light_directional"):
            m.light_directional[i] = 1

        if debug:
            print(f"[add_lights] {i}: pos={pos}, color={color}")


        

if __name__ == "__main__":
    bundle = make_so101_sim(with_cameras=True, headless=False, debug_print=True)
    # Example: render one frame if the camera exists
    if "robotwrist" in bundle.camera_names:
        
        renderer = make_renderer(bundle.sim)
        
        img = render_camera(renderer, bundle.sim, "robotwrist")
        
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.axis("off")
        plt.show()

        print(img.min(), img.max(), img.mean())

        print("Rendered wrist_cam frame:", img.shape, img.dtype)
