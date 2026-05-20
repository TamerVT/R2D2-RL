"""Environment factory for SO101 simulation.

Keeps scene/env creation in one place and delegates camera setup to camera_defs.py.
"""

from __future__ import annotations


import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any
PROJECT_ROOT = Path.cwd()
SCENE_XML = PROJECT_ROOT / "assets" / "scene_linker.xml"

print(PROJECT_ROOT)
from randomize_Cube_pos import Workspace2D, randomize_cube_positions

import numpy as np
import mujoco
from rcs.envs.configs import EmptyWorldSO101
from rcs.envs.base import ControlMode, RelativeTo


ws = Workspace2D(
    x_range=(0.10, 0.30),
    y_range=(-0.15, 0.15),
    yaw_range=(-np.pi, np.pi),
)

cube_names = [
    "blue_cube_body", "green_cube_body", "orange_cube_body",
    "purple_cube_body", "red_cube_body", "yellow_cube_body",
]


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
                   with_cameras = True,
                   headless = True,
                   debug_print = False,
                   background_noise = True,
                   noise_low = 0.3,
                   noise_high = 0.7,
                   seed = None) -> SimBundle:

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
    cfg.headless = True

    env = scene.create_env(cfg)
    sim = env.get_wrapper_attr("sim")
    _randomize_background(sim, noise_low, noise_high, seed)
    #dump_compiled_xml(sim)
    m, d = get_m_and_d(sim)
    
    randomize_cube_positions(
        sim,
        cube_names,
        ws,
        surface_z=0.0,              # floor plane z (or table top z)
        base_body_name="robotbase", # your robot root in the compiled model
        robot_body_prefix="robot",  # matches your body names
        debug=False,
    )

    mujoco.mj_forward(m, d)

    randomize_lights(sim, n_lights=3, debug=debug_print)
    mujoco.mj_forward(m, d)
    
    # Robot handle
    robots = env.get_wrapper_attr("robot")
    robot = robots["robot"] if isinstance(robots, dict) else robots
    
    for i in range(m.nbody):
        name = m.body(i).name
        print(name, d.xpos[i])

    robot_cfg = cfg.robot_cfgs[0] if isinstance(cfg.robot_cfgs, list) else list(cfg.robot_cfgs.values())[0]
    print(robot_cfg)
    print("fields:", [a for a in dir(robot_cfg) if "pos" in a or "quat" in a or "pose" in a])

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

def get_m_and_d(sim):
    m = sim.model
    d = getattr(sim, "mjdata", None) or getattr(sim, "data", None)
    if d is None:
        raise RuntimeError("No mjData handle found on sim (expected mjdata or data).")
    return m, d

def make_renderer(sim, width=640, height=480):
    import mujoco
    m, d = get_m_and_d(sim)

    mujoco.mj_forward(m, d)
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



def randomize_lights(
    sim,
    n_lights: int = 3,          # Increased default to allow richer lighting environments
    target=(0.0, 0.0, 0.0),
    radius_xy=(0.2, 1.2),       # Widened to allow steep overhead lights or low grazing angles
    height=(0.5, 2.0),          # Expanded height ranges
    diffuse_range=(0.1, 0.7),   # Randomized overall brightness intensity per light
    ambient_range=(0.0, 0.08),  # Random ambient fill levels
    specular_range=(0.01, 0.4), # Highly variable glare/shiny spots on the cubes
    allow_spotlights=True,      # Toggles between spotlights and directional/point lights
    seed: int | None = None,
    debug: bool = False,
):
    import numpy as np
    import mujoco
    
    m = sim.model
    d = sim.data if hasattr(sim, "data") else sim.mjdata
    rng = np.random.default_rng(seed)

    if m.nlight == 0:
        print("[randomize_lights] WARNING: Model has 0 lights. Nothing to randomize.")
        return

    # 0) Completely kill the built-in ambient headlight 
    if hasattr(m, "vis") and hasattr(m.vis, "headlight"):
        m.vis.headlight.active = 0
        m.vis.headlight.ambient[:] = 0.0
        m.vis.headlight.diffuse[:] = 0.0

    # 1) Clear out previous light state entirely
    m.light_active[:] = 0

    # 2) Pick a random subset of lights to activate
    n = min(n_lights, m.nlight)
    idxs = np.arange(m.nlight)
    rng.shuffle(idxs)
    idxs = idxs[:n]

    tgt = np.array(target, dtype=float)

    # 3) Define distinct color profiles (Daylight, Warm Incandescent, High-Contrast Monochromatic)
    color_profiles = [
        np.array([1.0, 0.95, 0.88]), # Soft Warm Sunlight
        np.array([0.85, 0.92, 1.0]), # Cool Skylight / Laboratory LED
        np.array([1.0, 0.80, 0.50]), # Heavy Tungsten / Halogen
    ]

    for i in idxs:
        # A. Position Sampling (3D cylindrical coordinates)
        r = rng.uniform(*radius_xy)
        ang = rng.uniform(0, 2 * np.pi)
        pos = np.array([r * np.cos(ang), r * np.sin(ang), rng.uniform(*height)], dtype=float)

        # B. Direction vectors pointing directly at workspace center with noise
        direction = (tgt - pos)
        direction /= (np.linalg.norm(direction) + 1e-8)
        # Add slight pointing deviations so shadows aren't perfectly symmetrical
        direction += rng.uniform(-0.05, 0.05, size=3)
        direction /= np.linalg.norm(direction)

        # C. Radical Color & Intensity Randomization
        # 15% chance of a highly saturated purely random color tint, otherwise choose a realistic profile
        if rng.random() < 0.15:
            base_color = rng.uniform(0.4, 1.0, size=3)
        else:
            base_color = rng.choice(color_profiles)
        
        color_jitter = rng.uniform(0.8, 1.2, size=3)
        final_color = np.clip(base_color * color_jitter, 0.0, 1.0)

        # Sample customized independent scalar intensifiers per light property
        diffuse_scale = rng.uniform(*diffuse_range)
        ambient_scale = rng.uniform(*ambient_range)
        specular_scale = rng.uniform(*specular_range)

        # Write fundamental properties to the model instance
        m.light_pos[i] = pos
        m.light_dir[i] = direction
        m.light_diffuse[i] = diffuse_scale * final_color
        m.light_ambient[i] = ambient_scale * final_color
        m.light_specular[i] = specular_scale * final_color
        m.light_active[i] = 1

        # D. Dynamic Light Type Randomization (Directional vs. Point vs. Spot)
        # Toggle shadow casting properties randomly
        m.light_castshadow[i] = 1 if rng.random() < 0.85 else 0

        # Decide whether it acts as a global sun, local bulb, or sharp spotlight cone
        light_type_roll = rng.random()
        
        if light_type_roll < 0.35:
            # Type 1: Directional light (Infinite distance sun effect)
            m.light_directional[i] = 1
        elif light_type_roll < 0.70 or not allow_spotlights:
            # Type 2: Omnidirectional Point Light (Local bulb with distance attenuation)
            m.light_directional[i] = 0
            # Set attenuation coefficients [constant, linear, quadratic] 
            # Forces light to decay over distance, creating beautiful localized gradients
            m.light_attenuation[i] = np.array([1.0, rng.uniform(0.5, 2.0), rng.uniform(0.2, 1.5)])
            m.light_cutoff[i] = 180.0 # Open up full 180 sphere
        else:
            # Type 3: Spotlight (Sharp directional cone with custom cutoff drop-off)
            m.light_directional[i] = 0
            m.light_attenuation[i] = np.array([1.0, rng.uniform(0.2, 1.0), 0.0])
            # Cutoff defines cone angle (in degrees). Smaller means narrower spotlight beam.
            m.light_cutoff[i] = rng.uniform(15.0, 45.0)
            # Exponent defines how fast the spotlight blurs/fades toward its outer edge
            m.light_exponent[i] = rng.uniform(5.0, 40.0)

        if debug:
            type_str = "DIR" if m.light_directional[i] else ("SPOT" if m.light_cutoff[i] < 90 else "POINT")
            print(f"[randomize_lights] Light[{i}] Type={type_str} | Pos={pos} | Shadows={m.light_castshadow[i]}")

    # Synchronize tracking and physics properties before camera snapshot reads
    mujoco.mj_forward(m, d)

from dataclasses import dataclass, field
from typing import Optional
import time
import threading


@dataclass
class MujocoGuiHandle:
    """Handle for a MuJoCo passive viewer attached to *the same* mjModel/mjData as RCS."""
    viewer: Any
    fps: Optional[float] = None  # if set and background=True, refresh rate
    _thread: Optional[threading.Thread] = None
    _stop: threading.Event = field(default_factory=threading.Event)

    def sync(self) -> None:
        """Render latest state. Call this after env.step() for deterministic updates."""
        v = self.viewer
        if v is None:
            return

        # Some viewers expose sync(), others render()
        if hasattr(v, "sync"):
            v.sync()
        elif hasattr(v, "render"):
            v.render()

    def close(self) -> None:
        """Stop background refresh and close the viewer if supported."""
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if hasattr(self.viewer, "close"):
            try:
                self.viewer.close()
            except Exception:
                pass


def open_mujoco_gui_from_rcs(
    env_or_sim: Any,
    *,
    background: bool = False,
    fps: float = 60.0,
    show_left_ui: bool = True,
    show_right_ui: bool = True,
) -> MujocoGuiHandle:
    """
    Explicitly open the MuJoCo GUI (viewer) attached to RCS' *actual* mjModel/mjData.

    - If you never call this, no GUI opens.
    - If background=False (recommended), call gui.sync() after each env.step().
    - If background=True, a refresh thread calls gui.sync() at `fps`.

    Parameters
    ----------
    env_or_sim:
        Either the RCS env (bundle.env) or the sim wrapper (bundle.sim).
    background:
        If True, refresh viewer in a background thread at `fps`.
    fps:
        Refresh rate for background mode.
    show_left_ui / show_right_ui:
        Pass-through UI toggles for MuJoCo viewer (if supported by your mujoco version).

    Returns
    -------
    MujocoGuiHandle
    """
    # Lazy import: ensures no viewer code runs unless this function is called
    import mujoco
    import mujoco.viewer

    # Accept either env or sim; in your factory you use env.get_wrapper_attr("sim")
    sim = env_or_sim
    if hasattr(env_or_sim, "get_wrapper_attr"):
        try:
            sim = env_or_sim.get_wrapper_attr("sim")
        except Exception:
            sim = env_or_sim

    m, d = get_m_and_d(sim)  # uses sim.model and sim.data/mjdata [1](https://ethz-my.sharepoint.com/personal/luppf_ethz_ch/Documents/Microsoft%20Copilot%20Chat%20Files/env_factory.py)

    # IMPORTANT: passive viewer renders the given mjModel/mjData (no internal stepping),
    # so it will always show the exact state RCS updates.
    viewer = mujoco.viewer.launch_passive(
        m, d,
        show_left_ui=show_left_ui,
        show_right_ui=show_right_ui,
    )

    gui = MujocoGuiHandle(viewer=viewer, fps=(fps if background else None))

    if background:
        # Background refresh loop.
        # NOTE: if your simulation steps in another thread, consider locking (see below).
        def _loop():
            dt = 1.0 / max(1e-6, fps)
            while not gui._stop.is_set():
                # If viewer provides a lock(), use it while syncing for thread-safety.
                if hasattr(viewer, "lock"):
                    with viewer.lock():
                        gui.sync()
                else:
                    gui.sync()
                time.sleep(dt)

        gui._thread = threading.Thread(target=_loop, daemon=True)
        gui._thread.start()

    return gui


def step_env_with_gui(env: Any, action: Any, gui: Optional[MujocoGuiHandle] = None):
    """
    Convenience wrapper: steps env and syncs GUI, with viewer locking if available.
    Use this for the simplest always-up-to-date GUI.

    Example:
        gui = open_mujoco_gui_from_rcs(env)
        obs, rew, term, trunc, info = step_env_with_gui(env, act, gui)
    """
    if gui is None or gui.viewer is None:
        return env.step(action)

    v = gui.viewer
    if hasattr(v, "lock"):
        with v.lock():
            out = env.step(action)
            gui.sync()
            return out
    else:
        out = env.step(action)
        gui.sync()
        return out        

if __name__ == "__main__":
    bundle = make_so101_sim(with_cameras=True, headless=False, debug_print=True)
    # Example: render one frame if the camera exists
    if "robotwrist" in bundle.camera_names:
        
        renderer = make_renderer(bundle.sim)
        
        img = render_camera(renderer, bundle.sim, "robotwrist")
        
        open_mujoco_gui_from_rcs(bundle.env, background=True, fps=60.0)
        
        import matplotlib.pyplot as plt
        plt.imshow(img)
        plt.axis("off")
        plt.show()
        
        print(img.min(), img.max(), img.mean())

        print("Rendered wrist_cam frame:", img.shape, img.dtype)
