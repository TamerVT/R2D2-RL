# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:58:56 2026

@author: felix
"""

from env_factory import build_so101_env
import mujoco
import numpy as np

class Environment:
    def __init__(self, render_scene, render_roboteye):
        # 1. Bind core MuJoCo C-pointers to the instance
        self.model, self.data = build_so101_env()
        
        # 2. Configure temporal tracking (Frequencies to explicit timesteps)
        self.policy_hz = 25                                # Neural network step rate (0.04s)
        self.model.opt.timestep = 1.0 / 1250               # Fixed internal physics rate (0.0008s -> 1250 Hz)
        self.substeps = int((1.0 / self.policy_hz) / self.model.opt.timestep) # Exactly 50 steps
        
        # 3. Handle placeholders
        self.renderer = None
        self._video_frames = []
        self.recording = False
        
        if render_scene or render_roboteye:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            # If roboteye is specified, lock the viewport onto that camera id
            if render_roboteye:
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'robotwrist') # hardcoded, do not change xml's!
                if cam_id != -1:
                    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    self.viewer.cam.fixedcamid = cam_id
          
    def randomize_background(self, low=0.3, high=0.7, seed=None):
        import numpy as np
        rng = np.random.default_rng(seed)
        if not hasattr(self.model, "tex_rgb"):
            return
        # overwrite entire texture buffer
        self.model.tex_rgb[:] = rng.uniform(low, high, size=self.model.tex_rgb.shape)
        
    def make_renderer(self, width=640, height=480):
        self.renderer = mujoco.Renderer(self.model, height, width)
        return self.renderer

    def render_camera(self, camera_name):
        """Captures a raw RGB frame from a specified camera in the scene graph."""
        if self.renderer is None:
            self.make_renderer()
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            
        if cam_id == -1:
            available_cams = [self.model.camera(i).name for i in range(self.model.ncam)]
            print("Camera not found. Avialable Cameras:", available_cams)
            return None
        # Synchronize renderer with current dynamic system state
        self.renderer.update_scene(self.data, camera=cam_id)
        return np.asarray(self.renderer.render())
    
    def randomize_lights(self,):
        pass # this is just a wrapper pointing to a seperate file, where the function is defined
        
    def update_scene(self):
        ...
    
    def simulate_scene(self):
        ...
    def reset_scene(self, randomize = False):
        # randomize is a list 
        ...
        
    def set_reset_state(self):
        ...
        
    def ik_solver(self):
        # maps end effector position to joint angle state vector
        ...
    def record_video(self):
        # records a video until the next reset
        ...
        
    def Cam_preprocessing(self):
        # preprocesses the image before passing to NN for better abstraction
        ... 
        
if __name__ == "__main__":
    env = Environment(render_scene=True, render_roboteye=None)
    input("Press enter to stop")

    