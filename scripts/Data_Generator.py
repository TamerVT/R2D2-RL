# -*- coding: utf-8 -*-
"""
Created on Wed May 20 19:30:53 2026

@author: felix
"""

import os
import sys
import numpy as np
import cv2
import mujoco

# 1. Get the directory where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Find the parent directory and append it to the path
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from env.env_factory import make_so101_sim
from env.Visual_Preprocessing import ImagePreprocessor

class SpatialDataGenerator:
    def __init__(self, target_camera="robotwrist", output_dir="collected_data"):
        """Initializes MuJoCo simulation environment and tracking targets."""
        print("====== Initializing Spatial Data Generator ======")
        self.bundle = make_so101_sim(with_cameras=True, headless=True, debug_print=False)
        self.sim = self.bundle.sim
        self.model = self.sim.model
        self.data = self.sim.data if hasattr(self.sim, "data") else self.sim.mjdata
        
        # Verify target camera asset
        self.camera_name = target_camera
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if self.cam_id == -1:
            available_cams = [self.model.camera(i).name for i in range(self.model.ncam)]
            if available_cams:
                self.camera_name = available_cams[0]
                self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
            else:
                raise ValueError("Fatal: No cameras found in the simulation model.")

        # Identify End Effector Body for tracking
        ee_candidates = ["end_effector", "hand", "gripper", "link_6", "flange"]
        self.ee_body_name = None
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if any(c in name.lower() for c in ee_candidates):
                self.ee_body_name = name
                break
        if not self.ee_body_name:
            self.ee_body_name = self.model.body(self.model.nbody - 1).name
        
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)
        print(f"✓ Connected to camera target: '{self.camera_name}'")
        print(f"✓ Target end-effector body mapped to: '{self.ee_body_name}'")

        # Processors and file path management
        self.preprocessor = ImagePreprocessor(target_size=(320, 320))
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self.output_dir = os.path.expanduser("~/RL_Proj/collected_data")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.cube_names = [self.model.body(i).name for i in range(self.model.nbody) 
                           if "cube_body" in self.model.body(i).name]
        print(f"✓ Found {len(self.cube_names)} sample target cubes in workspace.")
        print("=================================================\n")

    def get_cube_transform(self, cube_name):
        """Extracts global position and quaternion orientation of a cube."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
        pos = self.data.xpos[body_id].copy()
        
        rot_mat = self.data.xmat[body_id].reshape(3, 3)
        quat_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(quat_wxyz, rot_mat.flatten())
        
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        return pos, quat_xyzw

    def _solve_ik_with_upright_bias(self, target_pos, max_steps=200, tol=1e-4):
        """Drives the EE to target_pos while keeping the gripper horizontal."""
        target_rot = np.eye(3) 
        damping = 1e-3
        step_size = 0.2
        
        for step in range(max_steps):
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            
            current_pos = self.data.xpos[self.ee_id]
            pos_err = target_pos - current_pos
            
            current_rot = self.data.xmat[self.ee_id].reshape(3, 3)
            rot_err_mat = target_rot @ current_rot.T
            rot_err = np.array([
                rot_err_mat[2, 1] - rot_err_mat[1, 2],
                rot_err_mat[0, 2] - rot_err_mat[2, 0],
                rot_err_mat[1, 0] - rot_err_mat[0, 1]
            ]) * 0.5
            
            error_6d = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(error_6d) < tol:
                return True
                
            jac_p = np.zeros((3, self.model.nv))
            jac_r = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jac_p, jac_r, self.ee_id)
            jac_6d = np.vstack([jac_p, jac_r])
            
            delta_q = jac_6d.T @ np.linalg.solve(jac_6d @ jac_6d.T + damping * np.eye(6), error_6d)
            mujoco.mj_integratePos(self.model, self.data.qpos, delta_q * step_size, 1.0)
            
        return False

    def generate_samples(self, num_sequences=10, samples_per_sequence=5, noise_std=0.02):
        """Generates visual network dataset matching the required input/label shapes."""
        sample_count = 0
        metadata_records = []
        
        print(f"-> Commencing data generation. Target: {self.output_dir}")
        
        # FIX: Capture baseline joint coordinates directly instead of calling mj_getState bitmasks
        mujoco.mj_forward(self.model, self.data)
        baseline_qpos = self.data.qpos.copy()
        baseline_qvel = self.data.qvel.copy()
        
        for seq in range(num_sequences):
            target_cube = np.random.choice(self.cube_names)
            
            for s in range(samples_per_sequence):
                # FIX: Explicitly restore joint properties back to neutral configuration state
                self.data.qpos[:] = baseline_qpos
                self.data.qvel[:] = baseline_qvel
                mujoco.mj_forward(self.model, self.data)
                
                # Fetch cube positions from the fresh, reset coordinate setup
                cube_world_xyz, cube_world_xyzw = self.get_cube_transform(target_cube)
                anchor_pos = cube_world_xyz + np.array([0.0, 0.0, 0.15]) # 5cm above
                
                # Sample a localized tracking position around our 5cm target anchor
                target_pos = anchor_pos + np.random.uniform(-0.01, 0.01, size=3)
                
                # Execute Inverse Kinematics calculations
                ik_success = self._solve_ik_with_upright_bias(target_pos)
                
                # Force a full geometric pipeline update to align model and scene frames
                mujoco.mj_forward(self.model, self.data)
                
                # Extract finalized structural tracking variables
                ee_world_pos = self.data.xpos[self.ee_id].copy()
                ee_world_rot = self.data.xmat[self.ee_id].reshape(3, 3).copy()
                
                # Transform true cube coordinates into local End-Effector Space
                cube_local_xyz = ee_world_rot.T @ (cube_world_xyz - ee_world_pos)
                
                # Transform orientation quaternion into local End-Effector space
                ee_quat_wxyz = np.zeros(4)
                mujoco.mju_mat2Quat(ee_quat_wxyz, ee_world_rot.flatten())
                cube_quat_wxyz = np.zeros(4)
                mujoco.mju_mat2Quat(cube_quat_wxyz, self.data.xmat[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, target_cube)].flatten())
                
                ee_quat_inv = np.array([ee_quat_wxyz[0], -ee_quat_wxyz[1], -ee_quat_wxyz[2], -ee_quat_wxyz[3]])
                cube_local_wxyz = np.zeros(4)
                mujoco.mju_mulQuat(cube_local_wxyz, ee_quat_inv, cube_quat_wxyz)
                cube_local_xyzw = np.array([cube_local_wxyz[1], cube_local_wxyz[2], cube_local_wxyz[3], cube_local_wxyz[0]])
                
                # Inject tracking noise into true world-space coordinates for visual network input
                noise_xy = np.random.normal(0, noise_std, size=2)
                noisy_input_xy = cube_world_xyz[:2] + noise_xy
                
                # Update the graphics buffer frame and read processed matrix views
                self.renderer.update_scene(self.data, camera=self.camera_name)
                #edge_map_post = self.renderer.render()
                rendered_frame = self.renderer.render()
                edge_map_post = self.preprocessor.process(rendered_frame)
                
                # Save processed visual frame
                image_filename = f"sample_{sample_count:05d}_edge.png"
                cv2.imwrite(os.path.join(self.output_dir, image_filename), edge_map_post)
                
                true_relative_xyzw = np.concatenate([cube_local_xyz, cube_local_xyzw])
                
                record = {
                    "sample_id": sample_count,
                    "image_file": image_filename,
                    "input_noisy_xy": noisy_input_xy,
                    "label_relative_xyzw": true_relative_xyzw,
                    "ik_converged": int(ik_success)
                }
                metadata_records.append(record)
                
                status_str = "converged" if ik_success else "failed"
                print(f"  ↳ Sample {sample_count:05d} | Cube: {target_cube} | Offset Step: {s+1}/5 | IK: {status_str}")
                sample_count += 1
                
        np.save(os.path.join(self.output_dir, "visual_training_dataset.npy"), metadata_records)
        print(f"✓ Success. {sample_count} frames recorded safely.")

if __name__ == "__main__":
    generator = SpatialDataGenerator(target_camera="robotwrist")
    generator.generate_samples(num_sequences=4, samples_per_sequence=5)