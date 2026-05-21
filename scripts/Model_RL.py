import os
import random
import numpy as np
import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import cv2
import mujoco
from scipy.spatial.transform import Rotation as R
from pathlib import Path
# Import tracking handles directly from your local simulation factory
# 1. Dynamically locate the parent folder (RL_ProjFelix) relative to this file
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# 2. Append the project root to the python path if it isn't already there
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from env.env_factory import SimBundle, make_renderer, render_camera, get_m_and_d

# =============================================================================
# 1. SYMMETRIC CUBE ALIGNMENT UTILITY
# =============================================================================
class SymmetricCubeAligner:
    def __init__(self):
        self.symmetric_quats = []
        steps = [0, 90, 180, 270]
        unique_rotations = set()

        for x in steps:
            for y in steps:
                for z in steps:
                    r = R.from_euler('xyz', [x, y, z], degrees=True)
                    q = tuple(np.round(r.as_quat(), decimals=4))
                    unique_rotations.add(q)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sym_tensors = torch.tensor(list(unique_rotations), dtype=torch.float32, device=self.device)

    def get_closest_symmetry(self, q_target, q_eef):
        q_pred_norm = q_target / torch.norm(q_target, dim=1, keepdim=True)
        q_eef_norm = q_eef / torch.norm(q_eef, dim=1, keepdim=True)
        
        aligned_quats = []
        for b in range(q_pred_norm.shape[0]):
            qp = q_pred_norm[b] 
            qe = q_eef_norm[b]  
            
            candidates = self._quat_multiply_batch(qp, self.sym_tensors) 
            dot_products = torch.abs(torch.sum(candidates * qe, dim=1)) 
            
            best_idx = torch.argmax(dot_products)
            aligned_quats.append(candidates[best_idx])
            
        return torch.stack(aligned_quats, dim=0)

    def _quat_multiply_batch(self, q1, q2_batch):
        x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
        x2, y2, z2, w2 = q2_batch[:, 0], q2_batch[:, 1], q2_batch[:, 2], q2_batch[:, 3]
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return torch.stack([x, y, z, w], dim=1)

def solve_differential_ik(m, d, eef_body_name, cartesian_action):
    """
    Converts a 6D Cartesian action [dx, dy, dz, droll, dpitch, dyaw] 
    into Joint angle deltas using the MuJoCo Jacobian pseudo-inverse.
    """
    jacp = np.zeros((3, m.nv)) # Translation Jacobian
    jacr = np.zeros((3, m.nv)) # Rotation Jacobian
    
    eef_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, eef_body_name)
    
    # Calculate Jacobians for the current simulation state
    mujoco.mj_jacBody(m, d, jacp, jacr, eef_id)
    
    # Stack into a full 6xN matrix
    J = np.vstack((jacp, jacr))
    
    # Compute the pseudo-inverse (Damped least squares for stability)
    lambda_damp = 0.01 
    J_pinv = J.T @ np.linalg.inv(J @ J.T + (lambda_damp**2) * np.eye(6))
    
    # Calculate joint deltas
    delta_q = J_pinv @ cartesian_action
    
    return delta_q
# =============================================================================
# 2. EXPERIENCED REPLAY BUFFER
# =============================================================================
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (torch.FloatTensor(np.array(state)),
                torch.FloatTensor(np.array(action)),
                torch.FloatTensor(np.array(reward)).unsqueeze(1),
                torch.FloatTensor(np.array(next_state)),
                torch.FloatTensor(np.array(done)).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# 3. SAC NETWORKS (STATE-DIMENSION = 7)
# =============================================================================
class TwinCritic(nn.Module):
    def __init__(self, state_dim=7, action_dim=7, hidden_dim=128):
        super(TwinCritic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)


class GaussianActor(nn.Module):
    def __init__(self, state_dim=7, action_dim=7, hidden_dim=128):
        super(GaussianActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)
        return mu, log_std.exp()

    def sample_action(self, state):
        mu, std = self.forward(state)
        dist = Normal(mu, std)
        x_t = dist.rsample()  
        action = torch.tanh(x_t)  
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class SACAgent:
    def __init__(self, state_dim=7, action_dim=7, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha 

        self.actor = GaussianActor(state_dim, action_dim).to(self.device)
        self.critic = TwinCritic(state_dim, action_dim).to(self.device)
        self.critic_target = TwinCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if evaluate:
            mu, _ = self.actor(state)
            return torch.tanh(mu).detach().cpu().numpy()[0]
        else:
            action, _ = self.actor.sample_action(state)
            return action.detach().cpu().numpy()[0]

    def update_parameters(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
        
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.actor.sample_action(next_state_batch)
            target_q1, target_q2 = self.critic_target(next_state_batch, next_state_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * target_q

        curr_q1, curr_q2 = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(curr_q1, next_q_value) + F.mse_loss(curr_q2, next_q_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_pi = self.actor.sample_action(state_batch)
        q1_new, q2_new = self.critic(state_batch, new_action)
        expected_q = torch.min(q1_new, q2_new)
        actor_loss = ((self.alpha * log_pi) - expected_q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)


# =============================================================================
# 4. MULTIPHASE TIME-OPTIMIZED REWARD FUNCTION
# =============================================================================
def compute_time_optimized_multiphase_reward(
    true_cube_pos, true_cube_quat, current_eef_pos, current_eef_quat, 
    aligned_cube_quat, current_jaw_position, cube_initial_pos, cube_initial_quat
):
    OFFSET_TIP = 0.08  
    A_MAX = 0.15       
    LAMBDA_GATE = 3.0  
    
    dot_prod = np.clip(np.abs(np.sum(aligned_cube_quat * current_eef_quat)), 0.0, 1.0)
    theta_error = 2.0 * np.arccos(dot_prod)
    
    a = A_MAX * np.exp(-LAMBDA_GATE * theta_error)
    x_offs = current_eef_pos + (R.from_quat(current_eef_quat).as_matrix()[:, 2] * (OFFSET_TIP + a))
    
    physical_gripper_center = current_eef_pos + (R.from_quat(current_eef_quat).as_matrix()[:, 2] * OFFSET_TIP)
    distance_to_cube = np.linalg.norm(true_cube_pos - physical_gripper_center)
    
    approach_phase_weight = np.clip((distance_to_cube - 0.02) / 0.10, 0.0, 1.0)
    jaw_closed_penalty = -2.0 * current_jaw_position * approach_phase_weight
    reward_approach = -np.linalg.norm(true_cube_pos - x_offs) - (0.1 * theta_error)
    
    reward_grasp = 2.0 * current_jaw_position if distance_to_cube < 0.02 else 0.0
    
    reward_lift = 0.0
    cube_roll_penalty = 0.0
    z_delta = true_cube_pos[2] - cube_initial_pos[2]
    
    if z_delta > 0.003: 
        reward_lift += 10.0 * z_delta
        euler_angles = np.abs((R.from_quat(cube_initial_quat).inv() * R.from_quat(true_cube_quat)).as_euler('xyz', degrees=True))
        tilt_x = np.min([euler_angles[0] % 90, 90 - (euler_angles[0] % 90)])
        tilt_y = np.min([euler_angles[1] % 90, 90 - (euler_angles[1] % 90)])
        cube_roll_penalty = -0.05 * (tilt_x + tilt_y)

    return reward_approach + jaw_closed_penalty + reward_grasp + reward_lift + cube_roll_penalty - 0.05


# =============================================================================
# 5. ORACLE TRAINER WITH BACKGROUND PICTURE CAPTURE LOOP
# =============================================================================
def run_oracle_training_loop(sim_bundle: SimBundle, visual_head, data_generator, total_episodes=500, max_steps=150, batch_size=64):

    env = sim_bundle.env
    sim = sim_bundle.sim
    m, d = get_m_and_d(sim)
    
    renderer = data_generator.renderer
    camera_name = data_generator.camera_name
    camera_id = data_generator.cam_id
    ee_id = data_generator.ee_id

    agent = SACAgent(state_dim=7, action_dim=7)
    buffer = ReplayBuffer(capacity=100000)
    aligner = SymmetricCubeAligner()
    
    vision_dump_dir = "/nas/frainer/RL_ProjFelix/collected_data/"
    os.makedirs(vision_dump_dir, exist_ok=True)
    
    visual_record_index = 40000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Oracle Pre-Training Initialized. Learning trajectories using true cube states directly...")
    # Prior to the episode loop, extract the neutral baseline configuration once:
    mujoco.mj_forward(m, d)
    baseline_qpos = d.qpos.copy()
    baseline_qvel = d.qvel.copy()
    
    # Dynamically find all cube bodies present in the model workspace
    cube_names = [m.body(i).name for i in range(m.nbody) if "cube_body" in m.body(i).name]
    if not cube_names:
        cube_names = ["blue_cube_body"] # fallback
    for episode in range(total_episodes):
        # Record episode 0 immediately, and then every 50 episodes
        record_this_episode = (episode == 0) or ((episode + 1) % 50 == 0)
        video_writer = None
        if record_this_episode:
            video_dir = os.path.join(data_generator.output_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"episode_{episode:03d}.mp4")
            
            # Using mp4v codec at 25 Frames Per Second
            # The frame size must match your generator's renderer dimensions exactly!
            # Your Data_Generator.py defines: Renderer(self.model, height=480, width=640)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 25.0, (640, 480))
            print(f"Recording video for Episode {episode:03d} -> {video_path}")
        # -----------------------------------------------------------------
        # PHASE 2: SAMPLE NEW TARGET CUBE & GET ROBOT CLOSE
        # -----------------------------------------------------------------
        # 1. Randomly select one cube from the available pool
        target_cube = np.random.choice(data_generator.cube_names)
        cube_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, target_cube)
        
        # 2. Extract its true world coordinates
        cube_world_xyz, cube_world_xyzw = data_generator.get_cube_transform(target_cube)
        
        # 3. Define a randomized spatial tracking anchor 15cm above it
        anchor_pos = cube_world_xyz + np.array([0.0, 0.0, 0.15])
        target_pos = anchor_pos + np.random.uniform(-0.02, 0.02, size=3)
        
        # 4. Run the upright-biased analytical IK solver to snap the arm nearby
        ik_success = data_generator._solve_ik_with_upright_bias(target_pos)
        mujoco.mj_forward(m, d)
        
        # Freeze initial physics positions for this specific episode's reward checks
        cube_initial_pos = d.xpos[cube_id].copy()
        cube_initial_quat = R.from_matrix(d.xmat[cube_id].reshape(3,3)).as_quat()
        
        # Seed your tracking joint accumulator directly where the generator parked the arm
        target_arm_qpos = d.qpos[:6].copy() 
        
        episode_reward = 0

        target_arm_qpos = np.array(d.qpos[:6].copy())

        # 4. Pull live values for the very first step evaluation
        true_cube_pos = d.xpos[cube_id].copy()
        true_cube_quat = R.from_matrix(d.xmat[cube_id].reshape(3,3)).as_quat()
        current_eef_pos = d.xpos[ee_id].copy()
        current_eef_quat = R.from_matrix(d.xmat[ee_id].reshape(3,3)).as_quat()
        
        q_true_t = torch.FloatTensor(true_cube_quat).unsqueeze(0).to(device)
        q_eef_t = torch.FloatTensor(current_eef_quat).unsqueeze(0).to(device)
        with torch.no_grad():
            aligned_cube_quat = aligner.get_closest_symmetry(q_true_t, q_eef_t).cpu().numpy()[0]
        
        translation_error = true_cube_pos - current_eef_pos 
        q_eef_inv = np.array([-current_eef_quat[0], -current_eef_quat[1], -current_eef_quat[2], current_eef_quat[3]])
        
        dq_x =  aligned_cube_quat[3]*q_eef_inv[0] + aligned_cube_quat[0]*q_eef_inv[3] + aligned_cube_quat[1]*q_eef_inv[2] - aligned_cube_quat[2]*q_eef_inv[1]
        dq_y =  aligned_cube_quat[3]*q_eef_inv[1] - aligned_cube_quat[0]*q_eef_inv[2] + aligned_cube_quat[1]*q_eef_inv[3] + aligned_cube_quat[2]*q_eef_inv[0]
        dq_z =  aligned_cube_quat[3]*q_eef_inv[2] + aligned_cube_quat[0]*q_eef_inv[1] - aligned_cube_quat[1]*q_eef_inv[0] + aligned_cube_quat[2]*q_eef_inv[3]
        dq_w =  aligned_cube_quat[3]*q_eef_inv[3] - aligned_cube_quat[0]*q_eef_inv[0] - aligned_cube_quat[1]*q_eef_inv[1] - aligned_cube_quat[2]*q_eef_inv[2]
        rotation_error = np.array([dq_x, dq_y, dq_z, dq_w])
        
        # This state vector is now dynamically tuned to the randomly selected cube!
        state = np.concatenate([translation_error, rotation_error])
        for step in range(max_steps):
            if record_this_episode and video_writer is not None:
                # Update the scene using the dynamic data handle (pass camera id)
                renderer.update_scene(d, camera=camera_id)
                rendered_frame = renderer.render()
                
                # MuJoCo outputs standard RGB, but OpenCV handles video tracks in BGR
                bgr_frame = cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)
            # -----------------------------------------------------------------
            # STEP 1: READ DIRECT SIMULATOR TRUTH (No Visual Inference Errors!)
            # -----------------------------------------------------------------
            # Extract live physics arrays as explicit, frozen NumPy copies
            live_cube_pos = np.array(d.xpos[cube_id].copy())
            live_cube_quat = np.array(R.from_matrix(d.xmat[cube_id].reshape(3,3)).as_quat().copy())
            live_eef_pos = np.array(d.xpos[ee_id].copy())
            live_eef_quat = np.array(R.from_matrix(d.xmat[ee_id].reshape(3,3)).as_quat().copy())
            
            current_jaw_pos = sim_bundle.robot.get_joint_positions()[-1] if hasattr(sim_bundle.robot, "get_joint_positions") else 0.0
            # -----------------------------------------------------------------
            # STEP 2: CALCULATE CLOSER-SYMMETRIC ROTATION AND ERROR STATES
            # -----------------------------------------------------------------
            q_true_t = torch.FloatTensor(true_cube_quat).unsqueeze(0).to(device)
            q_eef_t = torch.FloatTensor(current_eef_quat).unsqueeze(0).to(device)
            with torch.no_grad():
                aligned_cube_quat = aligner.get_closest_symmetry(q_true_t, q_eef_t).cpu().numpy()[0]
            
            translation_error = true_cube_pos - current_eef_pos 
            q_eef_inv = np.array([-current_eef_quat[0], -current_eef_quat[1], -current_eef_quat[2], current_eef_quat[3]])
            
            dq_x =  aligned_cube_quat[3]*q_eef_inv[0] + aligned_cube_quat[0]*q_eef_inv[3] + aligned_cube_quat[1]*q_eef_inv[2] - aligned_cube_quat[2]*q_eef_inv[1]
            dq_y =  aligned_cube_quat[3]*q_eef_inv[1] - aligned_cube_quat[0]*q_eef_inv[2] + aligned_cube_quat[1]*q_eef_inv[3] + aligned_cube_quat[2]*q_eef_inv[0]
            dq_z =  aligned_cube_quat[3]*q_eef_inv[2] + aligned_cube_quat[0]*q_eef_inv[1] - aligned_cube_quat[1]*q_eef_inv[0] + aligned_cube_quat[2]*q_eef_inv[3]
            dq_w =  aligned_cube_quat[3]*q_eef_inv[3] - aligned_cube_quat[0]*q_eef_inv[0] - aligned_cube_quat[1]*q_eef_inv[1] - aligned_cube_quat[2]*q_eef_inv[2]
            rotation_error = np.array([dq_x, dq_y, dq_z, dq_w])
            
            state = np.concatenate([translation_error, rotation_error])
            
            # -----------------------------------------------------------------
            # STEP 3: REWARD EVALUATION & ACTION SELECTION
            # -----------------------------------------------------------------
            # Extract fresh copies right before calculating to bypass any shadowing bugs

            
            reward = compute_time_optimized_multiphase_reward(
                live_cube_pos, live_cube_quat, live_eef_pos, live_eef_quat,
                aligned_cube_quat, current_jaw_pos, cube_initial_pos, cube_initial_quat
            )
            
            # Agent outputs 7D: [dx, dy, dz, droll, dpitch, dyaw, djaw]
            raw_action = agent.select_action(state, evaluate=False)
            
            spatial_scale = np.array([0.02, 0.02, 0.02, 0.05, 0.05, 0.05])
            cartesian_delta = raw_action[:6] * spatial_scale
            jaw_action = raw_action[6:] 
            joint_deltas = solve_differential_ik(m, d, "robotwrist", cartesian_delta)
            ik_action = np.concatenate([joint_deltas[:6], jaw_action]).astype(np.float32) 
            
            # Use the environment wrapper to apply actions so RCS/RelativeActionSpace
            # semantics (clipping, relative-to-last-step handling, gripper wrapper, etc.)
            # are respected instead of bypassing them by writing to `d.ctrl` directly.
            #
            # Prepare action dict expected by the wrapper: `joints` + `gripper`.
            try:
                current_joint_positions = np.array(sim_bundle.robot.get_joint_positions())
            except Exception:
                # Fallback: infer number of robot joints from model (first 6 qpos entries)
                current_joint_positions = d.qpos[:6].copy()

            if expected_joints_len is None or expected_joints_len == 6:
                expected_joints_len = 5 

            # Take only the first 5 elements of joint_deltas to match the physical motors
            joints_action = np.array(joint_deltas[:expected_joints_len], dtype=np.float64)

            # Map jaw action from tanh range (-1,1) to gripper normalized [0,1]
            gripper_cmd = float(((jaw_action + 1.0) * 0.5).squeeze())

            env_action = {
                "joints": joints_action,  # Now strictly shape (5,)
                "gripper": np.array([gripper_cmd], dtype=np.float64),
            }

            # Execute action through the env wrapper so it advances the sim correctly
            # Wrap action for multi-env wrappers which expect {robot_key: action}
            outer_action = env_action
            outer_key = None
            try:
                if hasattr(env, "envs"):
                    outer_key = next(iter(env.envs.keys()))
                else:
                    robot_attr = env.get_wrapper_attr("robot")
                    if isinstance(robot_attr, dict):
                        outer_key = next(iter(robot_attr.keys()))
            except Exception:
                outer_key = None

            if outer_key is not None:
                outer_action = {outer_key: env_action}

            obs_step, env_reward, terminated, truncated, info = env.step(outer_action)
            done = bool(terminated or truncated)

            # Ensure MuJoCo forward kinematics are up-to-date
            m, d = get_m_and_d(sim)
            mujoco.mj_forward(m, d)

            # -----------------------------------------------------------------
            # STEP 4: BACKGROUND VISION ACQUISITION (For future Visual Head iterations)
            # -----------------------------------------------------------------
            # We capture a picture frame at every step to build up our vision repository
            img_frame = render_camera(renderer, sim, camera_name)
            if len(img_frame.shape) == 3:
                img_frame = cv2.cvtColor(img_frame, cv2.COLOR_RGB2GRAY)
            
            # Save snapshots on a regular interval or based on your chosen logic
            if step % 5 == 0:
                img_name = f"sample_{visual_record_index:05d}_edge.png"
                cv2.imwrite(os.path.join(vision_dump_dir, img_name), img_frame)
                
                # Synthetic input context to mock generator specifications
                noisy_xy = true_cube_pos[0:2] + np.random.uniform(-0.03, 0.03, size=2)
                
                meta_record = {
                    "image_file": img_name,
                    "input_noisy_xy": noisy_xy,
                    "label_relative_xyzw": np.concatenate([true_cube_pos - current_eef_pos, true_cube_quat])
                }
                # (You can serialize meta_record list elements directly to an array file here)
                visual_record_index += 1

            # -----------------------------------------------------------------
            # STEP 5: COMPUTE NEXT ORACLE STATE & COMMIT TO BUFFER
            # -----------------------------------------------------------------
            next_cube_pos = d.xpos[cube_id].copy()
            next_cube_quat = R.from_matrix(d.xmat[cube_id].reshape(3,3)).as_quat()
            next_eef_pos = d.xpos[ee_id].copy()
            next_eef_quat = R.from_matrix(d.xmat[ee_id].reshape(3,3)).as_quat()
            
            next_translation_error = next_cube_pos - next_eef_pos
            next_q_eef_inv = np.array([-next_eef_quat[0], -next_eef_quat[1], -next_eef_quat[2], next_eef_quat[3]])
            
            ndq_x =  aligned_cube_quat[3]*next_q_eef_inv[0] + aligned_cube_quat[0]*next_q_eef_inv[3] + aligned_cube_quat[1]*next_q_eef_inv[2] - aligned_cube_quat[2]*next_q_eef_inv[1]
            ndq_y =  aligned_cube_quat[3]*next_q_eef_inv[1] - aligned_cube_quat[0]*next_q_eef_inv[2] + aligned_cube_quat[1]*next_q_eef_inv[3] + aligned_cube_quat[2]*next_q_eef_inv[0]
            ndq_z =  aligned_cube_quat[3]*next_q_eef_inv[2] + aligned_cube_quat[0]*next_q_eef_inv[1] - aligned_cube_quat[1]*next_q_eef_inv[0] + aligned_cube_quat[2]*next_q_eef_inv[3]
            ndq_w =  aligned_cube_quat[3]*next_q_eef_inv[3] - aligned_cube_quat[0]*next_q_eef_inv[0] - aligned_cube_quat[1]*next_q_eef_inv[1] - aligned_cube_quat[2]*next_q_eef_inv[2]
            next_rotation_error = np.array([ndq_x, ndq_y, ndq_z, ndq_w])
            
            next_state = np.concatenate([next_translation_error, next_rotation_error])
            
            buffer.push(state, raw_action, reward, next_state, done)
            
            if len(buffer) > batch_size:
                agent.update_parameters(buffer, batch_size)
                
            # --- ADD THIS BLOCK TO ADVANCE THE STATE ---
            state = next_state
            true_cube_pos = next_cube_pos
            true_cube_quat = next_cube_quat
            current_eef_pos = next_eef_pos
            current_eef_quat = next_eef_quat
            # -------------------------------------------
            episode_reward += reward
            if done:
                break
        # Release video writer after episode finishes so file contains full frames
        if video_writer is not None:
            video_writer.release()

        if episode % 1 == 0:
            print(f"Episode: {episode:03d} | Steps: {step+1:03d} | Total Accumulated Reward: {episode_reward:.10f}")
        # -----------------------------------------------------------------
        # PERIODIC MODEL CHECKPOINT SAVING
        # -----------------------------------------------------------------
        if (episode + 1) % 50 == 0:
            checkpoint_dir = 'checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            actor_path = os.path.join(checkpoint_dir, f"sac_actor_ep{episode+1:03d}.pth")
            critic_path = os.path.join(checkpoint_dir, f"sac_critic_ep{episode+1:03d}.pth")
            
            # Save state dictionaries safely
            torch.save(agent.actor.state_dict(), actor_path)
            torch.save(agent.critic.state_dict(), critic_path)
            print(f"💾 Checkpoint saved: Actor -> {f'sac_actor_ep{episode+1:03d}.pth'}")

if __name__ == "__main__":
    from env.env_factory import make_so101_sim
    from scripts.Data_Generator import SpatialDataGenerator
    # 1. Initialize your custom MuJoCo/RCS simulation workspace
    print("Building MuJoCo simulation context via factory abstractions...")
    sim_bundle = make_so101_sim(
        with_cameras=True, 
        headless=True,       # Set to False if you are running locally and want the GUI
        debug_print=False
    )
    data_gen = SpatialDataGenerator(target_camera="robotwrist")
    sim_bundle = data_gen.bundle
    # 2. Define a dummy lambda function for your visual head 
    # Since we are using true cube states directly for Phase 1 oracle control, 
    # we just need a dummy function that returns an array of zeros to satisfy the inputs.
    dummy_visual_head = lambda img, prior: torch.zeros((1, 7))
    
    # 3. Fire up the training loop!
    run_oracle_training_loop(
        sim_bundle=sim_bundle,
        visual_head=dummy_visual_head,
        data_generator=data_gen,
        total_episodes=500,
        max_steps=150,
        batch_size=64
    )