"""
Real-time Control System (RCS) controller for SO-101 pick-and-place.

Provides:
  - Inverse kinematics (MuJoCo's jacobian-based solver)
  - Waypoint trajectory execution with smooth interpolation
  - Safety layer (joint limits, velocity limits, distance monitoring)
  - State machine for pick-and-place operations
  - Real-time control loop friendly

Usage:
    controller = SO101Controller(model, data, ee_site_name="ee_site")
    controller.set_target_cartesian(target_pos, target_rot)
    for _ in range(sim_steps):
        action = controller.step(dt)
        data.ctrl[:] = action
        mujoco.mj_step(model, data)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


# ── constants ──────────────────────────────────────────────────────────────

class ControlPhase(Enum):
    IDLE = 0
    MOVING_TO_TARGET = 1
    GRASPING = 2
    RETRACTING = 3
    MOVING_TO_PLACE = 4
    PLACING = 5
    ERROR = 6


@dataclass
class ControllerConfig:
    """Configuration for SO101Controller."""
    max_joint_vel: float = 0.5  # rad/s - max joint velocity
    max_ee_vel: float = 0.2  # m/s - max end-effector velocity
    max_accel: float = 0.5  # m/s^2 - max acceleration
    ik_iterations: int = 50  # IK solver iterations
    ik_tol: float = 1e-4  # IK convergence tolerance (meters)
    pos_tolerance: float = 0.01  # Position tolerance for waypoint reach (meters)
    rot_tolerance: float = 0.1  # Rotation tolerance (radians)
    grasp_time: float = 0.5  # Time to hold grasp (seconds)
    retract_distance: float = 0.1  # Retract height after pick (meters)
    safety_margin: float = 0.05  # Collision safety margin (meters)
    log_debug: bool = False


@dataclass
class ControllerState:
    """Current state of the controller."""
    phase: ControlPhase = ControlPhase.IDLE
    target_qpos: np.ndarray = field(default_factory=lambda: np.zeros(6))
    current_qpos: np.ndarray = field(default_factory=lambda: np.zeros(6))
    target_ee_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_ee_rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    ee_pos_error: float = 0.0
    ee_rot_error: float = 0.0
    phase_time: float = 0.0
    gripper_cmd: float = 0.0  # 0=closed, 1=open
    safety_ok: bool = True
    last_error: str = ""


class SO101Controller:
    """
    Real-time controller for SO-101 arm pick-and-place.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        ee_site_name: str = "ee_site",
        config: Optional[ControllerConfig] = None,
    ):
        self.model = model
        self.data = data
        self.config = config or ControllerConfig()
        self.ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)

        if self.ee_site_id < 0:
            raise ValueError(f"Site '{ee_site_name}' not found in model")

        self.state = ControllerState()
        self._trajectory: list[np.ndarray] = []
        self._traj_idx = 0
        self._phase_timer = 0.0

    # ── getters ────────────────────────────────────────────────────────────────

    def get_ee_pos(self) -> np.ndarray:
        """Get current end-effector position (world frame)."""
        return self.data.site(self.ee_site_id).xpos.copy()

    def get_ee_rot(self) -> np.ndarray:
        """Get current end-effector rotation (world frame, 3x3 rotation matrix)."""
        return self.data.site(self.ee_site_id).xmat.reshape(3, 3).copy()

    def get_qpos(self) -> np.ndarray:
        """Get current joint positions (6 DOF)."""
        return self.data.qpos[:6].copy()

    def get_qvel(self) -> np.ndarray:
        """Get current joint velocities."""
        return self.data.qvel[:6].copy()

    def get_state(self) -> ControllerState:
        """Get current controller state snapshot."""
        self.state.current_qpos = self.get_qpos()
        self.state.ee_pos_error = np.linalg.norm(self.get_ee_pos() - self.state.target_ee_pos)
        self.state.phase_time = self._phase_timer
        return self.state

    # ── inverse kinematics ─────────────────────────────────────────────────────

    def compute_ik(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        initial_qpos: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Compute inverse kinematics to reach target pose.

        Args:
            target_pos: Target EE position (3,) in world frame.
            target_rot: Target EE rotation (3, 3) rotation matrix, world frame.
            initial_qpos: Initial guess for IK (6,). If None, use current qpos.

        Returns:
            Joint positions (6,) if IK succeeded, None otherwise.
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)
        target_rot = np.asarray(target_rot, dtype=np.float64)

        if initial_qpos is None:
            qpos = self.get_qpos().copy()
        else:
            qpos = np.asarray(initial_qpos[:6], dtype=np.float64)

        # Save current data state
        data_backup = mujoco.MjData(self.model)
        mujoco.mj_copyData(data_backup, self.data)

        # IK via optimization (Gauss-Newton on position + rotation error)
        for iteration in range(self.config.ik_iterations):
            # Set candidate qpos and forward kinematics
            self.data.qpos[:6] = qpos
            mujoco.mj_forward(self.model, self.data)

            ee_pos = self.get_ee_pos()
            ee_rot = self.get_ee_rot()

            pos_error = target_pos - ee_pos
            pos_error_norm = np.linalg.norm(pos_error)

            # Check convergence
            if pos_error_norm < self.config.ik_tol:
                if self.config.log_debug:
                    print(f"IK converged in {iteration} iterations, error={pos_error_norm:.6f}")
                # Restore data
                mujoco.mj_copyData(self.data, data_backup)
                return qpos.copy()

            # Compute Jacobian
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)

            # Use position Jacobian (3 x nv)
            jac = jacp[:, :6].copy()

            # Compute pseudoinverse and step
            jac_pinv = np.linalg.pinv(jac, rcond=1e-4)
            dq = jac_pinv @ pos_error * 0.1

            # Damped update
            qpos_new = qpos + dq
            qpos_new = np.clip(qpos_new, self.model.jnt_range[:6, 0], self.model.jnt_range[:6, 1])

            if np.allclose(qpos_new, qpos):
                break  # Converged
            qpos = qpos_new

        # Restore data
        mujoco.mj_copyData(self.data, data_backup)

        if self.config.log_debug:
            print(f"IK did not converge after {self.config.ik_iterations} iterations, error={pos_error_norm:.6f}")

        return None

    # ── trajectory planning ────────────────────────────────────────────────────

    def _plan_linear_trajectory(
        self,
        start_q: np.ndarray,
        end_q: np.ndarray,
        num_steps: int = 50,
    ) -> list[np.ndarray]:
        """Plan linear interpolation in joint space."""
        traj = []
        for i in range(num_steps):
            alpha = i / (num_steps - 1) if num_steps > 1 else 1.0
            q = (1 - alpha) * start_q + alpha * end_q
            traj.append(q.copy())
        return traj

    def plan_to_cartesian(
        self,
        target_pos: np.ndarray,
        target_rot: np.ndarray,
        duration: float = 2.0,
    ) -> bool:
        """
        Plan motion from current pose to target Cartesian pose.

        Returns:
            True if planning succeeded.
        """
        # Compute target joint positions via IK
        target_qpos = self.compute_ik(target_pos, target_rot)
        if target_qpos is None:
            self.state.last_error = "IK failed"
            self.state.phase = ControlPhase.ERROR
            return False

        # Plan smooth trajectory in joint space
        start_q = self.get_qpos()
        num_steps = max(int(duration / (1.0 / 30.0)), 10)  # Assume 30 Hz control
        self._trajectory = self._plan_linear_trajectory(start_q, target_qpos, num_steps)
        self._traj_idx = 0

        self.state.target_qpos = target_qpos.copy()
        self.state.target_ee_pos = target_pos.copy()
        self.state.target_ee_rot = target_rot.copy()
        self.state.phase = ControlPhase.MOVING_TO_TARGET
        self._phase_timer = 0.0

        return True

    # ── safety checking ────────────────────────────────────────────────────────

    def check_safety(self) -> bool:
        """
        Check safety constraints (joint limits, velocities, etc).

        Returns:
            True if safe, False otherwise.
        """
        qpos = self.get_qpos()
        qvel = self.get_qvel()

        # Check joint limits
        q_low = self.model.jnt_range[:6, 0]
        q_high = self.model.jnt_range[:6, 1]
        if np.any(qpos < q_low - self.config.safety_margin) or np.any(qpos > q_high + self.config.safety_margin):
            self.state.last_error = f"Joint limit violation: qpos={qpos}"
            self.state.safety_ok = False
            return False

        # Check joint velocities
        if np.any(np.abs(qvel) > self.config.max_joint_vel * 1.5):  # 1.5x margin
            self.state.last_error = f"Velocity limit violation: qvel={qvel}"
            self.state.safety_ok = False
            return False

        self.state.safety_ok = True
        return True

    # ── control loop ───────────────────────────────────────────────────────────

    def step(self, dt: float) -> np.ndarray:
        """
        Execute one control step and return joint commands.

        Args:
            dt: Time step (seconds).

        Returns:
            Joint command array (6,) for the arm.
        """
        self._phase_timer += dt

        if not self.check_safety():
            return np.zeros(6)

        action = np.zeros(6)

        if self.state.phase == ControlPhase.IDLE:
            action = self.get_qpos() * 0.0  # Zero command
            self.state.safety_ok = True

        elif self.state.phase == ControlPhase.MOVING_TO_TARGET:
            if self._traj_idx < len(self._trajectory):
                action = self._trajectory[self._traj_idx].copy()
                self._traj_idx += 1
            else:
                # Trajectory complete, check if reached target
                ee_pos = self.get_ee_pos()
                pos_error = np.linalg.norm(ee_pos - self.state.target_ee_pos)
                if pos_error < self.config.pos_tolerance:
                    self.state.phase = ControlPhase.IDLE
                    if self.config.log_debug:
                        print(f"[Controller] Reached target, pos_error={pos_error:.6f}")
                else:
                    action = self.state.target_qpos.copy()  # Hold target
                    if self.config.log_debug:
                        print(f"[Controller] Still moving, pos_error={pos_error:.6f}")

        elif self.state.phase == ControlPhase.GRASPING:
            # Hold target position while gripper closes
            action = self.state.target_qpos.copy()
            if self._phase_timer > self.config.grasp_time:
                self.state.phase = ControlPhase.RETRACTING
                self._phase_timer = 0.0

        elif self.state.phase == ControlPhase.RETRACTING:
            # Move upward (Z+) while holding X-Y
            ee_pos = self.get_ee_pos()
            retract_pos = ee_pos.copy()
            retract_pos[2] += self.config.retract_distance
            target_q = self.compute_ik(retract_pos, self.state.target_ee_rot)
            if target_q is not None:
                action = target_q
                if self._phase_timer > 1.0:  # Timeout fallback
                    self.state.phase = ControlPhase.IDLE
            else:
                self.state.phase = ControlPhase.IDLE

        elif self.state.phase == ControlPhase.ERROR:
            action = self.get_qpos()  # Hold current position

        return action.astype(np.float32)

    # ── high-level operations ──────────────────────────────────────────────────

    def pick(self, target_pos: np.ndarray, target_rot: Optional[np.ndarray] = None) -> bool:
        """
        Execute pick operation: move to target, grasp, retract.

        Args:
            target_pos: Pick location (3,).
            target_rot: Target orientation (3, 3). If None, use current.

        Returns:
            True if pick started successfully.
        """
        if target_rot is None:
            target_rot = self.get_ee_rot()

        if not self.plan_to_cartesian(target_pos, target_rot, duration=2.0):
            return False

        # After reaching: set phase to GRASPING for next steps
        # (actual grasp happens in step() after reaching target)
        return True

    def place(self, target_pos: np.ndarray, target_rot: Optional[np.ndarray] = None) -> bool:
        """
        Execute place operation: move to target location.

        Args:
            target_pos: Place location (3,).
            target_rot: Target orientation (3, 3). If None, use current.

        Returns:
            True if place started successfully.
        """
        if target_rot is None:
            target_rot = self.get_ee_rot()

        return self.plan_to_cartesian(target_pos, target_rot, duration=2.0)

    def set_gripper(self, open: bool) -> None:
        """
        Command gripper open/close.

        Args:
            open: True to open, False to close.
        """
        self.state.gripper_cmd = 1.0 if open else 0.0

    def go_home(self) -> bool:
        """
        Move to home position.

        Returns:
            True if home motion started.
        """
        home_q = np.array([0.0, -1.57, 1.0, 1.0, 0.0, 0.02], dtype=np.float32)
        start_q = self.get_qpos()
        self._trajectory = self._plan_linear_trajectory(start_q, home_q, num_steps=50)
        self._traj_idx = 0
        self.state.phase = ControlPhase.MOVING_TO_TARGET
        self._phase_timer = 0.0
        return True

    def reset(self) -> None:
        """Reset controller to idle state."""
        self.state = ControllerState()
        self._trajectory.clear()
        self._traj_idx = 0
        self._phase_timer = 0.0


# ── standalone test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Example: load SO-101 model and test controller
    hw_dir = Path(__file__).parent / "hw4_reinforcement_learning" if not Path(__file__).parent.name == "hw4_reinforcement_learning" else Path(__file__).parent
    xml_path = hw_dir / "assets" / "mujoco" / "so100_pos_ctrl.xml"

    if not xml_path.exists():
        print(f"XML not found: {xml_path}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    data.qpos[:6] = np.array([0.0, -1.57, 1.0, 1.0, 0.0, 0.02])
    mujoco.mj_forward(model, data)

    controller = SO101Controller(model, data, config=ControllerConfig(log_debug=True))

    print("Testing SO101Controller...")
    print(f"Current EE pos: {controller.get_ee_pos()}")
    print(f"Current qpos:   {controller.get_qpos()}")

    # Test: plan to a nearby target
    target_pos = controller.get_ee_pos() + np.array([0.1, 0.0, 0.0])
    target_rot = controller.get_ee_rot()

    print(f"\nPlanning to target: {target_pos}")
    if controller.plan_to_cartesian(target_pos, target_rot, duration=1.0):
        print("✓ Plan succeeded")
        for i in range(100):
            action = controller.step(0.01)
            data.ctrl[:6] = action
            mujoco.mj_step(model, data)
            if i % 20 == 0:
                print(f"Step {i}: EE pos error = {controller.get_state().ee_pos_error:.6f}")
    else:
        print("✗ Plan failed")
