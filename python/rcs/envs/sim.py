import logging
from typing import Any, cast

import gymnasium as gym
import mujoco
import numpy as np
from numpy.random import random
from rcs._core.common import RobotPlatform
from rcs.envs.base import GripperWrapper
from rcs.envs.space_utils import ActObsInfoWrapper
from rcs.envs.task_helper import get_geom_pos, random_position_in_bounds

import rcs
from rcs import sim

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RobotSimWrapper(ActObsInfoWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert self.env.get_wrapper_attr("PLATFORM") == RobotPlatform.SIMULATION, "Base environment must be simulation."
        assert isinstance(self.get_wrapper_attr("robot"), sim.SimRobot), "Robot must be a sim.SimRobot instance."
        self.sim_robot = cast(sim.SimRobot, self.get_wrapper_attr("robot"))
        self.sim = cast(sim.Sim, self.get_wrapper_attr("sim"))

    def action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.sim_robot.clear_collision_flag()
        return action

    def observation(self, observation: dict, info: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        state = self.sim_robot.get_state()
        if "collision" not in info:
            info["collision"] = state.collision
        else:
            info["collision"] = info["collision"] or state.collision
        info["ik_success"] = state.ik_success
        info["is_sim_converged"] = self.sim.is_converged()
        return observation, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.sim_robot.clear_collision_flag()
        return super().reset(seed=seed, options=options)


class GripperWrapperSim(ActObsInfoWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert self.env.get_wrapper_attr("PLATFORM") == RobotPlatform.SIMULATION, "Base environment must be simulation."
        assert isinstance(
            self.get_wrapper_attr("gripper"), sim.SimGripper
        ), "Gripper must be a sim.SimGripper instance."
        self._gripper = cast(sim.SimGripper, self.get_wrapper_attr("gripper"))

    def action(self, action: dict[str, Any]) -> dict[str, Any]:
        self._gripper.clear_collision_flag()
        return action

    def observation(self, observation: dict[str, Any], info: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        state = self._gripper.get_state()
        if "collision" not in info or not info["collision"]:
            info["collision"] = state.collision
        info["gripper_width"] = self._gripper.get_normalized_width()
        info["is_grasped"] = self._gripper.get_normalized_width() > 0.01 and self._gripper.get_normalized_width() < 0.99
        return observation, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._gripper.clear_collision_flag()
        return super().reset(seed=seed, options=options)


class HandWrapperSim(ActObsInfoWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert self.env.get_wrapper_attr("PLATFORM") == RobotPlatform.SIMULATION, "Base environment must be simulation."
        assert isinstance(
            self.get_wrapper_attr("hand"), sim.SimTilburgHand
        ), "Hand must be a sim.SimTilburgHand instance."
        self._hand = cast(sim.SimTilburgHand, self.get_wrapper_attr("hand"))

    def action(self, action: dict[str, Any]) -> dict[str, Any]:
        if isinstance(action["hand"], int | float):
            return action
        if len(action["hand"]) == 18:
            action["hand"] = action["hand"][:16]
        assert len(action["hand"]) == 16 or len(action["hand"]) == 1, "Hand action must be of length 16 or 1"
        return action

    def observation(self, observation: dict[str, Any], info: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        state = self._hand.get_state()
        if "collision" not in info or not info["collision"]:
            info["collision"] = state.collision
        info["hand_position"] = self._hand.get_normalized_joint_poses()
        # info["is_grasped"] = self._hand.get_normalized_joint_poses() > 0.01 and self._hand.get_normalized_joint_poses() < 0.99
        return observation, info


class RandomObjectPos(gym.Wrapper):
    """
    Wrapper to randomly re-place an object in the lab environments.
    Given the object's joint name and initial pose, its x, y coordinates are randomized, while z remains fixed.
    If include_rotation is true, the object's z-axis rotation (yaw) is also randomized.

    Args:
        env (gym.Env): The environment to wrap.
        simulation (sim.Sim): The simulation instance.
        joint_name (str): The name of the free joint attached to the object to manipulate.
        init_object_pose (rcs.common.Pose): The initial pose of the object.
        include_rotation (bool): Whether to include rotation in the randomization.
    """

    def __init__(
        self,
        env: gym.Env,
        joint_name: str,
        init_object_pose: rcs.common.Pose,
        include_position: bool = True,
        include_rotation: bool = False,
        x_scale: float = 0.2,
        y_scale: float = 0.2,
        x_offset: float = 0.1,
        y_offset: float = 0.1,
    ):
        super().__init__(env)
        self.joint_name = joint_name
        self.init_object_pose = init_object_pose
        self.include_position = include_position
        self.include_rotation = include_rotation
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.x_offset = x_offset
        self.y_offset = y_offset

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if options is not None and "RandomObjectPos.init_object_pose" in options:
            assert isinstance(
                options["RandomObjectPos.init_object_pose"], rcs.common.Pose
            ), "RandomObjectPos.init_object_pose must be a rcs.common.Pose"

            self.init_object_pose = options["RandomObjectPos.init_object_pose"]
            print("Got random object pos!\n", self.init_object_pose)
            del options["RandomObjectPos.init_object_pose"]
        obs, info = super().reset(seed=seed, options=options)

        pos_z = self.init_object_pose.translation()[2]
        if self.include_position:
            pos_x = self.init_object_pose.translation()[0] + np.random.random() * self.x_scale + self.x_offset
            pos_y = self.init_object_pose.translation()[1] + np.random.random() * self.y_scale + self.y_offset
        else:
            pos_x = self.init_object_pose.translation()[0]
            pos_y = self.init_object_pose.translation()[1]

        quat = self.init_object_pose.rotation_q()  # xyzw format
        if self.include_rotation:
            random_z_rotation = (np.random.random() - 0.5) * (0.7071068 * 2)
            self.get_wrapper_attr("sim").data.joint(self.joint_name).qpos = [
                pos_x,
                pos_y,
                pos_z,
                quat[3] + random_z_rotation,
                quat[0],
                quat[1],
                quat[2] + random_z_rotation,
            ]
        else:
            self.get_wrapper_attr("sim").data.joint(self.joint_name).qpos = [
                pos_x,
                pos_y,
                pos_z,
                quat[3],
                quat[0],
                quat[1],
                quat[2],
            ]

        return obs, info


class RandomCubePos(gym.Wrapper):
    """Wrapper to randomly place cube in the lab environments.

    Works only for single robot
    """

    def __init__(self, env: gym.Env, include_rotation: bool = False, cube_joint_name="box_joint"):
        super().__init__(env)
        self.include_rotation = include_rotation
        self.cube_joint_name = cube_joint_name

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)

        iso_cube = np.array([0.498, 0.0, 0.226])
        iso_cube_pose = rcs.common.Pose(translation=np.array(iso_cube), rpy_vector=np.array([0, 0, 0]))  # type: ignore
        iso_cube = self.get_wrapper_attr("robot").to_pose_in_world_coordinates(iso_cube_pose).translation()
        pos_z = 0.0288
        pos_x = iso_cube[0] + np.random.random() * 0.2 - 0.1
        pos_y = iso_cube[1] + np.random.random() * 0.2 - 0.1

        if self.include_rotation:
            self.get_wrapper_attr("sim").data.joint(self.cube_joint_name).qpos = [
                pos_x,
                pos_y,
                pos_z,
                2 * np.random.random() - 1,
                0,
                0,
                1,
            ]
        else:
            self.get_wrapper_attr("sim").data.joint(self.cube_joint_name).qpos = [pos_x, pos_y, pos_z, 0, 0, 0, 1]

        return obs, info


class PickCubeSuccessWrapper(gym.Wrapper):
    """
    Wrapper to check if the cube is successfully picked up in the FR3SimplePickUpSim environment.
    Cube must be lifted 10 cm above the robot base.
    Computes a reward between 0 and 1 based on:
    - TCP to object distance
    - cube z height
    - whether the arm is standing still once the task is solved.
    """

    def __init__(self, env, cube_geom_name="box_geom"):
        super().__init__(env)
        assert isinstance(self.get_wrapper_attr("robot"), sim.SimRobot), "Robot must be a sim.SimRobot instance."
        self._robot = cast(sim.SimRobot, self.get_wrapper_attr("robot"))
        self.sim = self.env.get_wrapper_attr("sim")
        self.cube_geom_name = cube_geom_name
        self.home_pose = self._robot.get_cartesian_position()
        self._gripper_closing = 0
        self._gripper = self.get_wrapper_attr("gripper")

    def step(self, action: dict[str, Any]):  # type: ignore
        obs, reward, _, truncated, info = super().step(action)
        if (
            self._gripper.get_normalized_width() > 0.01
            and self._gripper.get_normalized_width() < 0.99
            and obs["gripper"] == GripperWrapper.BINARY_GRIPPER_CLOSED
        ):
            self._gripper_closing += 1
        else:
            self._gripper_closing = 0
        cube_pose = rcs.common.Pose(translation=self.sim.data.geom(self.cube_geom_name).xpos)
        cube_pose = self._robot.to_pose_in_robot_coordinates(cube_pose)
        tcp_to_obj_dist = np.linalg.norm(cube_pose.translation() - self._robot.get_cartesian_position().translation())
        obj_to_goal_dist = 0.10 - min(cube_pose.translation()[-1], 0.10)
        obj_to_goal_dist = np.linalg.norm(cube_pose.translation() - self.home_pose.translation())
        # NOTE: 4 depends on the time passing between each step.
        is_grasped = (
            self._gripper_closing >= 4  # gripper is closing since more than 4 steps
            and obs["gripper"] == GripperWrapper.BINARY_GRIPPER_CLOSED  # command is still close
            and tcp_to_obj_dist <= 0.01  # tcp to cube center is max 1cm
        )
        success = obj_to_goal_dist <= 0.022 and info["is_grasped"]
        movement = np.linalg.norm(self.sim.data.qvel)

        reaching_reward = 1 - np.tanh(5 * tcp_to_obj_dist)
        place_reward = 1 - np.tanh(5 * obj_to_goal_dist)
        static_reward = 1 - np.tanh(5 * movement)
        info["is_grasped"] = is_grasped
        info["success"] = success
        reward = reaching_reward + place_reward * is_grasped + static_reward * success
        reward /= 3  # type: ignore
        return obs, reward, success, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset()
        self.home_pose = self._robot.get_cartesian_position()
        return obs, info


class MazeTaskWrapper(gym.Wrapper):
    """
    wrapper to check if the maze task is succesful

    and to conduct a soft reset of the task
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(self.get_wrapper_attr("robot"), sim.SimRobot), "Robot must be a sim.SimRobot instance."
        self._robot = cast(sim.SimRobot, self.get_wrapper_attr("robot"))
        self.sim = self.env.get_wrapper_attr("sim")
        self.home_pose = self._robot.get_cartesian_position()
        self._gripper_closing = 0
        self._gripper = self.get_wrapper_attr("gripper")
        self.start_marker = "boxmarker_start"
        self.goal_marker = "boxmarker_start"
        self.ball_join = "boxball_joint"
        self.ball_name = "boxball"
        self.board_name = "boxboard"
        self.board_joint = "boxboard_joint"
        self.phase = [True, False]  # phase[0] approaching board phase[1] holding board and working on moving ball
        self.holds = ["hold1", "hold2", "hold3", "hold4"]
        self.goal_dist = 0.008

    def calculate_reward(self):

        # get concat pairs
        contacts = []
        reward = 0

        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]

            g1 = self.sim.model.geom(c.geom1).name
            g2 = self.sim.model.geom(c.geom2).name

            contacts.append((g1, g2))

        # check if phase 1 complete
        # check if two contact wiht the board
        contact_points = 0
        for n in self.holds:
            if any(n in (a, b) for a, b in contacts):
                contact_points += 1

        # check if board is lifted
        n = "floor"
        is_lifted = not any(n in (a, b) for a, b in contacts)

        if contact_points == 2 and is_lifted:
            self.phase = [False, True]
            reward = 0.5
        else:
            self.phase = [True, False]

        # check if ball on goal
        goal_pose = self.sim.data.body(self.goal_marker).xpos
        ball_pose = self.sim.data.body(self.ball_name).xpos

        dist = np.linalg.norm(goal_pose - ball_pose)

        if dist < self.goal_dist:
            reward = reward + 0.5

        return reward, self.phase

    def step(self, action: dict[str, Any]):  # type: ignore
        obs, reward, _, truncated, info = super().step(action)
        success = False

        reward, phase = self.calculate_reward()

        info["pahse"] = phase

        if reward == 1:
            success = True
        return obs, reward, success, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset()

        # TODO move to somewhere where it fits
        x_max_t = 1
        x_min_t = 1
        y_max_t = 1
        z_t = 0.1

        # randmoise the board position
        random.seed(seed)
        x_min = x_min_t
        x_max = x_max_t
        y_min = y_max_t
        y_max = y_max_t
        z = z_t

        old_pos = self.simdata.body(self.board_name).xpos
        pos = random_position_in_bounds(old_pos, x_min, x_max, y_min, y_max, z, random)
        joint_id = self.sim.model.joint(self.board_joint).id
        qpos_adr = self.sim.model.jnt_qposadr[joint_id]

        self.sim.data.qpos[qpos_adr : qpos_adr + 3] = pos

        print(pos)
        mujoco.mj_forward(self.sim.model, self.sim.data)

        # get_start_position
        pos = self.sim.data.body(self.start_marker).xpos
        print(pos)

        vertical_offset = 0.005

        pos[2] = pos[2] + vertical_offset
        joint_id = self.sim.model.joint(self.ball_join).id
        qpos_adr = self.sim.model.jnt_qposadr[joint_id]

        self.sim.data.qpos[qpos_adr : qpos_adr + 3] = pos

        return obs, info


class JoinBlocksTaskWrapper(gym.Wrapper):
    """
    wrapper to check if the maze task is succesful

    and to conduct a soft reset of the task
    """

    def __init__(self, env, cube_geom_name="box_geom"):
        super().__init__(env)
        assert isinstance(self.get_wrapper_attr("robot"), sim.SimRobot), "Robot must be a sim.SimRobot instance."
        self._robot = cast(sim.SimRobot, self.get_wrapper_attr("robot"))
        self.sim = self.env.get_wrapper_attr("sim")
        self.home_pose = self._robot.get_cartesian_position()
        self._gripper_closing = 0
        self._gripper = self.get_wrapper_attr("gripper")
        self.h_block = "h_blockhblock"
        self.marker_block = "h_blockmarker_block"
        self.marker_wall = "h_blockmarker_wall"
        self.h_block_joint = "h_blockhblock_joint"

        self.p_block = "p_blockpblock"
        self.block_peg_name = "p_blockblock_peg"
        self.p_block_joint = "p_blockpblock_joint"

        self.wall = "wallwall"
        self.wall_peg_name = "wallwall_peg"

        self.block_dist = 0.01
        self.wall_dist = 0.01

        self.phase = [True, False, False]  # [approaching, connecting blocks, connecting wall]

    def calculate_reward(self):

        contacts = []
        reward = 0

        # get all contact pairs
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]

            g1 = self.sim.model.geom(c.geom1).name
            g2 = self.sim.model.geom(c.geom2).name

            contacts.append((g1, g2))

        # get connection point positions

        block_peg_pos = get_geom_pos(self.sim.model, self.sim.data, self.block_peg_name)
        wall_peg_pos = get_geom_pos(self.sim.model, self.sim.data, self.wall_peg_name)
        wall_marker_pos = get_geom_pos(self.sim.model, self.sim.data, self.marker_wall)
        block_marker_pos = get_geom_pos(self.sim.model, self.sim.data, self.marker_block)

        # check approach phase:
        p_block_lift = any(self.p_block in (a, b) for a, b in contacts)
        h_block_lift = any(self.h_block in (a, b) for a, b in contacts)

        if h_block_lift and p_block_lift:
            self.phase[0] = False
            self.phase[1] = True
        else:
            self.phase = [True, False, False]

        # check block connect phase
        diff_block = abs(block_peg_pos - block_marker_pos)

        reward_block = 0
        if diff_block[0] < 0.005 and diff_block[1] < 0.005 and diff_block[2] < 0.05:

            reward_block = 0.5
            self.phase = [False, False, True]
        else:
            reward_block = 0

        # check wall connect phase

        diff_wall = abs(wall_peg_pos - wall_marker_pos)
        reward_wall = 0
        if diff_wall[0] < 0.005 and diff_wall[1] < 0.005 and diff_wall[2] < 0.01:
            reward_wall = 0.5
            self.phase = [False, False, False]
        else:
            reward_wall = 0

        reward = reward_wall + reward_block
        return reward, self.phase

    def step(self, action: dict[str, Any]):  # type: ignore
        obs, reward, _, truncated, info = super().step(action)
        success = False

        reward, phase = self.calculate_reward()

        info["pahse"] = phase

        if reward == 1:
            success = True
        return obs, reward, success, truncated, info

    def reset(self, scene_dict, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = super().reset()

        # TODO move to somewhere where it fits
        x_max_t = 1
        x_min_t = 1
        y_max_t = 1
        z_t = 0.1

        # randmoise the board position
        random.seed(seed)
        x_min = x_min_t
        x_max = x_max_t
        y_min = y_max_t
        y_max = y_max_t
        z = z_t

        # randmoise_p_block

        old_p_block_pos = self.sim.data.body(self.p_block).xpos
        pos = random_position_in_bounds(old_p_block_pos, x_min, x_max, y_min, y_max, z, random)
        joint_id = self.sim.model.joint(self.p_block_joint).id
        qpos_adr = self.sim.model.jnt_qposadr[joint_id]

        self.sim.data.qpos[qpos_adr : qpos_adr + 3] = pos
        mujoco.mj_forward(self.sim.model, self.sim.data)

        # randmosie h_block
        old_h_block_pos = self.sim.data.body(self.h_block).xpos
        pos = random_position_in_bounds(old_h_block_pos, x_min, x_max, y_min, y_max, z, random)
        joint_id = self.sim.model.joint(self.h_block_joint).id
        qpos_adr = self.sim.model.jnt_qposadr[joint_id]

        self.sim.data.qpos[qpos_adr : qpos_adr + 3] = pos
        mujoco.mj_forward(self.sim.model, self.sim.data)

        # randmise_wall

        old_wall_pos = self.sim.data.body(self.wall).xpos
        pos = random_position_in_bounds(old_wall_pos, x_min, x_max, y_min, y_max, z, random)
        body_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, self.wall)
        self.sim.model.body_pos[body_id] = pos
        mujoco.mj_forward(self.sim.model, self.sim.data)

        return obs, info


class DigitalTwin(gym.Wrapper):
    def __init__(self, env, twin_env):
        super().__init__(env)
        self.twin_env = twin_env

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        twin_obs, _, _, _, _ = self.twin_env.step(obs)
        info["twin_obs"] = twin_obs
        return obs, reward, terminated, truncated, info
