from rcs_panda._core import hw

import rcs
from rcs import common


def default_panda_hw_robot_cfg(ip: str = "", async_control: bool = False) -> hw.PandaConfig:
    robot_cfg = hw.PandaConfig(ip=ip)
    robot_cfg.robot_type = rcs.scenes["panda_empty_world"].robot_type
    robot_cfg.kinematic_model_path = rcs.scenes["panda_empty_world"].mjcf_robot
    robot_cfg.tcp_offset = common.Pose(common.FrankaHandTCPOffset())
    robot_cfg.attachment_site = "attachment_site_0"
    robot_cfg.speed_factor = 0.1
    robot_cfg.ik_solver = hw.IKSolver.rcs_ik
    robot_cfg.async_control = async_control
    return robot_cfg


def default_panda_hw_gripper_cfg(ip: str = "", async_control: bool = False) -> hw.FHConfig:
    gripper_cfg = hw.FHConfig(ip=ip)
    gripper_cfg.epsilon_inner = gripper_cfg.epsilon_outer = 0.1
    gripper_cfg.speed = 0.1
    gripper_cfg.force = 30
    gripper_cfg.async_control = async_control
    return gripper_cfg
