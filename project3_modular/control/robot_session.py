from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lerobot.cameras.configs import Cv2Backends

try:
    # Current public import path used in LeRobot docs
    from lerobot.cameras.opencv import OpenCVCameraConfig
except ImportError:
    # Fallback for slightly different LeRobot versions
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


@dataclass(frozen=True)
class RobotSessionConfig:
    """Configuration for the follower arm + wrist camera."""
    robot_port: str = "/dev/ttyACM2"
    robot_id: str = "my_awesome_follower_arm"

    camera_name: str | None = "front"
    camera_index_or_path: int | str = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    camera_fourcc: str = "MJPG"
    disable_torque_on_disconnect: bool = False


def make_follower_robot(config: RobotSessionConfig) -> SO101Follower:
    """Construct the SO101 follower robot, optionally with a wrist camera."""
    if config.camera_name is None:
        camera_config = {}
    else:
        camera_config = {
            config.camera_name: OpenCVCameraConfig(
                index_or_path=config.camera_index_or_path,
                backend=Cv2Backends.V4L2,
                fourcc=config.camera_fourcc,
                width=config.camera_width,
                height=config.camera_height,
                fps=config.camera_fps,
            )
        }

    robot_config = SO101FollowerConfig(
        port=config.robot_port,
        id=config.robot_id,
        cameras=camera_config,
        disable_torque_on_disconnect=config.disable_torque_on_disconnect,
    )

    return SO101Follower(robot_config)


class RobotSession:
    """
    Context manager that connects/disconnects the SO101 follower safely.

    Usage:
        with RobotSession(config) as robot:
            obs = robot.get_observation()
    """

    def __init__(self, config: RobotSessionConfig):
        self.config = config
        self.robot = make_follower_robot(config)

    def __enter__(self) -> SO101Follower:
        self.robot.connect()
        return self.robot

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.robot.disconnect()