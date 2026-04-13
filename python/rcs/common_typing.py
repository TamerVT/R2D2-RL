# ATTENTION: auto generated from C++ stub files, use `make stubgen` to update!
"""TypedDict helpers generated from `python/rcs/_core/common.pyi`."""
from __future__ import annotations

from typing import TypedDict

from rcs._core import common

__all__ = ["BaseCameraConfigKwargs", "RobotConfigKwargs"]


class BaseCameraConfigKwargs(TypedDict, total=False):
    identifier: str
    frame_rate: int
    resolution_width: int
    resolution_height: int


class RobotConfigKwargs(TypedDict, total=False):
    robot_type: common.RobotType
    robot_platform: common.RobotPlatform
    tcp_offset: common.Pose
    attachment_site: str
    kinematic_model_path: str
