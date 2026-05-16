from __future__ import annotations


def cube_table_xy_to_coarse_pregrasp_ee_xy(
    table_x: float,
    table_y: float,
) -> tuple[float, float]:
    """
    Convert a detected cube position in our table coordinate frame
    into a rough LeRobot EE-frame XY target for coarse pre-grasp motion.

    This is intentionally approximate: the RL pickup skill is expected
    to handle the remaining local alignment error.
    """
    ee_x = table_y - 0.0806
    ee_y = -table_x - 0.0076
    return ee_x, ee_y