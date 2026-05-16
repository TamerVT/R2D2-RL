from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_homography(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Homography file not found: {path}")

    H = np.load(path)

    if H.shape != (3, 3):
        raise ValueError(f"Expected homography shape (3, 3), got {H.shape}")

    return H.astype(np.float64)


def pixel_to_table_xy(
    pixel_uv: tuple[float, float] | np.ndarray,
    H_image_to_table: np.ndarray,
) -> np.ndarray:
    """
    Convert one image pixel coordinate (u, v) to table/robot xy coordinates.

    Args:
        pixel_uv:
            One pixel coordinate.
        H_image_to_table:
            3x3 homography mapping image pixels to table xy.

    Returns:
        np.ndarray of shape (2,), [x, y].
    """
    uv = np.asarray(pixel_uv, dtype=np.float64).reshape(1, 1, 2)
    xy = cv2.perspectiveTransform(uv, H_image_to_table)
    return xy.reshape(2)


def pixels_to_table_xy(
    pixels_uv: np.ndarray,
    H_image_to_table: np.ndarray,
) -> np.ndarray:
    """
    Convert multiple image pixel coordinates to table/robot xy coordinates.

    Args:
        pixels_uv:
            Array of shape (N, 2).
        H_image_to_table:
            3x3 homography mapping image pixels to table xy.

    Returns:
        Array of shape (N, 2).
    """
    pixels_uv = np.asarray(pixels_uv, dtype=np.float64)

    if pixels_uv.ndim != 2 or pixels_uv.shape[1] != 2:
        raise ValueError(f"Expected pixels_uv shape (N, 2), got {pixels_uv.shape}")

    uv = pixels_uv.reshape(-1, 1, 2)
    xy = cv2.perspectiveTransform(uv, H_image_to_table)
    return xy.reshape(-1, 2)
