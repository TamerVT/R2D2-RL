from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CubeDetection:
    """
    One detected cube candidate in image coordinates.
    """
    centroid_uv: tuple[float, float]
    bbox_xywh: tuple[int, int, int, int]
    area_px: float
    color: str
    contour: np.ndarray


@dataclass(frozen=True)
class CubeDetectorConfig:
    """
    Simple HSV-based cube detector.

    The thresholds are intentionally conservative starting values.
    They can be tuned from debug overlays if needed.
    """
    min_area_px: float = 80.0
    max_area_px: float = 12000.0

    # Shape sanity filter on bounding-box aspect ratio.
    min_aspect_ratio: float = 0.35
    max_aspect_ratio: float = 2.8

    # Morphological cleanup.
    morph_kernel_size: int = 5
    morph_open_iterations: int = 1
    morph_close_iterations: int = 2


# HSV ranges are in OpenCV convention:
# H ∈ [0, 179], S ∈ [0, 255], V ∈ [0, 255].
#
# These ranges are tightened around the known project cube palette.
# Earlier camera-color checks gave approximate OpenCV-HSV centers of:
#   red    ≈ H=1
#   orange ≈ H=6
#   yellow ≈ H=24
#   blue   ≈ H=108
#   purple ≈ H=136
#
# Green is kept somewhat wider for now because its earlier camera-vs-XML
# comparison was the least clean of the six colors.
#
# Goal:
#   - reduce false positive colored blobs from the robot/background,
#   - reduce overlap between red/orange/yellow,
#   - retain enough tolerance for lighting variation in the wrist scans.
COLOR_RANGES_HSV: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
    "red": [
        (
            np.array([0, 110, 80], dtype=np.uint8),
            np.array([4, 255, 255], dtype=np.uint8),
        ),
        (
            np.array([176, 110, 80], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8),
        ),
    ],
    "orange": [
        (
            np.array([5, 100, 80], dtype=np.uint8),
            np.array([15, 255, 255], dtype=np.uint8),
        ),
    ],
    "yellow": [
        (
            np.array([18, 90, 90], dtype=np.uint8),
            np.array([34, 255, 255], dtype=np.uint8),
        ),
    ],
    "green": [
        (
            # Project cube XML green rgba=(0.125, 0.522, 0.302, 1)
            # corresponds approximately to OpenCV HSV=(73, 194, 133).
            np.array([62, 90, 60], dtype=np.uint8),
            np.array([84, 255, 230], dtype=np.uint8),
        ),
    ],
    "blue": [
        (
            np.array([98, 110, 60], dtype=np.uint8),
            np.array([118, 255, 255], dtype=np.uint8),
        ),
    ],
    "purple": [
        (
            np.array([126, 70, 60], dtype=np.uint8),
            np.array([150, 255, 230], dtype=np.uint8),
        ),
    ],
}


def _build_color_mask(
    hsv_image: np.ndarray,
    color: str,
) -> np.ndarray:
    if color not in COLOR_RANGES_HSV:
        raise ValueError(
            f"Unknown color {color!r}. "
            f"Available colors: {sorted(COLOR_RANGES_HSV.keys())}"
        )

    masks = []
    for lower, upper in COLOR_RANGES_HSV[color]:
        masks.append(cv2.inRange(hsv_image, lower, upper))

    if len(masks) == 1:
        return masks[0]

    merged = masks[0]
    for mask in masks[1:]:
        merged = cv2.bitwise_or(merged, mask)
    return merged


def _clean_mask(mask: np.ndarray, config: CubeDetectorConfig) -> np.ndarray:
    k = config.morph_kernel_size
    kernel = np.ones((k, k), dtype=np.uint8)

    cleaned = mask.copy()

    if config.morph_open_iterations > 0:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            kernel,
            iterations=config.morph_open_iterations,
        )

    if config.morph_close_iterations > 0:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=config.morph_close_iterations,
        )

    return cleaned


def _detections_from_mask(
    mask: np.ndarray,
    color: str,
    config: CubeDetectorConfig,
) -> list[CubeDetection]:
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    detections: list[CubeDetection] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < config.min_area_px or area > config.max_area_px:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h <= 0 or w <= 0:
            continue

        aspect_ratio = w / h
        if not (
            config.min_aspect_ratio
            <= aspect_ratio
            <= config.max_aspect_ratio
        ):
            continue

        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-9:
            continue

        u = float(moments["m10"] / moments["m00"])
        v = float(moments["m01"] / moments["m00"])

        detections.append(
            CubeDetection(
                centroid_uv=(u, v),
                bbox_xywh=(x, y, w, h),
                area_px=area,
                color=color,
                contour=contour,
            )
        )

    return detections


def detect_colored_cubes(
    image_rgb: np.ndarray,
    *,
    target_color: str | None = None,
    config: CubeDetectorConfig | None = None,
) -> tuple[list[CubeDetection], dict[str, Any]]:
    """
    Detect colored cube-like regions in an RGB image.

    Args:
        image_rgb:
            RGB image of shape (H, W, 3).
        target_color:
            If provided, only detect that color.
            If None, detect across all configured cube colors.
        config:
            Optional detector config.

    Returns:
        detections:
            List of cube candidates.
        debug:
            Dictionary containing masks useful for visualization.
    """
    if config is None:
        config = CubeDetectorConfig()

    image_rgb = np.asarray(image_rgb)

    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image shape (H, W, 3), got {image_rgb.shape}."
        )

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    colors_to_check = (
        [target_color]
        if target_color is not None
        else list(COLOR_RANGES_HSV.keys())
    )

    detections: list[CubeDetection] = []
    raw_masks: dict[str, np.ndarray] = {}
    clean_masks: dict[str, np.ndarray] = {}

    for color in colors_to_check:
        raw_mask = _build_color_mask(hsv, color)
        clean_mask = _clean_mask(raw_mask, config)

        raw_masks[color] = raw_mask
        clean_masks[color] = clean_mask

        detections.extend(
            _detections_from_mask(
                clean_mask,
                color=color,
                config=config,
            )
        )

    # Larger colored blobs first. For Task 1, the top candidate is often the cube.
    detections.sort(key=lambda det: det.area_px, reverse=True)

    debug = {
        "hsv": hsv,
        "raw_masks": raw_masks,
        "clean_masks": clean_masks,
    }

    return detections, debug


def draw_cube_detections(
    image_rgb: np.ndarray,
    detections: list[CubeDetection],
    *,
    selected_index: int | None = None,
) -> np.ndarray:
    """
    Draw detector outputs onto a copy of the RGB image.
    """
    overlay_rgb = image_rgb.copy()

    for idx, detection in enumerate(detections):
        x, y, w, h = detection.bbox_xywh
        u, v = detection.centroid_uv

        color_rgb = (0, 255, 0)
        thickness = 2

        if selected_index is not None and idx == selected_index:
            color_rgb = (255, 0, 0)
            thickness = 3

        cv2.rectangle(
            overlay_rgb,
            (x, y),
            (x + w, y + h),
            color_rgb,
            thickness,
        )
        cv2.circle(
            overlay_rgb,
            (int(round(u)), int(round(v))),
            5,
            color_rgb,
            -1,
        )

        label = f"{idx}:{detection.color} A={detection.area_px:.0f}"
        cv2.putText(
            overlay_rgb,
            label,
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color_rgb,
            2,
            cv2.LINE_AA,
        )

    return overlay_rgb
