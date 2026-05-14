"""Color-based block detector for RGB wrist-camera images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class Detection2D:
    color: str
    centroid_uv: np.ndarray
    area_px: float
    bbox_xywh: tuple[int, int, int, int]
    confidence: float
    covariance_uv: np.ndarray
    mask: np.ndarray | None = None


class ColorBlockDetector:
    """Detect known colored blocks with HSV thresholding and contour filtering."""

    def __init__(self, config: dict[str, Any]):
        self.config = config.get("perception", config)
        self.colors = self.config.get("colors", {})
        if not self.colors:
            raise ValueError("perception.colors must define at least one target color.")
        self.min_area_px = float(self.config.get("min_area_px", 50))
        self.max_area_px = float(self.config.get("max_area_px", 10000))
        self.morph_kernel = int(self.config.get("morph_kernel", 3))
        self.confidence_min_area_px = float(
            self.config.get("confidence_min_area_px", self.min_area_px)
        )
        self.confidence_full_area_px = float(
            self.config.get("confidence_full_area_px", max(self.min_area_px * 4.0, 1.0))
        )
        self.pixel_sigma_base = float(self.config.get("pixel_sigma_base", 3.0))
        self.confidence_floor = float(self.config.get("confidence_floor", 0.2))
        self.return_masks = bool(self.config.get("return_masks", False))

    def detect(self, image_rgb: np.ndarray, target_color: str | None = None) -> list[Detection2D]:
        """Return detections for all configured colors or only `target_color`."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape (H, W, 3).")
        if image_rgb.dtype != np.uint8:
            raise ValueError("image_rgb must be uint8 RGB data in the [0, 255] range.")
        if target_color is not None and target_color not in self.colors:
            raise ValueError(f"Unknown target color '{target_color}'.")

        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        colors = [target_color] if target_color is not None else list(self.colors)
        detections: list[Detection2D] = []

        for color in colors:
            if color not in self.colors:
                continue
            mask = self._mask_for_color(hsv, self.colors[color])
            mask = self._clean_mask(mask)
            detections.extend(self._detections_from_mask(color, mask))

        detections.sort(key=lambda det: det.confidence, reverse=True)
        return detections

    def _mask_for_color(self, hsv: np.ndarray, color_cfg: dict[str, Any]) -> np.ndarray:
        ranges = color_cfg.get("hsv_ranges")
        if ranges is None:
            ranges = [
                {
                    "lower": color_cfg["hsv_lower"],
                    "upper": color_cfg["hsv_upper"],
                }
            ]

        masks = []
        for hsv_range in ranges:
            lower = _hsv_threshold(hsv_range["lower"])
            upper = _hsv_threshold(hsv_range["upper"])

            if lower[0] <= upper[0]:
                masks.append(cv2.inRange(hsv, lower, upper))
            else:
                low_a = lower.copy()
                high_a = np.array([179, upper[1], upper[2]], dtype=np.uint8)
                low_b = np.array([0, lower[1], lower[2]], dtype=np.uint8)
                high_b = upper.copy()
                masks.append(
                    cv2.bitwise_or(
                        cv2.inRange(hsv, low_a, high_a),
                        cv2.inRange(hsv, low_b, high_b),
                    )
                )

        out = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for mask in masks:
            out = cv2.bitwise_or(out, mask)
        return out

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.morph_kernel <= 1:
            return mask
        kernel = np.ones((self.morph_kernel, self.morph_kernel), dtype=np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    def _detections_from_mask(self, color: str, mask: np.ndarray) -> list[Detection2D]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection2D] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area_px or area > self.max_area_px:
                continue

            moments = cv2.moments(contour)
            if abs(moments["m00"]) < 1e-9:
                continue

            u = float(moments["m10"] / moments["m00"])
            v = float(moments["m01"] / moments["m00"])
            x, y, w, h = cv2.boundingRect(contour)
            confidence = self._confidence(area, contour)
            sigma = self.pixel_sigma_base / max(confidence, self.confidence_floor)
            covariance = np.eye(2, dtype=np.float64) * (sigma**2)
            component_mask = None
            if self.return_masks:
                component_mask = np.zeros_like(mask)
                cv2.drawContours(component_mask, [contour], -1, 255, thickness=-1)

            detections.append(
                Detection2D(
                    color=color,
                    centroid_uv=np.array([u, v], dtype=np.float64),
                    area_px=area,
                    bbox_xywh=(int(x), int(y), int(w), int(h)),
                    confidence=confidence,
                    covariance_uv=covariance,
                    mask=component_mask,
                )
            )
        return detections

    def _confidence(self, area: float, contour: np.ndarray) -> float:
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = max(float(w * h), 1.0)
        fill_ratio = np.clip(area / rect_area, 0.0, 1.0)
        area_score = (area - self.confidence_min_area_px) / max(
            self.confidence_full_area_px - self.confidence_min_area_px,
            1.0,
        )
        area_score = np.clip(area_score, 0.0, 1.0)
        return float(np.clip(0.6 * area_score + 0.4 * fill_ratio, 0.0, 1.0))


def _hsv_threshold(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.shape != (3,):
        raise ValueError("HSV thresholds must contain exactly 3 values.")
    if not (0 <= arr[0] <= 179 and np.all((0 <= arr[1:]) & (arr[1:] <= 255))):
        raise ValueError("HSV thresholds must be [h in 0..179, s/v in 0..255].")
    return arr.astype(np.uint8)
