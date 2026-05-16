from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from perception.cube_detector import (
    CubeDetection,
    CubeDetectorConfig,
    detect_colored_cubes,
)
from perception.homography import load_homography, pixel_to_table_xy


DEFAULT_POSES = ("center", "left", "right")


@dataclass(frozen=True)
class WorkspaceBounds:
    x_min: float = -0.30
    x_max: float = 0.30
    y_min: float = 0.05
    y_max: float = 0.65

    def contains(self, xy: np.ndarray) -> bool:
        x, y = float(xy[0]), float(xy[1])
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "x": [self.x_min, self.x_max],
            "y": [self.y_min, self.y_max],
        }


def load_rgb_image(path: str | Path) -> np.ndarray:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    return np.asarray(Image.open(path).convert("RGB"))


def map_detections_to_table_candidates(
    *,
    pose_name: str,
    detections: list[CubeDetection],
    H_image_to_table: np.ndarray,
    workspace_bounds: WorkspaceBounds,
) -> list[dict[str, Any]]:
    """
    Convert image-space cube detections into table-coordinate candidate reports.
    """
    candidates: list[dict[str, Any]] = []

    for detection_index, detection in enumerate(detections):
        table_xy = pixel_to_table_xy(
            detection.centroid_uv,
            H_image_to_table,
        )
        inside_workspace = workspace_bounds.contains(table_xy)

        candidates.append(
            {
                "pose_name": pose_name,
                "detection_index": detection_index,
                "color": detection.color,
                "centroid_uv": list(detection.centroid_uv),
                "bbox_xywh": list(detection.bbox_xywh),
                "area_px": float(detection.area_px),
                "table_xy": table_xy.tolist(),
                "inside_workspace": bool(inside_workspace),
            }
        )

    return candidates


def cluster_multiview_candidates(
    all_candidates: list[dict[str, Any]],
    *,
    cluster_radius_m: float = 0.02,
) -> list[dict[str, Any]]:
    """
    Cluster mapped cube candidates in table-coordinate space.

    Important rule:
    A cluster may contain at most one candidate from each observation pose.
    This prevents duplicate detections within one image from inflating a cluster.
    """
    clusters: list[dict[str, Any]] = []

    for candidate in all_candidates:
        xy = np.asarray(candidate["table_xy"], dtype=np.float64)

        assigned = False

        for cluster in clusters:
            center = np.asarray(cluster["center_xy"], dtype=np.float64)
            distance = float(np.linalg.norm(xy - center))

            if distance > cluster_radius_m:
                continue

            existing_poses = {
                member["pose_name"]
                for member in cluster["members"]
            }

            if candidate["pose_name"] in existing_poses:
                continue

            cluster["members"].append(candidate)

            member_xy = np.asarray(
                [member["table_xy"] for member in cluster["members"]],
                dtype=np.float64,
            )
            cluster["center_xy"] = np.median(member_xy, axis=0).tolist()

            assigned = True
            break

        if not assigned:
            clusters.append(
                {
                    "members": [candidate],
                    "center_xy": xy.tolist(),
                }
            )

    for cluster in clusters:
        members = cluster["members"]

        member_xy = np.asarray(
            [member["table_xy"] for member in members],
            dtype=np.float64,
        )

        fused_xy = np.median(member_xy, axis=0)
        residuals = np.linalg.norm(
            member_xy - fused_xy[None, :],
            axis=1,
        )

        distinct_views = sorted(
            {member["pose_name"] for member in members}
        )

        cluster["num_members"] = int(len(members))
        cluster["num_distinct_views"] = int(len(distinct_views))
        cluster["distinct_views"] = distinct_views
        cluster["fused_xy"] = fused_xy.tolist()
        cluster["max_residual_m"] = (
            float(residuals.max()) if len(residuals) else 0.0
        )
        cluster["mean_residual_m"] = (
            float(residuals.mean()) if len(residuals) else 0.0
        )
        cluster["total_area_px"] = float(
            sum(member["area_px"] for member in members)
        )

    # Priority:
    # 1. support from more distinct views
    # 2. more total members
    # 3. tighter spatial agreement
    # 4. larger total colored area as a weak final tie-break
    clusters.sort(
        key=lambda cluster: (
            cluster["num_distinct_views"],
            cluster["num_members"],
            -cluster["max_residual_m"],
            cluster["total_area_px"],
        ),
        reverse=True,
    )

    return clusters


def localize_cube_from_scan(
    *,
    scan_dir: str | Path,
    calibration_dir: str | Path,
    poses: Sequence[str] = DEFAULT_POSES,
    target_color: str | None = None,
    detector_config: CubeDetectorConfig | None = None,
    workspace_bounds: WorkspaceBounds | None = None,
    cluster_radius_m: float = 0.02,
    reliable_min_views: int = 2,
) -> dict[str, Any]:
    """
    Estimate the cube table position from a saved multi-view scan.

    Returns a dictionary with:
      - cube_xy: fused best estimate, or None
      - is_reliable: True if the best cluster is supported by enough views
      - best_cluster: chosen candidate cluster, or None
      - clusters: all candidate clusters
      - poses: per-pose images, detections, masks, and mapped candidates

    This function performs no plotting and saves no files.
    Debug scripts can build overlays from the returned data.
    """
    scan_dir = Path(scan_dir)
    calibration_dir = Path(calibration_dir)

    if detector_config is None:
        detector_config = CubeDetectorConfig()

    if workspace_bounds is None:
        workspace_bounds = WorkspaceBounds()

    pose_results: dict[str, Any] = {}
    all_valid_candidates: list[dict[str, Any]] = []

    for pose_name in poses:
        image_path = scan_dir / f"{pose_name}.png"
        homography_path = calibration_dir / f"homography_{pose_name}.npy"

        image_rgb = load_rgb_image(image_path)
        H_image_to_table = load_homography(homography_path)

        detections, detector_debug = detect_colored_cubes(
            image_rgb,
            target_color=target_color,
            config=detector_config,
        )

        candidates = map_detections_to_table_candidates(
            pose_name=pose_name,
            detections=detections,
            H_image_to_table=H_image_to_table,
            workspace_bounds=workspace_bounds,
        )

        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate["inside_workspace"]
        ]
        all_valid_candidates.extend(valid_candidates)

        pose_results[pose_name] = {
            "image_path": str(image_path),
            "homography_path": str(homography_path),
            "image_rgb": image_rgb,
            "homography": H_image_to_table,
            "detections": detections,
            "detector_debug": detector_debug,
            "candidates": candidates,
            "valid_candidates": valid_candidates,
        }

    clusters = cluster_multiview_candidates(
        all_valid_candidates,
        cluster_radius_m=cluster_radius_m,
    )

    if not clusters:
        return {
            "cube_xy": None,
            "is_reliable": False,
            "best_cluster": None,
            "clusters": [],
            "poses": pose_results,
            "workspace_bounds": workspace_bounds.to_dict(),
            "target_color": target_color,
        }

    best_cluster = clusters[0]
    cube_xy = np.asarray(best_cluster["fused_xy"], dtype=np.float64)

    is_reliable = (
        best_cluster["num_distinct_views"] >= reliable_min_views
    )

    return {
        "cube_xy": cube_xy,
        "is_reliable": bool(is_reliable),
        "best_cluster": best_cluster,
        "clusters": clusters,
        "poses": pose_results,
        "workspace_bounds": workspace_bounds.to_dict(),
        "target_color": target_color,
    }


def localization_result_to_jsonable(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert a localization result into a JSON-serializable summary.

    Heavy image arrays, homography arrays, contour objects, and masks are omitted.
    """
    cube_xy = result["cube_xy"]

    jsonable: dict[str, Any] = {
        "cube_xy": None if cube_xy is None else cube_xy.tolist(),
        "is_reliable": bool(result["is_reliable"]),
        "best_cluster": result["best_cluster"],
        "clusters": result["clusters"],
        "workspace_bounds": result["workspace_bounds"],
        "target_color": result["target_color"],
        "poses": {},
    }

    for pose_name, pose_result in result["poses"].items():
        jsonable["poses"][pose_name] = {
            "image_path": pose_result["image_path"],
            "homography_path": pose_result["homography_path"],
            "num_detections": len(pose_result["detections"]),
            "candidates": pose_result["candidates"],
            "valid_candidates": pose_result["valid_candidates"],
        }

    return jsonable
