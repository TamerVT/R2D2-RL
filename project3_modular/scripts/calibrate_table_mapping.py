from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "data" / "calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate per-observation-pose pixel-to-table homographies "
            "from a saved observation scan."
        )
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        required=True,
        help=(
            "Folder containing center.png, left.png, right.png from "
            "capture_observation_scan.py."
        ),
    )
    parser.add_argument(
        "--marker-points-json",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR / "marker_world_points.json",
        help="JSON file containing known marker table coordinates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Where to write homographies and clicked point metadata.",
    )
    parser.add_argument(
        "--poses",
        nargs="+",
        default=["center", "left", "right"],
        help="Pose/image names to calibrate.",
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=0.01,
        help=(
            "RANSAC reprojection threshold in target-coordinate units. "
            "Default 0.01 means 1 cm if using meters."
        ),
    )
    return parser.parse_args()


def load_marker_world_points(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]], dict[str, Any]]:
    """
    Load:
      - all known marker table coordinates, indexed by marker ID
      - visible marker IDs per observation pose
      - the raw JSON metadata
    """
    if not path.exists():
        raise FileNotFoundError(f"Marker coordinate file not found: {path}")

    data = json.loads(path.read_text())

    markers = data.get("markers")
    if not isinstance(markers, list) or len(markers) < 4:
        raise ValueError("marker_world_points.json must contain at least 4 markers.")

    marker_lookup: dict[str, np.ndarray] = {}

    for marker in markers:
        marker_id = str(marker["id"])
        x = float(marker["x"])
        y = float(marker["y"])

        if marker_id in marker_lookup:
            raise ValueError(f"Duplicate marker ID: {marker_id}")

        marker_lookup[marker_id] = np.array([x, y], dtype=np.float64)

    visible_markers = data.get("visible_markers")
    if not isinstance(visible_markers, dict):
        raise ValueError(
            "marker_world_points.json must contain a 'visible_markers' object."
        )

    for pose_name, marker_ids in visible_markers.items():
        if not isinstance(marker_ids, list):
            raise ValueError(
                f"visible_markers[{pose_name!r}] must be a list of marker IDs."
            )

        if len(marker_ids) < 4:
            raise ValueError(
                f"Pose {pose_name!r} has only {len(marker_ids)} visible markers. "
                "At least 4 are required for a homography."
            )

        unknown = [marker_id for marker_id in marker_ids if marker_id not in marker_lookup]
        if unknown:
            raise ValueError(
                f"Pose {pose_name!r} references unknown marker IDs: {unknown}"
            )

    return marker_lookup, visible_markers, data


def load_rgb_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    return np.asarray(Image.open(path).convert("RGB"))


def click_points_for_pose(
    image: np.ndarray,
    pose_name: str,
    marker_ids: list[str],
) -> np.ndarray:
    """
    Ask the user to click marker centers in the order given by marker_ids.

    Matplotlib's ginput supports:
      - left click: add point
      - middle click: remove last point
      - right click: finish early
    """
    n_points = len(marker_ids)
    click_order = " → ".join(marker_ids)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(
        f"{pose_name}: click marker centers in this order:\n{click_order}",
        fontsize=12,
    )
    ax.set_xlabel(
        "Left click = add point | Middle click = undo last point | "
        f"Need exactly {n_points} clicks"
    )
    ax.set_axis_off()

    print(f"\n[{pose_name}] Click {n_points} marker centers in order:")
    print("  " + click_order)
    print("  Left click = add point, middle click = undo last point.")

    points = plt.ginput(n_points, timeout=-1, show_clicks=True)
    plt.close(fig)

    if len(points) != n_points:
        raise RuntimeError(
            f"Pose {pose_name!r}: expected {n_points} clicks, got {len(points)}."
        )

    return np.asarray(points, dtype=np.float64)


def compute_homography(
    image_points: np.ndarray,
    world_points: np.ndarray,
    ransac_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute H mapping image pixel coordinates [u,v] -> table coordinates [x,y].
    """
    H, inlier_mask = cv2.findHomography(
        srcPoints=image_points,
        dstPoints=world_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
    )

    if H is None:
        raise RuntimeError("cv2.findHomography failed.")

    if inlier_mask is None:
        inlier_mask = np.ones((len(image_points), 1), dtype=np.uint8)

    return H.astype(np.float64), inlier_mask.reshape(-1).astype(bool)


def apply_homography(H: np.ndarray, image_points: np.ndarray) -> np.ndarray:
    """
    Apply image-pixel -> table-coordinate homography to Nx2 points.
    """
    pts = image_points.reshape(-1, 1, 2).astype(np.float64)
    transformed = cv2.perspectiveTransform(pts, H)
    return transformed.reshape(-1, 2)


def reprojection_report(
    marker_ids: list[str],
    predicted_world: np.ndarray,
    true_world: np.ndarray,
    inlier_mask: np.ndarray,
) -> dict[str, Any]:
    errors = np.linalg.norm(predicted_world - true_world, axis=1)
    rmse = float(np.sqrt(np.mean(errors**2)))
    mean_err = float(np.mean(errors))
    max_err = float(np.max(errors))

    per_marker = []
    for idx, marker_id in enumerate(marker_ids):
        per_marker.append(
            {
                "id": marker_id,
                "predicted_xy": predicted_world[idx].tolist(),
                "true_xy": true_world[idx].tolist(),
                "error": float(errors[idx]),
                "ransac_inlier": bool(inlier_mask[idx]),
            }
        )

    return {
        "rmse": rmse,
        "mean_error": mean_err,
        "max_error": max_err,
        "per_marker": per_marker,
    }


def save_debug_overlay(
    image: np.ndarray,
    pose_name: str,
    marker_ids: list[str],
    image_points: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(f"Calibration clicks: {pose_name}")
    ax.set_axis_off()

    for marker_id, (u, v) in zip(marker_ids, image_points):
        ax.scatter([u], [v], s=60)
        ax.text(u + 5, v + 5, marker_id, fontsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_pose_calibration(
    output_dir: Path,
    pose_name: str,
    H: np.ndarray,
    marker_ids: list[str],
    image_points: np.ndarray,
    world_points: np.ndarray,
    report: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    homography_path = output_dir / f"homography_{pose_name}.npy"
    points_path = output_dir / f"table_points_{pose_name}.json"

    np.save(homography_path, H)

    payload = {
        "pose_name": pose_name,
        "homography_file": str(homography_path),
        "mapping_direction": "image_pixels_uv_to_table_xy",
        "markers": [
            {
                "id": marker_id,
                "image_uv": image_points[idx].tolist(),
                "table_xy": world_points[idx].tolist(),
            }
            for idx, marker_id in enumerate(marker_ids)
        ],
        "reprojection_report": report,
    }

    points_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"Saved homography: {homography_path}")
    print(f"Saved point metadata: {points_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    marker_lookup, visible_markers, marker_config = load_marker_world_points(
        args.marker_points_json
    )

    summary: dict[str, Any] = {
        "scan_dir": str(args.scan_dir),
        "marker_points_json": str(args.marker_points_json),
        "coordinate_convention": marker_config.get("coordinate_convention", {}),
        "poses": {},
    }

    for pose_name in args.poses:
        if pose_name not in visible_markers:
            raise ValueError(
                f"No visible marker list found for pose {pose_name!r}. "
                f"Available poses: {sorted(visible_markers.keys())}"
            )

        marker_ids = visible_markers[pose_name]
        world_points = np.asarray(
            [marker_lookup[marker_id] for marker_id in marker_ids],
            dtype=np.float64,
        )

        image_path = args.scan_dir / f"{pose_name}.png"
        image = load_rgb_image(image_path)

        image_points = click_points_for_pose(
            image=image,
            pose_name=pose_name,
            marker_ids=marker_ids,
        )

        H, inlier_mask = compute_homography(
            image_points=image_points,
            world_points=world_points,
            ransac_threshold=args.ransac_threshold,
        )

        predicted_world = apply_homography(H, image_points)
        report = reprojection_report(
            marker_ids=marker_ids,
            predicted_world=predicted_world,
            true_world=world_points,
            inlier_mask=inlier_mask,
        )

        print(f"\n[{pose_name}] calibration quality:")
        print(f"  RMSE:       {report['rmse']:.5f}")
        print(f"  Mean error: {report['mean_error']:.5f}")
        print(f"  Max error:  {report['max_error']:.5f}")
        print(f"  Inliers:    {int(inlier_mask.sum())}/{len(inlier_mask)}")

        save_pose_calibration(
            output_dir=args.output_dir,
            pose_name=pose_name,
            H=H,
            marker_ids=marker_ids,
            image_points=image_points,
            world_points=world_points,
            report=report,
        )

        overlay_path = args.output_dir / f"calibration_overlay_{pose_name}.png"
        save_debug_overlay(
            image=image,
            pose_name=pose_name,
            marker_ids=marker_ids,
            image_points=image_points,
            output_path=overlay_path,
        )
        print(f"Saved click overlay: {overlay_path}")

        summary["poses"][pose_name] = report

    summary_path = args.output_dir / "homography_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\nCalibration complete.")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
