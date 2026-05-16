from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "data" / "calibration"

sys.path.insert(0, str(PROJECT_ROOT))

from perception.homography import load_homography, pixels_to_table_xy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate saved pixel-to-table homographies on independent test points."
        )
    )
    parser.add_argument(
        "--scan-dir",
        type=Path,
        required=True,
        help="Folder containing center.png, left.png, right.png.",
    )
    parser.add_argument(
        "--test-points-json",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR / "test_points.json",
        help="JSON file containing ground-truth test point coordinates.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=DEFAULT_CALIBRATION_DIR,
        help="Folder containing homography_center.npy, etc.",
    )
    parser.add_argument(
        "--poses",
        nargs="+",
        default=["center", "left", "right"],
        help="Pose/image names to test.",
    )
    return parser.parse_args()


def load_test_points(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, list[str]] | None, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Test-point file not found: {path}")

    data = json.loads(path.read_text())
    points = data.get("test_points")

    if not isinstance(points, list) or len(points) < 1:
        raise ValueError("test_points.json must contain at least one test point.")

    point_lookup: dict[str, np.ndarray] = {}

    for point in points:
        point_id = str(point["id"])
        x = float(point["x"])
        y = float(point["y"])

        if point_id in point_lookup:
            raise ValueError(f"Duplicate test point ID: {point_id}")

        point_lookup[point_id] = np.array([x, y], dtype=np.float64)

    visible = data.get("visible_test_points")
    if visible is not None:
        if not isinstance(visible, dict):
            raise ValueError("'visible_test_points' must be an object/dict.")

        for pose_name, point_ids in visible.items():
            if not isinstance(point_ids, list):
                raise ValueError(
                    f"visible_test_points[{pose_name!r}] must be a list."
                )

            unknown = [pid for pid in point_ids if pid not in point_lookup]
            if unknown:
                raise ValueError(
                    f"Pose {pose_name!r} references unknown test point IDs: {unknown}"
                )

    return point_lookup, visible, data


def load_rgb_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    return np.asarray(Image.open(path).convert("RGB"))


def click_points_for_pose(
    image: np.ndarray,
    pose_name: str,
    point_ids: list[str],
) -> np.ndarray:
    n_points = len(point_ids)
    click_order = " → ".join(point_ids)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(
        f"{pose_name}: click test marker centers in this order:\n{click_order}",
        fontsize=12,
    )
    ax.set_xlabel(
        "Left click = add point | Middle click = undo last point | "
        f"Need exactly {n_points} clicks"
    )
    ax.set_axis_off()

    print(f"\n[{pose_name}] Click {n_points} test point center(s) in order:")
    print("  " + click_order)

    points = plt.ginput(
        n_points,
        timeout=-1,
        show_clicks=True,
        mouse_pop=2,
        mouse_stop=3,
    )
    plt.close(fig)

    if len(points) != n_points:
        raise RuntimeError(
            f"Pose {pose_name!r}: expected {n_points} clicks, got {len(points)}."
        )

    return np.asarray(points, dtype=np.float64)


def compute_pose_report(
    point_ids: list[str],
    predicted_xy: np.ndarray,
    true_xy: np.ndarray,
) -> dict[str, Any]:
    errors = np.linalg.norm(predicted_xy - true_xy, axis=1)

    per_point = []
    for idx, point_id in enumerate(point_ids):
        per_point.append(
            {
                "id": point_id,
                "predicted_xy": predicted_xy[idx].tolist(),
                "true_xy": true_xy[idx].tolist(),
                "error_m": float(errors[idx]),
                "error_cm": float(errors[idx] * 100.0),
            }
        )

    return {
        "mean_error_m": float(errors.mean()),
        "mean_error_cm": float(errors.mean() * 100.0),
        "max_error_m": float(errors.max()),
        "max_error_cm": float(errors.max() * 100.0),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "rmse_cm": float(np.sqrt(np.mean(errors**2)) * 100.0),
        "per_point": per_point,
    }


def save_debug_overlay(
    image: np.ndarray,
    pose_name: str,
    point_ids: list[str],
    image_points: np.ndarray,
    predicted_xy: np.ndarray,
    true_xy: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_title(f"Homography validation clicks: {pose_name}")
    ax.set_axis_off()

    for idx, point_id in enumerate(point_ids):
        u, v = image_points[idx]
        pred_x, pred_y = predicted_xy[idx]
        true_x, true_y = true_xy[idx]
        err_cm = np.linalg.norm(predicted_xy[idx] - true_xy[idx]) * 100.0

        ax.scatter([u], [v], s=70)
        ax.text(
            u + 5,
            v + 5,
            f"{point_id}\nerr={err_cm:.2f} cm",
            fontsize=11,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    point_lookup, visible_test_points, raw_json = load_test_points(
        args.test_points_json
    )

    summary: dict[str, Any] = {
        "scan_dir": str(args.scan_dir),
        "test_points_json": str(args.test_points_json),
        "poses": {},
    }

    all_errors_m: list[float] = []

    for pose_name in args.poses:
        if visible_test_points is None:
            point_ids = list(point_lookup.keys())
        else:
            if pose_name not in visible_test_points:
                raise ValueError(
                    f"No visible test point list provided for pose {pose_name!r}."
                )
            point_ids = visible_test_points[pose_name]

        if len(point_ids) == 0:
            print(f"\n[{pose_name}] No visible test points. Skipping.")
            continue

        true_xy = np.asarray(
            [point_lookup[point_id] for point_id in point_ids],
            dtype=np.float64,
        )

        image_path = args.scan_dir / f"{pose_name}.png"
        image = load_rgb_image(image_path)

        H_path = args.calibration_dir / f"homography_{pose_name}.npy"
        H = load_homography(H_path)

        image_points = click_points_for_pose(
            image=image,
            pose_name=pose_name,
            point_ids=point_ids,
        )

        predicted_xy = pixels_to_table_xy(image_points, H)

        report = compute_pose_report(
            point_ids=point_ids,
            predicted_xy=predicted_xy,
            true_xy=true_xy,
        )

        print(f"\n[{pose_name}] independent validation:")
        print(f"  RMSE:       {report['rmse_cm']:.3f} cm")
        print(f"  Mean error: {report['mean_error_cm']:.3f} cm")
        print(f"  Max error:  {report['max_error_cm']:.3f} cm")

        for point in report["per_point"]:
            pred = point["predicted_xy"]
            true = point["true_xy"]
            print(
                f"    {point['id']}: "
                f"pred=({pred[0]:.4f}, {pred[1]:.4f}) "
                f"true=({true[0]:.4f}, {true[1]:.4f}) "
                f"err={point['error_cm']:.3f} cm"
            )
            all_errors_m.append(point["error_m"])

        overlay_path = (
            args.calibration_dir / f"homography_validation_overlay_{pose_name}.png"
        )
        save_debug_overlay(
            image=image,
            pose_name=pose_name,
            point_ids=point_ids,
            image_points=image_points,
            predicted_xy=predicted_xy,
            true_xy=true_xy,
            output_path=overlay_path,
        )
        print(f"Saved validation overlay: {overlay_path}")

        summary["poses"][pose_name] = report

    if all_errors_m:
        all_errors = np.asarray(all_errors_m, dtype=np.float64)
        summary["overall"] = {
            "rmse_m": float(np.sqrt(np.mean(all_errors**2))),
            "rmse_cm": float(np.sqrt(np.mean(all_errors**2)) * 100.0),
            "mean_error_m": float(all_errors.mean()),
            "mean_error_cm": float(all_errors.mean() * 100.0),
            "max_error_m": float(all_errors.max()),
            "max_error_cm": float(all_errors.max() * 100.0),
        }

        print("\nOverall validation:")
        print(f"  RMSE:       {summary['overall']['rmse_cm']:.3f} cm")
        print(f"  Mean error: {summary['overall']['mean_error_cm']:.3f} cm")
        print(f"  Max error:  {summary['overall']['max_error_cm']:.3f} cm")

    output_path = args.calibration_dir / "homography_validation_summary.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved summary: {output_path}")


if __name__ == "__main__":
    main()
