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
sys.path.insert(0, str(PROJECT_ROOT))

from perception.cube_detector import (
    CubeDetectorConfig,
    draw_cube_detections,
)
from perception.multiview_cube_localizer import (
    WorkspaceBounds,
    localize_cube_from_scan,
    localization_result_to_jsonable,
)


DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "data" / "calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test multi-view cube localization on a saved observation scan."
    )

    parser.add_argument(
        "--scan-dir",
        type=Path,
        required=True,
        help="Folder containing center.png, left.png, right.png.",
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
        help="Pose/image names to process.",
    )
    parser.add_argument(
        "--target-color",
        type=str,
        default=None,
        help=(
            "Optional target color. If omitted, detect any configured cube color. "
            "Useful later for Tasks 2/3."
        ),
    )

    parser.add_argument("--min-area-px", type=float, default=80.0)
    parser.add_argument("--max-area-px", type=float, default=12000.0)
    parser.add_argument("--min-aspect-ratio", type=float, default=0.35)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.8)

    parser.add_argument("--workspace-x-min", type=float, default=-0.30)
    parser.add_argument("--workspace-x-max", type=float, default=0.30)
    parser.add_argument("--workspace-y-min", type=float, default=0.05)
    parser.add_argument("--workspace-y-max", type=float, default=0.65)

    parser.add_argument(
        "--cluster-radius-m",
        type=float,
        default=0.02,
        help="Maximum XY distance for two detections to belong to the same cluster.",
    )
    parser.add_argument(
        "--reliable-min-views",
        type=int,
        default=2,
        help="Minimum number of distinct camera views required for a reliable result.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "cube_detection_debug",
    )

    return parser.parse_args()


def save_rgb_image(image_rgb: np.ndarray, path: Path) -> None:
    Image.fromarray(image_rgb.astype(np.uint8, copy=False)).save(path)


def build_mask_montage(clean_masks: dict[str, np.ndarray]) -> np.ndarray | None:
    if not clean_masks:
        return None

    rows = []
    for _, mask in clean_masks.items():
        mask_rgb = np.repeat(mask[..., None], 3, axis=2)
        rows.append(mask_rgb)

    return np.concatenate(rows, axis=0)


def selected_detection_indices_from_best_cluster(
    best_cluster: dict[str, Any] | None,
) -> dict[str, int]:
    """
    Return:
        {
            "center": detection_index,
            "left": detection_index,
            ...
        }
    for detections used in the selected multi-view cluster.
    """
    if best_cluster is None:
        return {}

    selected: dict[str, int] = {}

    for member in best_cluster["members"]:
        selected[member["pose_name"]] = int(member["detection_index"])

    return selected


def save_topdown_cluster_plot(
    *,
    clusters: list[dict[str, Any]],
    best_cluster: dict[str, Any] | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot all cluster members lightly.
    for cluster_idx, cluster in enumerate(clusters):
        members = cluster["members"]

        for member in members:
            x, y = member["table_xy"]
            ax.scatter([x], [y], s=45)
            ax.text(
                x + 0.004,
                y + 0.004,
                f"{member['pose_name']}",
                fontsize=8,
            )

        cx, cy = cluster["fused_xy"]
        ax.scatter([cx], [cy], s=90, marker="x")
        ax.text(cx + 0.004, cy - 0.004, f"cluster {cluster_idx}", fontsize=9)

    if best_cluster is not None:
        fx, fy = best_cluster["fused_xy"]
        ax.scatter([fx], [fy], s=220, marker="x", label="chosen fused cube xy")

    ax.set_title("Multi-view cube localization clusters")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    detector_config = CubeDetectorConfig(
        min_area_px=args.min_area_px,
        max_area_px=args.max_area_px,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
    )

    workspace_bounds = WorkspaceBounds(
        x_min=args.workspace_x_min,
        x_max=args.workspace_x_max,
        y_min=args.workspace_y_min,
        y_max=args.workspace_y_max,
    )

    result = localize_cube_from_scan(
        scan_dir=args.scan_dir,
        calibration_dir=args.calibration_dir,
        poses=args.poses,
        target_color=args.target_color,
        detector_config=detector_config,
        workspace_bounds=workspace_bounds,
        cluster_radius_m=args.cluster_radius_m,
        reliable_min_views=args.reliable_min_views,
    )

    best_cluster = result["best_cluster"]
    clusters = result["clusters"]
    cube_xy = result["cube_xy"]
    is_reliable = result["is_reliable"]

    selected_indices = selected_detection_indices_from_best_cluster(best_cluster)

    # ------------------------------------------------------------------
    # Save per-view debug overlays and masks
    # ------------------------------------------------------------------
    for pose_name, pose_result in result["poses"].items():
        image_rgb = pose_result["image_rgb"]
        detections = pose_result["detections"]
        detector_debug = pose_result["detector_debug"]

        selected_index = selected_indices.get(pose_name)

        overlay = draw_cube_detections(
            image_rgb,
            detections,
            selected_index=selected_index,
        )
        overlay_path = args.output_dir / f"detections_{pose_name}.png"
        save_rgb_image(overlay, overlay_path)

        mask_montage = build_mask_montage(detector_debug["clean_masks"])
        if mask_montage is not None:
            mask_path = args.output_dir / f"masks_{pose_name}.png"
            save_rgb_image(mask_montage, mask_path)

    # ------------------------------------------------------------------
    # Print result
    # ------------------------------------------------------------------
    if best_cluster is None or cube_xy is None:
        print("\nNo cube cluster found.")
    else:
        print("\nBest multi-view candidate cluster:")
        print(f"  views: {best_cluster['distinct_views']}")
        print(f"  num views: {best_cluster['num_distinct_views']}")
        print(
            f"  fused cube estimate: "
            f"x={cube_xy[0]:.4f}, y={cube_xy[1]:.4f}"
        )
        print(
            f"  max cluster disagreement: "
            f"{best_cluster['max_residual_m'] * 100:.2f} cm"
        )
        print(f"  reliable: {is_reliable}")

        print("\nCluster members:")
        for member in best_cluster["members"]:
            xy = member["table_xy"]
            uv = member["centroid_uv"]
            print(
                f"  {member['pose_name']:>6s}: "
                f"color={member['color']:<7s} "
                f"xy=({xy[0]:.4f}, {xy[1]:.4f}) "
                f"uv=({uv[0]:.1f}, {uv[1]:.1f})"
            )

    # ------------------------------------------------------------------
    # Save top-down cluster visualization
    # ------------------------------------------------------------------
    topdown_plot_path = args.output_dir / "cube_xy_clusters_topdown.png"
    save_topdown_cluster_plot(
        clusters=clusters,
        best_cluster=best_cluster,
        output_path=topdown_plot_path,
    )

    # ------------------------------------------------------------------
    # Save summary JSON
    # ------------------------------------------------------------------
    summary = localization_result_to_jsonable(result)
    summary["topdown_plot_path"] = str(topdown_plot_path)

    summary_path = args.output_dir / "cube_detection_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved top-down plot: {topdown_plot_path}")


if __name__ == "__main__":
    main()