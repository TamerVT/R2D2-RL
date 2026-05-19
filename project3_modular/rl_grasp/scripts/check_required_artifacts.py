#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASET_ROOT = (
    Path.home()
    / ".cache/huggingface/lerobot/frainer"
    / "p3_local_grasp_hil_multicolor_colorcond_v1"
)


@dataclass
class CheckResult:
    label: str
    ok: bool
    detail: str
    required: bool = True


def repo_root_from_this_file() -> Path:
    # .../project3_modular/rl_grasp/scripts/check_required_artifacts.py
    return Path(__file__).resolve().parents[3]


def exists_check(label: str, path: Path, *, required: bool = True) -> CheckResult:
    return CheckResult(
        label=label,
        ok=path.exists(),
        detail=str(path),
        required=required,
    )


def file_contains_check(
    label: str,
    path: Path,
    needle: str,
    *,
    required: bool = True,
) -> CheckResult:
    if not path.exists():
        return CheckResult(
            label=label,
            ok=False,
            detail=f"missing file: {path}",
            required=required,
        )

    text = path.read_text(errors="replace")
    ok = needle in text
    detail = f"{path} | looking for: {needle!r}"
    return CheckResult(label=label, ok=ok, detail=detail, required=required)


def json_shape_check(
    label: str,
    path: Path,
    keys: list[str],
    *,
    required: bool = True,
) -> CheckResult:
    if not path.exists():
        return CheckResult(
            label=label,
            ok=False,
            detail=f"missing file: {path}",
            required=required,
        )

    try:
        obj = json.loads(path.read_text())
    except Exception as exc:
        return CheckResult(
            label=label,
            ok=False,
            detail=f"could not parse JSON at {path}: {exc}",
            required=required,
        )

    missing = [key for key in keys if key not in obj]
    if missing:
        return CheckResult(
            label=label,
            ok=False,
            detail=f"{path} | missing top-level keys: {missing}",
            required=required,
        )

    return CheckResult(
        label=label,
        ok=True,
        detail=str(path),
        required=required,
    )


def print_result(result: CheckResult) -> None:
    if result.ok:
        status = "OK"
    elif result.required:
        status = "MISSING"
    else:
        status = "WARN"

    print(f"[{status:7}] {result.label}")
    print(f"          {result.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether the required Project 3 sim-training artifacts, "
            "datasets, third-party repos, and patches are available."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the real-world local grasp LeRobot dataset.",
    )
    args = parser.parse_args()

    root = repo_root_from_this_file()
    dataset_root = args.dataset_root.expanduser().resolve()

    rcs_root = root / "third_party" / "robot-control-stack"
    lerobot_root = root / "lerobot"

    pregrasp_ckpt = (
        root
        / "project3_modular"
        / "outputs"
        / "pregrasp_regressor"
        / "best_pregrasp_mlp.pt"
    )

    normalization_stats = (
        root
        / "project3_modular"
        / "rl_grasp"
        / "outputs"
        / "real_hil_bc_warmstart_no_currents_v2"
        / "normalization_stats.json"
    )

    rcs_patch = (
        root
        / "project3_modular"
        / "rl_grasp"
        / "patches"
        / "robot_control_stack_so101_local_grasp_sim.patch"
    )

    lerobot_patch = (
        root
        / "project3_modular"
        / "rl_grasp"
        / "patches"
        / "lerobot_hilserl_so101_local_grasp.patch"
    )

    rcs_so101_xml = rcs_root / "assets" / "robots" / "so101" / "so101.xml"
    rcs_scene_xml = rcs_root / "assets" / "scenes" / "empty_world" / "scene.xml"

    hil_compat_env = (
        root
        / "project3_modular"
        / "rl_grasp"
        / "envs"
        / "so101_local_grasp_hil_compat_env.py"
    )

    visual_sac_train_script = (
        root
        / "project3_modular"
        / "rl_grasp"
        / "scripts"
        / "train_visual_hil_compat_sac.py"
    )

    dataset_stats = dataset_root / "meta" / "stats.json"
    dataset_info = dataset_root / "meta" / "info.json"

    results: list[CheckResult] = []

    print()
    print("=" * 78)
    print("Project 3 SO101 sim-training setup check")
    print("=" * 78)
    print(f"Project root:  {root}")
    print(f"Dataset root:  {dataset_root}")
    print()

    # ------------------------------------------------------------------
    # Core repo/code checks
    # ------------------------------------------------------------------
    results.extend(
        [
            exists_check("Main project repo root", root),
            exists_check("LeRobot checkout", lerobot_root),
            exists_check("robot-control-stack checkout", rcs_root),
            exists_check("HIL-compatible sim environment", hil_compat_env),
            exists_check("Visual SAC sim training script", visual_sac_train_script),
        ]
    )

    # ------------------------------------------------------------------
    # Required non-Git artifacts
    # ------------------------------------------------------------------
    results.extend(
        [
            exists_check("Pregrasp regressor checkpoint", pregrasp_ckpt),
            json_shape_check(
                "Normalization stats JSON",
                normalization_stats,
                keys=["state_mean", "state_std"],
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Real-world dataset used for BC warm-start and stats
    # ------------------------------------------------------------------
    results.extend(
        [
            exists_check("Real-world grasp dataset root", dataset_root),
            json_shape_check(
                "Dataset stats.json",
                dataset_stats,
                keys=["observation.images.wrist", "observation.state", "action"],
            ),
            exists_check(
                "Dataset info.json",
                dataset_info,
                required=False,
            ),
        ]
    )

    # Optional but useful: check for some parquet/video files.
    parquet_files = list(dataset_root.rglob("*.parquet")) if dataset_root.exists() else []
    results.append(
        CheckResult(
            label="Dataset parquet episodes",
            ok=len(parquet_files) > 0,
            detail=f"found {len(parquet_files)} parquet file(s)",
            required=True,
        )
    )

    video_files = list(dataset_root.rglob("*.mp4")) if dataset_root.exists() else []
    results.append(
        CheckResult(
            label="Dataset videos",
            ok=len(video_files) > 0,
            detail=f"found {len(video_files)} video file(s)",
            required=False,
        )
    )

    # ------------------------------------------------------------------
    # Patch files committed in R2D2-RL
    # ------------------------------------------------------------------
    results.extend(
        [
            exists_check("RCS sim asset patch file", rcs_patch),
            exists_check("LeRobot HIL-SERL patch file", lerobot_patch, required=False),
        ]
    )

    # ------------------------------------------------------------------
    # Verify that the relevant RCS patch effects are actually present
    # ------------------------------------------------------------------
    results.extend(
        [
            file_contains_check(
                "RCS wrist camera exists",
                rcs_so101_xml,
                'camera name="robotwrist"',
            ),
            file_contains_check(
                "RCS calibrated wrist camera pose exists",
                rcs_so101_xml,
                '-0.00828381134 0.0463468991 -0.0553307671',
            ),
            file_contains_check(
                "RCS beige tabletop-like groundplane exists",
                rcs_scene_xml,
                'material name="groundplane" rgba="0.722 0.678 0.663 1"',
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    for result in results:
        print_result(result)

    required_failures = [result for result in results if result.required and not result.ok]
    optional_warnings = [result for result in results if not result.required and not result.ok]

    print()
    print("-" * 78)

    if not required_failures:
        print("All required checks passed.")
        if optional_warnings:
            print(f"There are {len(optional_warnings)} optional warning(s), but sim training can proceed.")
        print()
        print("Typical SB3 visual SAC training command:")
        print(
            "  PYTHONPATH=. python "
            "project3_modular/rl_grasp/scripts/train_visual_hil_compat_sac.py \\"
        )
        print(
            "    --output-dir "
            "project3_modular/rl_grasp/outputs/"
            "sac_visual_hil_bcpretrain_openreset_rewardshaped_v1 \\"
        )
        print(
            "    --normalization-stats "
            "project3_modular/rl_grasp/outputs/"
            "real_hil_bc_warmstart_no_currents_v2/normalization_stats.json \\"
        )
        print(f"    --bc-dataset-root {dataset_root} \\")
        print("    --bc-pretrain-epochs 60 \\")
        print("    --bc-zero-motor-currents \\")
        print("    --total-timesteps 50000 \\")
        print("    --buffer-size 10000 \\")
        print("    --checkpoint-freq 10000")
        return 0

    print(f"{len(required_failures)} required check(s) failed.")
    print()
    print("Most common fixes:")
    print("  - Unzip the shared dataset into:")
    print(f"      {DEFAULT_DATASET_ROOT}")
    print("  - Unzip the shared artifact archive from the project repo root.")
    print("  - Apply the RCS patch inside third_party/robot-control-stack.")
    print()
    print("RCS patch application:")
    print(
        "  git apply "
        "../../project3_modular/rl_grasp/patches/"
        "robot_control_stack_so101_local_grasp_sim.patch"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
