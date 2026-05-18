from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def describe_value(name: str, value) -> None:
    print(f"\n{name}:")
    print(f"  type:  {type(value)}")

    if hasattr(value, "shape"):
        print(f"  shape: {tuple(value.shape)}")
    if hasattr(value, "dtype"):
        print(f"  dtype: {value.dtype}")

    try:
        arr = np.asarray(value)
        if np.issubdtype(arr.dtype, np.number):
            print(f"  min:   {arr.min():.5f}")
            print(f"  max:   {arr.max():.5f}")
            print(f"  first: {arr.reshape(-1)[:8]}")
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Full local dataset directory, e.g. ~/.cache/.../p3_local_grasp_real_v1_...",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="frainer/p3_local_grasp_real_v1",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    args = parser.parse_args()

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        episodes=[args.episode_index],
    )

    print("=== Dataset summary ===")
    print("repo_id:", args.repo_id)
    print("root:   ", args.root)
    print("len:    ", len(dataset))
    print("fps:    ", dataset.fps)
    print("features:")
    for key, spec in dataset.features.items():
        print(f"  {key}: {spec}")

    sample = dataset[0]

    print("\n=== First sample keys ===")
    for key in sample:
        print(" ", key)

    print("\n=== First sample details ===")
    for key, value in sample.items():
        describe_value(key, value)


if __name__ == "__main__":
    main()
