#!/usr/bin/env python3

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_ROOT = Path("/home/frainer/.cache/huggingface/lerobot/frainer")

SOURCE_DATASETS = [
    ("blue",   "frainer/p3_local_grasp_hil_demos_v1",  BASE_ROOT / "p3_local_grasp_hil_demos_v1"),
    ("green",  "frainer/p3_local_grasp_hil_green_v1", BASE_ROOT / "p3_local_grasp_hil_green_v1"),
    ("purple", "frainer/p3_local_grasp_hil_purple_v1", BASE_ROOT / "p3_local_grasp_hil_purple_v1"),
    ("orange", "frainer/p3_local_grasp_hil_orange_v1", BASE_ROOT / "p3_local_grasp_hil_orange_v1"),
    ("yellow", "frainer/p3_local_grasp_hil_yellow_v1", BASE_ROOT / "p3_local_grasp_hil_yellow_v1"),
    ("red",    "frainer/p3_local_grasp_hil_red_v1", BASE_ROOT / "p3_local_grasp_hil_red_v1"),
]

COLOR_ORDER = ["blue", "green", "purple", "orange", "yellow", "red"]
COLOR_TO_ONEHOT = {
    color: np.eye(len(COLOR_ORDER), dtype=np.float32)[i]
    for i, color in enumerate(COLOR_ORDER)
}

OUTPUT_REPO_ID = "frainer/p3_local_grasp_hil_multicolor_colorcond_v1"
OUTPUT_ROOT = BASE_ROOT / "p3_local_grasp_hil_multicolor_colorcond_v1"


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def as_torch_float(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().to(torch.float32)
    return torch.tensor(x, dtype=torch.float32)


def make_output_features(reference_ds: LeRobotDataset) -> dict:
    """
    Recreate the relevant stored dataset features, but expand observation.state:
      18 robot features -> 24 robot+target-color features
    """
    action_feature = reference_ds.features["action"]
    image_feature = reference_ds.features["observation.images.wrist"]

    old_state_shape = tuple(reference_ds.features["observation.state"]["shape"])
    assert old_state_shape == (18,), f"Expected old state shape (18,), got {old_state_shape}"

    features = {
        "action": action_feature,
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": None,
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        },
        "complementary_info.discrete_penalty": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["discrete_penalty"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": tuple(image_feature["shape"]),
            "names": image_feature["names"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (24,),
            "names": None,
        },
    }
    return features


def main() -> None:
    print("Color order:")
    for i, color in enumerate(COLOR_ORDER):
        print(f"  {i}: {color} -> {COLOR_TO_ONEHOT[color].tolist()}")

    if OUTPUT_ROOT.exists():
        raise FileExistsError(
            f"Output dataset already exists:\n  {OUTPUT_ROOT}\n"
            "Delete it manually if you want to rebuild it."
        )

    # Load all sources once and sanity-check feature compatibility.
    loaded = []
    for color, repo_id, root in SOURCE_DATASETS:
        ds = LeRobotDataset(repo_id, root=str(root))
        loaded.append((color, repo_id, root, ds))
        print(
            f"Loaded {color:>6}: "
            f"{ds.meta.total_episodes:>2} episodes, {len(ds):>4} frames"
        )

    reference_ds = loaded[0][3]
    features = make_output_features(reference_ds)

    fps = reference_ds.meta.fps
    print(f"\nCreating merged dataset at:\n  {OUTPUT_ROOT}")
    print(f"repo_id: {OUTPUT_REPO_ID}")
    print(f"fps: {fps}")

    out_ds = LeRobotDataset.create(
        OUTPUT_REPO_ID,
        fps,
        root=str(OUTPUT_ROOT),
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        features=features,
    )

    total_written_frames = 0
    total_written_episodes = 0

    for color, repo_id, root, ds in loaded:
        color_vec = torch.tensor(COLOR_TO_ONEHOT[color], dtype=torch.float32)

        print(f"\nProcessing {color} from {repo_id} ...")

        for idx in range(len(ds)):
            sample = ds[idx]

            old_state = as_torch_float(sample["observation.state"])
            assert old_state.shape == (18,), (
                f"{color} frame {idx}: expected state shape (18,), got {tuple(old_state.shape)}"
            )

            new_state = torch.cat([old_state, color_vec], dim=0)
            assert new_state.shape == (24,)

            reward = as_numpy(sample["next.reward"]).astype(np.float32).reshape(1)
            done = as_numpy(sample["next.done"]).astype(bool).reshape(1)
            discrete_penalty = (
                as_numpy(sample["complementary_info.discrete_penalty"])
                .astype(np.float32)
                .reshape(1)
            )

            frame = {
                "observation.images.wrist": sample["observation.images.wrist"],
                "observation.state": new_state,
                "action": as_torch_float(sample["action"]),
                "next.reward": reward,
                "next.done": done,
                "complementary_info.discrete_penalty": discrete_penalty,
                "task": sample["task"],
            }

            out_ds.add_frame(frame)
            total_written_frames += 1

            # Each recorded HIL demo ends on next.done=True.
            if bool(done[0]):
                out_ds.save_episode()
                total_written_episodes += 1
                print(
                    f"  saved merged episode {total_written_episodes:>2} "
                    f"(source color: {color})"
                )

    print("\nFinished.")
    print(f"  frames written:   {total_written_frames}")
    print(f"  episodes written: {total_written_episodes}")
    print(f"  output root:      {OUTPUT_ROOT}")
    print(f"  output repo_id:   {OUTPUT_REPO_ID}")


if __name__ == "__main__":
    main()
