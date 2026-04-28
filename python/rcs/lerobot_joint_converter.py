from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torchvision.io import decode_jpeg
from torchvision.transforms import v2

DATASET_PATHS = [
    "data_grasp",
]
HF_DATA_DIR = "data_lerobot_joint_simple"
REPO_ID = "rcs/grasp_joint_simple"
ROBOT_TYPE = "fr3"
FPS = 30
ARM_KEY = "right"

CAMERAS = [
    ("head", "head"),
    ("image_left_wrist", "left_wrist"),
    ("image_right_wrist", "right_wrist"),
]
IMAGE_SIZE = (256, 256)
RESIZE = v2.Resize(IMAGE_SIZE)
IMAGE_BATCH_SIZE = 32


class JointDatasetConverter:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.conn = duckdb.connect()
        self.source_sql = self._build_source_sql(DATASET_PATHS)

        self.lrds = LeRobotDataset.create(
            repo_id=REPO_ID,
            robot_type=ROBOT_TYPE,
            root=self.root,
            fps=FPS,
            use_videos=False,
            features={
                "observation.images.head": {
                    "dtype": "image",
                    "shape": (*IMAGE_SIZE, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.images.image_left_wrist": {
                    "dtype": "image",
                    "shape": (*IMAGE_SIZE, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.images.image_right_wrist": {
                    "dtype": "image",
                    "shape": (*IMAGE_SIZE, 3),
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": [f"joint_{i}" for i in range(7)] + ["gripper"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (8,),
                    "names": [f"joint_{i}" for i in range(7)] + ["gripper"],
                },
            },
            image_writer_threads=0,
            image_writer_processes=0,
        )

    def _build_source_sql(self, dataset_paths: list[str | Path]) -> str:
        queries = []
        for path in dataset_paths:
            escaped = str(path).replace("'", "''")
            queries.append(f"SELECT * FROM read_parquet('{escaped}')")
        return " UNION ALL ".join(queries)

    def generate_examples(self, success: bool = True, n: int = -1):
        uuids = self.conn.execute(f"SELECT DISTINCT uuid FROM ({self.source_sql}) AS src ORDER BY uuid").fetchall()

        for (episode_id,) in uuids:
            table = self._fetch_transition_table(episode_id)

            converted = self.parse_episode(episode_id, table, success)
            if converted:
                n -= 1
                if n == 0:
                    break

        self.lrds.finalize()

    def _fetch_transition_table(self, episode_id: str) -> pd.DataFrame:
        return self.conn.execute(
            f"""
            WITH ordered AS (
                SELECT
                    uuid,
                    step,
                    success,
                    instruction,
                    obs.{ARM_KEY}.joints AS observation_joints,
                    obs.{ARM_KEY}.gripper AS observation_gripper,
                    action.{ARM_KEY}.gripper AS action_gripper,
                    LEAD(obs.{ARM_KEY}.joints) OVER (PARTITION BY uuid ORDER BY step) AS next_joints,
                    LEAD(obs.{ARM_KEY}.gripper) OVER (PARTITION BY uuid ORDER BY step) AS next_gripper
                FROM ({self.source_sql}) AS src
                WHERE uuid = ?
            )
            SELECT
                uuid,
                step,
                success,
                instruction,
                observation_joints,
                observation_gripper,
                action_gripper,
                next_joints,
                next_gripper
            FROM ordered
            WHERE next_joints IS NOT NULL
              AND NOT (
                  observation_joints = next_joints
                  AND observation_gripper = next_gripper
              )
            ORDER BY step
            """,
            [episode_id],
        ).df()

    def _fetch_episode_success(self, episode_id: str) -> bool:
        return bool(
            self.conn.execute(
                f"SELECT COALESCE(MAX(success), FALSE) FROM ({self.source_sql}) AS src WHERE uuid = ?",
                [episode_id],
            ).fetchone()[0]
        )

    def _image_query(self) -> str:
        return f"""
            WITH ordered AS (
                SELECT
                    uuid,
                    step,
                    obs.{ARM_KEY}.joints AS observation_joints,
                    obs.{ARM_KEY}.gripper AS observation_gripper,
                    obs.frames.head.rgb.data AS image_head,
                    obs.frames.left_wrist.rgb.data AS image_left_wrist,
                    obs.frames.right_wrist.rgb.data AS image_right_wrist,
                    LEAD(obs.{ARM_KEY}.joints) OVER (PARTITION BY uuid ORDER BY step) AS next_joints,
                    LEAD(obs.{ARM_KEY}.gripper) OVER (PARTITION BY uuid ORDER BY step) AS next_gripper
                FROM ({self.source_sql}) AS src
                WHERE uuid = ?
                  AND obs.frames.head.rgb.data IS NOT NULL
                  AND obs.frames.left_wrist.rgb.data IS NOT NULL
                  AND obs.frames.right_wrist.rgb.data IS NOT NULL
            )
            SELECT
                step,
                image_head,
                image_left_wrist,
                image_right_wrist
            FROM ordered
            WHERE next_joints IS NOT NULL
              AND NOT (
                  observation_joints = next_joints
                  AND observation_gripper = next_gripper
              )
            ORDER BY step
        """

    def parse_episode(self, episode_id: str, table: pd.DataFrame, success: bool):
        if len(table) == 0:
            return False

        if success and not self._fetch_episode_success(episode_id):
            return False

        df = table.reset_index(drop=True)  # noqa: PD901
        rows_by_step = {int(row["step"]): row for _, row in df.iterrows()}
        step_order = [int(step) for step in df["step"].tolist()]
        frames_by_step: dict[int, dict[str, np.ndarray]] = {}

        reader = self.conn.execute(self._image_query(), [episode_id]).fetch_record_batch(
            rows_per_batch=IMAGE_BATCH_SIZE
        )
        for batch in reader:
            self._decode_image_batch(batch, frames_by_step)

        for step in step_order:
            curr = rows_by_step[step]
            images = frames_by_step[step]

            state_vec = np.concatenate(
                [
                    np.asarray(curr["observation_joints"], dtype=np.float32),
                    np.asarray(curr["observation_gripper"], dtype=np.float32),
                ]
            ).astype(np.float32)

            action_gripper = curr["action_gripper"]
            if (
                action_gripper is None
                or action_gripper is pd.NA
                or (isinstance(action_gripper, float) and np.isnan(action_gripper))
            ):
                action_gripper_vec = np.asarray(curr["next_gripper"], dtype=np.float32)
            else:
                action_gripper_vec = np.asarray(action_gripper, dtype=np.float32)

            action_vec = np.concatenate(
                [
                    np.asarray(curr["next_joints"], dtype=np.float32),
                    action_gripper_vec,
                ]
            ).astype(np.float32)

            self.lrds.add_frame(
                {
                    "observation.images.head": images["head"],
                    "observation.images.image_left_wrist": images["image_left_wrist"],
                    "observation.images.image_right_wrist": images["image_right_wrist"],
                    "observation.state": state_vec,
                    "action": action_vec,
                    "task": str(curr["instruction"]),
                }
            )

        self.lrds.save_episode()
        return True

    def _decode_and_resize_batch(self, image_bytes_list: list[bytes]) -> np.ndarray:
        image_tensors = [
            torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8) for image_bytes in image_bytes_list
        ]
        decoded = decode_jpeg(image_tensors)
        batch = torch.stack(decoded)
        resized = RESIZE(batch)
        return resized.permute(0, 2, 3, 1).cpu().numpy()

    def _decode_image_batch(self, batch: pa.RecordBatch, frames_by_step: dict[int, dict[str, np.ndarray]]) -> None:
        batch_dict = batch.to_pydict()
        steps = [int(step) for step in batch_dict["step"]]
        decoded_images = {}
        for feature_name, column_name in CAMERAS:
            image_column = f"image_{column_name}" if column_name != "head" else "image_head"
            decoded_images[feature_name] = self._decode_and_resize_batch(batch_dict[image_column])

        for idx, step in enumerate(steps):
            frames_by_step[step] = {
                "head": decoded_images["head"][idx],
                "image_left_wrist": decoded_images["image_left_wrist"][idx],
                "image_right_wrist": decoded_images["image_right_wrist"][idx],
            }


if __name__ == "__main__":
    hf_ds = JointDatasetConverter(HF_DATA_DIR)
    hf_ds.generate_examples(success=True, n=-1)
