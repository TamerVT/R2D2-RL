from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class CamConversionConfig:
    name: str
    resolution: tuple[int, int]
    source_name: str | None = None

    @property
    def dataset_key(self) -> str:
        return f"observation.images.{self.name}"

    @property
    def frame_name(self) -> str:
        return self.source_name or self.name.removeprefix("image_")

    @property
    def image_column(self) -> str:
        return f"image_{self.name}"


CAMERAS = [
    CamConversionConfig(name="head", resolution=(256, 256)),
    CamConversionConfig(name="image_left_wrist", source_name="left_wrist", resolution=(256, 256)),
    CamConversionConfig(name="image_right_wrist", source_name="right_wrist", resolution=(256, 256)),
]
IMAGE_BATCH_SIZE = 32


class JointDatasetConverter:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.conn = duckdb.connect()
        self.source_sql = self._build_source_sql(DATASET_PATHS)
        self.camera_resizers = {
            camera.name: v2.Resize(camera.resolution) for camera in CAMERAS
        }

        self.lrds = LeRobotDataset.create(
            repo_id=REPO_ID,
            robot_type=ROBOT_TYPE,
            root=self.root,
            fps=FPS,
            use_videos=False,
            features=self._build_features(),
            image_writer_threads=0,
            image_writer_processes=0,
        )

    def _build_features(self) -> dict[str, dict[str, object]]:
        features = {
            camera.dataset_key: {
                "dtype": "image",
                "shape": (*camera.resolution, 3),
                "names": ["height", "width", "channel"],
            }
            for camera in CAMERAS
        }
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (8,),
            "names": [f"joint_{i}" for i in range(7)] + ["gripper"],
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (8,),
            "names": [f"joint_{i}" for i in range(7)] + ["gripper"],
        }
        return features

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
                    LEAD(obs.{ARM_KEY}.joints) OVER w AS next_joints,
                    LEAD(obs.{ARM_KEY}.gripper) OVER w AS next_gripper
                FROM ({self.source_sql}) AS src
                WHERE uuid = ?
                WINDOW w AS (PARTITION BY uuid ORDER BY step)
            ),
            action_annotated AS (
                SELECT
                    *,
                    COALESCE(action_gripper, next_gripper) AS effective_action_gripper,
                    LAG(next_joints) OVER (PARTITION BY uuid ORDER BY step) AS prev_action_joints,
                    LAG(COALESCE(action_gripper, next_gripper)) OVER (PARTITION BY uuid ORDER BY step) AS prev_action_gripper
                FROM ordered
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
            FROM action_annotated
            WHERE next_joints IS NOT NULL
              AND NOT (
                  prev_action_joints IS NOT NULL
                  AND next_joints = prev_action_joints
                  AND effective_action_gripper = prev_action_gripper
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
        image_selects = ",\n                    ".join(
            f"obs.frames.{camera.frame_name}.rgb.data AS {camera.image_column}" for camera in CAMERAS
        )
        image_not_null_checks = "\n                  ".join(
            f"AND obs.frames.{camera.frame_name}.rgb.data IS NOT NULL" for camera in CAMERAS
        )
        image_columns = ",\n                ".join(camera.image_column for camera in CAMERAS)

        return f"""
            WITH ordered AS (
                SELECT
                    uuid,
                    step,
                    action.{ARM_KEY}.gripper AS action_gripper,
                    {image_selects},
                    LEAD(obs.{ARM_KEY}.joints) OVER w AS next_joints,
                    LEAD(obs.{ARM_KEY}.gripper) OVER w AS next_gripper
                FROM ({self.source_sql}) AS src
                WHERE uuid = ?
                  {image_not_null_checks}
                WINDOW w AS (PARTITION BY uuid ORDER BY step)
            ),
            action_annotated AS (
                SELECT
                    *,
                    COALESCE(action_gripper, next_gripper) AS effective_action_gripper,
                    LAG(next_joints) OVER (PARTITION BY uuid ORDER BY step) AS prev_action_joints,
                    LAG(COALESCE(action_gripper, next_gripper)) OVER (PARTITION BY uuid ORDER BY step) AS prev_action_gripper
                FROM ordered
            )
            SELECT
                step,
                {image_columns}
            FROM action_annotated
            WHERE next_joints IS NOT NULL
              AND NOT (
                  prev_action_joints IS NOT NULL
                  AND next_joints = prev_action_joints
                  AND effective_action_gripper = prev_action_gripper
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

            frame = {
                camera.dataset_key: images[camera.name] for camera in CAMERAS
            }
            frame["observation.state"] = state_vec
            frame["action"] = action_vec
            frame["task"] = str(curr["instruction"])

            self.lrds.add_frame(frame)

        self.lrds.save_episode()
        return True

    def _decode_and_resize_batch(self, image_bytes_list: list[bytes], camera: CamConversionConfig) -> np.ndarray:
        image_tensors = [
            torch.frombuffer(bytearray(image_bytes), dtype=torch.uint8) for image_bytes in image_bytes_list
        ]
        decoded = decode_jpeg(image_tensors)
        batch = torch.stack(decoded)
        resized = self.camera_resizers[camera.name](batch)
        return resized.permute(0, 2, 3, 1).cpu().numpy()

    def _decode_image_batch(self, batch: pa.RecordBatch, frames_by_step: dict[int, dict[str, np.ndarray]]) -> None:
        batch_dict = batch.to_pydict()
        steps = [int(step) for step in batch_dict["step"]]
        decoded_images = {}
        for camera in CAMERAS:
            decoded_images[camera.name] = self._decode_and_resize_batch(batch_dict[camera.image_column], camera)

        for idx, step in enumerate(steps):
            frames_by_step[step] = {camera.name: decoded_images[camera.name][idx] for camera in CAMERAS}


if __name__ == "__main__":
    hf_ds = JointDatasetConverter(HF_DATA_DIR)
    hf_ds.generate_examples(success=True, n=-1)
