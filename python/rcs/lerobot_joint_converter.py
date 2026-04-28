from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import rcs
from rcs._core.common import RobotType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch
from torchvision.io import decode_jpeg
from torchvision.transforms import v2

DATASET_PATHS = [
    "data_grasp",
]
HF_DATA_DIR = "data_lerobot_joint_simple"
REPO_ID = "rcs/grasp_joint_simple"
ROBOT_TYPE = "fr3"
FPS = 30
ROBOT_KEYS = ["left", "right"]
JOINTS = False
TCP_OFFSET = rcs.GRIPPER_OFFSETS[rcs.common.GripperType("Robotiq2F85")]


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
PER_ROBOT_ARM_DIM = 7
PER_ROBOT_STATE_DIM = PER_ROBOT_ARM_DIM + 1

ik = rcs.common.Pin(
    rcs.ROBOTS[RobotType.FR3].mjcf_model_path,
    rcs.ROBOTS[RobotType.FR3].attachment_site,
)


class JointDatasetConverter:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.conn = duckdb.connect()
        self.source_sql = self._build_source_sql(DATASET_PATHS)
        self.state_dim = len(ROBOT_KEYS) * PER_ROBOT_STATE_DIM
        self.camera_resizers = {camera.name: v2.Resize(camera.resolution) for camera in CAMERAS}

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
        state_names = []
        for robot_key in ROBOT_KEYS:
            state_names.extend([f"{robot_key}_joint_{i}" for i in range(7)])
            state_names.append(f"{robot_key}_gripper")

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
            "shape": (self.state_dim,),
            "names": state_names,
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (self.state_dim,),
            "names": state_names,
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
        observation_selects = ",\n                    ".join(
            [
                f"obs.{robot_key}.joints AS observation_joints_{robot_key}"
                for robot_key in ROBOT_KEYS
            ]
            + [
                f"obs.{robot_key}.gripper AS observation_gripper_{robot_key}"
                for robot_key in ROBOT_KEYS
            ]
        )
        action_selects = ",\n                    ".join(
            [
                f"info.{robot_key}.absolute_action AS absolute_action_{robot_key}"
                for robot_key in ROBOT_KEYS
            ]
            + [
                f"env_action.{robot_key}.gripper AS action_gripper_{robot_key}"
                for robot_key in ROBOT_KEYS
            ]
        )

        return self.conn.execute(
            f"""
            SELECT
                uuid,
                step,
                success,
                instruction,
                {observation_selects},
                {action_selects}
            FROM ({self.source_sql}) AS src
            WHERE uuid = ?
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
                    {image_selects}
                FROM ({self.source_sql}) AS src
                WHERE uuid = ?
                  {image_not_null_checks}
            )
            SELECT
                step,
                {image_columns}
            FROM ordered
            ORDER BY step
        """

    def _is_missing(self, value: object) -> bool:
        if value is None or value is pd.NA:
            return True
        if isinstance(value, float):
            return bool(np.isnan(value))
        return False

    def _build_observation_state(self, row: pd.Series) -> np.ndarray:
        vectors = []
        for robot_key in ROBOT_KEYS:
            joints = row[f"observation_joints_{robot_key}"]
            gripper = row[f"observation_gripper_{robot_key}"]
            if self._is_missing(joints) or self._is_missing(gripper):
                msg = f"Missing observation state for robot '{robot_key}' at step {row['step']}"
                raise ValueError(msg)

            joints_vec = np.asarray(joints, dtype=np.float32)
            gripper_vec = np.asarray(gripper, dtype=np.float32)
            if joints_vec.shape != (PER_ROBOT_ARM_DIM,) or gripper_vec.shape != (1,):
                msg = (
                    f"Unexpected observation shapes for robot '{robot_key}' at step {row['step']}: "
                    f"joints={joints_vec.shape}, gripper={gripper_vec.shape}"
                )
                raise ValueError(msg)
            vectors.append(np.concatenate([joints_vec, gripper_vec]).astype(np.float32))

        return np.concatenate(vectors).astype(np.float32)

    def _convert_action_to_joint_space(self, row: pd.Series) -> np.ndarray:
        actions = []
        for robot_key in ROBOT_KEYS:
            observation_joints = row[f"observation_joints_{robot_key}"]
            absolute_action = row[f"absolute_action_{robot_key}"]
            action_gripper = row[f"action_gripper_{robot_key}"]
            if self._is_missing(observation_joints) or self._is_missing(absolute_action) or self._is_missing(action_gripper):
                msg = f"Missing action inputs for robot '{robot_key}' at step {row['step']}"
                raise ValueError(msg)

            observation_joints_vec = np.asarray(observation_joints, dtype=np.float64)
            absolute_action_vec = np.asarray(absolute_action, dtype=np.float64)
            action_gripper_vec = np.asarray(action_gripper, dtype=np.float32)
            if (
                observation_joints_vec.shape != (PER_ROBOT_ARM_DIM,)
                or absolute_action_vec.shape != (PER_ROBOT_ARM_DIM,)
                or action_gripper_vec.shape != (1,)
            ):
                msg = (
                    f"Unexpected action shapes for robot '{robot_key}' at step {row['step']}: "
                    f"observation_joints={observation_joints_vec.shape}, "
                    f"absolute_action={absolute_action_vec.shape}, action_gripper={action_gripper_vec.shape}"
                )
                raise ValueError(msg)

            if JOINTS:
                arm_action_vec = absolute_action_vec.astype(np.float32)
            else:
                target_pose = rcs.common.Pose(
                    translation=absolute_action_vec[:3],
                    quaternion=absolute_action_vec[3:7],
                )
                ik_joints = ik.inverse(target_pose, observation_joints_vec, tcp_offset=TCP_OFFSET)
                if ik_joints is None:
                    msg = f"IK failed for robot '{robot_key}' at step {row['step']}"
                    raise ValueError(msg)
                arm_action_vec = np.asarray(ik_joints, dtype=np.float32)

            actions.append(np.concatenate([arm_action_vec, action_gripper_vec]).astype(np.float32))

        concatenated = np.concatenate(actions).astype(np.float32)
        if concatenated.shape != (self.state_dim,):
            msg = f"Unexpected concatenated action shape {concatenated.shape} at step {row['step']}"
            raise ValueError(msg)
        return concatenated

    def _prepare_transition_table(self, table: pd.DataFrame) -> pd.DataFrame:
        if len(table) == 0:
            return table

        df = table.copy()
        df["observation_state"] = df.apply(self._build_observation_state, axis=1)
        df["action_vector"] = df.apply(self._convert_action_to_joint_space, axis=1)

        prev_action: np.ndarray | None = None
        keep_mask = []
        for action_vec in df["action_vector"]:
            assert isinstance(action_vec, np.ndarray)
            keep_mask.append(prev_action is None or not np.array_equal(action_vec, prev_action))
            prev_action = action_vec

        return df.loc[keep_mask].reset_index(drop=True)

    def parse_episode(self, episode_id: str, table: pd.DataFrame, success: bool):
        table = self._prepare_transition_table(table)
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

        num_frames_added = 0
        for step in step_order:
            curr = rows_by_step[step]
            if step not in frames_by_step:
                continue
            images = frames_by_step[step]

            frame = {
                camera.dataset_key: images[camera.name] for camera in CAMERAS
            }
            frame["observation.state"] = curr["observation_state"]
            frame["action"] = curr["action_vector"]
            frame["task"] = str(curr["instruction"])

            self.lrds.add_frame(frame)
            num_frames_added += 1

        if num_frames_added == 0:
            return False
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
