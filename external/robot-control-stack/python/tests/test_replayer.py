import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

import duckdb
import numpy as np
from rcs._core.sim import SimConfig
from rcs.envs.base import RelativeTo
from rcs.envs.configs import EmptyWorldFR3Duo
from rcs.envs.storage_wrapper import StorageWrapper
from rcs.envs.tasks import PickTaskConfig
from rcs.sim.replayer import (
    load_distinct_uuids,
    load_trajectory,
    replay_trajectory,
    restore_sim_step,
)
from rcs.sim.sim import ROOT_RELATIVE_FREE_STATE_ENCODING

import rcs


def _build_env(
    output_dir: Path,
    *,
    with_cameras: bool,
    instruction: str = "",
    scene_path: Path | None = None,
    root_frame_to_world: rcs.common.Pose | None = None,
) -> StorageWrapper:
    scene = EmptyWorldFR3Duo()
    cfg = scene.config()
    cfg.sim_cfg = SimConfig(async_control=True, realtime=False, frequency=30, max_convergence_steps=500)
    cfg.headless = True
    cfg.relative_to = RelativeTo.CONFIGURED_ORIGIN
    if cfg.root_frame_objects is None:
        cfg.root_frame_objects = {}
    cfg.task_cfg = PickTaskConfig(robot_name="right")
    if scene_path is not None:
        cfg.scene = str(scene_path)
    if root_frame_to_world is not None:
        cfg.root_frame_to_world = root_frame_to_world
    if not with_cameras:
        cfg.camera_cfgs = {}
    else:
        assert cfg.camera_cfgs is not None
        for camera_cfg in cfg.camera_cfgs.values():
            camera_cfg.resolution_width = 64
            camera_cfg.resolution_height = 48
            camera_cfg.frame_rate = 1

    env = scene.create_env(cfg)
    return StorageWrapper(
        env,
        str(output_dir),
        instruction,
        batch_size=2,
        max_rows_per_group=10,
        max_rows_per_file=10,
        always_record=True,
    )


def _record_source_dataset(
    dataset_dir: Path,
    *,
    limit: int,
    instruction: str,
    scene_path: Path | None = None,
    root_frame_to_world: rcs.common.Pose | None = None,
) -> None:
    env = _build_env(
        dataset_dir,
        with_cameras=False,
        instruction=instruction,
        scene_path=scene_path,
        root_frame_to_world=root_frame_to_world,
    )
    try:
        env.reset()
        action = {
            "left": {
                "tquat": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                "gripper": np.array([1.0], dtype=np.float32),
            },
            "right": {
                "tquat": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                "gripper": np.array([1.0], dtype=np.float32),
            },
        }
        for _ in range(limit):
            env.step(action)
    finally:
        env.close()


def _source_rows(dataset_dir: Path, limit: int):
    connection = duckdb.connect()
    try:
        uuid = load_distinct_uuids(dataset_dir)[0]
        return connection.execute(
            """
            SELECT step, obs, info, reward, success, action, env_action, instruction
            FROM read_parquet(?)
            WHERE uuid = ?
            ORDER BY step
            LIMIT ?
            """,
            [str(dataset_dir), uuid, limit],
        ).fetchall()
    finally:
        connection.close()


def _replay_rows(dataset_dir: Path):
    connection = duckdb.connect()
    try:
        return connection.execute(
            """
            SELECT step, obs, info, reward, success, action, env_action, instruction
            FROM read_parquet(?)
            ORDER BY step
            """,
            [str(dataset_dir)],
        ).fetchall()
    finally:
        connection.close()


def _replay_prefix(
    output_dir: Path,
    *,
    with_cameras: bool,
    limit: int,
    scene_path: Path | None = None,
    root_frame_to_world: rcs.common.Pose | None = None,
) -> None:
    source_dir = output_dir.parent / "source"
    env = _build_env(
        output_dir,
        with_cameras=with_cameras,
        scene_path=scene_path,
        root_frame_to_world=root_frame_to_world,
    )
    try:
        uuid = load_distinct_uuids(source_dir)[0]
        recorded_steps = load_trajectory(source_dir, uuid)[:limit]
        env.get_wrapper_attr("set_instruction")(recorded_steps[0].instruction)
        replay_trajectory(env, recorded_steps, True)
    finally:
        env.close()


def _write_scene_with_extra_fixed_body_and_camera(src: Path, dst: Path):
    tree = ET.parse(src)
    root = tree.getroot()
    for include in root.findall("include"):
        include_file = include.get("file")
        if include_file is not None and not Path(include_file).is_absolute():
            include.set("file", str((src.parent / include_file).resolve()))

    worldbody = root.find("worldbody")
    assert worldbody is not None

    worldbody.append(
        ET.Element(
            "camera",
            {
                "name": "replay_extra_cam",
                "pos": "1.4 0.0 0.9",
                "xyaxes": "0 1 0 -0.3 0 1",
            },
        )
    )
    body = ET.SubElement(worldbody, "body", {"name": "replay_extra_bg", "pos": "3 3 3"})
    ET.SubElement(body, "geom", {"name": "replay_extra_bg_geom", "type": "box", "size": "0.1 0.1 0.1"})
    tree.write(dst)


def _assert_nested_close(actual: Any, expected: Any, *, atol: float = 1e-6):
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_close(actual[key], expected[key], atol=atol)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_close(actual_item, expected_item, atol=atol)
        return
    if expected is None:
        assert actual is None
        return
    if isinstance(expected, bool):
        assert actual is expected
        return
    if isinstance(expected, (int, float)):
        assert np.isclose(actual, expected, rtol=0.0, atol=atol)
        return
    assert actual == expected


def _strip_unstable_info(info: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in info.items():
        if key in {"camera_available", "frame_timestamp"}:
            continue
        if isinstance(value, dict):
            cleaned[key] = {
                nested_key: nested_value
                for nested_key, nested_value in value.items()
                if nested_key not in {"sim_state", "is_sim_converged", "absolute_action"}
            }
        else:
            cleaned[key] = value
    return cleaned


def _strip_frames(obs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in obs.items() if key != "frames"}


def _tilted_root_frame_to_world() -> rcs.common.Pose:
    return rcs.common.Pose(
        translation=np.array([0.35, -0.2, 0.15]),
        quaternion=np.array([0.0, 0.0, 0.38268343, 0.92387953]),
    )


def _joint_qpos_from_state(state: np.ndarray, schema: dict[str, list[str] | list[int]], joint_name: str) -> np.ndarray:
    joint_names = cast(list[str], schema["joint_names"])
    joint_index = joint_names.index(joint_name)
    qpos_offset = sum(int(size) for size in schema["qpos_sizes"][:joint_index])
    qpos_size = int(schema["qpos_sizes"][joint_index])
    return np.asarray(state[qpos_offset : qpos_offset + qpos_size], dtype=np.float64)


def _joint_qpos_in_root_frame(env: StorageWrapper, joint_name: str, root_frame_to_world: rcs.common.Pose) -> np.ndarray:
    joint_qpos_world = np.asarray(env.get_wrapper_attr("sim").data.joint(joint_name).qpos, dtype=np.float64)
    joint_pose_world = rcs.common.Pose(
        translation=joint_qpos_world[:3],
        quaternion=np.array([joint_qpos_world[4], joint_qpos_world[5], joint_qpos_world[6], joint_qpos_world[3]]),
    )
    joint_pose_root = root_frame_to_world.inverse() * joint_pose_world
    return np.concatenate((joint_pose_root.translation(), joint_pose_root.rotation_q_wxyz()))


def test_replayer_reproduces_existing_parquet_prefix_without_cameras(tmp_path: Path):
    source_dir = tmp_path / "source"
    replay_dir = tmp_path / "replayed"
    limit = 3
    instruction = "pick up cube"

    _record_source_dataset(source_dir, limit=limit, instruction=instruction)
    _replay_prefix(replay_dir, with_cameras=False, limit=limit)

    source_rows = _source_rows(source_dir, limit)
    replay_rows = _replay_rows(replay_dir)

    assert len(source_rows) == len(replay_rows) == limit

    for replay_row, source_row in zip(replay_rows, source_rows, strict=True):
        (
            replay_step,
            replay_obs,
            replay_info,
            replay_reward,
            replay_success,
            replay_action,
            replay_env_action,
            replay_instruction,
        ) = replay_row
        (
            source_step,
            source_obs,
            source_info,
            source_reward,
            source_success,
            source_action,
            source_env_action,
            source_instruction,
        ) = source_row

        _assert_nested_close(replay_step, source_step)
        _assert_nested_close(replay_obs, source_obs, atol=1e-5)
        assert replay_info["camera_available"] is source_info["camera_available"]
        _assert_nested_close(_strip_unstable_info(replay_info), _strip_unstable_info(source_info), atol=1e-5)
        # Replay from a restored simulator state is deterministic up to tiny floating-point drift.
        _assert_nested_close(replay_reward, source_reward, atol=1e-7)
        _assert_nested_close(replay_success, source_success)
        _assert_nested_close(replay_action, source_action, atol=1e-8)
        _assert_nested_close(replay_env_action, source_env_action, atol=1e-8)
        _assert_nested_close(replay_instruction, source_instruction)


def test_replayer_restores_sim_state_across_fixed_scene_changes(tmp_path: Path):
    source_scene_path = Path(EmptyWorldFR3Duo().config().scene)
    modified_scene_path = tmp_path / "modified_scene.xml"
    _write_scene_with_extra_fixed_body_and_camera(source_scene_path, modified_scene_path)

    for record_scene_path, replay_scene_path in (
        (source_scene_path, modified_scene_path),
        (modified_scene_path, source_scene_path),
    ):
        case_dir = tmp_path / f"{record_scene_path.stem}-to-{replay_scene_path.stem}"
        source_dir = case_dir / "source"
        replay_dir = case_dir / "replayed"

        _record_source_dataset(source_dir, limit=3, instruction="pick up cube", scene_path=record_scene_path)
        _replay_prefix(replay_dir, with_cameras=False, limit=3, scene_path=replay_scene_path)

        source_uuid = load_distinct_uuids(source_dir)[0]
        replay_uuid = load_distinct_uuids(replay_dir)[0]
        source_steps = load_trajectory(source_dir, source_uuid)
        replay_steps = load_trajectory(replay_dir, replay_uuid)

        assert len(source_steps) == len(replay_steps) == 3
        for replay_step, source_step in zip(replay_steps, source_steps, strict=True):
            assert replay_step.sim_state_schema == source_step.sim_state_schema
            assert np.allclose(replay_step.sim_state, source_step.sim_state, atol=1e-5, rtol=0)


def test_replayer_adds_cameras_to_existing_episode_without_cameras(tmp_path: Path):
    source_dir = tmp_path / "source"
    replay_dir = tmp_path / "replayed_with_cameras"
    limit = 3
    instruction = "pick up cube"

    _record_source_dataset(source_dir, limit=limit, instruction=instruction)
    _replay_prefix(replay_dir, with_cameras=True, limit=limit)

    source_rows = _source_rows(source_dir, limit)
    replay_rows = _replay_rows(replay_dir)

    assert len(source_rows) == len(replay_rows) == limit

    for replay_row, source_row in zip(replay_rows, source_rows, strict=True):
        (
            replay_step,
            replay_obs,
            replay_info,
            replay_reward,
            replay_success,
            replay_action,
            replay_env_action,
            replay_instruction,
        ) = replay_row
        (
            source_step,
            source_obs,
            source_info,
            source_reward,
            source_success,
            source_action,
            source_env_action,
            source_instruction,
        ) = source_row

        assert "frames" in replay_obs
        assert set(replay_obs["frames"]) == {"head", "left_wrist", "right_wrist"}
        assert replay_info["camera_available"] is True
        assert "frame_timestamp" in replay_info
        _assert_nested_close(replay_step, source_step)
        _assert_nested_close(_strip_frames(replay_obs), source_obs, atol=1e-5)
        _assert_nested_close(_strip_unstable_info(replay_info), _strip_unstable_info(source_info), atol=1e-5)
        # Replay from a restored simulator state is deterministic up to tiny floating-point drift.
        _assert_nested_close(replay_reward, source_reward, atol=1e-7)
        _assert_nested_close(replay_success, source_success)
        _assert_nested_close(replay_action, source_action, atol=1e-8)
        _assert_nested_close(replay_env_action, source_env_action, atol=1e-8)
        _assert_nested_close(replay_instruction, source_instruction)


def test_replayer_restores_root_relative_free_joint_state_across_root_frame_changes(tmp_path: Path):
    source_dir = tmp_path / "source"
    replay_dir = tmp_path / "replayed"
    default_root = rcs.common.Pose()
    shifted_root = _tilted_root_frame_to_world()
    object_joint_name = "PickTask_box_joint"

    _record_source_dataset(
        source_dir,
        limit=3,
        instruction="pick up cube",
        root_frame_to_world=default_root,
    )
    _replay_prefix(
        replay_dir,
        with_cameras=False,
        limit=3,
        root_frame_to_world=shifted_root,
    )

    source_uuid = load_distinct_uuids(source_dir)[0]
    replay_uuid = load_distinct_uuids(replay_dir)[0]
    source_steps = load_trajectory(source_dir, source_uuid)
    replay_steps = load_trajectory(replay_dir, replay_uuid)

    assert len(source_steps) == len(replay_steps) == 3
    for replay_step, source_step in zip(replay_steps, source_steps, strict=True):
        assert replay_step.sim_state_schema == source_step.sim_state_schema
        assert replay_step.sim_state_schema is not None
        schema = replay_step.sim_state_schema
        joint_names = cast(list[str], schema["joint_names"])
        encodings = cast(list[str], schema["encodings"])
        object_joint_index = joint_names.index(object_joint_name)
        assert encodings[object_joint_index] == ROOT_RELATIVE_FREE_STATE_ENCODING
        assert np.allclose(replay_step.sim_state, source_step.sim_state, atol=1e-5, rtol=0)

    replay_env = _build_env(replay_dir / "inspection", with_cameras=False, root_frame_to_world=shifted_root)
    try:
        replay_env.reset()
        lead_env = replay_env.get_wrapper_attr("lead_env")
        for source_step in source_steps:
            restore_sim_step(replay_env, source_step)
            lead_env.step_sim()
            assert source_step.sim_state_schema is not None
            expected_joint_qpos = _joint_qpos_from_state(
                source_step.sim_state, source_step.sim_state_schema, object_joint_name
            )
            actual_joint_qpos = _joint_qpos_in_root_frame(replay_env, object_joint_name, shifted_root)
            assert np.allclose(actual_joint_qpos, expected_joint_qpos, atol=1e-5, rtol=0)
    finally:
        replay_env.close()
