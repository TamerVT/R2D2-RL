import atexit
import contextlib
import multiprocessing as mp
import typing
import uuid
from logging import getLogger
from multiprocessing.synchronize import Event as EventClass
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import mujoco as mj
import mujoco.viewer
import numpy as np
from rcs._core import common
from rcs._core.sim import DynamicJointSchema, DynamicJointState
from rcs._core.sim import GuiClient as _GuiClient
from rcs._core.sim import Sim as _Sim
from rcs.sim import SimConfig, egl_bootstrap
from rcs.sim.composer import ModelComposer
from rcs.utils import SimpleFrameRate

egl_bootstrap.bootstrap()
logger = getLogger(__name__)


# Target frames per second
FPS = 60
RAW_STATE_ENCODING = "raw"
ROOT_RELATIVE_FREE_STATE_ENCODING = "root_relative_free"


def gui_loop(gui_uuid: str, close_event):
    frame_rate = SimpleFrameRate(FPS, "gui_loop")
    gui_client = _GuiClient(gui_uuid)
    model_bytes = gui_client.get_model_bytes()
    with NamedTemporaryFile(mode="wb") as f:
        f.write(model_bytes)
        model = mujoco.MjModel.from_binary_path(f.name)
    data = mujoco.MjData(model)
    gui_client.set_model_and_data(model._address, data._address)
    mujoco.mj_step(model, data)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while not close_event.is_set():
            mujoco.mj_step(model, data)
            viewer.sync()
            gui_client.sync()
            frame_rate()


class Sim(_Sim):
    def __init__(self, mjmdl: str | PathLike | ModelComposer, cfg: SimConfig | None = None):
        if isinstance(mjmdl, ModelComposer):
            self.model = mjmdl.get_model()
        else:
            mjmdl = Path(mjmdl)
            if mjmdl.suffix == ".xml":
                self.model = mj.MjModel.from_xml_path(str(mjmdl))
            elif mjmdl.suffix == ".mjb":
                self.model = mj.MjModel.from_binary_path(str(mjmdl))
            else:
                msg = f"Filetype {mjmdl.suffix} is unknown"
                logger.error(msg)

        self.data = mj.MjData(self.model)
        super().__init__(self.model._address, self.data._address)
        self._mp_context = mp.get_context("spawn")
        self._gui_uuid: Optional[str] = None
        self._gui_client: Optional[_GuiClient] = None
        self._gui_process: Optional[mp.context.SpawnProcess] = None
        self._stop_event: Optional[EventClass] = None
        self._gui_atexit_registered = False
        self._root_frame_to_world = common.Pose()
        self._root_relative_replay_free_joints: set[str] = set()
        if cfg is not None:
            self.set_config(cfg)

    def configure_state_encodings(
        self,
        *,
        root_frame_to_world: common.Pose,
        root_relative_free_joints: typing.Iterable[str] = (),
    ):
        self._root_frame_to_world = common.Pose(root_frame_to_world)
        self._root_relative_replay_free_joints = set(root_relative_free_joints)

    def get_state_schema(self) -> dict[str, list[str] | list[int]]:
        schema = super().get_dynamic_joint_schema()
        return {
            "joint_names": list(schema.joint_names),
            "joint_types": list(schema.joint_types),
            "qpos_sizes": list(schema.qpos_sizes),
            "qvel_sizes": list(schema.qvel_sizes),
            "encodings": [
                (
                    ROOT_RELATIVE_FREE_STATE_ENCODING
                    if joint_name in self._root_relative_replay_free_joints
                    else RAW_STATE_ENCODING
                )
                for joint_name in schema.joint_names
            ],
        }

    def get_state_size(self, schema: dict[str, list[str] | list[int]] | None = None) -> int:
        state_schema = self.get_state_schema() if schema is None else schema
        qpos_size = sum(int(value) for value in state_schema["qpos_sizes"])
        qvel_size = sum(int(value) for value in state_schema["qvel_sizes"])
        return qpos_size + qvel_size

    def get_state(self) -> np.ndarray:
        state = super().get_dynamic_joint_state()
        qpos = np.array(state.qpos, copy=True)
        qvel = np.array(state.qvel, copy=True)
        self._encode_state_in_place(qpos, qvel, self.get_state_schema())
        return np.concatenate((qpos, qvel))

    def set_state(
        self,
        state: np.ndarray,
        schema: dict[str, list[str] | list[int]] | None = None,
    ):
        state_schema = self.get_state_schema() if schema is None else schema
        state_array = np.asarray(state, dtype=np.float64)
        expected_size = self.get_state_size(state_schema)
        if state_array.shape != (expected_size,):
            msg = f"Expected state with shape ({expected_size},), got {state_array.shape}."
            raise ValueError(msg)

        qpos_size = sum(int(value) for value in state_schema["qpos_sizes"])
        qpos = np.array(state_array[:qpos_size], copy=True)
        qvel = np.array(state_array[qpos_size:], copy=True)
        self._decode_state_in_place(qpos, qvel, state_schema)

        dynamic_joint_schema = DynamicJointSchema()
        dynamic_joint_schema.joint_names = typing.cast(list[str], list(state_schema["joint_names"]))
        dynamic_joint_schema.joint_types = [int(value) for value in state_schema["joint_types"]]
        dynamic_joint_schema.qpos_sizes = [int(value) for value in state_schema["qpos_sizes"]]
        dynamic_joint_schema.qvel_sizes = [int(value) for value in state_schema["qvel_sizes"]]

        dynamic_joint_state = DynamicJointState()  # type: ignore
        dynamic_joint_state.qpos = qpos
        dynamic_joint_state.qvel = qvel
        super().set_dynamic_joint_state(dynamic_joint_schema, dynamic_joint_state)

    def _encode_state_in_place(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        schema: dict[str, list[str] | list[int]],
    ):
        self._transform_state_in_place(qpos, qvel, schema, encode=True)

    def _decode_state_in_place(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        schema: dict[str, list[str] | list[int]],
    ):
        self._transform_state_in_place(qpos, qvel, schema, encode=False)

    def _transform_state_in_place(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        schema: dict[str, list[str] | list[int]],
        *,
        encode: bool,
    ):
        joint_names = typing.cast(list[str], list(schema["joint_names"]))
        joint_types = [int(value) for value in schema["joint_types"]]
        qpos_sizes = [int(value) for value in schema["qpos_sizes"]]
        qvel_sizes = [int(value) for value in schema["qvel_sizes"]]
        encodings = typing.cast(list[str], list(schema.get("encodings", [RAW_STATE_ENCODING] * len(joint_names))))

        qpos_offset = 0
        qvel_offset = 0
        free_joint_type = int(mj.mjtJoint.mjJNT_FREE)
        for joint_name, joint_type, joint_qpos_size, joint_qvel_size, encoding in zip(
            joint_names, joint_types, qpos_sizes, qvel_sizes, encodings, strict=True
        ):
            if encoding == RAW_STATE_ENCODING:
                pass
            elif encoding == ROOT_RELATIVE_FREE_STATE_ENCODING:
                if joint_type != free_joint_type or joint_qpos_size != 7 or joint_qvel_size != 6:
                    msg = (
                        f"Joint '{joint_name}' uses encoding '{ROOT_RELATIVE_FREE_STATE_ENCODING}' "
                        "but is not a free joint."
                    )
                    raise ValueError(msg)
                joint_qpos = qpos[qpos_offset : qpos_offset + joint_qpos_size]
                joint_qvel = qvel[qvel_offset : qvel_offset + joint_qvel_size]
                transformed_qpos, transformed_qvel = self._transform_root_relative_free_joint(
                    joint_qpos, joint_qvel, encode=encode
                )
                qpos[qpos_offset : qpos_offset + joint_qpos_size] = transformed_qpos
                qvel[qvel_offset : qvel_offset + joint_qvel_size] = transformed_qvel
            else:
                msg = f"Unsupported sim state encoding '{encoding}' for joint '{joint_name}'."
                raise ValueError(msg)

            qpos_offset += joint_qpos_size
            qvel_offset += joint_qvel_size

    def _transform_root_relative_free_joint(
        self,
        joint_qpos: np.ndarray,
        joint_qvel: np.ndarray,
        *,
        encode: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        joint_pose_world = common.Pose(
            translation=np.asarray(joint_qpos[:3], dtype=np.float64),
            quaternion=np.asarray([joint_qpos[4], joint_qpos[5], joint_qpos[6], joint_qpos[3]], dtype=np.float64),
        )
        root_inverse = self._root_frame_to_world.inverse()
        joint_pose = root_inverse * joint_pose_world if encode else self._root_frame_to_world * joint_pose_world

        rotation = root_inverse.rotation_m() if encode else self._root_frame_to_world.rotation_m()
        transformed_qvel = np.concatenate((rotation @ joint_qvel[:3], rotation @ joint_qvel[3:6]))

        return (
            np.concatenate((joint_pose.translation(), joint_pose.rotation_q_wxyz())),
            transformed_qvel,
        )

    def close_gui(self):
        if self._stop_event is not None:
            self._stop_event.set()
        if self._gui_process is not None:
            self._gui_process.join()
        self._stop_gui_server()
        self._gui_uuid = None
        self._gui_client = None
        self._gui_process = None
        self._stop_event = None
        if self._gui_atexit_registered:
            with contextlib.suppress(ValueError):
                atexit.unregister(self.close_gui)
            self._gui_atexit_registered = False

    def open_gui(self):
        if self._gui_uuid is None:
            self._gui_uuid = "rcs_" + str(uuid.uuid4())
            self._start_gui_server(self._gui_uuid)
        if self._gui_process is None or not self._gui_process.is_alive():
            self._stop_event = self._mp_context.Event()
            self._gui_process = self._mp_context.Process(
                target=gui_loop,
                args=(self._gui_uuid, self._stop_event),
            )
            self._gui_process.start()
        if not self._gui_atexit_registered:
            atexit.register(self.close_gui)
            self._gui_atexit_registered = True
