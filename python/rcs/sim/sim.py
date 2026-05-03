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
from rcs._core.sim import DynamicJointSchema
from rcs._core.sim import DynamicJointState
from rcs._core.sim import GuiClient as _GuiClient
from rcs._core.sim import Sim as _Sim
from rcs.sim import SimConfig, egl_bootstrap
from rcs.sim.composer import ModelComposer
from rcs.utils import SimpleFrameRate

egl_bootstrap.bootstrap()
logger = getLogger(__name__)


# Target frames per second
FPS = 60


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
        if cfg is not None:
            self.set_config(cfg)

    def get_state_schema(self) -> dict[str, list[str] | list[int]]:
        schema = super().get_dynamic_joint_schema()
        return {
            "joint_names": list(schema.joint_names),
            "joint_types": list(schema.joint_types),
            "qpos_sizes": list(schema.qpos_sizes),
            "qvel_sizes": list(schema.qvel_sizes),
        }

    def get_state_size(self, schema: dict[str, list[str] | list[int]] | None = None) -> int:
        state_schema = self.get_state_schema() if schema is None else schema
        qpos_size = sum(int(value) for value in state_schema["qpos_sizes"])
        qvel_size = sum(int(value) for value in state_schema["qvel_sizes"])
        return qpos_size + qvel_size

    def get_state(self) -> np.ndarray:
        state = super().get_dynamic_joint_state()
        return np.concatenate((state.qpos, state.qvel))

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


        dynamic_joint_schema = DynamicJointSchema()
        dynamic_joint_schema.joint_names = typing.cast(list[str], list(state_schema["joint_names"]))
        dynamic_joint_schema.joint_types = [int(value) for value in state_schema["joint_types"]]
        dynamic_joint_schema.qpos_sizes = [int(value) for value in state_schema["qpos_sizes"]]
        dynamic_joint_schema.qvel_sizes = [int(value) for value in state_schema["qvel_sizes"]]

        dynamic_joint_state = DynamicJointState() # type: ignore
        dynamic_joint_state.qpos = state_array[:qpos_size]
        dynamic_joint_state.qvel = state_array[qpos_size:]
        super().set_dynamic_joint_state(dynamic_joint_schema, dynamic_joint_state)

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
