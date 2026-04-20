from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco as mj
import numpy as np
import pyarrow.dataset as ds
from rcs._core.common import RobotPlatform
from rcs.camera.interface import CameraFrame, DataFrame, Frame, FrameSet
from rcs.envs.storage_wrapper import StorageWrapper

import rcs

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_local_module(module_name: str, relative_path: str):
    module_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        msg = f"Could not create an import spec for {module_name} from {module_path}."
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    parent_name, _, child_name = module_name.rpartition(".")
    if parent_name:
        parent_module = sys.modules[parent_name]
        setattr(parent_module, child_name, module)
    spec.loader.exec_module(module)
    return module


local_sim_module = _load_local_module("rcs.sim.sim", "python/rcs/sim/sim.py")
rcs.sim.__dict__["Sim"] = local_sim_module.Sim
_load_local_module("rcs.envs.sim", "python/rcs/envs/sim.py")
_load_local_module("rcs.sim_state_replay", "python/rcs/sim_state_replay.py")

from rcs.envs.sim import SimStateObservationWrapper  # noqa: E402
from rcs.sim.sim import Sim  # noqa: E402
from rcs.sim_state_replay import (  # noqa: E402
    load_trajectory,
    replay_trajectory,
    restore_sim_step,
)

XML = """
<mujoco>
  <worldbody>
    <camera name="main" pos="1 0 0.7" xyaxes="0 1 0 -0.5 0 1"/>
    <body name="box" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" rgba="0.2 0.6 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""


@dataclass
class DummyCameraSet:
    sim: Sim

    def get_latest_frames(self) -> FrameSet:
        color_value = int(np.clip(round((self.sim.data.qpos[0] + 1.0) * 80.0), 0, 255))
        rgb = np.full((8, 8, 3), color_value, dtype=np.uint8)
        return FrameSet(
            frames={
                "main": Frame(
                    camera=CameraFrame(
                        color=DataFrame(data=rgb),
                        depth=None,
                    ),
                )
            },
            avg_timestamp=None,
        )


class DummySimEnv(gym.Env):
    PLATFORM = RobotPlatform.SIMULATION

    def __init__(self, sim: Sim, camera_set: DummyCameraSet | None = None):
        super().__init__()
        self.sim = sim
        self.camera_set = camera_set
        self.action_space = gym.spaces.Dict(
            {
                "delta": gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64),
            }
        )
        self.observation_space = gym.spaces.Dict(
            {
                "qpos": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.sim.model.nq,), dtype=np.float64),
                "qvel": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.sim.model.nv,), dtype=np.float64),
            }
        )

    def _obs(self) -> dict[str, np.ndarray]:
        return {
            "qpos": self.sim.data.qpos.copy(),
            "qvel": self.sim.data.qvel.copy(),
        }

    def get_wrapper_attr(self, name: str):
        return getattr(self, name)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mj.mj_resetData(self.sim.model, self.sim.data)
        mj.mj_forward(self.sim.model, self.sim.data)
        return self._obs(), {}

    def step(self, action: dict[str, np.ndarray]):
        self.sim.data.qpos[0] += float(action["delta"][0])
        self.sim.data.qvel[:] = 0.0
        mj.mj_forward(self.sim.model, self.sim.data)
        return self._obs(), 0.0, False, False, {}

    def close(self):
        return None


def test_record_and_replay_sim_state(tmp_path: Path):
    model_path = tmp_path / "dummy.xml"
    model_path.write_text(XML)

    dataset_path = tmp_path / "dataset"
    record_env: gym.Env = DummySimEnv(Sim(model_path))
    record_env = SimStateObservationWrapper(record_env)
    record_env = StorageWrapper(record_env, str(dataset_path), "test sim replay", batch_size=1, always_record=True)

    obs, _ = record_env.reset()
    assert SimStateObservationWrapper.STATE_KEY in obs

    record_env.step({"delta": np.array([0.125], dtype=np.float64)})
    record_env.close()

    table = ds.dataset(str(dataset_path), format="parquet").to_table().sort_by([("step", "ascending")])
    rows = table.to_pylist()
    assert len(rows) == 1

    recorded_obs = rows[0]["obs"]
    assert SimStateObservationWrapper.STATE_KEY in recorded_obs
    assert SimStateObservationWrapper.STATE_SPEC_KEY in recorded_obs
    assert SimStateObservationWrapper.STATE_SIZE_KEY in recorded_obs
    assert (
        len(recorded_obs[SimStateObservationWrapper.STATE_KEY])
        == recorded_obs[SimStateObservationWrapper.STATE_SIZE_KEY]
    )

    recorded_steps = load_trajectory(dataset_path, rows[0]["uuid"], prefer_duckdb=True)
    assert len(recorded_steps) == 1
    assert np.allclose(recorded_steps[0].sim_state, np.asarray(recorded_obs[SimStateObservationWrapper.STATE_KEY]))

    replay_sim = Sim(model_path)
    replay_env: gym.Env = DummySimEnv(replay_sim, camera_set=DummyCameraSet(replay_sim))
    replay_env = SimStateObservationWrapper(replay_env)
    render_dir = tmp_path / "rendered"

    replay_env.reset()
    restore_sim_step(replay_env, recorded_steps[0])
    assert np.allclose(
        replay_env.get_wrapper_attr("sim").data.qpos, np.asarray(recorded_obs["qpos"]), atol=1e-9, rtol=0
    )
    assert np.allclose(
        replay_env.get_wrapper_attr("sim").data.qvel, np.asarray(recorded_obs["qvel"]), atol=1e-9, rtol=0
    )

    replay_trajectory(replay_env, recorded_steps, output_dir=render_dir)

    rendered_files = sorted(path.name for path in render_dir.glob("*.png"))
    assert rendered_files == ["step-000000-main.png"]
