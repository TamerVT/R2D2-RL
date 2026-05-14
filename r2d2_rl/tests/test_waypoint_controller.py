"""Unit tests for ``control.waypoint_controller`` step shaping + gripper preservation.

The tests do not bring up a real RCS env; they swap in a tiny fake that records
``step`` actions so we can assert the controller commands the right deltas and
gripper values.
"""

import unittest
from typing import Any

import numpy as np

from control.waypoint_controller import RcsWaypointController, _proportional_step
from planning.hybrid_waypoint_planner import Waypoint


class _Box:
    def __init__(self, spaces: dict[str, object]) -> None:
        self.spaces = spaces


class _ActionSpace:
    def __init__(self) -> None:
        self.spaces = {"robot": _Box({"tquat": object(), "gripper": object()})}


class _FakeEnv:
    def __init__(self) -> None:
        self.action_space = _ActionSpace()
        self.actions: list[dict[str, Any]] = []
        self.xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.gripper = 0.5

    def reset(self, seed: int | None = None):
        self.xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.gripper = 0.5
        return self._obs(), {}

    def step(self, action: dict[str, Any]):
        self.actions.append(action)
        robot = action["robot"]
        delta = np.asarray(robot["tquat"], dtype=np.float64)[:3]
        self.xyz = self.xyz + delta
        gripper = robot.get("gripper")
        if gripper is not None:
            self.gripper = float(np.asarray(gripper, dtype=np.float64).reshape(-1)[0])
        return self._obs(), 0.0, False, False, {}

    def _obs(self) -> dict[str, Any]:
        return {
            "robot": {
                "tquat": np.array([*self.xyz, 0.0, 0.0, 0.0, 1.0]),
                "gripper": np.array([self.gripper], dtype=np.float32),
            },
        }


class ProportionalStepTest(unittest.TestCase):
    def test_short_vector_passes_through_unchanged(self):
        v = np.array([0.01, 0.005, -0.01], dtype=np.float64)
        step = _proportional_step(v, max_step=0.05)
        np.testing.assert_allclose(step, v)

    def test_long_vector_is_scaled_to_max_step(self):
        v = np.array([0.30, 0.0, -0.40], dtype=np.float64)
        step = _proportional_step(v, max_step=0.05)
        self.assertAlmostEqual(float(np.linalg.norm(step)), 0.05)
        np.testing.assert_allclose(step / np.linalg.norm(step), v / np.linalg.norm(v))

    def test_zero_vector_is_zero(self):
        step = _proportional_step(np.zeros(3), max_step=0.05)
        np.testing.assert_array_equal(step, np.zeros(3))


class RcsWaypointControllerTest(unittest.TestCase):
    def test_reaches_waypoint_with_diagonal_steps(self):
        env = _FakeEnv()
        controller = RcsWaypointController(env, max_step_m=0.05, position_tolerance_m=0.005)
        controller.reset()

        waypoint = Waypoint(
            name="diag",
            xyz_base=np.array([0.10, 0.10, 0.10], dtype=np.float64),
            rpy_base=np.zeros(3),
        )
        ok = controller.execute([waypoint])

        self.assertTrue(ok)
        # Last action should have moved the env close to the waypoint.
        self.assertLess(float(np.linalg.norm(env.xyz - waypoint.xyz_base)), 0.005)
        # No single action delta should exceed max_step.
        for act in env.actions:
            delta = np.asarray(act["robot"]["tquat"])[:3]
            self.assertLessEqual(float(np.linalg.norm(delta)), 0.05 + 1e-9)

    def test_passing_gripper_none_preserves_current_state(self):
        env = _FakeEnv()
        env.gripper = 0.0  # gripper is closed (e.g., holding a cube)
        controller = RcsWaypointController(env)
        controller.reset()
        controller.last_obs["robot"]["gripper"] = np.array([0.0], dtype=np.float32)

        controller.step_delta(np.zeros(3), gripper=None)
        # The action should have sent gripper=0.0 (preserved), not 1.0 (open).
        last_action = env.actions[-1]
        gripper_cmd = float(np.asarray(last_action["robot"]["gripper"]).reshape(-1)[0])
        self.assertAlmostEqual(gripper_cmd, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
