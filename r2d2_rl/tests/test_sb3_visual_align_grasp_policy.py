"""Tests for the SB3 visual align-grasp hybrid adapter."""

import unittest

import numpy as np

from r2d2_rl.rl.lerobot_compat import IMAGE_KEY, STATE_KEY, target_color_onehot
from r2d2_rl.runtime.sb3_visual_align_grasp_policy import SB3VisualAlignGraspPolicy


class _FakeRobot:
    def __init__(self) -> None:
        self.last_q: np.ndarray | None = None

    def set_joint_position(self, q: np.ndarray) -> None:
        self.last_q = np.asarray(q, dtype=np.float64)


class _FakeEnv:
    def __init__(self, robot: _FakeRobot) -> None:
        self.robot = robot

    def get_wrapper_attr(self, name: str):
        if name == "robot":
            return self.robot
        raise AttributeError(name)


class _FakeController:
    def __init__(self, obs: dict | None = None, robot: _FakeRobot | None = None) -> None:
        self.last_obs = obs
        self.env = _FakeEnv(robot or _FakeRobot())
        self.last_gripper: float | None = None

    def step_delta(self, delta_xyz: np.ndarray, gripper: float | None = None):
        self.last_gripper = gripper
        return self.last_obs or {}, 0.0, False, False, {}


def _policy_without_model(**attrs) -> SB3VisualAlignGraspPolicy:
    policy = object.__new__(SB3VisualAlignGraspPolicy)
    defaults = {
        "controller": _FakeController(),
        "camera_name": "wrist",
        "image_size": 128,
        "real_gripper_max": 35.0,
        "compat_dt_s": 0.1,
        "joint_delta_deg": 5.0,
        "action_scale": 1.0,
        "_prev_real_positions": None,
    }
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(policy, key, value)
    return policy


class SB3VisualAlignGraspPolicyTest(unittest.TestCase):
    def test_build_obs_matches_lerobot_visual_boundary(self):
        rgb = np.zeros((64, 96, 3), dtype=np.uint8)
        rgb[20:40, 30:50] = np.array([0, 255, 0], dtype=np.uint8)
        obs = {
            "robot": {
                "joints": np.deg2rad(np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)),
                "gripper": np.array([0.5], dtype=np.float32),
            },
            "frames": {"wrist": {"rgb": {"data": rgb}}},
        }
        policy = _policy_without_model(controller=_FakeController(obs=obs))

        converted = policy._build_obs("green")

        self.assertIsNotNone(converted)
        assert converted is not None
        self.assertEqual(converted[IMAGE_KEY].shape, (3, 128, 128))
        self.assertEqual(converted[IMAGE_KEY].dtype, np.uint8)
        self.assertEqual(converted[STATE_KEY].shape, (24,))
        np.testing.assert_allclose(
            converted[STATE_KEY][:6],
            np.array([1.0, -2.0, 3.0, -4.0, 5.0, 17.5], dtype=np.float32),
            atol=1e-5,
        )
        np.testing.assert_allclose(converted[STATE_KEY][6:12], np.zeros(6, dtype=np.float32))
        np.testing.assert_allclose(converted[STATE_KEY][18:24], target_color_onehot("green"))

    def test_apply_action_uses_bounded_lerobot_joint_step(self):
        robot = _FakeRobot()
        controller = _FakeController(robot=robot)
        policy = _policy_without_model(controller=controller)
        policy._prev_real_positions = np.array([0.0, 10.0, -10.0, 0.0, 4.0, 20.0], dtype=np.float32)

        action = np.array([20.0, 0.0, -30.0, 10.0, 9.0, 17.5], dtype=np.float32)
        policy._apply_lerobot_action(action)

        expected_deg = np.array([5.0, 5.0, -15.0, 5.0, 9.0], dtype=np.float64)
        np.testing.assert_allclose(robot.last_q, np.deg2rad(expected_deg), atol=1e-8)
        self.assertAlmostEqual(controller.last_gripper, 0.5)


if __name__ == "__main__":
    unittest.main()
