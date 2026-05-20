"""Unit tests for the run_eval_sequence.py config helpers + multi-cube wiring."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


class ConfigHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        from scripts.run_eval_sequence import _cube_specs_from_config, _goals_from_config

        self._cube_specs_from_config = _cube_specs_from_config
        self._goals_from_config = _goals_from_config

    def test_scene_cubes_parsed(self) -> None:
        config = {
            "scene": {
                "cubes": [
                    {"color": "red", "xy": [0.20, 0.05]},
                    {"color": "green", "xy": [0.22, -0.04], "z": 0.02, "yaw": 0.3},
                ],
            },
            "eval": {"goals": [{"target_color": "red", "bowl_xyz_base": [0.3, 0.1, 0.05]}]},
        }
        specs = self._cube_specs_from_config(config)
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].color, "red")
        self.assertEqual(specs[0].xy, (0.20, 0.05))
        self.assertEqual(specs[1].color, "green")
        self.assertAlmostEqual(specs[1].z, 0.02)
        self.assertAlmostEqual(specs[1].yaw, 0.3)

    def test_scene_falls_back_to_first_goal(self) -> None:
        config = {
            "eval": {"goals": [{"target_color": "blue", "bowl_xyz_base": [0.30, 0.10, 0.05]}]},
        }
        specs = self._cube_specs_from_config(config)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].color, "blue")

    def test_any_target_color_resolves(self) -> None:
        config = {"eval": {"goals": [{"target_color": "any", "bowl_xyz_base": [0.3, 0.1, 0.05]}]}}
        specs = self._cube_specs_from_config(config)
        self.assertEqual(specs[0].color, "green")  # default fallback

    def test_goals_parsed(self) -> None:
        config = {
            "eval": {
                "goals": [
                    {"target_color": "red", "bowl_xyz_base": [0.3, 0.1, 0.05]},
                    {"target_color": "blue", "bowl_xyz_base": [0.2, -0.15, 0.05]},
                    {"target_color": "green", "bowl_xyz_base": [0.3, 0.0, 0.05]},
                ],
            },
        }
        goals = self._goals_from_config(config)
        self.assertEqual(len(goals), 3)
        self.assertEqual(goals[0].target_color, "red")
        self.assertEqual(goals[1].target_color, "blue")
        np.testing.assert_allclose(goals[2].bowl_xyz_base, [0.3, 0.0, 0.05])

    def test_goal_any_falls_back(self) -> None:
        config = {"eval": {"goals": [{"target_color": "any", "bowl_xyz_base": [0.3, 0.1, 0.05]}]}}
        goals = self._goals_from_config(config, fallback_color="purple")
        self.assertEqual(goals[0].target_color, "purple")


class EvalYamlIntegrationTest(unittest.TestCase):
    """Verify the shipped eval2/eval3 YAMLs are loadable + multi-cube ready."""

    def setUp(self) -> None:
        from hybrid_control_rl.config import load_yaml_config
        from scripts.run_eval_sequence import _cube_specs_from_config, _goals_from_config

        self._load = load_yaml_config
        self._cube_specs = _cube_specs_from_config
        self._goals = _goals_from_config

    def _config_path(self, name: str) -> Path:
        from tests import BASE_CONFIG_PATH

        return BASE_CONFIG_PATH.parent / name

    def test_eval2_yaml_loads(self) -> None:
        cfg = self._load(self._config_path("eval2.yaml"))
        specs = self._cube_specs(cfg)
        goals = self._goals(cfg)
        self.assertGreaterEqual(len(specs), 2, "Eval 2 must declare multiple distractor cubes.")
        self.assertEqual(len(goals), 1, "Eval 2 should have exactly one target goal.")
        target = goals[0].target_color
        colors_in_scene = {s.color for s in specs}
        self.assertIn(target, colors_in_scene, "Eval 2 target color must be present in scene.")

    def test_eval3_yaml_loads(self) -> None:
        cfg = self._load(self._config_path("eval3.yaml"))
        specs = self._cube_specs(cfg)
        goals = self._goals(cfg)
        self.assertGreaterEqual(len(goals), 2, "Eval 3 must declare >=2 goals.")
        scene_colors = {s.color for s in specs}
        for goal in goals:
            self.assertIn(
                goal.target_color, scene_colors,
                f"Eval 3 goal color {goal.target_color} not in scene cubes.",
            )

    def test_eval2_has_distractors(self) -> None:
        cfg = self._load(self._config_path("eval2.yaml"))
        specs = self._cube_specs(cfg)
        goals = self._goals(cfg)
        target_color = goals[0].target_color
        distractors = [s for s in specs if s.color != target_color]
        self.assertGreaterEqual(
            len(distractors), 1,
            "Eval 2 must include at least one distractor cube.",
        )


if __name__ == "__main__":
    unittest.main()
