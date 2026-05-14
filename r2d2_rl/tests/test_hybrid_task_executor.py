import unittest

from tests import BASE_CONFIG_PATH

import numpy as np

from estimation.block_belief import BlockBelief
from hybrid_control_rl.config import load_yaml_config
from planning.hybrid_waypoint_planner import HybridWaypointPlanner
from runtime.hybrid_task_executor import HybridTaskExecutor, HybridTaskState, TaskGoal


def _belief(color: str = "green", xy=(0.20, -0.03)) -> BlockBelief:
    return BlockBelief(
        color=color,
        mean_xy=np.asarray(xy, dtype=np.float64),
        covariance_xy=np.eye(2) * 1e-5,
        last_seen_time=0.0,
        confidence=1.0,
        initialized=True,
    )


class MockObserver:
    def __init__(self, beliefs):
        self.beliefs = list(beliefs)
        self.calls = []

    def observe(self, target_color):
        self.calls.append(target_color)
        if not self.beliefs:
            return None
        return self.beliefs.pop(0)


class MockController:
    def __init__(self):
        self.calls = []

    def execute(self, waypoints):
        names = tuple(waypoint.name for waypoint in waypoints)
        self.calls.append(names)
        return True


class MockPolicy:
    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    def run(self, phase, target_color, belief):
        self.calls.append((phase, target_color, belief.color))
        return self.ok


class MockVisibility:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def is_visible(self, target_color):
        self.calls.append(target_color)
        if not self.values:
            return True
        return self.values.pop(0)


class HybridTaskExecutorTest(unittest.TestCase):
    def _executor(self, config, observer, controller, policy, visibility=None):
        planner = HybridWaypointPlanner(config)
        return HybridTaskExecutor(
            config=config,
            planner=planner,
            observer=observer,
            controller=controller,
            local_policy=policy,
            visibility_checker=visibility,
        )

    def test_successful_goal_runs_align_grasp_only(self):
        config = load_yaml_config(str(BASE_CONFIG_PATH))
        observer = MockObserver([_belief()])
        controller = MockController()
        policy = MockPolicy()
        executor = self._executor(config, observer, controller, policy, MockVisibility([True] * 20))

        result = executor.run_goal(TaskGoal("green", np.array([0.30, 0.10, 0.05])))

        self.assertTrue(result.success, msg=result.failure_reason)
        self.assertEqual(result.final_state, HybridTaskState.DONE)
        self.assertEqual(result.attempts, 0)
        self.assertEqual(policy.calls, [("align_grasp", "green", "green")])
        executed = [name for call in controller.calls for name in call]
        self.assertIn("pregrasp", executed)
        self.assertIn("lift", executed)
        self.assertIn("release", executed)
        trace_states = [event.state for event in result.trace]
        self.assertIn(HybridTaskState.OBSERVE, trace_states)
        self.assertIn(HybridTaskState.TRANSPORT, trace_states)
        self.assertIn(HybridTaskState.RELEASE, trace_states)

    def test_lost_target_triggers_recovery_and_retry(self):
        config = load_yaml_config(str(BASE_CONFIG_PATH))
        config["recovery"]["max_lost_frames"] = 1
        config["recovery"]["max_attempts"] = 2
        config["planning"]["max_waypoint_step_m"] = 0.04
        observer = MockObserver([_belief(), _belief()])
        controller = MockController()
        policy = MockPolicy()
        visibility = MockVisibility([False, True, True, True, True])
        executor = self._executor(config, observer, controller, policy, visibility)

        result = executor.run_goal(TaskGoal("green", np.array([0.30, 0.10, 0.05])))

        self.assertTrue(result.success, msg=result.failure_reason)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(observer.calls, ["green", "green"])
        executed = [name for call in controller.calls for name in call]
        self.assertIn("return_to_safe_pose", executed)
        trace_states = [event.state for event in result.trace]
        self.assertIn(HybridTaskState.RECOVERY, trace_states)

    def test_align_grasp_failure_fails_without_release(self):
        config = load_yaml_config(str(BASE_CONFIG_PATH))
        observer = MockObserver([_belief()])
        controller = MockController()
        executor = self._executor(config, observer, controller, MockPolicy(ok=False))

        result = executor.run_goal(TaskGoal("green", np.array([0.30, 0.10, 0.05])))

        self.assertFalse(result.success)
        self.assertIn("align_grasp", result.failure_reason)
        executed = [name for call in controller.calls for name in call]
        self.assertIn("pregrasp", executed)
        self.assertNotIn("release", executed)

    def test_unsupported_rl_phases_are_rejected(self):
        config = load_yaml_config(str(BASE_CONFIG_PATH))
        config["rl"]["phases"] = ["align_grasp", "release"]

        with self.assertRaisesRegex(ValueError, "unsupported phases"):
            self._executor(config, MockObserver([]), MockController(), MockPolicy())


if __name__ == "__main__":
    unittest.main()
