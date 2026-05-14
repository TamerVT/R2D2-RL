import unittest

import numpy as np

from estimation.block_belief import BlockBelief
from hybrid_control_rl.config import load_yaml_config
from planning.hybrid_waypoint_planner import HybridWaypointPlanner, PlanningError


def _belief(color: str = "green", xy=(0.20, -0.03), variance: float = 1e-5) -> BlockBelief:
    return BlockBelief(
        color=color,
        mean_xy=np.asarray(xy, dtype=np.float64),
        covariance_xy=np.eye(2) * variance,
        last_seen_time=0.0,
        confidence=1.0,
        initialized=True,
    )


class HybridWaypointPlannerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_yaml_config("configs/hybrid_control_rl/base.yaml")

    def test_pregrasp_waypoint_uses_belief_xy_and_configured_height(self):
        planner = HybridWaypointPlanner(self.config)

        plan = planner.plan_pregrasp(_belief())

        self.assertEqual(plan.phase, "approach")
        self.assertEqual(len(plan.waypoints), 1)
        waypoint = plan.waypoints[0]
        self.assertEqual(waypoint.name, "pregrasp")
        np.testing.assert_allclose(waypoint.xyz_base, [0.20, -0.03, 0.10])
        np.testing.assert_allclose(waypoint.rpy_base, self.config["planning"]["grasp_orientation_rpy"])
        self.assertEqual(waypoint.gripper, 1.0)

    def test_high_uncertainty_belief_is_rejected(self):
        planner = HybridWaypointPlanner(self.config)

        with self.assertRaisesRegex(PlanningError, "too uncertain"):
            planner.plan_pregrasp(_belief(variance=0.01))

    def test_out_of_workspace_target_is_rejected(self):
        planner = HybridWaypointPlanner(self.config)

        with self.assertRaisesRegex(PlanningError, "outside workspace"):
            planner.plan_pregrasp(_belief(xy=(0.80, 0.0)))

    def test_post_grasp_plan_splits_classical_phases(self):
        config = load_yaml_config("configs/hybrid_control_rl/base.yaml")
        config["planning"]["max_waypoint_step_m"] = 0.04
        planner = HybridWaypointPlanner(config)

        post = planner.plan_post_grasp(_belief(), np.array([0.30, 0.10, 0.05]))

        self.assertEqual(post.lift.phase, "lift")
        self.assertEqual(post.transport.phase, "transport")
        self.assertEqual(post.release.phase, "release")
        np.testing.assert_allclose(post.lift.waypoints[0].xyz_base, [0.20, -0.03, 0.15])
        self.assertEqual(post.lift.waypoints[0].gripper, 0.0)
        self.assertGreater(len(post.transport.waypoints), 1)
        np.testing.assert_allclose(post.transport.waypoints[-1].xyz_base, [0.30, 0.10, 0.15])
        self.assertEqual(post.transport.waypoints[-1].gripper, 0.0)
        self.assertEqual(post.release.waypoints[0].name, "release")
        self.assertEqual(post.release.waypoints[0].gripper, 1.0)
        self.assertEqual(post.release.waypoints[-1].name, "retreat_after_release")

    def test_recovery_plan_uses_safe_pose_from_config(self):
        planner = HybridWaypointPlanner(self.config)

        recovery = planner.plan_recovery()

        self.assertEqual(recovery.phase, "recovery")
        np.testing.assert_allclose(
            recovery.waypoints[0].xyz_base,
            self.config["recovery"]["return_xyz_base"],
        )
        self.assertEqual(recovery.waypoints[0].gripper, 1.0)


if __name__ == "__main__":
    unittest.main()
