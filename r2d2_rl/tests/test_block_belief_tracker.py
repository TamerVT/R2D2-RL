"""Unit tests for ``estimation.block_belief.BlockBeliefTracker``.

Each test feeds synthetic measurements with known covariance and checks the
filter's behavior. None of the tests depend on MuJoCo or OpenCV.
"""

import unittest

import numpy as np

from r2d2_rl.estimation.block_belief import BlockBeliefTracker


def _config(**overrides) -> dict:
    cfg = {
        "estimation": {
            "process_noise_xy": 1e-5,
            "process_noise_contact_xy": 1e-2,
            "initial_covariance_xy": 1e-3,
            "contact_decay_s": 1.0,
        }
    }
    cfg["estimation"].update(overrides)
    return cfg


class BlockBeliefTrackerTest(unittest.TestCase):
    def test_repeated_measurements_reduce_covariance(self):
        tracker = BlockBeliefTracker(_config())
        meas_cov = np.eye(2) * 0.01
        first = tracker.update("red", np.array([0.1, 0.0]), meas_cov, timestamp=0.0)

        last_trace = np.trace(first.covariance_xy)
        for k in range(1, 6):
            updated = tracker.update("red", np.array([0.1, 0.0]), meas_cov, timestamp=float(k))
            self.assertLess(np.trace(updated.covariance_xy), last_trace)
            last_trace = np.trace(updated.covariance_xy)

    def test_initial_covariance_setting_is_applied(self):
        tracker = BlockBeliefTracker(_config(initial_covariance_xy=0.02))
        meas_cov = np.eye(2) * 0.01

        belief = tracker.update("red", np.array([0.1, 0.0]), meas_cov, timestamp=0.0)

        np.testing.assert_allclose(belief.covariance_xy, np.eye(2) * 0.03, atol=1e-12)

    def test_high_noise_measurements_weighted_less(self):
        tracker_low = BlockBeliefTracker(_config())
        tracker_high = BlockBeliefTracker(_config())

        prior = np.array([0.20, 0.00])
        tracker_low.update("red", prior, np.eye(2) * 0.0001, timestamp=0.0)
        tracker_high.update("red", prior, np.eye(2) * 0.0001, timestamp=0.0)

        noisy_meas = np.array([0.40, 0.00])
        belief_low = tracker_low.update("red", noisy_meas, np.eye(2) * 0.0001, timestamp=1.0)
        belief_high = tracker_high.update("red", noisy_meas, np.eye(2) * 1.0, timestamp=1.0)

        # The high-variance measurement should move the mean far less.
        shift_low = abs(belief_low.mean_xy[0] - prior[0])
        shift_high = abs(belief_high.mean_xy[0] - prior[0])
        self.assertGreater(shift_low, shift_high)
        self.assertLess(shift_high, 0.01)

    def test_predict_increases_covariance(self):
        tracker = BlockBeliefTracker(_config(process_noise_xy=1e-3))
        tracker.update("red", np.array([0.1, 0.0]), np.eye(2) * 1e-4, timestamp=0.0)
        before = tracker.get("red")
        tracker.predict(dt=2.0)
        after = tracker.get("red")
        self.assertGreater(np.trace(after.covariance_xy), np.trace(before.covariance_xy))

    def test_contact_inflates_process_noise(self):
        tracker_quiet = BlockBeliefTracker(_config())
        tracker_quiet.update("red", np.array([0.1, 0.0]), np.eye(2) * 1e-4, timestamp=0.0)
        tracker_quiet.predict(dt=0.1)
        after_quiet = tracker_quiet.get("red")

        tracker_contact = BlockBeliefTracker(_config())
        tracker_contact.update("red", np.array([0.1, 0.0]), np.eye(2) * 1e-4, timestamp=0.0)
        tracker_contact.mark_contact("red")
        tracker_contact.predict(dt=0.1)
        after_contact = tracker_contact.get("red")

        self.assertGreater(
            np.trace(after_contact.covariance_xy),
            np.trace(after_quiet.covariance_xy),
        )

    def test_contact_decays(self):
        tracker = BlockBeliefTracker(_config(contact_decay_s=0.2))
        tracker.update("red", np.array([0.1, 0.0]), np.eye(2) * 1e-4, timestamp=0.0)
        tracker.mark_contact("red")
        tracker.predict(dt=0.5)  # exceeds decay window
        trace_after_decay = np.trace(tracker.get("red").covariance_xy)
        tracker.predict(dt=0.1)
        trace_after_more = np.trace(tracker.get("red").covariance_xy)
        # second predict should add only nominal Q*dt = small change
        self.assertLess(trace_after_more - trace_after_decay, 1e-4)

    def test_per_color_isolation(self):
        tracker = BlockBeliefTracker(_config())
        tracker.update("red", np.array([0.10, 0.00]), np.eye(2) * 1e-4, timestamp=0.0)
        tracker.update("blue", np.array([0.20, 0.05]), np.eye(2) * 1e-4, timestamp=0.0)

        red = tracker.get("red")
        blue = tracker.get("blue")
        self.assertIsNotNone(red)
        self.assertIsNotNone(blue)
        np.testing.assert_allclose(red.mean_xy, [0.10, 0.00], atol=1e-9)
        np.testing.assert_allclose(blue.mean_xy, [0.20, 0.05], atol=1e-9)

    def test_get_unknown_color_returns_none(self):
        tracker = BlockBeliefTracker(_config())
        self.assertIsNone(tracker.get("green"))

    def test_reset_clears_state(self):
        tracker = BlockBeliefTracker(_config())
        tracker.update("red", np.array([0.1, 0.0]), np.eye(2) * 1e-4, timestamp=0.0)
        tracker.reset()
        self.assertEqual(tracker.get_all(), {})
        self.assertIsNone(tracker.get("red"))

    def test_singular_covariances_stay_finite_and_psd(self):
        tracker = BlockBeliefTracker(_config(initial_covariance_xy=0.0))
        tracker.update("red", np.array([0.1, 0.0]), np.zeros((2, 2)), timestamp=0.0)

        belief = tracker.update("red", np.array([0.1, 0.0]), np.zeros((2, 2)), timestamp=1.0)

        self.assertTrue(np.all(np.isfinite(belief.covariance_xy)))
        self.assertGreaterEqual(np.linalg.eigvalsh(belief.covariance_xy)[0], 0.0)

    def test_negative_covariance_is_rejected(self):
        tracker = BlockBeliefTracker(_config())

        with self.assertRaises(ValueError):
            tracker.update("red", np.array([0.1, 0.0]), -np.eye(2), timestamp=0.0)

    def test_negative_noise_config_is_rejected(self):
        with self.assertRaises(ValueError):
            BlockBeliefTracker(_config(process_noise_xy=-1e-3))


if __name__ == "__main__":
    unittest.main()
