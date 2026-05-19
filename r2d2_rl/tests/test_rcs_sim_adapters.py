import unittest

import numpy as np

from r2d2_rl.estimation.pixel_to_table import PixelToTableProjector
from r2d2_rl.perception.color_block_detector import Detection2D
from r2d2_rl.runtime.rcs_sim_adapters import _RCS_PROJECTOR_CACHE, project_detection_to_table


def _downward_frame(width: int = 640, height: int = 480) -> tuple[dict, np.ndarray]:
    K = np.array(
        [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    T_B_C = np.eye(4)
    T_B_C[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T_B_C[:3, 3] = np.array([0.20, -0.03, 0.50])
    frame = {
        "intrinsics": np.column_stack([K, np.zeros(3)]),
        "extrinsics": np.linalg.inv(T_B_C),
        "data": np.zeros((height, width, 3), dtype=np.uint8),
    }
    return frame, T_B_C


class RcsSimAdaptersTest(unittest.TestCase):
    def setUp(self) -> None:
        _RCS_PROJECTOR_CACHE.clear()

    def _detection(self, uv=(320.0, 240.0)) -> Detection2D:
        return Detection2D(
            color="green",
            centroid_uv=np.array(uv, dtype=np.float64),
            area_px=1000.0,
            bbox_xywh=(300, 220, 40, 40),
            confidence=1.0,
            covariance_uv=np.eye(2) * 4.0,
        )

    def _config(self) -> dict:
        return {
            "workspace": {
                "z_object_center": 0.02,
                "bounds_xy": {"x": [0.0, 0.5], "y": [-0.2, 0.2]},
            },
            "uncertainty": {"calibration_sigma_xy": 0.0},
        }

    def test_project_detection_to_table_uses_rcs_extrinsics(self):
        frame, _ = _downward_frame()
        estimate = project_detection_to_table(self._detection(), frame, self._config())

        self.assertTrue(estimate.valid, msg=estimate.reason)
        np.testing.assert_allclose(estimate.xyz_base, [0.20, -0.03, 0.02], atol=1e-12)
        self.assertGreater(estimate.covariance_xy[0, 0], 0.0)
        self.assertGreater(estimate.covariance_xy[1, 1], 0.0)

    def test_matches_pixel_to_table_projector_directly(self):
        """Adapter and ``PixelToTableProjector.project_from_T_BC`` must agree."""
        frame, T_B_C = _downward_frame()
        config = self._config()
        detection = self._detection(uv=(330.0, 260.0))

        adapter_estimate = project_detection_to_table(detection, frame, config)

        projector = PixelToTableProjector(
            {
                "camera": {"K": frame["intrinsics"][:3, :3].tolist(), "width": 640, "height": 480},
                "workspace": config["workspace"],
                "uncertainty": config["uncertainty"],
            }
        )
        direct_estimate = projector.project_from_T_BC(
            uv=detection.centroid_uv,
            T_B_C=T_B_C,
            covariance_uv=detection.covariance_uv,
        )

        self.assertTrue(adapter_estimate.valid)
        self.assertTrue(direct_estimate.valid)
        np.testing.assert_allclose(adapter_estimate.xy_base, direct_estimate.xy_base, atol=1e-12)
        np.testing.assert_allclose(adapter_estimate.covariance_xy, direct_estimate.covariance_xy, atol=1e-12)

    def test_outside_workspace_marked_invalid(self):
        frame, _ = _downward_frame()
        config = self._config()
        config["workspace"]["bounds_xy"] = {"x": [0.30, 0.40], "y": [-0.10, 0.10]}
        estimate = project_detection_to_table(self._detection(), frame, config)
        self.assertFalse(estimate.valid)
        self.assertIn("workspace", estimate.reason)


if __name__ == "__main__":
    unittest.main()
