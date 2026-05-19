"""Synthetic geometry tests for ``estimation.pixel_to_table``.

The tests do not depend on MuJoCo. Each test builds a known camera intrinsic,
end-effector pose, and end-effector-to-camera transform, picks a true table
point in the base frame, projects it analytically into pixel coordinates, then
feeds that pixel back into ``PixelToTableProjector`` and asserts the projector
recovers the original table point.
"""

import math
import unittest

import numpy as np

from r2d2_rl.estimation.pixel_to_table import (
    PixelToTableProjector,
    quat_xyzw_to_rotation,
    transform_from_translation_quat,
)


def _world_point_to_pixel(
    point_base: np.ndarray,
    K: np.ndarray,
    T_B_E: np.ndarray,
    T_E_C: np.ndarray,
) -> np.ndarray:
    """Forward-project a base-frame 3D point into pixel coords."""
    T_B_C = T_B_E @ T_E_C
    R_B_C = T_B_C[:3, :3]
    t_B_C = T_B_C[:3, 3]
    point_cam = R_B_C.T @ (point_base - t_B_C)
    if point_cam[2] <= 0:
        raise ValueError("Point is behind the camera under the supplied poses.")
    pixel_h = K @ point_cam
    return pixel_h[:2] / pixel_h[2]


def _world_point_to_distorted_pixel(
    point_base: np.ndarray,
    K: np.ndarray,
    T_B_E: np.ndarray,
    T_E_C: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    T_B_C = T_B_E @ T_E_C
    R_B_C = T_B_C[:3, :3]
    t_B_C = T_B_C[:3, 3]
    point_cam = R_B_C.T @ (point_base - t_B_C)
    x = point_cam[0] / point_cam[2]
    y = point_cam[1] / point_cam[2]
    k1, k2, p1, p2, k3 = dist_coeffs
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    pixel_h = K @ np.array([x_d, y_d, 1.0])
    return pixel_h[:2] / pixel_h[2]


def _downward_looking_config(width: int = 640, height: int = 480) -> dict:
    """Camera looking straight down at the table from above the EE."""
    fx, fy, cx, cy = 600.0, 600.0, width / 2.0, height / 2.0
    return {
        "camera": {
            "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "width": width,
            "height": height,
        },
        "transforms": {
            "T_E_C": {
                "translation": [0.0, 0.0, 0.0],
                "quaternion_xyzw": [1.0, 0.0, 0.0, 0.0],
            }
        },
        "workspace": {
            "z_object_center": 0.02,
            "bounds_xy": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
        },
        "uncertainty": {
            "calibration_sigma_xy": 0.0,
            "pixel_sigma_base": 1.0,
        },
    }


def _ee_pose_above(table_z: float, height: float, xy_base: np.ndarray) -> np.ndarray:
    """End-effector pose centered at ``xy_base`` and ``table_z + height``.

    The EE rotation is identity in base frame; combined with the test's
    ``T_E_C`` rotation of 180 deg about X, the camera's optical axis points
    along ``-z_base``, so the wrist camera looks straight down at the table.
    """
    T = np.eye(4)
    T[:3, 3] = [xy_base[0], xy_base[1], table_z + height]
    return T


class PixelToTableProjectionTest(unittest.TestCase):
    def test_roundtrip_directly_above(self):
        cfg = _downward_looking_config()
        projector = PixelToTableProjector(cfg)
        true_xy = np.array([0.08, -0.05])
        true_point = np.array([true_xy[0], true_xy[1], cfg["workspace"]["z_object_center"]])
        T_B_E = _ee_pose_above(cfg["workspace"]["z_object_center"], 0.40, np.array([0.0, 0.0]))

        K = np.asarray(cfg["camera"]["K"], dtype=np.float64)
        T_E_C = transform_from_translation_quat(
            cfg["transforms"]["T_E_C"]["translation"],
            cfg["transforms"]["T_E_C"]["quaternion_xyzw"],
        )
        uv = _world_point_to_pixel(true_point, K, T_B_E, T_E_C)

        est = projector.project_from_T_BE(uv, T_B_E)

        self.assertTrue(est.valid, msg=est.reason)
        np.testing.assert_allclose(est.xy_base, true_xy, atol=1e-9)
        np.testing.assert_allclose(est.xyz_base[2], cfg["workspace"]["z_object_center"])

    def test_roundtrip_oblique_view(self):
        cfg = _downward_looking_config()
        projector = PixelToTableProjector(cfg)
        true_xy = np.array([0.12, 0.04])
        true_point = np.array([true_xy[0], true_xy[1], cfg["workspace"]["z_object_center"]])

        T_B_E = _ee_pose_above(cfg["workspace"]["z_object_center"], 0.35, np.array([0.06, -0.02]))

        K = np.asarray(cfg["camera"]["K"], dtype=np.float64)
        T_E_C = transform_from_translation_quat(
            cfg["transforms"]["T_E_C"]["translation"],
            cfg["transforms"]["T_E_C"]["quaternion_xyzw"],
        )
        uv = _world_point_to_pixel(true_point, K, T_B_E, T_E_C)

        est = projector.project_from_T_BE(uv, T_B_E)

        self.assertTrue(est.valid, msg=est.reason)
        np.testing.assert_allclose(est.xy_base, true_xy, atol=1e-9)

    def test_pixel_outside_image_rejected(self):
        cfg = _downward_looking_config(width=320, height=240)
        projector = PixelToTableProjector(cfg)
        T_B_E = _ee_pose_above(cfg["workspace"]["z_object_center"], 0.20, np.array([0.0, 0.0]))

        est = projector.project_from_T_BE(np.array([-1.0, 5.0]), T_B_E)

        self.assertFalse(est.valid)
        self.assertIn("image bounds", est.reason)

    def test_subpixel_inside_last_pixel_is_accepted(self):
        cfg = _downward_looking_config(width=320, height=240)
        projector = PixelToTableProjector(cfg)
        T_B_E = _ee_pose_above(cfg["workspace"]["z_object_center"], 0.20, np.array([0.0, 0.0]))

        est = projector.project_from_T_BE(np.array([319.75, 239.75]), T_B_E)

        self.assertTrue(est.valid, msg=est.reason)

    def test_outside_workspace_marked_invalid_but_geometry_kept(self):
        cfg = _downward_looking_config()
        cfg["workspace"]["bounds_xy"] = {"x": [0.0, 0.05], "y": [-0.02, 0.02]}
        projector = PixelToTableProjector(cfg)
        T_B_E = _ee_pose_above(cfg["workspace"]["z_object_center"], 0.40, np.array([0.0, 0.0]))

        true_xy = np.array([0.08, 0.05])
        true_point = np.array([true_xy[0], true_xy[1], cfg["workspace"]["z_object_center"]])
        K = np.asarray(cfg["camera"]["K"], dtype=np.float64)
        T_E_C = transform_from_translation_quat(
            cfg["transforms"]["T_E_C"]["translation"],
            cfg["transforms"]["T_E_C"]["quaternion_xyzw"],
        )
        uv = _world_point_to_pixel(true_point, K, T_B_E, T_E_C)

        est = projector.project_from_T_BE(uv, T_B_E)

        self.assertFalse(est.valid)
        self.assertIn("workspace", est.reason)
        np.testing.assert_allclose(est.xy_base, true_xy, atol=1e-6)

    def test_grazing_ray_rejected(self):
        cfg = _downward_looking_config()
        projector = PixelToTableProjector(cfg)

        cam_pitch = math.pi / 2 - 1e-6
        s, c = math.sin(cam_pitch / 2.0), math.cos(cam_pitch / 2.0)
        quat = [s, 0.0, 0.0, c]
        T_B_E = transform_from_translation_quat(
            translation=[0.10, 0.0, cfg["workspace"]["z_object_center"] + 0.20],
            quat_xyzw=quat,
        )

        est = projector.project_from_T_BE(np.array([320.0, 240.0]), T_B_E)
        self.assertFalse(est.valid)

    def test_table_plane_d_uses_signed_plane_equation(self):
        cfg = _downward_looking_config()
        del cfg["workspace"]["z_object_center"]
        cfg["workspace"]["table_plane_base"] = {"normal": [0.0, 0.0, 1.0], "d": -0.10}
        cfg["workspace"]["object_height"] = 0.04
        projector = PixelToTableProjector(cfg)
        z_obj = 0.12

        T_B_E = _ee_pose_above(z_obj, 0.30, np.array([0.0, 0.0]))
        est = projector.project_from_T_BE(np.array([320.0, 240.0]), T_B_E)

        self.assertTrue(est.valid, msg=est.reason)
        np.testing.assert_allclose(est.xyz_base[2], z_obj, atol=1e-12)

    def test_distorted_pixels_are_undistorted_before_projection(self):
        cfg = _downward_looking_config()
        dist_coeffs = np.array([0.2, -0.04, 0.001, -0.001, 0.0], dtype=np.float64)
        cfg["camera"]["dist_coeffs"] = dist_coeffs.tolist()
        projector = PixelToTableProjector(cfg)

        true_xy = np.array([0.05, -0.04])
        true_point = np.array([true_xy[0], true_xy[1], cfg["workspace"]["z_object_center"]])
        T_B_E = _ee_pose_above(cfg["workspace"]["z_object_center"], 0.35, np.array([0.0, 0.0]))
        K = np.asarray(cfg["camera"]["K"], dtype=np.float64)
        T_E_C = transform_from_translation_quat(
            cfg["transforms"]["T_E_C"]["translation"],
            cfg["transforms"]["T_E_C"]["quaternion_xyzw"],
        )
        uv = _world_point_to_distorted_pixel(true_point, K, T_B_E, T_E_C, dist_coeffs)

        est = projector.project_from_T_BE(uv, T_B_E)

        self.assertTrue(est.valid, msg=est.reason)
        np.testing.assert_allclose(est.xy_base, true_xy, atol=1e-6)

    def test_covariance_isotropic_directly_above(self):
        cfg = _downward_looking_config()
        cfg["uncertainty"]["calibration_sigma_xy"] = 0.0
        projector = PixelToTableProjector(cfg)
        T_B_E = _ee_pose_above(cfg["workspace"]["z_object_center"], 0.18, np.array([0.0, 0.0]))

        K = np.asarray(cfg["camera"]["K"], dtype=np.float64)
        T_E_C = transform_from_translation_quat(
            cfg["transforms"]["T_E_C"]["translation"],
            cfg["transforms"]["T_E_C"]["quaternion_xyzw"],
        )
        uv = _world_point_to_pixel(
            np.array([0.0, 0.0, cfg["workspace"]["z_object_center"]]), K, T_B_E, T_E_C
        )

        cov_uv = np.diag([4.0, 4.0])
        est = projector.project_from_T_BE(uv, T_B_E, covariance_uv=cov_uv)

        self.assertTrue(est.valid, msg=est.reason)
        cov = est.covariance_xy
        self.assertGreater(cov[0, 0], 0)
        self.assertGreater(cov[1, 1], 0)
        self.assertAlmostEqual(cov[0, 0], cov[1, 1], places=10)
        self.assertAlmostEqual(cov[0, 1], 0.0, places=10)
        self.assertAlmostEqual(cov[1, 0], 0.0, places=10)

    def test_covariance_grows_with_height(self):
        cfg = _downward_looking_config()
        cfg["uncertainty"]["calibration_sigma_xy"] = 0.0
        projector = PixelToTableProjector(cfg)
        cov_uv = np.diag([9.0, 9.0])

        K = np.asarray(cfg["camera"]["K"], dtype=np.float64)
        T_E_C = transform_from_translation_quat(
            cfg["transforms"]["T_E_C"]["translation"],
            cfg["transforms"]["T_E_C"]["quaternion_xyzw"],
        )
        z_obj = cfg["workspace"]["z_object_center"]

        cov_traces = []
        for height in [0.10, 0.20, 0.40]:
            T_B_E = _ee_pose_above(z_obj, height, np.array([0.0, 0.0]))
            uv = _world_point_to_pixel(np.array([0.0, 0.0, z_obj]), K, T_B_E, T_E_C)
            est = projector.project_from_T_BE(uv, T_B_E, covariance_uv=cov_uv)
            self.assertTrue(est.valid, msg=est.reason)
            cov_traces.append(np.trace(est.covariance_xy))

        self.assertLess(cov_traces[0], cov_traces[1])
        self.assertLess(cov_traces[1], cov_traces[2])

    def test_kinematics_path_matches_direct_path(self):
        cfg = _downward_looking_config()
        T_B_E = _ee_pose_above(cfg["workspace"]["z_object_center"], 0.30, np.array([0.02, 0.0]))

        class _StubKin:
            def forward(self, q):
                return T_B_E

        projector = PixelToTableProjector(cfg, kinematics=_StubKin())

        K = np.asarray(cfg["camera"]["K"], dtype=np.float64)
        T_E_C = transform_from_translation_quat(
            cfg["transforms"]["T_E_C"]["translation"],
            cfg["transforms"]["T_E_C"]["quaternion_xyzw"],
        )
        true_point = np.array([0.06, -0.04, cfg["workspace"]["z_object_center"]])
        uv = _world_point_to_pixel(true_point, K, T_B_E, T_E_C)

        est_direct = projector.project_from_T_BE(uv, T_B_E)
        est_kin = projector.project(uv, q=np.zeros(6))

        self.assertTrue(est_direct.valid)
        self.assertTrue(est_kin.valid)
        np.testing.assert_allclose(est_kin.xy_base, est_direct.xy_base, atol=1e-12)

    def test_kinematics_path_accepts_pose_like_object(self):
        cfg = _downward_looking_config()
        T_B_E = _ee_pose_above(cfg["workspace"]["z_object_center"], 0.30, np.array([0.02, 0.0]))

        class _PoseLike:
            def pose_matrix(self):
                return T_B_E

        class _StubKin:
            def forward(self, q):
                return _PoseLike()

        projector = PixelToTableProjector(cfg, kinematics=_StubKin())
        est = projector.project(np.array([320.0, 240.0]), q=np.zeros(6))

        self.assertTrue(est.valid, msg=est.reason)


if __name__ == "__main__":
    unittest.main()
