"""Pixel-to-table-plane projection for the wrist camera.

Given a pixel `(u, v)` detected in the wrist-camera image and a current
end-effector pose `T_B_E`, this module casts a ray through the calibrated
pinhole camera, intersects it with the configured table plane in the robot
base frame, and returns the table point in robot-frame coordinates with an
optional covariance estimate.

Frames follow the project convention:

- ``B``: robot base frame
- ``E``: end-effector frame
- ``C``: wrist camera frame (rigidly attached to ``E`` via ``T_E_C``)
- ``I``: image pixel coordinates
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class TablePointEstimate:
    """Result of projecting a single pixel onto the table plane.

    ``valid`` is ``False`` whenever the ray-plane intersection cannot be trusted
    (grazing ray, outside workspace, NaN, etc.). The geometric fields are still
    populated when possible so callers can inspect failure modes.
    """

    xy_base: np.ndarray
    xyz_base: np.ndarray
    covariance_xy: np.ndarray
    valid: bool
    reason: str = ""


class Kinematics(Protocol):
    """Minimal kinematics protocol: ``q`` -> pose-like ``T_B_E``.

    ``forward`` may return a 4x4 ndarray or an object with ``pose_matrix()``,
    such as ``rcs.common.Pin.forward``.
    """

    def forward(self, q: np.ndarray) -> Any: ...


def quat_xyzw_to_rotation(q_xyzw: np.ndarray) -> np.ndarray:
    """Return a 3x3 rotation matrix from a unit quaternion in ``[x, y, z, w]``."""
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-9:
        raise ValueError("Quaternion must be non-zero and finite.")
    x, y, z, w = q / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def transform_from_translation_quat(translation: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from translation + xyzw quaternion."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rotation(quat_xyzw)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


class PixelToTableProjector:
    """Project image pixels to a fixed-height table plane in the base frame.

    The projector expects a config that follows ``calibration.yaml``: it reads
    intrinsics from ``camera.K``, the end-effector-to-camera transform from
    ``transforms.T_E_C`` (translation + xyzw quaternion), the table height
    from ``workspace.z_object_center``, and optional uncertainty values from
    ``uncertainty.calibration_sigma_xy`` (sigma applied isotropically to xy).

    Two entry points are supported:

    - ``project_from_T_BE(uv, T_B_E, ...)``: pure-numeric, what tests use.
    - ``project(uv, q, ...)``: convenience wrapper that uses ``kinematics.forward(q)``.
    """

    def __init__(self, config: dict[str, Any], kinematics: Kinematics | None = None):
        cam_cfg = config.get("camera") or {}
        K = np.asarray(cam_cfg.get("K"), dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError("camera.K must be a 3x3 matrix in the config.")
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.image_width = int(cam_cfg.get("width", 0))
        self.image_height = int(cam_cfg.get("height", 0))
        self.dist_coeffs = np.asarray(cam_cfg.get("dist_coeffs", []), dtype=np.float64).reshape(-1)

        transforms = config.get("transforms") or {}
        t_e_c = transforms.get("T_E_C")
        if t_e_c is None:
            # T_E_C is only needed for the kinematics-based ``project()`` /
            # ``project_from_T_BE()`` entry points. RCS callers that hand us
            # T_B_C directly (via ``project_from_T_BC``) can omit it.
            self.T_E_C = None
        else:
            translation = t_e_c.get("translation")
            quat = t_e_c.get("quaternion_xyzw")
            if translation is None or quat is None:
                raise ValueError("T_E_C must define both translation and quaternion_xyzw.")
            self.T_E_C = transform_from_translation_quat(translation, quat)

        workspace = config.get("workspace") or {}
        if "z_object_center" in workspace:
            self.z_object = float(workspace["z_object_center"])
        else:
            plane = workspace.get("table_plane_base", {})
            normal = np.asarray(plane.get("normal", [0.0, 0.0, 1.0]), dtype=np.float64).reshape(3)
            if not np.allclose(normal[:2], 0.0) or abs(normal[2]) < 1e-9:
                raise ValueError("Only z-aligned table_plane_base is supported without z_object_center.")
            self.z_object = -float(plane.get("d", 0.0)) / float(normal[2]) + (
                float(workspace.get("object_height", 0.0)) / 2.0
            )

        bounds = workspace.get("bounds_xy") or {}
        x_lo, x_hi = bounds.get("x", [-np.inf, np.inf])
        y_lo, y_hi = bounds.get("y", [-np.inf, np.inf])
        self.workspace_x = (float(x_lo), float(x_hi))
        self.workspace_y = (float(y_lo), float(y_hi))

        uncertainty = config.get("uncertainty") or {}
        sigma_xy = float(uncertainty.get("calibration_sigma_xy", 0.0))
        self.calibration_cov = np.eye(2, dtype=np.float64) * (sigma_xy**2)
        self.pixel_sigma_base = float(uncertainty.get("pixel_sigma_base", 3.0))

        self.parallel_ray_tol = float(uncertainty.get("parallel_ray_tol", 1e-3))
        self.fd_step_px = float(uncertainty.get("fd_step_px", 0.5))

        self.kinematics = kinematics

    def project(
        self,
        uv: np.ndarray,
        q: np.ndarray,
        covariance_uv: np.ndarray | None = None,
    ) -> TablePointEstimate:
        if self.kinematics is None:
            return _invalid_estimate("kinematics not provided to projector")
        T_B_E = self.kinematics.forward(np.asarray(q, dtype=np.float64))
        return self.project_from_T_BE(uv, T_B_E, covariance_uv=covariance_uv)

    def project_from_T_BE(
        self,
        uv: np.ndarray,
        T_B_E: np.ndarray,
        covariance_uv: np.ndarray | None = None,
    ) -> TablePointEstimate:
        uv = np.asarray(uv, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(uv)):
            return _invalid_estimate("pixel coordinates are not finite")

        if self.image_width > 0 and self.image_height > 0:
            u, v = float(uv[0]), float(uv[1])
            if not (0.0 <= u < self.image_width and 0.0 <= v < self.image_height):
                return _invalid_estimate("pixel outside image bounds")

        if self.T_E_C is None:
            return _invalid_estimate(
                "T_E_C not configured; pass T_B_C directly via project_from_T_BC instead."
            )
        T_B_E = _pose_like_to_matrix(T_B_E)
        if not np.all(np.isfinite(T_B_E)):
            return _invalid_estimate("end-effector pose is not finite")

        T_B_C = T_B_E @ self.T_E_C
        return self._project_from_T_BC_inner(uv, T_B_C, covariance_uv)

    def project_from_T_BC(
        self,
        uv: np.ndarray,
        T_B_C: np.ndarray,
        covariance_uv: np.ndarray | None = None,
    ) -> TablePointEstimate:
        """Project a pixel through a known camera-to-base transform.

        Used by RCS-backed callers that already have ``T_B_C`` from
        ``inv(obs['frames'][cam]['rgb']['extrinsics'])`` and therefore do not
        need to compose ``T_B_E @ T_E_C`` from kinematics + calibration.
        """
        uv = np.asarray(uv, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(uv)):
            return _invalid_estimate("pixel coordinates are not finite")

        if self.image_width > 0 and self.image_height > 0:
            u, v = float(uv[0]), float(uv[1])
            if not (0.0 <= u < self.image_width and 0.0 <= v < self.image_height):
                return _invalid_estimate("pixel outside image bounds")

        T_B_C = np.asarray(T_B_C, dtype=np.float64).reshape(4, 4)
        if not np.all(np.isfinite(T_B_C)):
            return _invalid_estimate("camera-to-base pose is not finite")
        return self._project_from_T_BC_inner(uv, T_B_C, covariance_uv)

    def _project_from_T_BC_inner(
        self,
        uv: np.ndarray,
        T_B_C: np.ndarray,
        covariance_uv: np.ndarray | None,
    ) -> TablePointEstimate:
        R_B_C = T_B_C[:3, :3]
        o_B = T_B_C[:3, 3]

        xyz, r_B = self._intersect_ray(uv, R_B_C, o_B)
        if xyz is None:
            return _invalid_estimate("ray nearly parallel to table plane")

        if not np.all(np.isfinite(xyz)):
            return _invalid_estimate("intersection contains NaN or Inf")

        cov_xy = self._propagate_covariance(uv, R_B_C, o_B, covariance_uv)

        in_workspace = (
            self.workspace_x[0] <= xyz[0] <= self.workspace_x[1]
            and self.workspace_y[0] <= xyz[1] <= self.workspace_y[1]
        )
        if not in_workspace:
            return TablePointEstimate(
                xy_base=xyz[:2].copy(),
                xyz_base=xyz.copy(),
                covariance_xy=cov_xy,
                valid=False,
                reason="intersection outside workspace bounds",
            )

        return TablePointEstimate(
            xy_base=xyz[:2].copy(),
            xyz_base=xyz.copy(),
            covariance_xy=cov_xy,
            valid=True,
            reason="",
        )

    def _intersect_ray(
        self,
        uv: np.ndarray,
        R_B_C: np.ndarray,
        o_B: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        r_C = self._pixel_to_camera_ray(uv)
        r_B = R_B_C @ r_C
        denom = r_B[2]
        if not np.isfinite(denom) or abs(denom) < self.parallel_ray_tol:
            return None, r_B
        lam = (self.z_object - o_B[2]) / denom
        if lam <= 0:
            return None, r_B
        xyz = o_B + lam * r_B
        return xyz, r_B

    def _propagate_covariance(
        self,
        uv: np.ndarray,
        R_B_C: np.ndarray,
        o_B: np.ndarray,
        covariance_uv: np.ndarray | None,
    ) -> np.ndarray:
        if covariance_uv is None:
            sigma = self.pixel_sigma_base
            cov_uv = np.eye(2, dtype=np.float64) * (sigma**2)
        else:
            cov_uv = np.asarray(covariance_uv, dtype=np.float64).reshape(2, 2)

        J = np.zeros((2, 2), dtype=np.float64)
        h = self.fd_step_px
        for i in range(2):
            delta = np.zeros(2)
            delta[i] = h
            xyz_plus, _ = self._intersect_ray(uv + delta, R_B_C, o_B)
            xyz_minus, _ = self._intersect_ray(uv - delta, R_B_C, o_B)
            if xyz_plus is None or xyz_minus is None:
                return np.full((2, 2), np.nan)
            J[:, i] = (xyz_plus[:2] - xyz_minus[:2]) / (2.0 * h)

        return J @ cov_uv @ J.T + self.calibration_cov

    def _pixel_to_camera_ray(self, uv: np.ndarray) -> np.ndarray:
        if self.dist_coeffs.size == 0 or np.allclose(self.dist_coeffs, 0.0):
            pixel_h = np.array([uv[0], uv[1], 1.0], dtype=np.float64)
            return self.K_inv @ pixel_h

        try:
            import cv2
        except ImportError as exc:
            raise ImportError("OpenCV is required when camera.dist_coeffs are non-zero.") from exc

        points = np.asarray(uv, dtype=np.float64).reshape(1, 1, 2)
        undistorted = cv2.undistortPoints(points, self.K, self.dist_coeffs).reshape(2)
        return np.array([undistorted[0], undistorted[1], 1.0], dtype=np.float64)


def _pose_like_to_matrix(pose_like: Any) -> np.ndarray:
    if hasattr(pose_like, "pose_matrix"):
        pose_like = pose_like.pose_matrix()
    matrix = np.asarray(pose_like, dtype=np.float64)
    if matrix.shape == (16,):
        matrix = matrix.reshape(4, 4)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 pose matrix, got shape {matrix.shape}.")
    return matrix


def _invalid_estimate(reason: str) -> TablePointEstimate:
    nan_xy = np.full(2, np.nan)
    nan_xyz = np.full(3, np.nan)
    return TablePointEstimate(
        xy_base=nan_xy,
        xyz_base=nan_xyz,
        covariance_xy=np.full((2, 2), np.nan),
        valid=False,
        reason=reason,
    )
