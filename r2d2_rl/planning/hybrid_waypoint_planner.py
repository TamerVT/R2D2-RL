"""Pure-numeric waypoint planning for the hybrid Project 3 controller.

The planner does not import RCS, MuJoCo, or LeRobot. It converts a target
block belief and bowl coordinates into conservative base-frame waypoints that
an environment-specific controller can execute later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from estimation.block_belief import BlockBelief


class PlanningError(ValueError):
    """Raised when a waypoint plan would be unsafe or underspecified."""


@dataclass(frozen=True)
class Waypoint:
    """One base-frame Cartesian command for the low-level controller.

    ``gripper`` follows the RCS convention used in this project: ``1.0`` is
    open and ``0.0`` is closed. Use ``None`` when the waypoint should leave the
    gripper command unchanged.
    """

    name: str
    xyz_base: np.ndarray
    rpy_base: np.ndarray
    gripper: float | None = None
    frame: str = "base"
    timeout_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz_base, dtype=np.float64).reshape(3)
        rpy = np.asarray(self.rpy_base, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz)):
            raise PlanningError(f"Waypoint '{self.name}' has non-finite xyz.")
        if not np.all(np.isfinite(rpy)):
            raise PlanningError(f"Waypoint '{self.name}' has non-finite rpy.")
        if self.gripper is not None and not 0.0 <= float(self.gripper) <= 1.0:
            raise PlanningError(f"Waypoint '{self.name}' gripper must be in [0, 1].")
        object.__setattr__(self, "xyz_base", xyz)
        object.__setattr__(self, "rpy_base", rpy)
        if self.gripper is not None:
            object.__setattr__(self, "gripper", float(self.gripper))


@dataclass(frozen=True)
class WaypointPlan:
    """Named waypoint sequence for one hybrid-controller phase."""

    phase: str
    waypoints: tuple[Waypoint, ...]

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise PlanningError(f"Waypoint plan '{self.phase}' is empty.")


@dataclass(frozen=True)
class PostGraspPlan:
    """Classical phases executed after the learned align/grasp policy."""

    lift: WaypointPlan
    transport: WaypointPlan
    release: WaypointPlan


class HybridWaypointPlanner:
    """Generate conservative Project 3 waypoints from config + beliefs."""

    def __init__(self, config: dict[str, Any]):
        planning = config.get("planning") or {}
        workspace = config.get("workspace") or {}
        control = config.get("control") or {}
        recovery = config.get("recovery") or {}

        self.z_pregrasp = float(planning.get("z_pregrasp", 0.10))
        self.z_grasp = float(planning.get("z_grasp", 0.025))
        self.z_lift = float(planning.get("z_lift", 0.15))
        self.pregrasp_xy_offset = _as_vector(planning.get("pregrasp_xy_offset", [0.0, 0.0]), 2)
        self.max_belief_std_for_grasp = float(planning.get("max_belief_std_for_grasp", 0.015))
        self.max_waypoint_step_m = float(planning.get("max_waypoint_step_m", 0.05))
        self.grasp_orientation_rpy = _as_vector(planning.get("grasp_orientation_rpy", [np.pi, 0.0, 0.0]), 3)
        self.timeout_s = float(control.get("waypoint_timeout_s", 5.0))
        self.table_collision_margin = float(control.get("table_collision_margin", 0.02))
        self.release_retreat_dz = float(planning.get("release_retreat_dz", 0.05))

        object_height = float(workspace.get("object_height", 0.04))
        if "z_object_center" in workspace:
            self.z_table = float(workspace["z_object_center"]) - object_height / 2.0
        else:
            plane = workspace.get("table_plane_base", {})
            normal = _as_vector(plane.get("normal", [0.0, 0.0, 1.0]), 3)
            if abs(normal[2]) < 1e-9:
                raise PlanningError("workspace.table_plane_base normal must have a non-zero z component.")
            self.z_table = -float(plane.get("d", 0.0)) / float(normal[2])

        bounds = workspace.get("bounds_xy") or {}
        self.workspace_x = _bounds_pair(bounds.get("x", [-np.inf, np.inf]), "workspace.bounds_xy.x")
        self.workspace_y = _bounds_pair(bounds.get("y", [-np.inf, np.inf]), "workspace.bounds_xy.y")
        self.return_xyz_base = _as_vector(recovery.get("return_xyz_base", [0.20, 0.0, 0.18]), 3)

        if self.max_belief_std_for_grasp <= 0:
            raise PlanningError("planning.max_belief_std_for_grasp must be positive.")
        if self.max_waypoint_step_m <= 0:
            raise PlanningError("planning.max_waypoint_step_m must be positive.")
        if self.timeout_s <= 0:
            raise PlanningError("control.waypoint_timeout_s must be positive.")
        for name, z in (("z_pregrasp", self.z_pregrasp), ("z_grasp", self.z_grasp), ("z_lift", self.z_lift)):
            self._validate_z(name, z)
        self._validate_xyz("recovery.return_xyz_base", self.return_xyz_base, check_xy=False)

    def plan_pregrasp(self, belief: BlockBelief) -> WaypointPlan:
        """Move above the target before handing off to ``align_grasp`` RL."""
        target_xy = self._validated_target_xy(belief)
        pregrasp_xy = target_xy + self.pregrasp_xy_offset
        self._validate_xy("pregrasp target", pregrasp_xy)
        waypoint = self._waypoint(
            name="pregrasp",
            xyz=[pregrasp_xy[0], pregrasp_xy[1], self.z_pregrasp],
            gripper=1.0,
            target_color=belief.color,
        )
        return WaypointPlan("approach", (waypoint,))

    def plan_post_grasp(self, belief: BlockBelief, bowl_xyz_base: np.ndarray) -> PostGraspPlan:
        """Plan classical lift, transport, release, and retreat waypoints."""
        target_xy = self._validated_target_xy(belief)
        bowl_xyz = _as_vector(bowl_xyz_base, 3)
        self._validate_xyz("bowl_xyz_base", bowl_xyz)

        lift = self._waypoint(
            name="lift",
            xyz=[target_xy[0], target_xy[1], self.z_lift],
            gripper=0.0,
            target_color=belief.color,
        )
        transport_z = max(self.z_lift, bowl_xyz[2] + self.release_retreat_dz)
        transport_waypoints = self._interpolate_waypoints(
            start_xyz=lift.xyz_base,
            end_xyz=np.array([bowl_xyz[0], bowl_xyz[1], transport_z], dtype=np.float64),
            name_prefix="transport",
            gripper=0.0,
            target_color=belief.color,
        )
        release = self._waypoint(
            name="release",
            xyz=bowl_xyz,
            gripper=1.0,
            target_color=belief.color,
        )
        retreat = self._waypoint(
            name="retreat_after_release",
            xyz=[bowl_xyz[0], bowl_xyz[1], transport_z],
            gripper=1.0,
            target_color=belief.color,
        )
        return PostGraspPlan(
            lift=WaypointPlan("lift", (lift,)),
            transport=WaypointPlan("transport", tuple(transport_waypoints)),
            release=WaypointPlan("release", (release, retreat)),
        )

    def plan_recovery(self) -> WaypointPlan:
        """Return to a safe observation pose before re-localizing."""
        waypoint = self._waypoint(
            name="return_to_safe_pose",
            xyz=self.return_xyz_base,
            gripper=1.0,
            target_color=None,
        )
        return WaypointPlan("recovery", (waypoint,))

    def _validated_target_xy(self, belief: BlockBelief) -> np.ndarray:
        if not belief.initialized:
            raise PlanningError(f"Belief for color '{belief.color}' is not initialized.")
        xy = np.asarray(belief.mean_xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(xy)):
            raise PlanningError(f"Belief for color '{belief.color}' has non-finite mean.")
        cov = np.asarray(belief.covariance_xy, dtype=np.float64).reshape(2, 2)
        if not np.all(np.isfinite(cov)):
            raise PlanningError(f"Belief for color '{belief.color}' has non-finite covariance.")
        cov = 0.5 * (cov + cov.T)
        eigvals = np.linalg.eigvalsh(cov)
        if eigvals[0] < -1e-10:
            raise PlanningError(f"Belief for color '{belief.color}' has a negative covariance eigenvalue.")
        max_std = float(np.sqrt(max(eigvals[-1], 0.0)))
        if max_std > self.max_belief_std_for_grasp:
            raise PlanningError(
                f"Belief for color '{belief.color}' is too uncertain for grasp: "
                f"std={max_std:.4f} m > {self.max_belief_std_for_grasp:.4f} m."
            )
        self._validate_xy(f"{belief.color} belief", xy)
        return xy

    def _waypoint(
        self,
        *,
        name: str,
        xyz: np.ndarray | list[float],
        gripper: float | None,
        target_color: str | None,
    ) -> Waypoint:
        xyz_arr = _as_vector(xyz, 3)
        self._validate_xyz(name, xyz_arr)
        metadata = {}
        if target_color is not None:
            metadata["target_color"] = target_color
        return Waypoint(
            name=name,
            xyz_base=xyz_arr,
            rpy_base=self.grasp_orientation_rpy.copy(),
            gripper=gripper,
            timeout_s=self.timeout_s,
            metadata=metadata,
        )

    def _interpolate_waypoints(
        self,
        *,
        start_xyz: np.ndarray,
        end_xyz: np.ndarray,
        name_prefix: str,
        gripper: float | None,
        target_color: str | None,
    ) -> list[Waypoint]:
        delta = end_xyz - start_xyz
        distance = float(np.linalg.norm(delta))
        steps = max(1, int(np.ceil(distance / self.max_waypoint_step_m)))
        waypoints = []
        for idx in range(1, steps + 1):
            alpha = idx / steps
            name = f"{name_prefix}_{idx:02d}" if steps > 1 else f"{name_prefix}_above_bowl"
            waypoints.append(
                self._waypoint(
                    name=name,
                    xyz=start_xyz + alpha * delta,
                    gripper=gripper,
                    target_color=target_color,
                )
            )
        return waypoints

    def _validate_xyz(self, label: str, xyz: np.ndarray, *, check_xy: bool = True) -> None:
        if not np.all(np.isfinite(xyz)):
            raise PlanningError(f"{label} must be finite.")
        if check_xy:
            self._validate_xy(label, xyz[:2])
        self._validate_z(label, float(xyz[2]))

    def _validate_xy(self, label: str, xy: np.ndarray) -> None:
        if not (
            self.workspace_x[0] <= xy[0] <= self.workspace_x[1]
            and self.workspace_y[0] <= xy[1] <= self.workspace_y[1]
        ):
            raise PlanningError(
                f"{label} xy={xy.tolist()} outside workspace "
                f"x={self.workspace_x}, y={self.workspace_y}."
            )

    def _validate_z(self, label: str, z: float) -> None:
        if not np.isfinite(z):
            raise PlanningError(f"{label} z must be finite.")
        min_z = self.z_table + self.table_collision_margin
        if z < min_z:
            raise PlanningError(f"{label} z={z:.4f} is below table collision margin {min_z:.4f}.")


def _as_vector(value: Any, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(size)
    if not np.all(np.isfinite(arr)):
        raise PlanningError(f"Expected a finite vector of length {size}.")
    return arr


def _bounds_pair(value: Any, label: str) -> tuple[float, float]:
    lo_hi = _as_vector(value, 2)
    lo, hi = float(lo_hi[0]), float(lo_hi[1])
    if lo > hi:
        raise PlanningError(f"{label} lower bound must be <= upper bound.")
    return lo, hi
