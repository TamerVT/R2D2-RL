"""Finite-state task executor for the Project 3 hybrid controller.

This module intentionally depends only on pure project dataclasses/protocols.
The RCS-specific controller, camera, and policy adapters can be plugged in
behind these protocols without changing the high-level state machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence

import numpy as np

from estimation.block_belief import BlockBelief
from planning.hybrid_waypoint_planner import (
    HybridWaypointPlanner,
    PlanningError,
    Waypoint,
    WaypointPlan,
)


class HybridTaskState(str, Enum):
    IDLE = "idle"
    OBSERVE = "observe"
    APPROACH = "approach"
    ALIGN_GRASP = "align_grasp"
    LIFT = "lift"
    TRANSPORT = "transport"
    RELEASE = "release"
    RECOVERY = "recovery"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class TaskGoal:
    """One Project 3 target-color to bowl placement goal."""

    target_color: str
    bowl_xyz_base: np.ndarray

    def __post_init__(self) -> None:
        xyz = np.asarray(self.bowl_xyz_base, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz)):
            raise ValueError("bowl_xyz_base must be finite.")
        object.__setattr__(self, "bowl_xyz_base", xyz)


@dataclass(frozen=True)
class TraceEvent:
    state: HybridTaskState
    detail: str = ""
    attempt: int = 0


@dataclass(frozen=True)
class ExecutorResult:
    success: bool
    final_state: HybridTaskState
    attempts: int
    trace: tuple[TraceEvent, ...]
    failure_reason: str = ""


class BeliefObserver(Protocol):
    def observe(self, target_color: str) -> BlockBelief | None: ...


class WaypointController(Protocol):
    def execute(self, waypoints: Sequence[Waypoint]) -> bool: ...


class LocalPolicy(Protocol):
    def run(self, phase: str, target_color: str, belief: BlockBelief) -> bool: ...


class VisibilityChecker(Protocol):
    def is_visible(self, target_color: str) -> bool: ...


@dataclass
class HybridTaskExecutor:
    """Run the observe/approach/align-grasp/classical-transport loop."""

    config: dict[str, Any]
    planner: HybridWaypointPlanner
    observer: BeliefObserver
    controller: WaypointController
    local_policy: LocalPolicy
    visibility_checker: VisibilityChecker | None = None
    trace: list[TraceEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        phases = list((self.config.get("rl") or {}).get("phases", ["align_grasp"]))
        unsupported = [phase for phase in phases if phase != "align_grasp"]
        if unsupported:
            raise ValueError(
                "Only the simplified align_grasp RL phase is supported; "
                f"remove unsupported phases: {unsupported}."
            )

        recovery = self.config.get("recovery") or {}
        self.max_lost_frames = int(recovery.get("max_lost_frames", 5))
        self.max_attempts = int(recovery.get("max_attempts", 3))
        if self.max_lost_frames < 1:
            raise ValueError("recovery.max_lost_frames must be >= 1.")
        if self.max_attempts < 0:
            raise ValueError("recovery.max_attempts must be >= 0.")

    def run_goal(self, goal: TaskGoal) -> ExecutorResult:
        """Execute one target-color placement goal.

        ``attempts`` in the result counts recovery/retry attempts after the
        first try. A successful first pass therefore returns ``attempts == 0``.
        """
        self.trace.clear()
        attempts = 0
        self._record(HybridTaskState.IDLE, "start", attempts)

        while attempts <= self.max_attempts:
            belief = self._observe(goal.target_color, attempts)
            if belief is None:
                return self._failed("target belief unavailable", attempts)

            try:
                approach_plan = self.planner.plan_pregrasp(belief)
            except PlanningError as exc:
                return self._failed(str(exc), attempts)

            if not self._execute_plan(HybridTaskState.APPROACH, approach_plan, attempts):
                return self._failed("approach waypoint execution failed", attempts)

            self._record(HybridTaskState.ALIGN_GRASP, goal.target_color, attempts)
            if not bool(self.local_policy.run("align_grasp", goal.target_color, belief)):
                return self._failed("align_grasp policy failed", attempts)

            try:
                post_grasp = self.planner.plan_post_grasp(belief, goal.bowl_xyz_base)
            except PlanningError as exc:
                return self._failed(str(exc), attempts)

            if not self._execute_plan(HybridTaskState.LIFT, post_grasp.lift, attempts):
                return self._failed("lift waypoint execution failed", attempts)

            transport_ok = self._execute_transport(post_grasp.transport, goal.target_color, attempts)
            if not transport_ok:
                if attempts >= self.max_attempts:
                    return self._failed("target lost during transport; recovery attempts exhausted", attempts)
                if not self._recover(attempts):
                    return self._failed("recovery waypoint execution failed", attempts)
                attempts += 1
                continue

            if not self._execute_plan(HybridTaskState.RELEASE, post_grasp.release, attempts):
                return self._failed("release waypoint execution failed", attempts)

            self._record(HybridTaskState.DONE, goal.target_color, attempts)
            return ExecutorResult(
                success=True,
                final_state=HybridTaskState.DONE,
                attempts=attempts,
                trace=tuple(self.trace),
            )

        return self._failed("maximum attempts exceeded", attempts)

    def run_sequence(self, goals: Sequence[TaskGoal]) -> ExecutorResult:
        """Execute a Project 3 Eval-3 style sequence with re-observation per goal."""
        all_trace: list[TraceEvent] = []
        total_attempts = 0
        for goal in goals:
            result = self.run_goal(goal)
            all_trace.extend(result.trace)
            total_attempts += result.attempts
            if not result.success:
                return ExecutorResult(
                    success=False,
                    final_state=result.final_state,
                    attempts=total_attempts,
                    trace=tuple(all_trace),
                    failure_reason=result.failure_reason,
                )
        return ExecutorResult(
            success=True,
            final_state=HybridTaskState.DONE,
            attempts=total_attempts,
            trace=tuple(all_trace),
        )

    def _observe(self, target_color: str, attempt: int) -> BlockBelief | None:
        self._record(HybridTaskState.OBSERVE, target_color, attempt)
        return self.observer.observe(target_color)

    def _execute_plan(self, state: HybridTaskState, plan: WaypointPlan, attempt: int) -> bool:
        self._record(state, plan.phase, attempt)
        return bool(self.controller.execute(plan.waypoints))

    def _execute_transport(self, plan: WaypointPlan, target_color: str, attempt: int) -> bool:
        self._record(HybridTaskState.TRANSPORT, plan.phase, attempt)
        lost_frames = 0
        for waypoint in plan.waypoints:
            if not bool(self.controller.execute((waypoint,))):
                return False
            if self.visibility_checker is None:
                continue
            if self.visibility_checker.is_visible(target_color):
                lost_frames = 0
            else:
                lost_frames += 1
                self._record(
                    HybridTaskState.TRANSPORT,
                    f"lost_frame={lost_frames}/{self.max_lost_frames}",
                    attempt,
                )
                if lost_frames >= self.max_lost_frames:
                    return False
        return True

    def _recover(self, attempt: int) -> bool:
        recovery_plan = self.planner.plan_recovery()
        self._record(HybridTaskState.RECOVERY, recovery_plan.phase, attempt)
        return bool(self.controller.execute(recovery_plan.waypoints))

    def _failed(self, reason: str, attempts: int) -> ExecutorResult:
        self._record(HybridTaskState.FAILED, reason, attempts)
        return ExecutorResult(
            success=False,
            final_state=HybridTaskState.FAILED,
            attempts=attempts,
            trace=tuple(self.trace),
            failure_reason=reason,
        )

    def _record(self, state: HybridTaskState, detail: str, attempt: int) -> None:
        self.trace.append(TraceEvent(state=state, detail=detail, attempt=attempt))
