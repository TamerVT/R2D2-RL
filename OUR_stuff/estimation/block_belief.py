"""Per-color block belief tracker.

Each block is modeled as a stationary 2D point in the robot base frame with
diagonal process noise. Detections from ``PixelToTableProjector`` arrive as
``(mean_xy, covariance_xy)`` pairs and update the belief with a standard
linear Kalman filter on the identity observation model.

Process noise can be temporarily inflated through :meth:`mark_contact` to
reflect that recent contact with the gripper may have moved the block. The
tracker holds one belief per color, so beliefs for different colors stay
independent and can be retrieved individually.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BlockBelief:
    color: str
    mean_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    covariance_xy: np.ndarray = field(default_factory=lambda: np.eye(2))
    last_seen_time: float = -np.inf
    confidence: float = 0.0
    initialized: bool = False

    def copy(self) -> "BlockBelief":
        return BlockBelief(
            color=self.color,
            mean_xy=self.mean_xy.copy(),
            covariance_xy=self.covariance_xy.copy(),
            last_seen_time=self.last_seen_time,
            confidence=self.confidence,
            initialized=self.initialized,
        )


class BlockBeliefTracker:
    """Per-color static Kalman tracker with optional contact-aware predict.

    Configuration is read from the ``estimation`` block of the project YAML:

    - ``process_noise_xy``         (default 1e-4): nominal sigma^2 per second per axis
    - ``process_noise_contact_xy`` (default 2.5e-3): inflated value for ``mark_contact``
    - ``initial_covariance_xy``    (default 2.5e-3): variance used on first observation
    - ``contact_decay_s``          (default 1.0): seconds before contact noise relaxes
    """

    def __init__(self, config: dict[str, Any]):
        cfg = config.get("estimation") or {}
        self.process_noise_xy = float(cfg.get("process_noise_xy", 1e-4))
        self.process_noise_contact_xy = float(cfg.get("process_noise_contact_xy", 2.5e-3))
        self.initial_covariance_xy = float(cfg.get("initial_covariance_xy", 2.5e-3))
        self.contact_decay_s = float(cfg.get("contact_decay_s", 1.0))
        if min(
            self.process_noise_xy,
            self.process_noise_contact_xy,
            self.initial_covariance_xy,
            self.contact_decay_s,
        ) < 0:
            raise ValueError("Belief tracker noise and contact decay settings must be non-negative.")
        self._beliefs: dict[str, BlockBelief] = {}
        self._contact_state: dict[str, float] = {}

    def reset(self) -> None:
        self._beliefs.clear()
        self._contact_state.clear()

    def mark_contact(self, color: str) -> None:
        """Flag a block as recently contacted: next ``predict`` uses inflated Q."""
        self._contact_state[color] = self.contact_decay_s

    def predict(self, dt: float) -> None:
        """Advance every belief by ``dt`` seconds, inflating Q during contact."""
        if dt <= 0:
            return
        for color, belief in self._beliefs.items():
            if not belief.initialized:
                continue
            q = self._effective_process_noise(color)
            belief.covariance_xy = belief.covariance_xy + np.eye(2) * (q * dt)
            if color in self._contact_state:
                self._contact_state[color] = max(0.0, self._contact_state[color] - dt)
                if self._contact_state[color] <= 0:
                    del self._contact_state[color]

    def update(
        self,
        color: str,
        measurement_xy: np.ndarray,
        measurement_cov: np.ndarray,
        timestamp: float,
        confidence: float | None = None,
    ) -> BlockBelief:
        """Fuse a new ``(xy, cov)`` measurement into the per-color belief."""
        measurement_xy = np.asarray(measurement_xy, dtype=np.float64).reshape(2)
        measurement_cov = _sanitize_covariance(measurement_cov)
        if not np.all(np.isfinite(measurement_xy)):
            raise ValueError("measurement_xy must be finite.")

        belief = self._beliefs.get(color)
        if belief is None or not belief.initialized:
            belief = BlockBelief(
                color=color,
                mean_xy=measurement_xy.copy(),
                covariance_xy=measurement_cov.copy() + np.eye(2) * self.initial_covariance_xy,
                last_seen_time=timestamp,
                confidence=confidence if confidence is not None else 1.0,
                initialized=True,
            )
            self._beliefs[color] = belief
            return belief.copy()

        # Identity observation model: y = x + v, v ~ N(0, R).
        P = belief.covariance_xy
        R = measurement_cov
        S = P + R
        K = np.linalg.solve(S.T, P.T).T
        innovation = measurement_xy - belief.mean_xy
        belief.mean_xy = belief.mean_xy + K @ innovation
        I_K = np.eye(2) - K
        belief.covariance_xy = _sanitize_covariance(I_K @ P @ I_K.T + K @ R @ K.T)
        belief.last_seen_time = timestamp
        if confidence is not None:
            belief.confidence = max(belief.confidence, float(confidence))
        return belief.copy()

    def get(self, color: str) -> BlockBelief | None:
        belief = self._beliefs.get(color)
        return belief.copy() if belief and belief.initialized else None

    def get_all(self) -> dict[str, BlockBelief]:
        return {c: b.copy() for c, b in self._beliefs.items() if b.initialized}

    def _effective_process_noise(self, color: str) -> float:
        if color in self._contact_state and self._contact_state[color] > 0:
            return self.process_noise_contact_xy
        return self.process_noise_xy


def _sanitize_covariance(covariance: np.ndarray) -> np.ndarray:
    cov = np.asarray(covariance, dtype=np.float64).reshape(2, 2)
    if not np.all(np.isfinite(cov)):
        raise ValueError("measurement_cov must be finite.")
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    if eigvals[0] < -1e-10:
        raise ValueError("measurement_cov must be positive semidefinite.")
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T
