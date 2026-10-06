from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .types import LayerSnapshot


class CompetitiveAttractorLayer:
    """Stateful 2D attractor landscape with competition, recurrence and fatigue."""

    def __init__(
        self,
        name: str,
        labels: Sequence[str],
        *,
        dt_s: float,
        centers: np.ndarray | None = None,
        tau_s: float = 0.120,
        fatigue_tau_s: float = 0.650,
        evidence_gain: float = 1.55,
        self_excitation: float = 0.62,
        lateral_inhibition: float = 0.44,
        fatigue_gain: float = 0.52,
        base_depth: float = 0.35,
        activation_depth_gain: float = 2.8,
        fatigue_depth_gain: float = 0.85,
        basin_sigma: float = 0.25,
        activation_threshold: float = 0.70,
        activation_slope: float = 6.0,
    ) -> None:
        self.name = name
        self.labels = tuple(labels)
        self.dt_s = float(dt_s)
        self.tau_s = float(tau_s)
        self.fatigue_tau_s = float(fatigue_tau_s)
        self.evidence_gain = float(evidence_gain)
        self.self_excitation = float(self_excitation)
        self.lateral_inhibition = float(lateral_inhibition)
        self.fatigue_gain = float(fatigue_gain)
        self.base_depth = float(base_depth)
        self.activation_depth_gain = float(activation_depth_gain)
        self.fatigue_depth_gain = float(fatigue_depth_gain)
        self.basin_sigma = float(basin_sigma)
        self.activation_threshold = float(activation_threshold)
        self.activation_slope = float(activation_slope)

        n = len(self.labels)
        centers = self._ring_centers(n) if centers is None else np.asarray(centers, dtype=np.float64)
        if centers.shape != (n, 2):
            raise ValueError(f"centers must have shape {(n, 2)}, got {centers.shape}")
        self.centers = centers
        self.activation = np.zeros(n, dtype=np.float64)
        self.fatigue = np.zeros(n, dtype=np.float64)
        self.depths = np.full(n, self.base_depth, dtype=np.float64)
        self.position = np.array([0.5, 0.5], dtype=np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)

    @staticmethod
    def _ring_centers(n: int) -> np.ndarray:
        if n <= 0:
            return np.zeros((0, 2), dtype=np.float64)
        angles = np.linspace(-math.pi / 2.0, 3.0 * math.pi / 2.0, n, endpoint=False)
        radius = 0.34
        return np.column_stack((0.5 + radius * np.cos(angles), 0.5 + radius * np.sin(angles)))

    def reset(self) -> None:
        self.activation.fill(0.0)
        self.fatigue.fill(0.0)
        self.depths.fill(self.base_depth)
        self.position[:] = 0.5
        self.velocity.fill(0.0)

    def step(self, evidence: np.ndarray) -> LayerSnapshot:
        evidence = np.asarray(evidence, dtype=np.float64)
        if evidence.shape != self.activation.shape:
            raise ValueError(f"evidence must have shape {self.activation.shape}, got {evidence.shape}")
        evidence = np.clip(evidence, 0.0, 1.0)
        competitors = (
            (np.sum(self.activation) - self.activation) / max(1, self.activation.size - 1)
            if self.activation.size
            else self.activation
        )
        net = (
            self.evidence_gain * evidence
            + self.self_excitation * self.activation
            - self.lateral_inhibition * competitors
            - self.fatigue_gain * self.fatigue
        )
        target = 1.0 / (1.0 + np.exp(-self.activation_slope * (net - self.activation_threshold)))
        alpha = 1.0 - math.exp(-self.dt_s / max(self.tau_s, 1e-6))
        self.activation += alpha * (target - self.activation)
        self.activation[:] = np.clip(self.activation, 0.0, 1.0)

        fatigue_alpha = 1.0 - math.exp(-self.dt_s / max(self.fatigue_tau_s, 1e-6))
        self.fatigue += fatigue_alpha * (self.activation - self.fatigue)
        self.fatigue[:] = np.clip(self.fatigue, 0.0, 1.0)
        self.depths = np.clip(
            self.base_depth + self.activation_depth_gain * self.activation - self.fatigue_depth_gain * self.fatigue,
            0.05,
            None,
        )
        self._integrate_position()
        return self.snapshot()

    def _integrate_position(self) -> None:
        if not self.labels:
            return
        delta = self.centers - self.position[np.newaxis, :]
        r2 = np.sum(delta * delta, axis=1)
        influence = self.depths * np.exp(-r2 / (2.0 * self.basin_sigma**2))
        force = np.sum(influence[:, np.newaxis] * delta, axis=0)
        self.velocity = 0.76 * self.velocity + 0.34 * force
        self.position += self.velocity
        self.position[:] = np.clip(self.position, 0.03, 0.97)

    def potential_grid(self, resolution: int = 96) -> np.ndarray:
        axis = np.linspace(0.0, 1.0, max(8, int(resolution)))
        xx, yy = np.meshgrid(axis, axis)
        potential = np.zeros_like(xx)
        for center, depth in zip(self.centers, self.depths, strict=True):
            r2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
            potential -= depth * np.exp(-r2 / (2.0 * self.basin_sigma**2))
        return potential

    def snapshot(self) -> LayerSnapshot:
        return LayerSnapshot(
            self.name,
            self.labels,
            self.activation.copy(),
            self.fatigue.copy(),
            self.depths.copy(),
            self.position.copy(),
        )
