from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .types import LayerSnapshot


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64) / max(float(temperature), 1e-6)
    values = values - float(np.max(values, initial=0.0))
    exp = np.exp(values)
    total = float(np.sum(exp))
    if total <= 1e-12 or not np.isfinite(total):
        return np.full_like(exp, 1.0 / max(len(exp), 1))
    return exp / total


@dataclass(frozen=True)
class AttractorSpec:
    label: str
    center: tuple[float, float]
    base_depth: float = 1.0
    width: float = 0.72


@dataclass(frozen=True)
class AttractorConfig:
    name: str
    attractors: tuple[AttractorSpec, ...]
    evidence_gain: float = 2.6
    evidence_tau_s: float = 0.16
    fatigue_gain: float = 0.35
    fatigue_tau_s: float = 0.90
    state_tau_s: float = 0.15
    velocity_damping: float = 0.72
    input_pull: float = 0.90
    temperature: float = 0.45
    state_score_weight: float = 1.0
    trace_score_weight: float = 0.65
    inhibition_gain: float = 0.20
    noise_std: float = 0.0


class AdaptiveAttractorLayer:
    """2D latent attractor field with stimulus-dependent basin depths.

    The geometry is explanatory. Basin centers are fixed latent anchors while
    evidence, adaptation, competition, and history change their effective depth.
    """

    def __init__(self, config: AttractorConfig, seed: int = 0) -> None:
        if not config.attractors:
            raise ValueError("attractor layer needs at least one attractor")
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._centers = np.asarray(
            [attractor.center for attractor in config.attractors],
            dtype=np.float64,
        )
        self._base_depths = np.asarray(
            [attractor.base_depth for attractor in config.attractors],
            dtype=np.float64,
        )
        self._widths = np.asarray(
            [attractor.width for attractor in config.attractors],
            dtype=np.float64,
        )
        self.reset()

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(a.label for a in self.config.attractors)

    @property
    def centers(self) -> np.ndarray:
        return self._centers.copy()

    def reset(self) -> None:
        n = len(self._centers)
        self.position = np.mean(self._centers, axis=0).astype(np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.evidence = np.zeros(n, dtype=np.float64)
        self.trace = np.zeros(n, dtype=np.float64)
        self.fatigue = np.zeros(n, dtype=np.float64)
        self.depths = self._base_depths.copy()
        self.activation = np.full(n, 1.0 / n, dtype=np.float64)

    def _score_activation(self) -> np.ndarray:
        delta = self.position[np.newaxis, :] - self._centers
        distance2 = np.sum(delta * delta, axis=1)
        score = (
            np.log(np.maximum(self.depths, 1e-6))
            + self.config.trace_score_weight * self.trace
            - self.config.state_score_weight
            * distance2
            / (2.0 * np.maximum(self._widths, 1e-6) ** 2)
        )
        return _softmax(score, self.config.temperature)

    def step(self, evidence: Sequence[float], dt_s: float) -> LayerSnapshot:
        incoming = np.clip(np.asarray(evidence, dtype=np.float64), 0.0, 1.0)
        if incoming.shape != self.trace.shape:
            raise ValueError(
                f"{self.config.name}: expected evidence shape {self.trace.shape}, "
                f"got {incoming.shape}"
            )
        self.evidence = incoming.copy()

        trace_alpha = 1.0 - np.exp(
            -dt_s / max(self.config.evidence_tau_s, 1e-9)
        )
        fatigue_alpha = 1.0 - np.exp(
            -dt_s / max(self.config.fatigue_tau_s, 1e-9)
        )
        self.trace += trace_alpha * (incoming - self.trace)
        self.fatigue += fatigue_alpha * (self.activation - self.fatigue)

        competitor = np.maximum(0.0, np.sum(self.trace) - self.trace)
        self.depths = np.maximum(
            0.06,
            self._base_depths
            + self.config.evidence_gain * self.trace
            - self.config.fatigue_gain * self.fatigue
            - self.config.inhibition_gain * competitor,
        )

        delta = self.position[np.newaxis, :] - self._centers
        width2 = np.maximum(self._widths, 1e-6) ** 2
        gaussian = np.exp(-np.sum(delta * delta, axis=1) / (2.0 * width2))
        force = np.sum(
            (self.depths * gaussian / width2)[:, np.newaxis] * (-delta),
            axis=0,
        )

        total = float(np.sum(incoming))
        if total > 1e-9:
            target = np.average(
                self._centers,
                axis=0,
                weights=incoming + 1e-12,
            )
            force += self.config.input_pull * (target - self.position)

        if self.config.noise_std > 0.0:
            force += self._rng.normal(0.0, self.config.noise_std, size=2)

        step_scale = dt_s / max(self.config.state_tau_s, 1e-9)
        self.velocity = (
            self.config.velocity_damping * self.velocity
            + step_scale * force
        )
        speed = float(np.linalg.norm(self.velocity))
        if speed > 0.75:
            self.velocity *= 0.75 / speed
        self.position = np.clip(self.position + self.velocity, -2.7, 2.7)
        self.activation = self._score_activation()
        return self.snapshot()

    def snapshot(self) -> LayerSnapshot:
        return LayerSnapshot(
            name=self.config.name,
            labels=self.labels,
            evidence=self.evidence.copy(),
            activation=self.activation.copy(),
            fatigue=self.fatigue.copy(),
            depths=self.depths.copy(),
            position=self.position.copy(),
        )


class LeakyDelayConnection:
    """Fixed projection with an explicit conduction delay and leaky integration."""

    def __init__(
        self,
        weights: np.ndarray,
        dt_s: float,
        delay_s: float = 0.0,
        tau_s: float = 0.10,
        normalize: bool = True,
    ) -> None:
        matrix = np.asarray(weights, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("connection weights must be a 2D matrix")
        self.weights = np.clip(matrix, 0.0, None)
        self.dt_s = float(dt_s)
        self.delay_s = max(0.0, float(delay_s))
        self.tau_s = max(1e-6, float(tau_s))
        self.normalize = normalize
        self.delay_steps = int(round(self.delay_s / max(self.dt_s, 1e-9)))
        self._source_size = matrix.shape[1]
        self._target_size = matrix.shape[0]
        self.reset()

    def reset(self) -> None:
        self._buffer: deque[np.ndarray] = deque(
            [np.zeros(self._source_size, dtype=np.float64)
             for _ in range(self.delay_steps)],
            maxlen=max(self.delay_steps + 1, 1),
        )
        self.state = np.zeros(self._target_size, dtype=np.float64)

    def step(self, source: Sequence[float]) -> np.ndarray:
        source_arr = np.asarray(source, dtype=np.float64)
        if source_arr.shape != (self._source_size,):
            raise ValueError(
                f"expected source shape {(self._source_size,)}, got {source_arr.shape}"
            )

        if self.delay_steps == 0:
            delayed = source_arr
        else:
            self._buffer.append(source_arr.copy())
            delayed = self._buffer.popleft()

        projected = self.weights @ delayed
        if self.normalize:
            max_value = float(np.max(projected, initial=0.0))
            if max_value > 1e-12:
                projected = projected / max_value

        alpha = 1.0 - np.exp(-self.dt_s / self.tau_s)
        self.state += alpha * (projected - self.state)
        return np.clip(self.state.copy(), 0.0, 1.0)
