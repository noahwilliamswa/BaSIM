from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np


Array = np.ndarray


def _softmax(values: Array, temperature: float = 1.0) -> Array:
    values = np.asarray(values, dtype=float) / max(float(temperature), 1e-6)
    values = values - np.max(values)
    exp = np.exp(values)
    total = float(exp.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full_like(exp, 1.0 / len(exp))
    return exp / total


@dataclass(frozen=True)
class AcousticSegment:
    label: str
    start_ms: float
    end_ms: float
    vector: tuple[float, float]
    strength: float = 1.0


@dataclass(frozen=True)
class StimulusProgram:
    name: str
    segments: tuple[AcousticSegment, ...]
    duration_ms: float = 1000.0
    edge_ms: float = 70.0

    def sample(self, t_ms: float) -> tuple[Array, float]:
        if t_ms < 0.0 or t_ms >= self.duration_ms:
            return np.zeros(2, dtype=float), 0.0

        weighted = np.zeros(2, dtype=float)
        total = 0.0
        edge = max(self.edge_ms, 1e-6)
        for seg in self.segments:
            if t_ms < seg.start_ms - edge or t_ms > seg.end_ms + edge:
                continue
            fade_in = np.clip((t_ms - (seg.start_ms - edge)) / edge, 0.0, 1.0)
            fade_out = np.clip(((seg.end_ms + edge) - t_ms) / edge, 0.0, 1.0)
            weight = float(min(fade_in, fade_out)) * seg.strength
            weighted += weight * np.asarray(seg.vector, dtype=float)
            total += weight

        if total <= 1e-9:
            return np.zeros(2, dtype=float), 0.0
        return weighted / total, min(total, 1.0)


@dataclass(frozen=True)
class AttractorSpec:
    label: str
    center: tuple[float, float]
    base_depth: float = 1.0
    width: float = 0.65


@dataclass
class LayerConfig:
    name: str
    attractors: tuple[AttractorSpec, ...]
    evidence_gain: float = 2.4
    evidence_tau_ms: float = 180.0
    fatigue_gain: float = 0.45
    fatigue_tau_ms: float = 900.0
    state_tau_ms: float = 140.0
    velocity_damping: float = 0.72
    input_pull: float = 1.0
    temperature: float = 0.45
    activation_state_weight: float = 1.0
    activation_drive_weight: float = 0.0
    noise: float = 0.0


@dataclass(frozen=True)
class LayerSnapshot:
    state: tuple[float, float]
    activations: tuple[float, ...]
    depths: tuple[float, ...]
    drive_trace: tuple[float, ...]


class AdaptiveAttractorLayer:
    """A small adaptive potential field used as an explanatory dynamical system.

    Incoming evidence deepens compatible wells over a leaky integration window.
    Recently occupied wells fatigue slightly, which prevents the display from
    behaving like a permanent winner-take-all latch. The state itself evolves
    downhill in the deformed potential field.
    """

    def __init__(self, config: LayerConfig, seed: int = 0):
        self.config = config
        self._rng = np.random.default_rng(seed)
        self._centers = np.asarray([a.center for a in config.attractors], dtype=float)
        self._base_depths = np.asarray([a.base_depth for a in config.attractors], dtype=float)
        self._widths = np.asarray([a.width for a in config.attractors], dtype=float)
        self.reset()

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(a.label for a in self.config.attractors)

    def reset(self) -> None:
        self.state = np.mean(self._centers, axis=0).astype(float)
        self.velocity = np.zeros(2, dtype=float)
        self.drive_trace = np.zeros(len(self._centers), dtype=float)
        self.fatigue = np.zeros(len(self._centers), dtype=float)
        self.activations = np.full(len(self._centers), 1.0 / len(self._centers), dtype=float)
        self.depths = self._base_depths.copy()

    def _activation_from_state(self, depths: Array) -> Array:
        delta = self.state[None, :] - self._centers
        d2 = np.sum(delta * delta, axis=1)
        scores = (
            np.log(np.maximum(depths, 1e-5))
            + self.config.activation_drive_weight * self.drive_trace
            - self.config.activation_state_weight
            * d2
            / (2.0 * np.maximum(self._widths, 1e-5) ** 2)
        )
        return _softmax(scores, self.config.temperature)

    def step(self, evidence: Sequence[float], dt_ms: float) -> LayerSnapshot:
        evidence_arr = np.asarray(evidence, dtype=float)
        if evidence_arr.shape != self.drive_trace.shape:
            raise ValueError(
                f"{self.config.name}: expected evidence shape {self.drive_trace.shape}, "
                f"got {evidence_arr.shape}"
            )
        evidence_arr = np.clip(evidence_arr, 0.0, 1.0)

        drive_alpha = 1.0 - np.exp(-dt_ms / max(self.config.evidence_tau_ms, 1e-6))
        fatigue_alpha = 1.0 - np.exp(-dt_ms / max(self.config.fatigue_tau_ms, 1e-6))
        self.drive_trace += drive_alpha * (evidence_arr - self.drive_trace)
        self.fatigue += fatigue_alpha * (self.activations - self.fatigue)

        self.depths = np.maximum(
            0.08,
            self._base_depths
            + self.config.evidence_gain * self.drive_trace
            - self.config.fatigue_gain * self.fatigue,
        )

        delta = self.state[None, :] - self._centers
        width2 = np.maximum(self._widths, 1e-5) ** 2
        gaussian = np.exp(-np.sum(delta * delta, axis=1) / (2.0 * width2))
        # Negative gradient of -depth * Gaussian: pull toward each center.
        force = np.sum(
            (self.depths * gaussian / width2)[:, None] * (-delta),
            axis=0,
        )

        evidence_total = float(evidence_arr.sum())
        if evidence_total > 1e-9:
            target = np.average(self._centers, axis=0, weights=evidence_arr + 1e-9)
            force += self.config.input_pull * (target - self.state)

        if self.config.noise > 0.0:
            force += self._rng.normal(0.0, self.config.noise, size=2)

        dt_scale = dt_ms / max(self.config.state_tau_ms, 1e-6)
        self.velocity = self.config.velocity_damping * self.velocity + dt_scale * force
        self.state = self.state + self.velocity
        self.activations = self._activation_from_state(self.depths)

        return self.snapshot()

    def snapshot(self) -> LayerSnapshot:
        return LayerSnapshot(
            state=(float(self.state[0]), float(self.state[1])),
            activations=tuple(float(v) for v in self.activations),
            depths=tuple(float(v) for v in self.depths),
            drive_trace=tuple(float(v) for v in self.drive_trace),
        )


@dataclass(frozen=True)
class WordCandidate:
    word: str
    phonemes: tuple[str, ...]
    prior: float = 1.0


class CohortTracker:
    """Incremental word-candidate evidence from a changing phoneme distribution.

    The tracker does not pretend to be a full spoken-word-recognition model.
    It keeps a soft prefix score for each word and only advances its discrete
    phoneme history when the dominant phoneme is stable long enough. That gives
    the lexical layer a changing cohort rather than a single post-hoc label.
    """

    def __init__(
        self,
        phoneme_labels: Sequence[str],
        candidates: Sequence[WordCandidate],
        min_confidence: float = 0.42,
        min_dwell_ms: float = 98.0,
    ):
        self.phoneme_labels = tuple(phoneme_labels)
        self.candidates = tuple(candidates)
        self.min_confidence = min_confidence
        self.min_dwell_ms = min_dwell_ms
        self.reset()

    def reset(self) -> None:
        self.observed: list[str] = []
        self._current: str | None = None
        self._dwell_ms = 0.0
        self._committed_current = False
        self._scores = np.asarray(
            [np.log(max(c.prior, 1e-6)) for c in self.candidates],
            dtype=float,
        )

    def _commit(self, phoneme: str, distribution: Array) -> None:
        if self.observed and self.observed[-1] == phoneme:
            return
        self.observed.append(phoneme)
        pos = len(self.observed) - 1
        floor = 0.04
        for i, candidate in enumerate(self.candidates):
            if pos < len(candidate.phonemes):
                expected = candidate.phonemes[pos]
                try:
                    expected_idx = self.phoneme_labels.index(expected)
                    p_expected = float(distribution[expected_idx])
                except ValueError:
                    p_expected = floor
                if expected == phoneme:
                    self._scores[i] += np.log(max(0.55 + 0.45 * p_expected, floor))
                else:
                    self._scores[i] += np.log(max(0.06 + 0.34 * p_expected, floor))
            else:
                self._scores[i] += np.log(0.08)

    def step(
        self,
        phoneme_distribution: Sequence[float],
        dt_ms: float,
        active: bool,
    ) -> Array:
        probs = np.asarray(phoneme_distribution, dtype=float)
        if probs.shape != (len(self.phoneme_labels),):
            raise ValueError("phoneme distribution has wrong shape")

        if active:
            idx = int(np.argmax(probs))
            label = self.phoneme_labels[idx]
            confidence = float(probs[idx])
            if confidence >= self.min_confidence:
                if label != self._current:
                    self._current = label
                    self._dwell_ms = dt_ms
                    self._committed_current = False
                else:
                    self._dwell_ms += dt_ms
                if self._dwell_ms >= self.min_dwell_ms and not self._committed_current:
                    self._commit(label, probs)
                    self._committed_current = True

        # A small unfinished-word penalty keeps exact-length candidates competitive
        # once input stops, but does not force a decision before the sound unfolds.
        adjusted = self._scores.copy()
        if not active and self.observed:
            for i, candidate in enumerate(self.candidates):
                remaining = max(0, len(candidate.phonemes) - len(self.observed))
                adjusted[i] -= 0.7 * remaining
        return _softmax(adjusted, temperature=0.55)


@dataclass(frozen=True)
class FrameRecord:
    t_ms: float
    stimulus: tuple[float, float]
    stimulus_strength: float
    acoustic: LayerSnapshot
    phoneme: LayerSnapshot
    lexical: LayerSnapshot
    lexical_evidence: tuple[float, ...]
    observed_phonemes: tuple[str, ...]


class SpeechAttractorSimulation:
    def __init__(
        self,
        stimulus: StimulusProgram,
        acoustic: AdaptiveAttractorLayer,
        phoneme: AdaptiveAttractorLayer,
        lexical: AdaptiveAttractorLayer,
        acoustic_to_phoneme: Array,
        cohort: CohortTracker,
        dt_ms: float = 49.0,
        total_ms: float = 5000.0,
    ):
        self.stimulus = stimulus
        self.acoustic = acoustic
        self.phoneme = phoneme
        self.lexical = lexical
        self.acoustic_to_phoneme = np.asarray(acoustic_to_phoneme, dtype=float)
        self.cohort = cohort
        self.dt_ms = float(dt_ms)
        self.total_ms = float(total_ms)
        if self.acoustic_to_phoneme.shape != (
            len(self.phoneme.labels),
            len(self.acoustic.labels),
        ):
            raise ValueError("acoustic_to_phoneme matrix shape does not match layer sizes")
        if len(self.lexical.labels) != len(self.cohort.candidates):
            raise ValueError("lexical attractors and cohort candidates must align")
        self.reset()

    def reset(self) -> None:
        self.acoustic.reset()
        self.phoneme.reset()
        self.lexical.reset()
        self.cohort.reset()
        self.t_ms = 0.0
        self.records: list[FrameRecord] = []

    def _acoustic_evidence(self, vector: Array, strength: float) -> Array:
        if strength <= 0.0:
            return np.zeros(len(self.acoustic.labels), dtype=float)
        centers = np.asarray(
            [a.center for a in self.acoustic.config.attractors],
            dtype=float,
        )
        widths = np.asarray(
            [max(a.width, 1e-4) for a in self.acoustic.config.attractors],
            dtype=float,
        )
        delta = centers - vector[None, :]
        sim = np.exp(-np.sum(delta * delta, axis=1) / (2.0 * widths * widths))
        if float(sim.max()) > 0.0:
            sim /= float(sim.max())
        return np.clip(sim * strength, 0.0, 1.0)

    def step(self) -> FrameRecord:
        vector, strength = self.stimulus.sample(self.t_ms)
        acoustic_evidence = self._acoustic_evidence(vector, strength)
        acoustic_snap = self.acoustic.step(acoustic_evidence, self.dt_ms)

        acoustic_activation = np.asarray(acoustic_snap.activations, dtype=float)
        phoneme_evidence = self.acoustic_to_phoneme @ acoustic_activation
        max_ev = float(phoneme_evidence.max()) if len(phoneme_evidence) else 0.0
        if max_ev > 0.0:
            phoneme_evidence = phoneme_evidence / max_ev
        phoneme_evidence *= strength
        phoneme_snap = self.phoneme.step(phoneme_evidence, self.dt_ms)

        lexical_evidence = self.cohort.step(
            phoneme_snap.activations,
            self.dt_ms,
            active=strength > 0.05,
        )
        lexical_snap = self.lexical.step(lexical_evidence, self.dt_ms)

        record = FrameRecord(
            t_ms=self.t_ms,
            stimulus=(float(vector[0]), float(vector[1])),
            stimulus_strength=float(strength),
            acoustic=acoustic_snap,
            phoneme=phoneme_snap,
            lexical=lexical_snap,
            lexical_evidence=tuple(float(v) for v in lexical_evidence),
            observed_phonemes=tuple(self.cohort.observed),
        )
        self.records.append(record)
        self.t_ms += self.dt_ms
        return record

    @property
    def finished(self) -> bool:
        return self.t_ms >= self.total_ms

    def run(self) -> list[FrameRecord]:
        while not self.finished:
            self.step()
        return self.records

    def winner(self) -> tuple[str, float]:
        activations = np.asarray(self.lexical.activations, dtype=float)
        idx = int(np.argmax(activations))
        return self.lexical.labels[idx], float(activations[idx])


@dataclass(frozen=True)
class SimulationBundle:
    simulation: SpeechAttractorSimulation
    metadata: Mapping[str, str] = field(default_factory=dict)
