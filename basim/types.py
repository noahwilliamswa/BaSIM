from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AcousticFrame:
    """Compact acoustic controls used to synthesize the explainer's spectrum."""

    f1: float = 0.0
    f2: float = 0.0
    voicing: float = 0.0
    burst: float = 0.0
    frication: float = 0.0
    amplitude: float = 0.0

    def clipped(self) -> "AcousticFrame":
        values = [self.f1, self.f2, self.voicing, self.burst, self.frication, self.amplitude]
        return AcousticFrame(*(float(np.clip(v, 0.0, 1.0)) for v in values))


@dataclass(frozen=True)
class StimulusSegment:
    label: str
    duration_s: float
    frame: AcousticFrame


@dataclass
class StimulusProgram:
    name: str
    segments: list[StimulusSegment]

    @property
    def duration_s(self) -> float:
        return sum(max(0.0, segment.duration_s) for segment in self.segments)

    @property
    def boundaries_s(self) -> tuple[float, ...]:
        cursor = 0.0
        values: list[float] = []
        for segment in self.segments[:-1]:
            cursor += max(0.0, segment.duration_s)
            values.append(cursor)
        return tuple(values)

    def frame_at(self, t_s: float) -> tuple[str, AcousticFrame]:
        if t_s < 0.0:
            return "silence", AcousticFrame()
        cursor = 0.0
        for segment in self.segments:
            cursor += max(0.0, segment.duration_s)
            if t_s < cursor:
                return segment.label, segment.frame.clipped()
        return "silence", AcousticFrame()


@dataclass(frozen=True)
class Spectrogram:
    """Explicit time-frequency stimulus representation.

    `power` is shaped (time, frequency) and normalized to [0, 1].
    """

    times_s: np.ndarray
    frequencies_hz: np.ndarray
    power: np.ndarray

    def __post_init__(self) -> None:
        if self.power.ndim != 2:
            raise ValueError("spectrogram power must be 2D")
        if self.power.shape != (len(self.times_s), len(self.frequencies_hz)):
            raise ValueError("spectrogram axes do not match power shape")

    def sample(self, t_s: float) -> np.ndarray:
        if len(self.times_s) == 0 or t_s < 0.0 or t_s > float(self.times_s[-1]):
            return np.zeros(len(self.frequencies_hz), dtype=np.float64)
        idx = int(np.searchsorted(self.times_s, t_s, side="left"))
        if idx <= 0:
            return self.power[0].copy()
        if idx >= len(self.times_s):
            return self.power[-1].copy()

        t0 = float(self.times_s[idx - 1])
        t1 = float(self.times_s[idx])
        if t1 <= t0:
            return self.power[idx].copy()
        alpha = (t_s - t0) / (t1 - t0)
        return (1.0 - alpha) * self.power[idx - 1] + alpha * self.power[idx]


@dataclass(frozen=True)
class SimulationConfig:
    dt_s: float = 0.049
    duration_s: float = 5.0
    spectrum_frame_s: float = 0.010
    acoustic_to_phoneme_delay_s: float = 0.049
    acoustic_to_phoneme_tau_s: float = 0.098
    phoneme_to_lexical_delay_s: float = 0.098
    phoneme_to_lexical_tau_s: float = 0.147
    lexical_feedback_gain: float = 0.05
    seed: int = 7


@dataclass
class LayerSnapshot:
    name: str
    labels: tuple[str, ...]
    evidence: np.ndarray
    activation: np.ndarray
    fatigue: np.ndarray
    depths: np.ndarray
    position: np.ndarray


@dataclass
class SimulationSnapshot:
    time_s: float
    stimulus_label: str
    spectrum: np.ndarray
    acoustic_features: np.ndarray
    feature: LayerSnapshot
    phoneme: LayerSnapshot
    lexical: LayerSnapshot
    lexical_evidence: np.ndarray

    @property
    def lexical_winner(self) -> str:
        if self.lexical.activation.size == 0:
            return ""
        return self.lexical.labels[int(np.argmax(self.lexical.activation))]
