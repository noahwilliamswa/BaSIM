from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AcousticFrame:
    """Normalized, formant-ish acoustic feature frame used by the explainer."""

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
class SimulationConfig:
    dt_s: float = 0.049
    duration_s: float = 5.0
    lexical_feedback_gain: float = 0.16
    seed: int = 7


@dataclass
class LayerSnapshot:
    name: str
    labels: tuple[str, ...]
    activation: np.ndarray
    fatigue: np.ndarray
    depths: np.ndarray
    position: np.ndarray


@dataclass
class SimulationSnapshot:
    time_s: float
    stimulus_label: str
    stimulus_frame: AcousticFrame
    feature: LayerSnapshot
    phoneme: LayerSnapshot
    lexical: LayerSnapshot

    @property
    def lexical_winner(self) -> str:
        if self.lexical.activation.size == 0:
            return ""
        return self.lexical.labels[int(np.argmax(self.lexical.activation))]
