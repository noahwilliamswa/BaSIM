from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DEFAULT_FREQUENCY_BANDS_HZ = (250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 4500.0)


@dataclass(frozen=True)
class AcousticFrame:
    """Legacy normalized feature frame kept for compatibility with early BaSIM scripts."""

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
class SpectrumFrame:
    """Energy in fixed frequency bands at one simulation instant."""

    energies: tuple[float, ...]
    bands_hz: tuple[float, ...] = DEFAULT_FREQUENCY_BANDS_HZ

    def __post_init__(self) -> None:
        if len(self.energies) != len(self.bands_hz):
            raise ValueError("energies and bands_hz must have equal length")
        if not self.energies:
            raise ValueError("SpectrumFrame requires at least one frequency band")

    def array(self) -> np.ndarray:
        return np.clip(np.asarray(self.energies, dtype=np.float64), 0.0, 1.0)

    @property
    def amplitude(self) -> float:
        return float(np.max(self.array(), initial=0.0))

    @classmethod
    def silence(cls, bands_hz: tuple[float, ...] = DEFAULT_FREQUENCY_BANDS_HZ) -> "SpectrumFrame":
        return cls(tuple(0.0 for _ in bands_hz), bands_hz)


@dataclass(frozen=True)
class StimulusSegment:
    label: str
    duration_s: float
    frame: SpectrumFrame | AcousticFrame


@dataclass
class StimulusProgram:
    name: str
    segments: list[StimulusSegment]
    crossfade_s: float = 0.049

    @property
    def duration_s(self) -> float:
        return sum(max(0.0, segment.duration_s) for segment in self.segments)

    def frame_at(self, t_s: float) -> tuple[str, SpectrumFrame | AcousticFrame]:
        if t_s < 0.0 or not self.segments:
            return "silence", SpectrumFrame.silence()

        cursor = 0.0
        for index, segment in enumerate(self.segments):
            start = cursor
            end = cursor + max(0.0, segment.duration_s)
            if t_s < end:
                # Crossfade only spectral frames. Legacy AcousticFrame scripts keep their
                # original piecewise-constant behavior.
                if (
                    index > 0
                    and self.crossfade_s > 0.0
                    and isinstance(segment.frame, SpectrumFrame)
                    and isinstance(self.segments[index - 1].frame, SpectrumFrame)
                    and t_s < start + self.crossfade_s
                ):
                    prev = self.segments[index - 1].frame
                    current = segment.frame
                    alpha = float(np.clip((t_s - start) / self.crossfade_s, 0.0, 1.0))
                    mixed = (1.0 - alpha) * prev.array() + alpha * current.array()
                    return f"{self.segments[index - 1].label}→{segment.label}", SpectrumFrame(
                        tuple(float(v) for v in mixed), current.bands_hz
                    )
                return segment.label, segment.frame
            cursor = end

        last = self.segments[-1].frame
        if isinstance(last, SpectrumFrame):
            return "silence", SpectrumFrame.silence(last.bands_hz)
        return "silence", AcousticFrame()


@dataclass(frozen=True)
class ConnectionConfig:
    delay_s: float = 0.049
    tau_s: float = 0.098
    gain: float = 1.0


@dataclass(frozen=True)
class SimulationConfig:
    dt_s: float = 0.049
    duration_s: float = 5.0
    lexical_feedback_gain: float = 0.16
    seed: int = 7
    feature_to_phoneme: ConnectionConfig = field(
        default_factory=lambda: ConnectionConfig(delay_s=0.049, tau_s=0.098, gain=1.0)
    )
    phoneme_to_lexical: ConnectionConfig = field(
        default_factory=lambda: ConnectionConfig(delay_s=0.049, tau_s=0.147, gain=1.0)
    )
    lexical_to_phoneme: ConnectionConfig = field(
        default_factory=lambda: ConnectionConfig(delay_s=0.098, tau_s=0.196, gain=1.0)
    )


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
    stimulus_frame: SpectrumFrame | AcousticFrame
    spectrum: np.ndarray
    feature: LayerSnapshot
    phoneme: LayerSnapshot
    lexical: LayerSnapshot

    @property
    def lexical_winner(self) -> str:
        if self.lexical.activation.size == 0:
            return ""
        return self.lexical.labels[int(np.argmax(self.lexical.activation))]
