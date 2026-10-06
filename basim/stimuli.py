from __future__ import annotations

import numpy as np

from .types import AcousticFrame, StimulusProgram, StimulusSegment


class AcousticEncoder:
    labels = ("F1 low", "F1 high", "F2 low", "F2 high", "voiced", "burst", "frication", "energy")

    @classmethod
    def encode(cls, frame: AcousticFrame) -> np.ndarray:
        f = frame.clipped()
        return f.amplitude * np.array(
            [1.0 - f.f1, f.f1, 1.0 - f.f2, f.f2, f.voicing, f.burst, f.frication, f.amplitude],
            dtype=np.float64,
        )


PHONEME_FRAMES: dict[str, AcousticFrame] = {
    "B": AcousticFrame(0.36, 0.40, 0.95, 0.75, 0.08, 1.0),
    "P": AcousticFrame(0.36, 0.40, 0.12, 0.88, 0.12, 1.0),
    "K": AcousticFrame(0.47, 0.68, 0.10, 0.92, 0.16, 1.0),
    "AE": AcousticFrame(0.78, 0.62, 0.98, 0.03, 0.02, 1.0),
    "EH": AcousticFrame(0.58, 0.66, 0.98, 0.03, 0.02, 1.0),
    "IH": AcousticFrame(0.40, 0.80, 0.98, 0.03, 0.02, 1.0),
    "T": AcousticFrame(0.46, 0.74, 0.10, 0.76, 0.64, 1.0),
    "D": AcousticFrame(0.46, 0.72, 0.88, 0.62, 0.32, 1.0),
}

LEXICON: dict[str, tuple[str, ...]] = {
    "BAT": ("B", "AE", "T"),
    "BAD": ("B", "AE", "D"),
    "CAT": ("K", "AE", "T"),
    "BET": ("B", "EH", "T"),
    "BIT": ("B", "IH", "T"),
}


def default_stimulus() -> StimulusProgram:
    return StimulusProgram(
        "BAT",
        [
            StimulusSegment("B", 0.22, PHONEME_FRAMES["B"]),
            StimulusSegment("AE", 0.48, PHONEME_FRAMES["AE"]),
            StimulusSegment("T", 0.30, PHONEME_FRAMES["T"]),
        ],
    )


def ambiguous_onset_stimulus() -> StimulusProgram:
    ambiguous = AcousticFrame(0.36, 0.40, 0.56, 0.82, 0.10, 1.0)
    return StimulusProgram(
        "?AT",
        [
            StimulusSegment("B/P?", 0.22, ambiguous),
            StimulusSegment("AE", 0.48, PHONEME_FRAMES["AE"]),
            StimulusSegment("T", 0.30, PHONEME_FRAMES["T"]),
        ],
    )


def phoneme_prototypes(labels: tuple[str, ...]) -> np.ndarray:
    return np.vstack([AcousticEncoder.encode(PHONEME_FRAMES[label]) for label in labels])


def prototype_similarity(current: np.ndarray, prototypes: np.ndarray, sigma: float = 0.28) -> np.ndarray:
    weights = np.array([1.0, 1.0, 1.0, 1.0, 2.4, 1.7, 1.9, 0.6], dtype=np.float64)
    distance2 = np.sum(weights * (prototypes - current[np.newaxis, :]) ** 2, axis=1) / np.sum(weights)
    return np.exp(-distance2 / (2.0 * sigma**2))
