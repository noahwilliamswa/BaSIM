from __future__ import annotations

import numpy as np

from .types import (
    DEFAULT_FREQUENCY_BANDS_HZ,
    AcousticFrame,
    SpectrumFrame,
    StimulusProgram,
    StimulusSegment,
)


class SpectralEncoder:
    bands_hz = DEFAULT_FREQUENCY_BANDS_HZ
    labels = tuple(f"{int(hz)} Hz" for hz in bands_hz)

    @classmethod
    def encode(cls, frame: SpectrumFrame | AcousticFrame) -> np.ndarray:
        if isinstance(frame, SpectrumFrame):
            if frame.bands_hz != cls.bands_hz:
                raise ValueError("SpectrumFrame bands do not match the configured encoder bands")
            return frame.array()
        # Compatibility map for early hand-authored AcousticFrame experiments.
        f = frame.clipped()
        return np.clip(
            f.amplitude
            * np.array(
                [
                    0.72 * f.voicing + 0.20 * (1.0 - f.f1),
                    0.62 * f.voicing + 0.38 * (1.0 - f.f1),
                    0.30 * f.voicing + 0.70 * f.f1,
                    0.25 * f.voicing + 0.55 * f.f1 + 0.20 * f.burst,
                    0.20 * f.voicing + 0.55 * f.f2 + 0.25 * f.burst,
                    0.15 * f.voicing + 0.60 * f.f2 + 0.25 * f.frication,
                    0.55 * f.frication + 0.45 * f.burst,
                    0.72 * f.frication + 0.28 * f.burst,
                ],
                dtype=np.float64,
            ),
            0.0,
            1.0,
        )


# Compatibility alias: early code imported AcousticEncoder.
AcousticEncoder = SpectralEncoder


def _spectrum(*values: float) -> SpectrumFrame:
    return SpectrumFrame(tuple(float(v) for v in values), SpectralEncoder.bands_hz)


PHONEME_SPECTRA: dict[str, SpectrumFrame] = {
    "B": _spectrum(0.86, 0.90, 0.58, 0.38, 0.34, 0.22, 0.12, 0.06),
    "P": _spectrum(0.10, 0.18, 0.34, 0.54, 0.73, 0.60, 0.31, 0.14),
    "K": _spectrum(0.05, 0.08, 0.15, 0.36, 0.82, 1.00, 0.48, 0.16),
    "AE": _spectrum(0.18, 0.42, 1.00, 0.58, 0.96, 0.34, 0.12, 0.05),
    "EH": _spectrum(0.28, 0.88, 0.66, 0.44, 0.54, 0.94, 0.22, 0.08),
    "IH": _spectrum(0.42, 1.00, 0.47, 0.34, 0.32, 0.62, 0.86, 0.16),
    "T": _spectrum(0.02, 0.04, 0.07, 0.13, 0.24, 0.46, 0.91, 1.00),
    "D": _spectrum(0.72, 0.82, 0.46, 0.25, 0.28, 0.42, 0.58, 0.47),
}

# Old name remains importable, but values are now spectrotemporal prototypes.
PHONEME_FRAMES = PHONEME_SPECTRA

LEXICON: dict[str, tuple[str, ...]] = {
    "BAT": ("B", "AE", "T"),
    "BAD": ("B", "AE", "D"),
    "PAT": ("P", "AE", "T"),
    "CAT": ("K", "AE", "T"),
    "BET": ("B", "EH", "T"),
    "BIT": ("B", "IH", "T"),
}


def default_stimulus() -> StimulusProgram:
    return StimulusProgram(
        "BAT",
        [
            StimulusSegment("B", 0.22, PHONEME_SPECTRA["B"]),
            StimulusSegment("AE", 0.48, PHONEME_SPECTRA["AE"]),
            StimulusSegment("T", 0.30, PHONEME_SPECTRA["T"]),
        ],
        crossfade_s=0.049,
    )


def ambiguous_onset_stimulus() -> StimulusProgram:
    ambiguous = _spectrum(*(
        0.5 * (PHONEME_SPECTRA["B"].array() + PHONEME_SPECTRA["P"].array())
    ))
    return StimulusProgram(
        "?AT",
        [
            StimulusSegment("B/P?", 0.22, ambiguous),
            StimulusSegment("AE", 0.48, PHONEME_SPECTRA["AE"]),
            StimulusSegment("T", 0.30, PHONEME_SPECTRA["T"]),
        ],
        crossfade_s=0.049,
    )


def phoneme_prototypes(labels: tuple[str, ...]) -> np.ndarray:
    return np.vstack([SpectralEncoder.encode(PHONEME_SPECTRA[label]) for label in labels])


def prototype_similarity(current: np.ndarray, prototypes: np.ndarray, sigma: float = 0.24) -> np.ndarray:
    current = np.asarray(current, dtype=np.float64)
    prototypes = np.asarray(prototypes, dtype=np.float64)
    if prototypes.ndim != 2 or current.shape != (prototypes.shape[1],):
        raise ValueError("current spectrum and prototype matrix have incompatible shapes")
    # Log-frequency bands are already coarse; keep the metric transparent and isotropic.
    distance2 = np.mean((prototypes - current[np.newaxis, :]) ** 2, axis=1)
    return np.exp(-distance2 / (2.0 * sigma**2))
