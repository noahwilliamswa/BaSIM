from __future__ import annotations

import math

import numpy as np

from .model import (
    AcousticSegment,
    AdaptiveAttractorLayer,
    AttractorSpec,
    CohortTracker,
    LayerConfig,
    SimulationBundle,
    SpeechAttractorSimulation,
    StimulusProgram,
    WordCandidate,
)


FREQUENCY_BANDS_HZ = (250, 500, 1000, 2000, 3000, 4000, 6000, 8000)


def _ring_points(n: int, radius: float = 1.7, phase: float = 0.0):
    return [
        (
            radius * math.cos(phase + 2.0 * math.pi * i / n),
            radius * math.sin(phase + 2.0 * math.pi * i / n),
        )
        for i in range(n)
    ]


def cat_demo(dt_ms: float = 49.0, total_ms: float = 5000.0) -> SimulationBundle:
    """Build a deterministic synthetic /k ae t/ spoken-word competition demo."""

    acoustic_specs = (
        AttractorSpec("burst", (-1.45, 0.65), 0.95, 0.72),
        AttractorSpec("front-vowel", (0.0, 1.55), 0.95, 0.78),
        AttractorSpec("central-vowel", (0.2, -1.45), 0.92, 0.82),
        AttractorSpec("coronal-release", (1.5, 0.6), 0.95, 0.70),
        AttractorSpec("voiced-low", (1.3, -0.9), 0.88, 0.75),
    )
    acoustic_layer = AdaptiveAttractorLayer(
        LayerConfig(
            name="Acoustic feature field",
            attractors=acoustic_specs,
            evidence_gain=2.8,
            evidence_tau_ms=105.0,
            state_tau_ms=115.0,
            fatigue_gain=0.25,
            input_pull=1.35,
            temperature=0.48,
            activation_state_weight=0.24,
            activation_drive_weight=2.15,
        ),
        seed=1,
    )

    # Rows align with acoustic_specs; columns align with FREQUENCY_BANDS_HZ.
    # These are synthetic spectral envelopes chosen for legible competition,
    # not measured human speech spectra.
    acoustic_templates = np.asarray(
        [
            [0.05, 0.08, 0.18, 0.55, 1.00, 0.82, 0.25, 0.10],
            [0.28, 0.72, 1.00, 0.80, 0.42, 0.20, 0.08, 0.03],
            [0.70, 1.00, 0.76, 0.36, 0.18, 0.09, 0.04, 0.02],
            [0.02, 0.03, 0.05, 0.10, 0.22, 0.45, 0.90, 1.00],
            [1.00, 0.82, 0.50, 0.24, 0.12, 0.07, 0.03, 0.02],
        ],
        dtype=float,
    )

    phoneme_labels = ("K", "AE", "T", "P", "B", "AH")
    phoneme_specs = tuple(
        AttractorSpec(label, center, 1.0, 0.72)
        for label, center in zip(
            phoneme_labels,
            _ring_points(len(phoneme_labels), 1.8, 0.18),
        )
    )
    phoneme_layer = AdaptiveAttractorLayer(
        LayerConfig(
            name="Phoneme field",
            attractors=phoneme_specs,
            evidence_gain=3.2,
            evidence_tau_ms=145.0,
            state_tau_ms=135.0,
            fatigue_gain=0.52,
            fatigue_tau_ms=650.0,
            input_pull=1.05,
            temperature=0.46,
            activation_state_weight=0.30,
            activation_drive_weight=2.00,
        ),
        seed=2,
    )

    candidates = (
        WordCandidate("cat", ("K", "AE", "T"), 1.00),
        WordCandidate("cap", ("K", "AE", "P"), 0.92),
        WordCandidate("cab", ("K", "AE", "B"), 0.88),
        WordCandidate("cut", ("K", "AH", "T"), 0.95),
        WordCandidate("bat", ("B", "AE", "T"), 0.90),
    )
    lexical_specs = tuple(
        AttractorSpec(candidate.word, center, 0.65, 1.20)
        for candidate, center in zip(
            candidates,
            _ring_points(len(candidates), 1.7, -0.35),
        )
    )
    lexical_layer = AdaptiveAttractorLayer(
        LayerConfig(
            name="Lexical field",
            attractors=lexical_specs,
            evidence_gain=3.8,
            evidence_tau_ms=330.0,
            state_tau_ms=390.0,
            fatigue_gain=0.16,
            fatigue_tau_ms=1400.0,
            input_pull=0.35,
            temperature=0.72,
            activation_state_weight=0.42,
            activation_drive_weight=1.65,
        ),
        seed=3,
    )

    # Rows: K, AE, T, P, B, AH. Columns: acoustic feature attractors above.
    acoustic_to_phoneme = np.asarray(
        [
            [1.00, 0.06, 0.03, 0.35, 0.08],
            [0.04, 1.00, 0.28, 0.02, 0.04],
            [0.22, 0.03, 0.02, 1.00, 0.12],
            [0.35, 0.03, 0.02, 0.82, 0.06],
            [0.10, 0.04, 0.08, 0.45, 1.00],
            [0.03, 0.30, 1.00, 0.03, 0.06],
        ],
        dtype=float,
    )

    burst, front_vowel, _, coronal_release, _ = acoustic_templates
    stimulus = StimulusProgram(
        name="CAT",
        duration_ms=1000.0,
        edge_ms=55.0,
        segments=(
            AcousticSegment("/k/", 0.0, 245.0, tuple(burst)),
            AcousticSegment("/ae/", 245.0, 690.0, tuple(front_vowel)),
            AcousticSegment("/t/", 690.0, 1000.0, tuple(coronal_release)),
        ),
    )

    cohort = CohortTracker(
        phoneme_labels=phoneme_labels,
        candidates=candidates,
        min_confidence=0.36,
        min_dwell_ms=98.0,
        change_threshold=0.30,
    )

    simulation = SpeechAttractorSimulation(
        stimulus=stimulus,
        acoustic=acoustic_layer,
        phoneme=phoneme_layer,
        lexical=lexical_layer,
        acoustic_templates=acoustic_templates,
        acoustic_to_phoneme=acoustic_to_phoneme,
        cohort=cohort,
        dt_ms=dt_ms,
        total_ms=total_ms,
    )
    return SimulationBundle(
        simulation=simulation,
        metadata={
            "stimulus": "synthetic 8-band /k ae t/ spectral trajectory",
            "frequency_bands_hz": ",".join(str(v) for v in FREQUENCY_BANDS_HZ),
            "interpretation": (
                "explanatory attractor dynamics; coordinates are latent, not anatomical"
            ),
        },
    )
