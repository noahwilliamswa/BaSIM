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


def _ring_points(n: int, radius: float = 1.7, phase: float = 0.0):
    return [
        (
            radius * math.cos(phase + 2.0 * math.pi * i / n),
            radius * math.sin(phase + 2.0 * math.pi * i / n),
        )
        for i in range(n)
    ]


def cat_demo(dt_ms: float = 49.0, total_ms: float = 5000.0) -> SimulationBundle:
    """Build a deterministic /k ae t/ spoken-word competition demo.

    Coordinates are explanatory latent coordinates, not cortical locations.
    """

    acoustic_specs = (
        AttractorSpec("burst", (-1.45, 0.65), 0.95, 0.72),
        AttractorSpec("vowel-front", (0.0, 1.55), 0.95, 0.78),
        AttractorSpec("vowel-central", (0.2, -1.45), 0.92, 0.82),
        AttractorSpec("closure", (1.5, 0.6), 0.95, 0.70),
        AttractorSpec("voiced-stop", (1.3, -0.9), 0.88, 0.75),
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
            temperature=0.42,
        ),
        seed=1,
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
            temperature=0.34,
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
            [1.00, 0.06, 0.03, 0.35, 0.08],  # K
            [0.04, 1.00, 0.28, 0.02, 0.04],  # AE
            [0.22, 0.03, 0.02, 1.00, 0.12],  # T
            [0.35, 0.03, 0.02, 0.82, 0.06],  # P
            [0.10, 0.04, 0.08, 0.45, 1.00],  # B
            [0.03, 0.30, 1.00, 0.03, 0.06],  # AH
        ],
        dtype=float,
    )

    stimulus = StimulusProgram(
        name="CAT",
        duration_ms=1000.0,
        edge_ms=55.0,
        segments=(
            AcousticSegment("/k/", 0.0, 245.0, (-1.42, 0.63)),
            AcousticSegment("/ae/", 245.0, 690.0, (0.0, 1.54)),
            AcousticSegment("/t/", 690.0, 1000.0, (1.48, 0.62)),
        ),
    )

    cohort = CohortTracker(
        phoneme_labels=phoneme_labels,
        candidates=candidates,
        min_confidence=0.38,
        min_dwell_ms=98.0,
    )

    simulation = SpeechAttractorSimulation(
        stimulus=stimulus,
        acoustic=acoustic_layer,
        phoneme=phoneme_layer,
        lexical=lexical_layer,
        acoustic_to_phoneme=acoustic_to_phoneme,
        cohort=cohort,
        dt_ms=dt_ms,
        total_ms=total_ms,
    )
    return SimulationBundle(
        simulation=simulation,
        metadata={
            "stimulus": "synthetic /k ae t/ feature trajectory",
            "interpretation": (
                "explanatory attractor dynamics; coordinates are latent, not anatomical"
            ),
        },
    )
