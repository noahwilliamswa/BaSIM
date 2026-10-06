from __future__ import annotations

import numpy as np

from .attractor import CompetitiveAttractorLayer
from .lexical import OnlineLexicalMatcher
from .stimuli import AcousticEncoder, LEXICON, PHONEME_FRAMES, default_stimulus, phoneme_prototypes, prototype_similarity
from .types import SimulationConfig, SimulationSnapshot, StimulusProgram


class BaSimEngine:
    """Continuous acoustic-features → phonemes → words simulation."""

    def __init__(self, config: SimulationConfig | None = None, stimulus: StimulusProgram | None = None) -> None:
        self.config = config or SimulationConfig()
        if self.config.dt_s <= 0 or self.config.duration_s <= 0:
            raise ValueError("dt_s and duration_s must be > 0")
        self.stimulus = stimulus or default_stimulus()
        dt = self.config.dt_s
        self.feature = CompetitiveAttractorLayer(
            "Acoustic features", AcousticEncoder.labels, dt_s=dt, tau_s=0.055, fatigue_tau_s=0.40,
            evidence_gain=1.50, self_excitation=0.18, lateral_inhibition=0.08, fatigue_gain=0.12,
            basin_sigma=0.24, activation_threshold=0.50, activation_slope=5.0,
        )
        self.phoneme_labels = tuple(PHONEME_FRAMES)
        self.phoneme = CompetitiveAttractorLayer(
            "Phoneme candidates", self.phoneme_labels, dt_s=dt, tau_s=0.135, fatigue_tau_s=0.75,
            evidence_gain=1.48, self_excitation=0.72, lateral_inhibition=0.56, basin_sigma=0.23,
        )
        self.lexical_labels = tuple(LEXICON)
        self.lexical = CompetitiveAttractorLayer(
            "Lexical candidates", self.lexical_labels, dt_s=dt, tau_s=0.190, fatigue_tau_s=1.15,
            evidence_gain=1.64, self_excitation=0.82, lateral_inhibition=0.63, fatigue_gain=0.36,
            basin_sigma=0.25, activation_threshold=0.52, activation_slope=5.4,
        )
        self.matcher = OnlineLexicalMatcher(LEXICON, self.phoneme_labels)
        self.prototypes = phoneme_prototypes(self.phoneme_labels)
        self.lexical_feedback = np.zeros(len(self.phoneme_labels))
        self.time_s = 0.0
        self.history: list[SimulationSnapshot] = []

    @property
    def done(self) -> bool:
        return self.time_s >= self.config.duration_s - 1e-12

    def reset(self, stimulus: StimulusProgram | None = None) -> None:
        if stimulus is not None:
            self.stimulus = stimulus
        self.feature.reset(); self.phoneme.reset(); self.lexical.reset(); self.matcher.reset()
        self.lexical_feedback.fill(0.0)
        self.time_s = 0.0
        self.history.clear()

    def step(self) -> SimulationSnapshot:
        if self.done:
            return self.snapshot()
        label, frame = self.stimulus.frame_at(self.time_s)
        feature_snapshot = self.feature.step(AcousticEncoder.encode(frame))

        if frame.amplitude > 0.0 or np.max(self.feature.activation, initial=0.0) > 0.05:
            bottom_up = prototype_similarity(self.feature.activation, self.prototypes)
            bottom_up *= float(np.clip(np.max(self.feature.activation) * 1.35, 0.0, 1.0))
        else:
            bottom_up = np.zeros(len(self.phoneme_labels))

        phoneme_evidence = np.clip(
            bottom_up + self.config.lexical_feedback_gain * self.lexical_feedback, 0.0, 1.0
        )
        phoneme_snapshot = self.phoneme.step(phoneme_evidence)
        lexical_evidence, feedback_basis = self.matcher.step(self.phoneme.activation)
        lexical_snapshot = self.lexical.step(lexical_evidence)
        self.lexical_feedback = feedback_basis * float(np.max(self.lexical.activation, initial=0.0))

        snapshot = SimulationSnapshot(
            self.time_s, label, frame, feature_snapshot, phoneme_snapshot, lexical_snapshot
        )
        self.history.append(snapshot)
        self.time_s = min(self.config.duration_s, self.time_s + self.config.dt_s)
        return snapshot

    def snapshot(self) -> SimulationSnapshot:
        label, frame = self.stimulus.frame_at(self.time_s)
        return SimulationSnapshot(
            self.time_s, label, frame, self.feature.snapshot(), self.phoneme.snapshot(), self.lexical.snapshot()
        )

    def run(self) -> list[SimulationSnapshot]:
        while not self.done:
            self.step()
        return list(self.history)

    def top_candidates(self, layer: str, n: int = 3) -> list[tuple[str, float]]:
        target = {"feature": self.feature, "phoneme": self.phoneme, "lexical": self.lexical}[layer]
        order = np.argsort(target.activation)[::-1][: max(0, n)]
        return [(target.labels[int(i)], float(target.activation[int(i)])) for i in order]
