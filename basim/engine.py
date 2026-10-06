from __future__ import annotations

import numpy as np

from .attractor import CompetitiveAttractorLayer
from .connections import LeakyDelayConnection
from .lexical import OnlineLexicalMatcher
from .stimuli import LEXICON, PHONEME_SPECTRA, SpectralEncoder, default_stimulus, phoneme_prototypes, prototype_similarity
from .domain import SimulationConfig, SimulationSnapshot, StimulusProgram


class BaSimEngine:
    """Continuous spectrum → phonemes → words simulation with delayed leaky coupling."""

    def __init__(self, config: SimulationConfig | None = None, stimulus: StimulusProgram | None = None) -> None:
        self.config = config or SimulationConfig()
        if self.config.dt_s <= 0 or self.config.duration_s <= 0:
            raise ValueError("dt_s and duration_s must be > 0")
        self.stimulus = stimulus or default_stimulus()
        dt = self.config.dt_s

        self.feature = CompetitiveAttractorLayer(
            "Frequency bands", SpectralEncoder.labels, dt_s=dt, tau_s=0.060, fatigue_tau_s=0.42,
            evidence_gain=1.55, self_excitation=0.16, lateral_inhibition=0.06, fatigue_gain=0.10,
            basin_sigma=0.24, activation_threshold=0.48, activation_slope=5.0,
        )
        self.phoneme_labels = tuple(PHONEME_SPECTRA)
        self.phoneme = CompetitiveAttractorLayer(
            "Phoneme candidates", self.phoneme_labels, dt_s=dt, tau_s=0.135, fatigue_tau_s=0.75,
            evidence_gain=1.52, self_excitation=0.68, lateral_inhibition=0.52, fatigue_gain=0.40,
            basin_sigma=0.23, activation_threshold=0.68, activation_slope=5.8,
        )
        self.lexical_labels = tuple(LEXICON)
        self.lexical = CompetitiveAttractorLayer(
            "Lexical candidates", self.lexical_labels, dt_s=dt, tau_s=0.205, fatigue_tau_s=1.20,
            evidence_gain=1.72, self_excitation=0.78, lateral_inhibition=0.58, fatigue_gain=0.30,
            basin_sigma=0.25, activation_threshold=0.50, activation_slope=5.2,
        )

        self.matcher = OnlineLexicalMatcher(LEXICON, self.phoneme_labels, dt_s=dt)
        self.prototypes = phoneme_prototypes(self.phoneme_labels)
        self.feature_to_phoneme = LeakyDelayConnection(
            len(self.feature.labels), dt, self.config.feature_to_phoneme
        )
        self.phoneme_to_lexical = LeakyDelayConnection(
            len(self.phoneme_labels), dt, self.config.phoneme_to_lexical
        )
        self.lexical_to_phoneme = LeakyDelayConnection(
            len(self.phoneme_labels), dt, self.config.lexical_to_phoneme
        )
        self.feedback_basis = np.zeros(len(self.phoneme_labels), dtype=np.float64)
        self.time_s = 0.0
        self.history: list[SimulationSnapshot] = []

    @property
    def done(self) -> bool:
        return self.time_s >= self.config.duration_s - 1e-12

    def reset(self, stimulus: StimulusProgram | None = None) -> None:
        if stimulus is not None:
            self.stimulus = stimulus
        self.feature.reset()
        self.phoneme.reset()
        self.lexical.reset()
        self.matcher.reset()
        self.feature_to_phoneme.reset()
        self.phoneme_to_lexical.reset()
        self.lexical_to_phoneme.reset()
        self.feedback_basis.fill(0.0)
        self.time_s = 0.0
        self.history.clear()

    def step(self) -> SimulationSnapshot:
        if self.done:
            return self.snapshot()

        label, frame = self.stimulus.frame_at(self.time_s)
        spectrum = SpectralEncoder.encode(frame)
        feature_snapshot = self.feature.step(spectrum)

        delayed_features = self.feature_to_phoneme.step(self.feature.activation)
        feature_strength = float(np.max(delayed_features, initial=0.0))
        if feature_strength > 0.015:
            bottom_up = prototype_similarity(delayed_features, self.prototypes) ** 3.0
            if np.max(bottom_up, initial=0.0) > 0.0:
                bottom_up /= np.max(bottom_up)
            bottom_up *= float(np.clip(feature_strength * 1.45, 0.0, 1.0))
        else:
            bottom_up = np.zeros(len(self.phoneme_labels), dtype=np.float64)

        feedback = self.lexical_to_phoneme.step(self.feedback_basis)
        phoneme_evidence = np.clip(
            bottom_up + self.config.lexical_feedback_gain * feedback,
            0.0,
            1.0,
        )
        phoneme_snapshot = self.phoneme.step(phoneme_evidence)

        delayed_phonemes = self.phoneme_to_lexical.step(self.phoneme.activation)
        phoneme_strength = float(np.max(delayed_phonemes, initial=0.0))
        phoneme_mass = np.clip(delayed_phonemes, 0.0, None) ** 2.0
        if float(np.sum(phoneme_mass)) > 1e-9:
            phoneme_mass = (phoneme_mass / float(np.sum(phoneme_mass))) * phoneme_strength
        lexical_evidence, feedback_basis = self.matcher.step(phoneme_mass)
        lexical_snapshot = self.lexical.step(lexical_evidence)
        lexical_strength = float(np.max(self.lexical.activation, initial=0.0))
        self.feedback_basis = feedback_basis * lexical_strength

        snapshot = SimulationSnapshot(
            time_s=self.time_s,
            stimulus_label=label,
            stimulus_frame=frame,
            spectrum=spectrum.copy(),
            feature=feature_snapshot,
            phoneme=phoneme_snapshot,
            lexical=lexical_snapshot,
        )
        self.history.append(snapshot)
        self.time_s = min(self.config.duration_s, self.time_s + self.config.dt_s)
        return snapshot

    def snapshot(self) -> SimulationSnapshot:
        label, frame = self.stimulus.frame_at(self.time_s)
        spectrum = SpectralEncoder.encode(frame)
        return SimulationSnapshot(
            self.time_s,
            label,
            frame,
            spectrum,
            self.feature.snapshot(),
            self.phoneme.snapshot(),
            self.lexical.snapshot(),
        )

    def run(self) -> list[SimulationSnapshot]:
        while not self.done:
            self.step()
        return list(self.history)

    def top_candidates(self, layer: str, n: int = 3) -> list[tuple[str, float]]:
        target = {"feature": self.feature, "phoneme": self.phoneme, "lexical": self.lexical}[layer]
        order = np.argsort(target.activation)[::-1][: max(0, n)]
        return [(target.labels[int(i)], float(target.activation[int(i)])) for i in order]
