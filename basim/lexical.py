from __future__ import annotations

import math

import numpy as np


class OnlineLexicalMatcher:
    """Soft forward alignment from phoneme populations to competing word templates.

    Every word owns probability-like mass over prefix positions. Mass can remain at
    its current prefix or advance when the expected next phoneme is active. There
    are no hard phoneme commits, winner thresholds, or symbolic transition events.
    """

    def __init__(
        self,
        lexicon: dict[str, tuple[str, ...]],
        phoneme_labels: tuple[str, ...],
        dt_s: float = 0.049,
        transition_tau_s: float = 0.105,
        memory_tau_s: float = 1.35,
    ) -> None:
        self.words = tuple(lexicon)
        self.lexicon = lexicon
        self.phoneme_labels = phoneme_labels
        self.phoneme_index = {label: i for i, label in enumerate(phoneme_labels)}
        self.dt_s = float(dt_s)
        self.transition_tau_s = float(transition_tau_s)
        self.memory_tau_s = float(memory_tau_s)
        self.mass = {
            word: np.zeros(len(tokens) + 1, dtype=np.float64)
            for word, tokens in lexicon.items()
        }
        self.completion_trace = {word: 0.0 for word in lexicon}
        self.reset()

    def reset(self) -> None:
        for word, value in self.mass.items():
            value.fill(0.0)
            value[0] = 1.0
            self.completion_trace[word] = 0.0

    def step(self, phoneme_activation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        activation = np.clip(np.asarray(phoneme_activation, dtype=np.float64), 0.0, 1.0)
        if activation.shape != (len(self.phoneme_labels),):
            raise ValueError("phoneme activation has wrong shape")

        advance_alpha = 1.0 - math.exp(-self.dt_s / max(self.transition_tau_s, 1e-6))
        retain = math.exp(-self.dt_s / max(self.memory_tau_s, 1e-6))
        evidence = np.zeros(len(self.words), dtype=np.float64)
        feedback = np.zeros(len(self.phoneme_labels), dtype=np.float64)

        for word_i, word in enumerate(self.words):
            tokens = self.lexicon[word]
            old = self.mass[word]
            new = old * retain
            new[0] = 1.0

            # Update from the end so one phoneme population cannot skip multiple
            # template positions during a single 49 ms integration step.
            for prefix in range(len(tokens), 0, -1):
                token = tokens[prefix - 1]
                emission = float(activation[self.phoneme_index[token]])
                transition = old[prefix - 1] * advance_alpha * emission
                new[prefix] = np.clip(new[prefix] + transition, 0.0, 1.0)

            self.mass[word] = new
            completion = float(new[-1])
            self.completion_trace[word] = max(retain * self.completion_trace[word], completion)

            prefix_weights = np.linspace(0.0, 1.0, len(new))
            partial = float(np.max(new * prefix_weights))
            evidence[word_i] = np.clip(0.55 * partial + 1.35 * self.completion_trace[word], 0.0, 1.0)

            # Feedback is also soft: expected phonemes are weighted by how much
            # probability mass currently occupies the prefix immediately before them.
            for prefix, token in enumerate(tokens):
                feedback[self.phoneme_index[token]] += new[prefix] * (0.35 + 0.65 * evidence[word_i])

        if np.max(feedback, initial=0.0) > 0.0:
            feedback /= np.max(feedback)
        return evidence, feedback
