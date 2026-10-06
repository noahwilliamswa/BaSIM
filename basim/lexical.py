from __future__ import annotations

import numpy as np


class OnlineLexicalMatcher:
    """Continuously align phoneme activations with competing word templates."""

    def __init__(self, lexicon: dict[str, tuple[str, ...]], phoneme_labels: tuple[str, ...]) -> None:
        self.words = tuple(lexicon)
        self.lexicon = lexicon
        self.phoneme_labels = phoneme_labels
        self.phoneme_index = {label: i for i, label in enumerate(phoneme_labels)}
        self.mass = {word: np.zeros(len(tokens), dtype=np.float64) for word, tokens in lexicon.items()}
        self.started = {word: False for word in lexicon}
        self.completion_trace = {word: 0.0 for word in lexicon}
        self.previous_phoneme = np.zeros(len(phoneme_labels), dtype=np.float64)

    def reset(self) -> None:
        for word, value in self.mass.items():
            value.fill(0.0)
            self.started[word] = False
            self.completion_trace[word] = 0.0
        self.previous_phoneme.fill(0.0)

    def step(self, phoneme_activation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        novelty = np.clip(phoneme_activation - 0.90 * self.previous_phoneme, 0.0, 1.0)
        evidence = np.zeros(len(self.words), dtype=np.float64)
        feedback = np.zeros(len(self.phoneme_labels), dtype=np.float64)

        for word_i, word in enumerate(self.words):
            tokens = self.lexicon[word]
            old = self.mass[word]
            new = np.zeros_like(old)
            indices = [self.phoneme_index[token] for token in tokens]
            emissions = np.array([phoneme_activation[i] for i in indices])
            changes = np.array([novelty[i] for i in indices])

            start = 0.0
            if not self.started[word] and changes[0] > 0.015 and emissions[0] > 0.12:
                start = changes[0] * emissions[0]
                if start > 0.004:
                    self.started[word] = True

            new[0] = max(0.985 * old[0] * (0.78 + 0.22 * emissions[0]), start)
            for j in range(1, len(tokens)):
                stay = 0.985 * old[j] * (0.80 + 0.20 * emissions[j])
                transition = old[j - 1] * (0.18 * emissions[j] + 2.25 * changes[j])
                new[j] = max(stay, transition)

            self.mass[word] = np.clip(new, 0.0, 1.0)
            prefix = float(np.max(new)) if new.size else 0.0
            completion_now = float(new[-1]) if new.size else 0.0
            self.completion_trace[word] = max(0.992 * self.completion_trace[word], completion_now)
            evidence[word_i] = np.clip(prefix + 2.40 * self.completion_trace[word], 0.0, 1.0)
            for j, index in enumerate(indices):
                feedback[index] += new[j]

        if np.max(feedback, initial=0.0) > 0.0:
            feedback /= np.max(feedback)
        self.previous_phoneme = phoneme_activation.copy()
        return evidence, feedback
