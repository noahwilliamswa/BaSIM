from __future__ import annotations

import numpy as np


class OnlineLexicalMatcher:
    """Soft online sequence alignment over competing lexical templates.

    Each word keeps distributed progress mass over its token positions.
    Evidence can persist at a token, or advance one token when the next
    phoneme becomes active. Nothing in the matcher requires a hard phoneme
    decision or a single winning path.
    """

    def __init__(
        self,
        lexicon: dict[str, tuple[str, ...]],
        phoneme_labels: tuple[str, ...],
        reference_dt_s: float = 0.049,
    ) -> None:
        self.words = tuple(lexicon)
        self.lexicon = lexicon
        self.phoneme_labels = phoneme_labels
        self.phoneme_index = {label: i for i, label in enumerate(phoneme_labels)}
        self.reference_dt_s = float(reference_dt_s)
        self.mass = {
            word: np.zeros(len(tokens), dtype=np.float64)
            for word, tokens in lexicon.items()
        }
        self.completion_trace = {word: 0.0 for word in lexicon}
        self.reset()

    def reset(self) -> None:
        for word, value in self.mass.items():
            value.fill(0.0)
            self.completion_trace[word] = 0.0

    def step(
        self,
        phoneme_activation: np.ndarray,
        dt_s: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        activation = np.clip(
            np.asarray(phoneme_activation, dtype=np.float64),
            0.0,
            1.0,
        )
        if activation.shape != (len(self.phoneme_labels),):
            raise ValueError("phoneme activation has wrong shape")

        dt = self.reference_dt_s if dt_s is None else float(dt_s)
        hold_decay = float(np.exp(-dt / 0.55))
        completion_decay = float(np.exp(-dt / 7.0))
        transition_gain = 1.35 * max(
            dt / max(self.reference_dt_s, 1e-9),
            1e-6,
        )

        evidence = np.zeros(len(self.words), dtype=np.float64)
        feedback = np.zeros(len(self.phoneme_labels), dtype=np.float64)

        for word_i, word in enumerate(self.words):
            tokens = self.lexicon[word]
            indices = [self.phoneme_index[token] for token in tokens]
            emissions = np.asarray([activation[index] for index in indices])
            old = self.mass[word]
            new = np.zeros_like(old)

            new[0] = max(hold_decay * old[0], float(emissions[0]))

            for j in range(1, len(tokens)):
                stay = hold_decay * old[j]
                emission = float(emissions[j])
                advance = (
                    old[j - 1]
                    * (emission ** 2.6)
                    * transition_gain
                )
                new[j] = max(stay, advance)

            new = np.clip(new, 0.0, 1.0)
            self.mass[word] = new

            completion_now = float(new[-1]) if new.size else 0.0
            self.completion_trace[word] = max(
                completion_decay * self.completion_trace[word],
                completion_now,
            )

            prefix = float(np.max(new)) if new.size else 0.0
            evidence[word_i] = np.clip(
                0.03 * prefix + 100.0 * self.completion_trace[word],
                0.0,
                1.0,
            )

            for j, index in enumerate(indices):
                feedback[index] += new[j]

        max_feedback = float(np.max(feedback, initial=0.0))
        if max_feedback > 0.0:
            feedback /= max_feedback
        return evidence, feedback
