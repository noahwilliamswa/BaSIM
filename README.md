# BaSIM

**BaSIM** is an interactive explainer for layered attractor dynamics in speech perception. It is a fresh implementation built around continuous propagation rather than the older "settle one basin, then drop into the next" behavior.

The default demo presents a 1-second schematic `/B AE T/` acoustic event and then lets the model continue evolving for 5 seconds at ~49 ms per step:

`acoustic features → phoneme candidates → lexical candidates`

Every layer updates on every step. Candidate activation, lateral competition, recurrence, and adaptation reshape the displayed basin landscape while the stimulus is still entering the system. The lexical layer performs online sequence matching and can optionally send weak feedback to the phoneme layer.

This is intentionally a **toy explanatory model**, not a claim that the displayed 2D manifolds or hand-authored acoustic prototypes are literal neural geometry.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
python -m basim
```

or:

```bash
basim
```

## Test

```bash
pip install -e ".[dev]"
pytest
```

## Current model

- **49 ms** simulation step by default.
- **1.0 s** built-in acoustic event inside a **5.0 s** simulation.
- Dynamic acoustic-feature, phoneme-candidate, and lexical-candidate attractor layers.
- Continuous feed-forward propagation; no layer waits for the previous layer to settle.
- Recurrent support, lateral inhibition, and slow adaptation/fatigue.
- Online lexical sequence matching, so partially matching words remain active competitors.
- Optional lexical → phoneme feedback.
- PySide6 visualization of changing basin depths, state trajectory, and top activations.
- Alternate ambiguous-onset demo for exploring competition.

See [`docs/model.md`](docs/model.md) for the model contract, scientific boundaries, and background reading.

## Next logical steps

The clean extension path is to replace schematic feature frames with a real time-frequency front end, move phoneme/word prototypes into data/config files, add trial recording/export, and then compare feed-forward-only vs interactive feedback conditions under controlled ambiguous stimuli.
