# BaSIM

BaSIM is a lightweight interactive explainer for **time-varying, layered attractor dynamics** in spoken-word recognition.

The current refactor models a synthetic one-second acoustic stimulus flowing through three continuously updating representational layers:

1. **Acoustic feature field** — a low-dimensional feature trajectory deforms a set of acoustic attractors.
2. **Phoneme field** — acoustic activations continuously bias competing phoneme attractors.
3. **Lexical field** — a soft cohort tracker turns the evolving phoneme distribution into word-candidate evidence, which deforms a final competitive attractor field.

The simulation runs for five seconds at **49 ms per integration step**. The external sound is present for the first second, but all layers retain state and continue evolving afterward. Each layer emits a representation on every tick, so downstream layers integrate a changing upstream state rather than receiving one final settled point.

## Scientific boundary

BaSIM is an explanatory dynamical model, not a biological reconstruction of auditory cortex. The 2D coordinates are latent visualization coordinates. Basin depth, position, fatigue, and deformation are deliberately inspectable abstractions for reasoning about competition, temporal integration, persistence, and changing candidate structure.

The design is inspired by broad properties of spoken-word-recognition models such as continuous graded activation, parallel candidate competition, incremental evidence accumulation, and cohort change over time. It does **not** claim that phonemes or lexical items literally occupy Gaussian wells in neural tissue.

## Current demo

The bundled preset presents a synthetic `/k æ t/` trajectory and a small lexical cohort:

- `cat`
- `cap`
- `cab`
- `cut`
- `bat`

Early in the signal, multiple onset-compatible candidates remain viable. As later phonemic evidence arrives, the candidate set is progressively reweighted and the lexical field settles toward `cat`.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
python -m basim
```

Run tests with:

```bash
pip install -e ".[dev]"
pytest
```

## Architecture

`src/basim/model.py` contains the simulation contract and no Qt code. `AdaptiveAttractorLayer` owns leaky evidence integration, adaptive basin depth, occupancy fatigue, field dynamics, and activation readout. `CohortTracker` is a deliberately separate transformation between phoneme and lexical representations so that sequence-sensitive evidence is not hidden inside the geometry.

`src/basim/presets.py` defines the current synthetic stimulus, layer geometries, acoustic-to-phoneme projection, and lexical cohort. `src/basim/ui.py` is only a viewer/controller over the core model.

This separation is intentional: future BaSIM experiments should be able to replace the evidence mapping or dynamics without rewriting the interface.
