# BaSIM

**BaSIM** is an interactive explainer for layered attractor dynamics in speech perception. It models a time-varying acoustic signal propagating continuously through three competitive dynamical layers:

`frequency bands → phoneme candidates → lexical candidates`

The default demo presents a synthetic 1-second `/B AE T/` event inside a 5-second simulation. The model advances in **49 ms** steps. Every layer updates on every step, connection delays are explicit, and state continues to evolve after the external sound ends.

BaSIM is intentionally a **toy explanatory model**. The frequency bands are coarse, the phoneme spectra are hand-authored, and the 2D basin geometry is a visualization of competition rather than a claim about literal cortical geometry.

## What is implemented

- Eight-band spectrotemporal input representation.
- Synthetic `/BAT/` and ambiguous `/B/P/AT/` stimuli with one-step spectral crossfades.
- Dynamic frequency-band, phoneme, and lexical attractor layers.
- Recurrent support, lateral inhibition, and slow adaptation/fatigue within each layer.
- Configurable **delayed, leaky connections** between layers instead of instantaneous hand-off.
- Fully soft online lexical sequence matching: word candidates accumulate mass across phoneme positions without hard phoneme commits.
- Optional delayed lexical → phoneme feedback.
- A PySide6 viewer with a live spectrogram, changing basin landscapes, and ranked candidate activations.
- Regression tests for timing, causal delay, propagation, ambiguity, adaptive basin depth, and lexical convergence.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
python main.py
```

The package entry points remain available too:

```bash
python -m basim
# or
basim
```

## Test

```bash
pip install -e ".[dev]"
pytest
```

## Current default timing

- simulation step: **49 ms**
- acoustic event: **1.0 s**
- simulation window: **5.0 s**
- frequency → phoneme delay: **49 ms**, with **98 ms** leaky integration
- phoneme → lexical delay: **49 ms**, with **147 ms** leaky integration
- lexical → phoneme feedback delay: **98 ms**, with **196 ms** leaky integration

See [`docs/model.md`](docs/model.md) for the model contract and the line between the explanatory mechanism and biological claims.

## Next useful experiments

The next model-facing work should be driven by comparisons rather than more UI surface area: vary connection delays and integration constants, compare feedback-on vs feed-forward-only ambiguous trials, add multiple word families, and replace the synthetic band sequence with an STFT/audio front end while preserving the same downstream contracts.
