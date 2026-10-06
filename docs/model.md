# BaSIM model contract

BaSIM is an explanatory dynamical-systems toy, not a biologically faithful auditory or language model. Its purpose is to make continuous competition, persistence, and cross-level influence visible.

## Time

- Default simulation step: **49 ms**.
- Default acoustic event: **1.0 s**.
- Default simulation window: **5.0 s**.
- Layers update on every simulation step. A layer does **not** wait to settle before the next layer receives information.
- State persists after the acoustic event and decays/adapts rather than disappearing immediately.

## Three manifolds

### 1. Acoustic-feature manifold

Input is a small normalized feature vector: two formant-ish dimensions plus voicing, burst, frication, and amplitude. The displayed basins correspond to feature channels, not literal cortical columns or measured neural populations.

### 2. Phoneme-candidate manifold

Phoneme candidates receive similarity-weighted support from the feature layer. Candidates compete laterally, retain activation recurrently, and accumulate fatigue. These terms continuously alter basin depth, so the landscape changes while the stimulus is arriving.

### 3. Lexical-candidate manifold

Word candidates are matched online against the evolving phoneme population. Each candidate carries probability-like mass across positions in its phoneme template. A rise in the expected next phoneme advances that mass. Nearby word candidates therefore become partially active in parallel instead of waiting for a hard phoneme decision.

A weak optional lexical feedback signal is returned to the phoneme layer on the next step. This is included to expose the consequences of an interactive architecture, not to claim that lexical feedback is the uniquely correct account of human speech perception.

## Attractor dynamics

For each layer, candidate activation is updated from:

- current bottom-up evidence,
- recurrent self-support,
- lateral inhibition from competitors,
- slow adaptation/fatigue.

Candidate activation changes the depth of its 2D Gaussian basin. A visible state point moves downhill through the resulting landscape with inertia. The 2D position is a visualization of competition; it is not asserted to be a recovered neural state-space coordinate.

## Why this architecture

The design borrows a few qualitative commitments from interactive-activation accounts of spoken-word recognition: acoustic/feature information unfolds over time; phoneme and lexical candidates can be partially active simultaneously; within-level candidates compete; activation persists after transient input; and feedback can be explored as a parameter rather than baked in as a truth claim.

Useful background:

- McClelland & Elman-style TRACE architecture summarized in: https://pmc.ncbi.nlm.nih.gov/articles/PMC3759031/
- Continuous lexical competition in real-time spoken-word recognition: https://pmc.ncbi.nlm.nih.gov/articles/PMC1177386/
- Lexically guided perceptual tuning / interactive Hebbian account: https://pmc.ncbi.nlm.nih.gov/articles/PMC2291357/
- Neural evidence for context-dependent warping during speech categorization: https://pmc.ncbi.nlm.nih.gov/articles/PMC8984957/

## Explicit non-claims

BaSIM currently does not model cochlear mechanics, a real spectrogram, spike trains, cortical anatomy, synaptic plasticity, measured attractor geometry, realistic phonetic acoustics, or a production-quality recognizer. The built-in phoneme prototypes are deliberately schematic. They exist to make the system inspectable before real data and more constrained models are connected later.
