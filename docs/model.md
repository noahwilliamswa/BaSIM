# Model contract

## Clock and propagation

The default clock is 49 ms. At each tick:

1. Sample the external time-frequency stimulus.
2. Compare its spectral profile with acoustic feature templates to produce graded evidence over acoustic attractors.
3. Integrate and update the acoustic attractor field.
4. Project the acoustic activation vector into phoneme evidence.
5. Integrate and update the phoneme attractor field.
6. Feed the current **soft phoneme distribution** into the cohort tracker.
7. Use the resulting word-candidate evidence to update the lexical attractor field.
8. Record all layer states, basin depths, activations, spectrum, and inferred phoneme-event history.

This is a streaming pipeline. No layer waits for the preceding layer to "finish."

## Acoustic representation

The demo stimulus is an eight-band spectral energy vector at 250, 500, 1000, 2000, 3000, 4000, 6000, and 8000 Hz. The `/k/`, `/æ/`, and `/t/` segments are synthetic spectral envelopes, not recordings and not fitted auditory-nerve responses. Crossfades make the acoustic evidence continuous across segment boundaries.

Acoustic attractors still live in a 2D latent plotting space, but **the input no longer lives in that space**. Spectral similarity drives the attractor field, which keeps visualization geometry separate from the actual stimulus representation.

## Adaptive attractor layer

Each attractor has a fixed latent center and width, plus a time-varying depth. Incoming evidence is low-pass filtered:

`drive <- drive + alpha * (evidence - drive)`

Depth is then computed from baseline depth, current evidence trace, and a slow fatigue term:

`depth = base + evidence_gain * drive - fatigue_gain * fatigue`

The layer state moves down the resulting summed Gaussian potential. A direct pull toward the evidence-weighted center of compatible basins keeps input coupled to state while still allowing the potential landscape to matter.

The important behavior is that **the landscape itself changes while the stimulus is arriving**. The same state can therefore evolve differently at different moments because basin depths are history-dependent.

## Lexical sequence evidence

A static vector average would discard phoneme order. `CohortTracker` therefore detects stable changes in the phoneme distribution and stores each event as the **entire probability vector**, not just its winning label. Candidate words are scored by how much probability mass the event assigns to the phoneme expected at that sequence position. Argmax labels are retained only for display.

For the demo, early `/k/` evidence leaves `cat`, `cap`, `cab`, and `cut` viable; the soft `/æ/` event suppresses `cut`; the final `/t/` event strongly favors `cat`.

This is intentionally a small, legible approximation of incremental lexical competition rather than a full TRACE, Shortlist, or neural speech-recognition implementation.

## What should change next

The next model-facing increments should be:

- accept a WAV file and compute the band-energy trajectory from audio rather than using a hand-authored synthetic spectrum;
- make inter-layer transmission delay and integration constants configurable per connection;
- expose noise, adaptation, inhibition, and time constants in the UI;
- add recorded trajectories and scrubbable time playback before richer 3D visualization;
- add ambiguous continua and alternate lexical cohorts so the model can be tested against explicit expected behaviors;
- separate short-lived stimulus adaptation from longer-lived learning/plasticity if learning is added later.
