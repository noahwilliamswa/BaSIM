# Model contract

## Clock and propagation

The default clock is 49 ms. At each tick:

1. Sample the external acoustic stimulus.
2. Convert that sample to graded evidence over acoustic attractors.
3. Integrate and update the acoustic attractor field.
4. Project the acoustic activation vector into phoneme evidence.
5. Integrate and update the phoneme attractor field.
6. Feed the current phoneme distribution into the cohort tracker.
7. Use the resulting word-candidate evidence to update the lexical attractor field.
8. Record all layer states, basin depths, activations, and inferred phoneme history.

This is a streaming pipeline. No layer waits for the preceding layer to "finish."

## Adaptive attractor layer

Each attractor has a fixed latent center and width, plus a time-varying depth. Incoming evidence is low-pass filtered:

`drive <- drive + alpha * (evidence - drive)`

Depth is then computed from baseline depth, current evidence trace, and a slow fatigue term:

`depth = base + evidence_gain * drive - fatigue_gain * fatigue`

The layer state moves down the resulting summed Gaussian potential. A direct pull toward the evidence-weighted center of compatible basins keeps input coupled to state while still allowing the potential landscape to matter.

The important behavior is that **the landscape itself changes while the stimulus is arriving**. The same state can therefore evolve differently at different moments because basin depths are history-dependent.

## Lexical sequence evidence

A static vector average would discard phoneme order, so lexical evidence is not produced by a simple matrix multiplication. `CohortTracker` watches the changing phoneme distribution, commits a phoneme after a short dwell interval, and reweights word candidates by prefix compatibility.

For the demo, `/k/` supports `cat`, `cap`, `cab`, and `cut`; `/æ/` reduces support for `cut`; `/t/` strongly favors `cat` over the remaining cohort.

This is intentionally a small, legible approximation of incremental lexical competition rather than a full TRACE, Shortlist, or neural speech-recognition implementation.

## What should change next

The next model-facing increments should be:

- replace the hand-authored synthetic feature trajectory with an explicit time-frequency stimulus representation;
- retain soft phoneme evidence without relying on a hard committed label for lexical sequencing;
- make inter-layer delay and integration constants configurable per connection;
- expose noise, adaptation, inhibition, and time constants in the UI;
- add recorded trajectories and scrubbable time playback before adding richer 3D visualization;
- add alternate lexical cohorts and ambiguity experiments so the model can be falsified against its own intended behavior.
