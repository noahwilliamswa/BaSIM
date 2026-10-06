# BaSIM model contract

BaSIM is an explanatory dynamical-systems model for reasoning about incremental speech integration. It is not a biologically faithful auditory-cortex simulation and it is not intended to be a production speech recognizer.

## Clock and causal propagation

The default integration step is **49 ms**. A built-in speech event lasts **1.0 s**, while the model continues through a **5.0 s** window.

At every tick:

1. sample the current spectrum;
2. update the frequency-band attractor layer;
3. send that population through a delayed, leaky connection;
4. compute graded phoneme evidence and update the phoneme layer;
5. send the phoneme population through another delayed, leaky connection;
6. update soft lexical sequence evidence and the lexical attractor layer;
7. send optional lexical feedback through its own delayed, leaky connection;
8. record the full state.

No layer waits for another layer to settle. Equally important, no downstream layer receives information instantaneously: inter-layer delay and integration are explicit model objects.

### Default connections

| connection | delay | integration time constant |
| --- | ---: | ---: |
| frequency → phoneme | 49 ms | 98 ms |
| phoneme → lexical | 49 ms | 147 ms |
| lexical → phoneme | 98 ms | 196 ms |

`LeakyDelayConnection` first delays the source population by an integer number of model ticks and then applies a first-order low-pass filter. These values are parameters for experiments, not measured biological constants.

## Input: time × frequency

The native input is now a `SpectrumFrame`: energy in eight coarse frequency bands centered at 250, 500, 750, 1000, 1500, 2000, 3000, and 4500 Hz. A `StimulusProgram` supplies one frame at each time point and crossfades adjacent synthetic segments over 49 ms.

The built-in phoneme spectra are hand-authored templates designed to create controlled ambiguity and separability. They are not empirical average spectra. The older `AcousticFrame` type is retained only as a compatibility input and is mapped into the fixed-band representation.

## Layer dynamics

Each `CompetitiveAttractorLayer` keeps a population activation vector, a fatigue vector, dynamic basin depths, and a 2D visualization state.

Population activation is driven by:

- current incoming evidence;
- recurrent self-support;
- lateral inhibition from competitors;
- slow adaptation/fatigue.

Activation then changes the depth of each visible Gaussian basin. The 2D point moves through the resulting potential field with inertia. **Categorical activation is not read from the 2D point.** The geometry is a visualization of the population competition, which avoids allowing arbitrary display coordinates to determine the model's answer.

## Frequency → phoneme mapping

Each phoneme has a synthetic eight-band prototype. The delayed frequency population is compared with every prototype using a transparent radial similarity metric. Similarity is sharpened before it is used as phoneme evidence so nearby but incompatible templates can still compete without every candidate remaining strongly active.

## Soft lexical sequence integration

`OnlineLexicalMatcher` never commits a single phoneme label. Each word owns soft mass over prefix positions:

- position 0 means no phonemes matched yet;
- position 1 means evidence has supported the first phoneme;
- subsequent positions represent progressively longer prefixes;
- mass persists with a finite memory time constant;
- mass advances in proportion to the **graded activation** of the expected next phoneme.

For example, after an ambiguous `/B/P/` onset, both `BAT` and `PAT` can carry non-zero lexical evidence. Later `/AE/` and `/T/` evidence can resolve the competition without requiring an earlier discrete B-or-P decision.

This is a deliberately small sequence integrator, not TRACE, Shortlist, an HMM recognizer, or a neural language model.

## Feedback

Lexical candidates can provide a soft prediction over phonemes that is delayed and low-pass filtered before being added back into phoneme evidence. The feedback gain can be set to zero for a feed-forward condition.

Feedback exists here as an experimental manipulation. BaSIM does not treat lexical feedback as settled biological fact.

## Scientific boundaries

BaSIM currently does **not** model:

- cochlear mechanics or auditory-nerve encoding;
- an empirical spectrogram front end;
- spiking neurons or synaptic conductances;
- cortical anatomy;
- learned phonetic categories;
- measured attractor geometry;
- realistic lexical frequency statistics;
- motor or semantic systems.

The valuable claim is narrower: continuous evidence, recurrent competition, finite integration windows, adaptation, causal delays, and soft sequence accumulation can be made inspectable in one small model.

## Validation targets

The regression suite currently checks that:

- a 5 s run at 49 ms produces the expected number of frames;
- the synthetic signal is truly represented as frequency-band energy and ends at 1 s;
- a two-tick connection delay does not leak evidence early;
- phoneme and lexical layers become active before the sound is finished;
- stimulus history changes basin depths;
- `/BAT/` eventually beats close competitors;
- an ambiguous `/B/P/` onset preserves both candidates early;
- the lexical matcher keeps multiple word hypotheses alive without hard phoneme commits.

The next useful validation work is experimental: sweep delays/time constants, quantify ambiguity resolution, compare feedback conditions, and then substitute real audio-derived spectra while keeping these behavioral tests.
