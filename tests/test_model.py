import numpy as np

from basim.connections import LeakyDelayConnection
from basim.lexical import OnlineLexicalMatcher
from basim.model import (
    BaSimEngine,
    ConnectionConfig,
    SimulationConfig,
    SpectrumFrame,
    ambiguous_onset_stimulus,
    default_stimulus,
)
from basim.stimuli import LEXICON, PHONEME_SPECTRA


def test_default_timing_is_roughly_49_ms_for_five_seconds():
    engine = BaSimEngine(SimulationConfig(dt_s=0.049, duration_s=5.0))
    history = engine.run()
    assert len(history) == 103
    assert history[0].time_s == 0.0
    assert 4.95 <= history[-1].time_s < 5.0


def test_default_stimulus_is_explicit_time_frequency_input():
    stimulus = default_stimulus()
    assert abs(stimulus.duration_s - 1.0) < 1e-9
    label, frame = stimulus.frame_at(0.10)
    assert label == "B"
    assert isinstance(frame, SpectrumFrame)
    assert len(frame.energies) == 8
    assert frame.amplitude > 0.0
    assert stimulus.frame_at(1.0)[0] == "silence"
    assert stimulus.frame_at(1.0)[1].amplitude == 0.0


def test_leaky_connection_obeys_two_tick_delay():
    connection = LeakyDelayConnection(
        width=2,
        dt_s=0.049,
        config=ConnectionConfig(delay_s=0.098, tau_s=0.0, gain=1.0),
    )
    source = np.array([1.0, 0.25])
    assert np.allclose(connection.step(source), 0.0)
    assert np.allclose(connection.step(source), 0.0)
    assert np.allclose(connection.step(source), source)


def test_layers_propagate_before_stimulus_has_finished():
    engine = BaSimEngine()
    snapshot = engine.snapshot()
    while engine.time_s < 0.80:
        snapshot = engine.step()
    assert float(np.max(snapshot.phoneme.activation)) > 0.15
    assert float(np.max(snapshot.lexical.activation)) > 0.03


def test_stimulus_changes_attractor_depths():
    engine = BaSimEngine()
    initial = engine.phoneme.depths.copy()
    for _ in range(10):
        engine.step()
    assert not np.allclose(initial, engine.phoneme.depths)
    assert float(np.max(engine.phoneme.depths)) > float(np.min(engine.phoneme.depths))


def test_default_bat_eventually_beats_close_lexical_competitors():
    engine = BaSimEngine()
    while engine.time_s < 1.55:
        engine.step()
    rankings = engine.top_candidates("lexical", n=6)
    assert rankings[0][0] == "BAT"
    assert rankings[0][1] > rankings[1][1]


def test_ambiguous_onset_preserves_b_and_p_competition_early():
    engine = BaSimEngine(stimulus=ambiguous_onset_stimulus())
    while engine.time_s < 0.20:
        engine.step()
    scores = dict(engine.top_candidates("phoneme", n=len(engine.phoneme_labels)))
    assert scores["B"] > 0.08
    assert scores["P"] > 0.08
    assert abs(scores["B"] - scores["P"]) < 0.18


def test_soft_lexical_matcher_keeps_multiple_words_alive_without_hard_commits():
    labels = tuple(PHONEME_SPECTRA)
    matcher = OnlineLexicalMatcher(LEXICON, labels, dt_s=0.049)
    b = labels.index("B")
    p = labels.index("P")
    activation = np.zeros(len(labels))
    activation[b] = 0.55
    activation[p] = 0.50
    evidence = None
    for _ in range(4):
        evidence, _ = matcher.step(activation)
    by_word = dict(zip(matcher.words, evidence, strict=True))
    assert by_word["BAT"] > 0.0
    assert by_word["PAT"] > 0.0
    assert abs(by_word["BAT"] - by_word["PAT"]) < 0.12
