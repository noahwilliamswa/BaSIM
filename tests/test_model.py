import numpy as np

from basim.model import BaSimEngine, SimulationConfig, default_stimulus


def test_default_timing_is_roughly_49_ms_for_five_seconds():
    engine = BaSimEngine(SimulationConfig(dt_s=0.049, duration_s=5.0))
    history = engine.run()
    assert len(history) == 103
    assert history[0].time_s == 0.0
    assert 4.95 <= history[-1].time_s < 5.0


def test_default_stimulus_lasts_one_second_then_turns_off():
    stimulus = default_stimulus()
    assert abs(stimulus.duration_s - 1.0) < 1e-9
    assert stimulus.frame_at(0.99)[0] != "silence"
    assert stimulus.frame_at(1.0)[0] == "silence"
    assert stimulus.frame_at(1.0)[1].amplitude == 0.0


def test_layers_propagate_before_stimulus_has_finished():
    engine = BaSimEngine()
    while engine.time_s < 0.75:
        snapshot = engine.step()
    assert float(np.max(snapshot.phoneme.activation)) > 0.2
    assert float(np.max(snapshot.lexical.activation)) > 0.05


def test_stimulus_changes_attractor_depths():
    engine = BaSimEngine()
    initial = engine.phoneme.depths.copy()
    for _ in range(8):
        engine.step()
    assert not np.allclose(initial, engine.phoneme.depths)
    assert float(np.max(engine.phoneme.depths)) > float(np.min(engine.phoneme.depths))


def test_default_bat_eventually_beats_close_lexical_competitors():
    engine = BaSimEngine()
    while engine.time_s < 1.35:
        engine.step()
    rankings = engine.top_candidates("lexical", n=5)
    assert rankings[0][0] == "BAT"
