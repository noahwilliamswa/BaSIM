import math

from basim.presets import cat_demo


def test_simulation_uses_requested_clock():
    sim = cat_demo().simulation
    records = sim.run()
    assert len(records) == math.ceil(5000.0 / 49.0)
    assert records[0].t_ms == 0.0
    assert records[-1].t_ms < 5000.0


def test_stimulus_stops_but_network_keeps_evolving():
    sim = cat_demo().simulation
    records = sim.run()
    after = [r for r in records if r.t_ms >= 1029.0]
    assert after
    assert all(r.stimulus_strength == 0.0 for r in after)
    assert after[0].lexical.state != after[-1].lexical.state


def test_cat_demo_forms_incremental_cohort_and_converges():
    sim = cat_demo().simulation
    records = sim.run()
    observed_lengths = [
        len(r.observed_phonemes)
        for r in records
        if r.t_ms < 1000.0
    ]
    assert max(observed_lengths) >= 2
    winner, confidence = sim.winner()
    assert winner == "cat"
    assert confidence > 0.50
