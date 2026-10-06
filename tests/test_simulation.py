import math

from basim.presets import FREQUENCY_BANDS_HZ, cat_demo


def test_simulation_uses_requested_clock():
    sim = cat_demo().simulation
    records = sim.run()
    assert len(records) == math.ceil(5000.0 / 49.0)
    assert records[0].t_ms == 0.0
    assert records[-1].t_ms < 5000.0


def test_stimulus_is_time_frequency_and_stops_after_one_second():
    sim = cat_demo().simulation
    records = sim.run()
    assert len(records[0].stimulus) == len(FREQUENCY_BANDS_HZ) == 8
    assert max(records[0].stimulus) > 0.0
    after = [r for r in records if r.t_ms >= 1029.0]
    assert after
    assert all(r.stimulus_strength == 0.0 for r in after)
    assert all(max(r.stimulus) == 0.0 for r in after)
    assert after[0].lexical.state != after[-1].lexical.state


def test_cat_demo_forms_soft_incremental_cohort_and_converges():
    sim = cat_demo().simulation
    records = sim.run()
    assert sim.cohort.observed == ["K", "AE", "T"]
    assert len(sim.cohort.observed_events) == 3
    first_event = sim.cohort.observed_events[0]
    assert 0.5 < first_event[sim.phoneme.labels.index("K")] < 1.0

    early = min(records, key=lambda r: abs(r.t_ms - 196.0))
    middle = min(records, key=lambda r: abs(r.t_ms - 588.0))
    late = min(records, key=lambda r: abs(r.t_ms - 980.0))
    assert early.lexical_evidence[0] < 0.5
    assert middle.lexical_evidence[0] < 0.5
    assert late.lexical_evidence[0] > 0.9

    winner, confidence = sim.winner()
    assert winner == "cat"
    assert confidence > 0.90
