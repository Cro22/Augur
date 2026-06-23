import numpy as np
import pytest

from augur_predict.emit import emit
from augur_predict.model import fit
from augur_predict.trace import Record


def _model(noise=10.0, n=40):
    rng = np.random.default_rng(3)
    recs = []
    for i in range(n):
        x = 100 + i * 40
        y = 40 + 0.25 * x + rng.normal(0, noise)
        recs.append(Record("s1", f"r{i}", 0, "m", int(x), int(round(y)),
                            cached_tokens=int(x * 0.1)))
    return fit(recs)


def test_emit_count_matches_runs():
    m = _model()
    recs = emit(m, runs=10, seed=0)
    run_ids = {r.run_id for r in recs}
    assert len(run_ids) == 10


def test_emit_is_deterministic_for_a_seed():
    m = _model()
    a = emit(m, runs=8, seed=42)
    b = emit(m, runs=8, seed=42)
    assert [r.output_tokens for r in a] == [r.output_tokens for r in b]


def test_emit_seed_changes_the_sample():
    m = _model()
    a = emit(m, runs=8, seed=1)
    b = emit(m, runs=8, seed=2)
    assert [r.output_tokens for r in a] != [r.output_tokens for r in b]


def test_emit_records_are_valid_for_go_aggregate():
    """cost.Usage.Validate on the Go side rejects negatives and cached>input."""
    m = _model()
    for r in emit(m, runs=20, seed=7, input_scale=2.0):
        assert r.input_tokens >= 0
        assert r.output_tokens >= 0
        assert r.cached_tokens >= 0
        assert r.cached_tokens <= r.input_tokens
        assert r.status == 200
        assert r.model == "m"
        assert r.scenario_id  # non-empty


def test_input_scale_inflates_prompt_and_output():
    m = _model(noise=0.0)  # deterministic line, no noise
    base = emit(m, runs=4, seed=0, input_scale=1.0)
    scaled = emit(m, runs=4, seed=0, input_scale=2.0)
    # same seed, same templates: compare matched calls
    assert scaled[0].input_tokens > base[0].input_tokens
    # bigger prompt predicts a longer completion (the slope is positive)
    assert scaled[0].output_tokens > base[0].output_tokens


def test_emit_zero_runs_is_empty():
    assert emit(_model(), runs=0) == []


def test_emit_rejects_nonpositive_scale():
    with pytest.raises(ValueError, match="positive"):
        emit(_model(), runs=2, input_scale=0)


def test_scenario_filter_restricts_templates():
    recs = [Record("s1", "r1", 0, "m", 100, 50),
            Record("s2", "r1", 0, "m", 200, 60)]
    m = fit(recs)
    out = emit(m, runs=4, seed=0, scenario_filter="s2")
    assert {r.scenario_id for r in out} == {"s2"}


def test_unknown_scenario_filter_raises():
    m = _model()
    with pytest.raises(ValueError, match="nope"):
        emit(m, runs=2, scenario_filter="nope")


def test_run_ids_use_prefix():
    out = emit(_model(), runs=3, seed=0, run_prefix="whatif")
    assert all(r.run_id.startswith("whatif-") for r in out)
