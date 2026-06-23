import numpy as np
import pytest

from augur_predict.model import Model, ModelFit, fit, _MIN_FIT_POINTS
from augur_predict.trace import Record


def _linear_records(model="m", a=40.0, b=0.25, n=40, noise=0.0, seed=1):
    """n successful calls whose output is a known linear function of input."""
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        x = 100 + i * 50
        y = a + b * x + (rng.normal(0, noise) if noise else 0.0)
        recs.append(Record("s1", f"r{i}", 0, model, int(x), int(round(y))))
    return recs


def test_ols_recovers_known_slope_and_intercept():
    m = fit(_linear_records(a=40.0, b=0.25, noise=0.0))
    f = m.fit_for("m")
    assert f.method == "ols"
    # outputs are integer token counts, so the recovered line is exact only up
    # to the rounding of y — a few thousandths on the slope, well within noise.
    assert f.slope == pytest.approx(0.25, abs=1e-2)
    assert f.intercept == pytest.approx(40.0, abs=1.0)
    assert f.r2 == pytest.approx(1.0, abs=1e-3)
    assert f.resid_std == pytest.approx(0.0, abs=1.0)


def test_ols_predict_matches_line():
    f = fit(_linear_records(a=40.0, b=0.25, noise=0.0)).fit_for("m")
    assert f.predict(1000) == pytest.approx(40 + 0.25 * 1000, abs=1.0)


def test_noisy_fit_has_lower_r2_and_positive_resid_std():
    f = fit(_linear_records(noise=30.0)).fit_for("m")
    assert f.method == "ols"
    assert 0.0 < f.r2 <= 1.0
    assert f.resid_std > 0.0


def test_fallback_to_mean_when_too_few_points():
    recs = _linear_records(n=_MIN_FIT_POINTS - 1)
    f = fit(recs).fit_for("m")
    assert f.method == "mean"
    assert f.slope == 0.0
    outputs = [r.output_tokens for r in recs]
    assert f.intercept == pytest.approx(np.mean(outputs))


def test_fallback_to_mean_when_no_input_spread():
    # Enough points but every input identical -> a slope would be noise.
    recs = [Record("s", f"r{i}", 0, "m", 500, 100 + i) for i in range(20)]
    f = fit(recs).fit_for("m")
    assert f.method == "mean"
    assert f.slope == 0.0


def test_predict_clipped_at_zero():
    # Negative intercept, tiny input -> raw line goes negative, must clip.
    recs = [Record("s", f"r{i}", 0, "m", 1000 + i * 10, 50 + i) for i in range(20)]
    f = fit(recs).fit_for("m")
    f.intercept = -100.0
    f.slope = 0.01
    assert f.predict(0) == 0.0


def test_band_is_ordered_and_nonnegative():
    f = fit(_linear_records(noise=30.0)).fit_for("m")
    lo, hi = f.band(1000)
    assert 0.0 <= lo <= hi


def test_failed_calls_excluded_from_fit_but_kept_in_templates():
    recs = _linear_records(n=20)
    # add a 429 with wild output that would wreck the regression if included
    recs.append(Record("s1", "rX", 1, "m", 300, 9999, status=429))
    m = fit(recs)
    f = m.fit_for("m")
    # slope stays close to the clean 0.25 because the 429 was excluded
    assert f.slope == pytest.approx(0.25, abs=0.05)
    # but the failed call still appears in a run template (call graph structure)
    all_calls = [c for t in m.templates for c in t.calls]
    assert any(c.input_tokens == 300 for c in all_calls)


def test_per_model_fits_are_independent():
    recs = _linear_records(model="cheap", a=10, b=0.1) + \
           _linear_records(model="verbose", a=200, b=0.5)
    m = fit(recs)
    assert m.fit_for("cheap").slope == pytest.approx(0.1, abs=1e-6)
    assert m.fit_for("verbose").slope == pytest.approx(0.5, abs=1e-6)


def test_templates_group_by_scenario_run_and_order_by_seq():
    recs = [
        Record("s1", "r1", 1, "m", 10, 5),
        Record("s1", "r1", 0, "m", 20, 6),
        Record("s2", "r1", 0, "m", 30, 7),
    ]
    m = fit(recs)
    assert len(m.templates) == 2
    t = next(t for t in m.templates if t.scenario_id == "s1")
    assert [c.seq for c in t.calls] == [0, 1]  # sorted by seq


def test_model_json_round_trip(tmp_path):
    m = fit(_linear_records(noise=10.0))
    m.source = "trace.jsonl"
    path = tmp_path / "model.json"
    m.save(str(path))
    back = Model.load(str(path))
    assert back.source == "trace.jsonl"
    assert back.n_records == m.n_records
    f0, f1 = m.fit_for("m"), back.fit_for("m")
    assert f1.slope == pytest.approx(f0.slope)
    assert f1.intercept == pytest.approx(f0.intercept)
    assert f1.r2 == pytest.approx(f0.r2)
    assert len(back.templates) == len(m.templates)


def test_fit_for_unknown_model_is_none():
    m = fit(_linear_records())
    assert m.fit_for("nope") is None


def test_empty_trace_yields_empty_model():
    m = fit([])
    assert m.n_records == 0
    assert m.fits == {}
    assert m.templates == []
