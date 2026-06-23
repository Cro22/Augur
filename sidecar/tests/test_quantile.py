import numpy as np
import pytest

from augur_predict.model import (
    fit, Model, _phi, _z_for_tau, _MIN_QR_POINTS, DEFAULT_TAUS,
)
from augur_predict.trace import Record


def _skewed_records(model="m", n=120, seed=4):
    """Right-skewed output: lognormal completion whose scale grows with input.

    Both the skew and the input-dependent spread are things the gaussian band
    can't represent but the quantile grid can.
    """
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        x = 200 + rng.uniform(0, 1800)
        # median output ~ 30 + 0.2x, multiplicative lognormal noise (skewed),
        # spread widening with x.
        med = 30 + 0.2 * x
        y = med * np.exp(rng.normal(0, 0.3 + 0.0002 * x))
        recs.append(Record("s1", f"r{i}", 0, model, int(x), int(max(1, round(y)))))
    return recs


def test_phi_and_inverse_are_consistent():
    for tau in (0.05, 0.25, 0.5, 0.75, 0.95):
        assert _phi(_z_for_tau(tau)) == pytest.approx(tau, abs=1e-6)


def test_quantile_fit_uses_regression_with_enough_data():
    m = fit(_skewed_records(), dist="quantile")
    f = m.fit_for("m")
    assert m.dist == "quantile"
    assert f.dist == "quantile"
    assert f.method == "quantile_reg"
    assert [q.tau for q in f.quantiles] == sorted(DEFAULT_TAUS)


def test_quantiles_are_ordered_p95_above_p50_above_p05():
    f = fit(_skewed_records(), dist="quantile").fit_for("m")
    for x in (300, 1000, 1900):
        p05 = f.quantile_at(x, 0.05)
        p50 = f.quantile_at(x, 0.5)
        p95 = f.quantile_at(x, 0.95)
        assert p05 <= p50 <= p95


def test_quantile_band_is_asymmetric_for_skewed_data():
    """The whole point: the upper tail is farther from the median than the lower
    tail, which a symmetric gaussian band cannot express."""
    f = fit(_skewed_records(), dist="quantile").fit_for("m")
    x = 1500
    p05 = f.quantile_at(x, 0.05)
    p50 = f.quantile_at(x, 0.5)
    p95 = f.quantile_at(x, 0.95)
    upper = p95 - p50
    lower = p50 - p05
    assert upper > lower * 1.2  # clearly right-skewed


def test_quantile_p95_exceeds_gaussian_p95_on_skewed_data():
    """A heavy upper tail is exactly what gaussian underestimates."""
    recs = _skewed_records()
    x = 1500
    qf = fit(recs, dist="quantile").fit_for("m")
    gf = fit(recs, dist="gaussian").fit_for("m")
    q95 = qf.quantile_at(x, 0.95)
    g95 = gf.predict(x) + 1.6448536 * gf.resid_std  # gaussian ~p95
    assert q95 > g95


def test_quantile_at_clamps_outside_fitted_range():
    f = fit(_skewed_records(), dist="quantile").fit_for("m")
    x = 1000
    lo_end = f.quantile_at(x, f.quantiles[0].tau)
    hi_end = f.quantile_at(x, f.quantiles[-1].tau)
    assert f.quantile_at(x, 0.001) == pytest.approx(lo_end)
    assert f.quantile_at(x, 0.999) == pytest.approx(hi_end)


def test_sample_is_monotone_in_z():
    f = fit(_skewed_records(), dist="quantile").fit_for("m")
    x = 1000
    lows = f.sample(x, -1.5)
    mids = f.sample(x, 0.0)
    highs = f.sample(x, 1.5)
    assert lows <= mids <= highs


def test_empirical_quantile_fallback_when_few_points():
    recs = _skewed_records(n=_MIN_QR_POINTS - 2)
    f = fit(recs, dist="quantile").fit_for("m")
    assert f.method == "empirical_quantile"
    # flat in input (no trustworthy slope) ...
    assert all(q.slope == 0.0 for q in f.quantiles)
    # ... but still asymmetric (skewed marginal)
    p50 = f.quantile_at(0, 0.5)
    p95 = f.quantile_at(0, 0.95)
    p05 = f.quantile_at(0, 0.05)
    assert (p95 - p50) > (p50 - p05)


def test_empirical_fallback_when_no_input_spread():
    recs = [Record("s", f"r{i}", 0, "m", 500, int(50 + i)) for i in range(30)]
    f = fit(recs, dist="quantile").fit_for("m")
    assert f.method == "empirical_quantile"


def test_pseudo_r1_in_range():
    f = fit(_skewed_records(), dist="quantile").fit_for("m")
    assert 0.0 <= f.r2 <= 1.0


def test_quantile_model_json_round_trip(tmp_path):
    m = fit(_skewed_records(), dist="quantile")
    m.source = "trace.jsonl"
    path = tmp_path / "model.json"
    m.save(str(path))
    back = Model.load(str(path))
    assert back.dist == "quantile"
    f0, f1 = m.fit_for("m"), back.fit_for("m")
    assert f1.method == f0.method
    assert len(f1.quantiles) == len(f0.quantiles)
    for a, b in zip(f0.quantiles, f1.quantiles):
        assert b.tau == pytest.approx(a.tau)
        assert b.slope == pytest.approx(a.slope)
        assert b.intercept == pytest.approx(a.intercept)


def test_gaussian_model_loads_without_dist_field(tmp_path):
    """Backward compat: an old artifact with no 'dist' key reads as gaussian."""
    import json
    m = fit(_skewed_records(), dist="gaussian")
    obj = m.to_json()
    del obj["dist"]
    for f in obj["models"].values():
        f.pop("dist", None)
    path = tmp_path / "old.json"
    path.write_text(json.dumps(obj))
    back = Model.load(str(path))
    assert back.dist == "gaussian"
    assert back.fit_for("m").dist == "gaussian"


def test_unknown_dist_raises():
    with pytest.raises(ValueError, match="unknown dist"):
        fit(_skewed_records(), dist="poisson")
