"""The predictive output-length model.

Two distribution modes, one shared structure:

* **gaussian** (default) — per model, ``output ~ a + b·input_tokens`` via OLS,
  with residuals modelled as a symmetric Gaussian spread. The honest linear
  baseline: the PreflightLLMCost direction, fit per billed model because
  verbosity differs sharply across models. With too few points or no input
  spread it degrades to predicting the mean output (``method == "mean"``) and
  says so.

* **quantile** — per model, a *grid* of conditional quantiles fit directly by
  quantile regression (``output_τ ~ a_τ + b_τ·input``). This targets what the
  gate actually uses (a conditional quantile, not the mean) and represents a
  skewed, right-tailed output distribution instead of a symmetric band — the
  cost surprise lives in that tail. The grid is a discretised conditional CDF:
  ``emit-trace`` samples it by inverse transform. With too few points it falls
  back to the *empirical marginal quantiles* of the output, which is still
  asymmetric (and so more honest than the gaussian mean fallback).

Either way we also capture each observed ``(scenario, run)`` as a *run template*
— its call-graph (which models, what input sizes, in what order) — so
``emit-trace`` can resample real run shapes rather than inventing structure.

The model never imports the Go code; it only reads Records and writes a JSON
artifact. numpy does the OLS and the sampling; scipy's ``linprog`` solves the
quantile-regression LP (quantile mode only).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .trace import Record

# z for an approximate 95% prediction band from the residual spread. The same
# 1.96 the Go projection uses for its normal-approximation intervals.
_Z95 = 1.959963984540054

# Below this many successful calls for a model we do not trust an OLS slope and
# fall back to predicting the mean output.
_MIN_FIT_POINTS = 8

# Quantile regression at the extreme taus needs more data than a median fit, so
# its fallback threshold is higher: below this we use empirical marginal
# quantiles instead of regressing.
_MIN_QR_POINTS = 12

# The conditional-quantile grid fit in quantile mode. 0.5 must be present (it is
# the point prediction); 0.05/0.95 bound the band the gate cares about. We stop
# at 0.95 — a p99 from a few dozen runs would be false precision.
DEFAULT_TAUS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _phi(z: float) -> float:
    """Standard-normal CDF, for turning a copula latent z into a uniform u."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass
class QuantileLine:
    """One conditional-quantile line: output_tau ≈ intercept + slope·input."""

    tau: float
    intercept: float
    slope: float

    def at(self, input_tokens: float) -> float:
        return self.intercept + self.slope * float(input_tokens)

    def to_json(self) -> dict:
        return {"tau": self.tau, "intercept": self.intercept, "slope": self.slope}

    @classmethod
    def from_json(cls, obj: dict) -> "QuantileLine":
        return cls(tau=float(obj["tau"]), intercept=float(obj["intercept"]),
                   slope=float(obj["slope"]))


@dataclass
class ModelFit:
    """The fitted output-length relationship for a single billed model.

    ``dist`` selects how the fields are read:

    * ``gaussian`` — ``intercept``/``slope`` are the OLS mean line; ``resid_std``
      is the symmetric prediction spread; ``method`` is ``"ols"`` or ``"mean"``.
    * ``quantile`` — ``quantiles`` holds the conditional-quantile grid;
      ``intercept``/``slope`` mirror the median line for convenience; ``method``
      is ``"quantile_reg"`` or ``"empirical_quantile"``. ``resid_std`` is the
      spread of residuals about the median, kept only for the report.
    """

    model: str
    n: int
    method: str
    intercept: float
    slope: float
    r2: float
    resid_std: float
    output_mean: float
    output_std: float
    input_mean: float
    input_min: float
    input_max: float
    dist: str = "gaussian"
    quantiles: List[QuantileLine] = field(default_factory=list)

    # --- prediction -------------------------------------------------------

    def predict(self, input_tokens: float) -> float:
        """Point prediction (clipped at zero).

        gaussian: the OLS mean. quantile: the median (tau=0.5) of the grid.
        """
        if self.dist == "quantile":
            return self.quantile_at(input_tokens, 0.5)
        return max(0.0, self.intercept + self.slope * float(input_tokens))

    def quantile_at(self, input_tokens: float, tau: float) -> float:
        """The tau-quantile of output at this input (quantile mode only).

        Evaluates every grid line at the input, enforces monotonicity across
        taus (independently-fit quantiles can cross), then interpolates at tau.
        Outside the fitted tau range np.interp clamps to the ends — we do not
        extrapolate a tail we never fit.
        """
        if not self.quantiles:
            # gaussian fit asked for a quantile: use the normal approximation.
            from_z = _z_for_tau(tau)
            return max(0.0, self.predict(input_tokens) + from_z * self.resid_std)
        taus = np.array([q.tau for q in self.quantiles])
        vals = np.maximum.accumulate(
            np.array([q.at(input_tokens) for q in self.quantiles]))
        return max(0.0, float(np.interp(tau, taus, vals)))

    def band(self, input_tokens: float, z: float = _Z95) -> tuple[float, float]:
        """Approximate ~95% band (lo, hi), both clipped at zero.

        gaussian: mean ± z·resid_std. quantile: the 0.05 and 0.95 grid lines —
        an asymmetric band straight from the fitted tails.
        """
        if self.dist == "quantile" and self.quantiles:
            lo = self.quantile_at(input_tokens, self.quantiles[0].tau)
            hi = self.quantile_at(input_tokens, self.quantiles[-1].tau)
            return lo, hi
        mid = self.predict(input_tokens)
        return max(0.0, mid - z * self.resid_std), max(0.0, mid + z * self.resid_std)

    def sample(self, input_tokens: float, z: float) -> int:
        """Draw one output length given a standard-normal latent ``z``.

        ``z`` carries the run-level correlation (see emit.py): the same z shared
        across a run's calls makes a verbose run verbose throughout. gaussian
        uses z directly as the standardised residual; quantile maps z→u=Φ(z) and
        inverts the conditional CDF, so the *skew* of the fitted grid is
        preserved rather than collapsed to a symmetric band.
        """
        if self.dist == "quantile":
            u = _phi(z)
            return int(round(self.quantile_at(input_tokens, u)))
        val = self.predict(input_tokens) + z * self.resid_std
        return int(round(max(0.0, val)))

    # --- serialization ----------------------------------------------------

    def to_json(self) -> dict:
        d = {
            "model": self.model,
            "n": self.n,
            "method": self.method,
            "dist": self.dist,
            "intercept": self.intercept,
            "slope": self.slope,
            "r2": self.r2,
            "resid_std": self.resid_std,
            "output_mean": self.output_mean,
            "output_std": self.output_std,
            "input_mean": self.input_mean,
            "input_min": self.input_min,
            "input_max": self.input_max,
        }
        if self.quantiles:
            d["quantiles"] = [q.to_json() for q in self.quantiles]
        return d

    @classmethod
    def from_json(cls, obj: dict) -> "ModelFit":
        return cls(
            model=obj["model"], n=int(obj["n"]), method=obj["method"],
            intercept=float(obj["intercept"]), slope=float(obj["slope"]),
            r2=float(obj["r2"]), resid_std=float(obj["resid_std"]),
            output_mean=float(obj["output_mean"]), output_std=float(obj["output_std"]),
            input_mean=float(obj["input_mean"]), input_min=float(obj["input_min"]),
            input_max=float(obj["input_max"]),
            dist=obj.get("dist", "gaussian"),
            quantiles=[QuantileLine.from_json(q) for q in obj.get("quantiles", [])],
        )


def _z_for_tau(tau: float) -> float:
    """Inverse standard-normal CDF via a rational approximation (Acklam).

    Used only when a gaussian fit is asked for a quantile (the report's p95 on a
    gaussian model). Accurate to ~1e-9 over (0,1), and avoids a scipy import on
    the gaussian path.
    """
    if tau <= 0.0:
        return -math.inf
    if tau >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if tau < plow:
        q = math.sqrt(-2 * math.log(tau))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if tau > phigh:
        q = math.sqrt(-2 * math.log(1 - tau))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = tau - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


@dataclass
class Call:
    """One call within a run template: the structure emit-trace replays."""

    seq: int
    model: str
    input_tokens: int
    cached_tokens: int
    endpoint: str = ""

    def to_json(self) -> dict:
        d = {
            "seq": self.seq,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
        }
        if self.endpoint:
            d["endpoint"] = self.endpoint
        return d

    @classmethod
    def from_json(cls, obj: dict) -> "Call":
        return cls(
            seq=int(obj["seq"]),
            model=obj["model"],
            input_tokens=int(obj["input_tokens"]),
            cached_tokens=int(obj.get("cached_tokens", 0)),
            endpoint=obj.get("endpoint", ""),
        )


@dataclass
class RunTemplate:
    """The observed call-graph of one (scenario, run): its sequence of calls."""

    scenario_id: str
    calls: List[Call]

    def to_json(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "calls": [c.to_json() for c in self.calls],
        }

    @classmethod
    def from_json(cls, obj: dict) -> "RunTemplate":
        return cls(
            scenario_id=obj.get("scenario_id", ""),
            calls=[Call.from_json(c) for c in obj.get("calls", [])],
        )


@dataclass
class Model:
    """The full artifact: per-model fits plus the observed run templates."""

    version: int
    n_records: int
    fits: Dict[str, ModelFit]
    templates: List[RunTemplate]
    dist: str = "gaussian"
    source: str = ""

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "source": self.source,
            "dist": self.dist,
            "n_records": self.n_records,
            "models": {m: f.to_json() for m, f in self.fits.items()},
            "run_templates": [t.to_json() for t in self.templates],
        }

    @classmethod
    def from_json(cls, obj: dict) -> "Model":
        return cls(
            version=int(obj.get("version", 1)),
            source=obj.get("source", ""),
            dist=obj.get("dist", "gaussian"),
            n_records=int(obj.get("n_records", 0)),
            fits={m: ModelFit.from_json(f) for m, f in obj.get("models", {}).items()},
            templates=[RunTemplate.from_json(t) for t in obj.get("run_templates", [])],
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, indent=2)
            f.write("\n")

    @classmethod
    def load(cls, path: str) -> "Model":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_json(json.load(f))

    def fit_for(self, model: str) -> Optional[ModelFit]:
        """The fit for a model, or None if the trace never exercised it."""
        return self.fits.get(model)


def _common_stats(xs: np.ndarray, ys: np.ndarray) -> dict:
    n = int(xs.size)
    return {
        "n": n,
        "output_mean": float(ys.mean()) if n else 0.0,
        "output_std": float(ys.std(ddof=1)) if n > 1 else 0.0,
        "input_mean": float(xs.mean()) if n else 0.0,
        "input_min": float(xs.min()) if n else 0.0,
        "input_max": float(xs.max()) if n else 0.0,
    }


def _fit_gaussian(model: str, xs: np.ndarray, ys: np.ndarray) -> ModelFit:
    """OLS mean line with a symmetric residual spread, or the mean fallback."""
    s = _common_stats(xs, ys)
    n = s["n"]
    enough = n >= _MIN_FIT_POINTS
    has_spread = n > 1 and float(xs.std()) > 0.0
    if not (enough and has_spread):
        return ModelFit(
            model=model, method="mean", dist="gaussian",
            intercept=s["output_mean"], slope=0.0, r2=0.0,
            resid_std=s["output_std"], **s)

    slope, intercept = np.polyfit(xs, ys, 1)
    pred = intercept + slope * xs
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    resid_std = float(np.sqrt(ss_res / (n - 2))) if n > 2 else 0.0
    return ModelFit(
        model=model, method="ols", dist="gaussian",
        intercept=float(intercept), slope=float(slope), r2=r2,
        resid_std=resid_std, **s)


def _qr_lp(xs: np.ndarray, ys: np.ndarray, tau: float) -> tuple[float, float]:
    """Linear quantile regression for one tau, as an exact LP.

    Quantile regression minimises the pinball loss, which is piecewise linear,
    so it is a linear program. With residual r = y - (a + b·x) split into
    positive/negative parts r = u⁺ - u⁻ (u⁺,u⁻ ≥ 0), the program is

        min  Σ τ·u⁺ᵢ + (1-τ)·u⁻ᵢ
        s.t. yᵢ - a - b·xᵢ = u⁺ᵢ - u⁻ᵢ

    over free (a, b) and non-negative (u⁺, u⁻). Solved with HiGHS via linprog.
    """
    from scipy.optimize import linprog

    n = ys.size
    X = np.column_stack([np.ones(n), xs])              # n×2 design (a, b)
    c = np.concatenate([np.zeros(2), tau * np.ones(n), (1 - tau) * np.ones(n)])
    A_eq = np.hstack([X, np.eye(n), -np.eye(n)])
    bounds = [(None, None), (None, None)] + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=ys, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"quantile LP failed at tau={tau}: {res.message}")
    a, b = res.x[0], res.x[1]
    return float(a), float(b)


def _fit_quantile(model: str, xs: np.ndarray, ys: np.ndarray,
                  taus: Sequence[float]) -> ModelFit:
    """Conditional-quantile grid by regression, or empirical marginal fallback."""
    s = _common_stats(xs, ys)
    n = s["n"]
    taus = sorted(taus)
    enough = n >= _MIN_QR_POINTS
    has_spread = n > 1 and float(xs.std()) > 0.0

    if not (enough and has_spread):
        # No trustworthy input→output trend: flat lines at the marginal output
        # quantiles. Still asymmetric, so it beats a symmetric mean fallback.
        lines = [QuantileLine(tau=t, intercept=float(np.quantile(ys, t)), slope=0.0)
                 for t in taus]
        median = next(l for l in lines if abs(l.tau - 0.5) < 1e-9)
        return ModelFit(
            model=model, method="empirical_quantile", dist="quantile",
            intercept=median.intercept, slope=0.0, r2=0.0,
            resid_std=s["output_std"], quantiles=lines, **s)

    lines: List[QuantileLine] = []
    for t in taus:
        try:
            a, b = _qr_lp(xs, ys, t)
        except RuntimeError:
            # A single tau failing the LP degrades to its marginal quantile
            # rather than dropping the whole grid.
            a, b = float(np.quantile(ys, t)), 0.0
        lines.append(QuantileLine(tau=t, intercept=a, slope=b))

    median = next(l for l in lines if abs(l.tau - 0.5) < 1e-9)
    pred = median.intercept + median.slope * xs
    resid = ys - pred
    resid_std = float(resid.std(ddof=1)) if n > 1 else 0.0
    # Koenker–Machado pseudo-R¹ for the median fit: 1 − (pinball of the model) /
    # (pinball of the unconditional median). The QR analogue of R², reported so
    # the user can judge the median fit's strength.
    r1 = _pseudo_r1(xs, ys, median, 0.5)
    return ModelFit(
        model=model, method="quantile_reg", dist="quantile",
        intercept=median.intercept, slope=median.slope, r2=r1,
        resid_std=resid_std, quantiles=lines, **s)


def _pinball(resid: np.ndarray, tau: float) -> float:
    return float(np.sum(np.where(resid >= 0, tau * resid, (tau - 1) * resid)))


def _pseudo_r1(xs: np.ndarray, ys: np.ndarray, line: QuantileLine, tau: float) -> float:
    v_model = _pinball(ys - (line.intercept + line.slope * xs), tau)
    v_null = _pinball(ys - np.quantile(ys, tau), tau)
    return 1.0 - v_model / v_null if v_null > 0 else 0.0


def _templates(records: List[Record]) -> List[RunTemplate]:
    """Group records into per-(scenario, run) call-graph templates.

    Order within a run follows ``seq`` so a replayed run reproduces the observed
    call ordering — the fan-out/retry structure the cost depends on. Insertion
    order of the runs themselves is preserved so emit-trace is deterministic.
    """
    order: List[tuple[str, str]] = []
    groups: Dict[tuple[str, str], List[Record]] = {}
    for r in records:
        key = (r.scenario_id, r.run_id)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(r)

    out: List[RunTemplate] = []
    for key in order:
        recs = sorted(groups[key], key=lambda r: r.seq)
        calls = [
            Call(seq=r.seq, model=r.model, input_tokens=r.input_tokens,
                 cached_tokens=r.cached_tokens, endpoint=r.endpoint)
            for r in recs
        ]
        out.append(RunTemplate(scenario_id=key[0], calls=calls))
    return out


def fit(records: List[Record], dist: str = "gaussian",
        taus: Sequence[float] = DEFAULT_TAUS) -> Model:
    """Fit the output-length model from a recorded trace.

    ``dist`` is ``"gaussian"`` (OLS mean + symmetric spread, the default) or
    ``"quantile"`` (a conditional-quantile grid by regression). Only successful
    calls feed the per-model fits (see Record.succeeded); every call, success or
    not, contributes its structure to the run templates, because a failed-but-
    billed call is part of the call graph emit-trace should reproduce.
    """
    if dist not in ("gaussian", "quantile"):
        raise ValueError(f"unknown dist {dist!r}: want 'gaussian' or 'quantile'")

    by_model: Dict[str, tuple[list, list]] = {}
    for r in records:
        if not r.succeeded():
            continue
        xs, ys = by_model.setdefault(r.model, ([], []))
        xs.append(r.input_tokens)
        ys.append(r.output_tokens)

    fits: Dict[str, ModelFit] = {}
    for model, (xs, ys) in by_model.items():
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        fits[model] = (_fit_quantile(model, x, y, taus) if dist == "quantile"
                       else _fit_gaussian(model, x, y))

    return Model(version=1, n_records=len(records), fits=fits,
                 templates=_templates(records), dist=dist)
