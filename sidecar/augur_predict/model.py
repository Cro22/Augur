"""The predictive output-length model.

What it learns, and why this shape:

* **Per model, output ~ input_tokens via OLS.** Completion length is the thing
  you cannot guess for an agent and the cost surprise lives in its tail (Augur's
  whole thesis). Prompt length is the one cheap signal that correlates with it
  (longer asks tend to get longer answers), so a one-feature linear fit is the
  honest baseline — the PreflightLLMCost direction, not a deep model. It is fit
  per billed model because verbosity differs sharply across models.

* **An intercept-only fallback.** With too few points, or no spread in the
  inputs, a slope is noise. The model degrades to predicting the mean output and
  *says so* (``method == "mean"``) rather than inventing a trend. Honesty about
  fit quality is a first-class output here, mirroring the Go side reporting CIs
  instead of bare point estimates.

* **Run templates.** To estimate cost *without running*, we need the call-graph
  shape, not just a single call's economics: how many calls a run makes, which
  models, what input sizes. We capture each observed (scenario, run) as a
  template of calls; ``emit-trace`` resamples them, rescales inputs, and predicts
  the outputs, producing a synthetic trace the Go pipeline prices as usual.

The model never imports the Go code; it only reads Records and writes a JSON
artifact. numpy does the linear algebra — no scikit-learn, because a one-feature
OLS plus residual spread does not justify the dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .trace import Record

# z for an approximate 95% prediction band from the residual spread. The same
# 1.96 the Go projection uses for its normal-approximation intervals.
_Z95 = 1.959963984540054

# Below this many successful calls for a model we do not trust a slope and fall
# back to predicting the mean output.
_MIN_FIT_POINTS = 8


@dataclass
class ModelFit:
    """The fitted output-length relationship for a single billed model.

    ``method`` is ``"ols"`` when a slope was fit and ``"mean"`` for the
    intercept-only fallback. ``resid_std`` is the spread used for prediction
    bands: the residual standard error for OLS, the plain output stddev for the
    mean fallback.
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

    def predict(self, input_tokens: float) -> float:
        """Point prediction of output tokens for a given prompt size.

        Clipped at zero — a model can predict a small negative for tiny inputs
        when the intercept is low, and a negative completion length is
        meaningless.
        """
        y = self.intercept + self.slope * float(input_tokens)
        return max(0.0, y)

    def band(self, input_tokens: float, z: float = _Z95) -> tuple[float, float]:
        """Approximate prediction interval (lo, hi), both clipped at zero."""
        mid = self.predict(input_tokens)
        lo = max(0.0, mid - z * self.resid_std)
        hi = max(0.0, mid + z * self.resid_std)
        return lo, hi

    def to_json(self) -> dict:
        return {
            "model": self.model,
            "n": self.n,
            "method": self.method,
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

    @classmethod
    def from_json(cls, obj: dict) -> "ModelFit":
        return cls(**{k: obj[k] for k in (
            "model", "n", "method", "intercept", "slope", "r2", "resid_std",
            "output_mean", "output_std", "input_mean", "input_min", "input_max",
        )})


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
    source: str = ""

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "source": self.source,
            "n_records": self.n_records,
            "models": {m: f.to_json() for m, f in self.fits.items()},
            "run_templates": [t.to_json() for t in self.templates],
        }

    @classmethod
    def from_json(cls, obj: dict) -> "Model":
        return cls(
            version=int(obj.get("version", 1)),
            source=obj.get("source", ""),
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


def _fit_one(model: str, xs: np.ndarray, ys: np.ndarray) -> ModelFit:
    """Fit a single model's output-length relationship.

    Uses OLS when there is enough data and real spread in the inputs; otherwise
    falls back to predicting the mean. R^2 is the coefficient of determination;
    for the mean fallback it is 0 by definition (the mean explains no variance
    beyond itself).
    """
    n = int(xs.size)
    output_mean = float(ys.mean()) if n else 0.0
    output_std = float(ys.std(ddof=1)) if n > 1 else 0.0
    input_mean = float(xs.mean()) if n else 0.0
    input_min = float(xs.min()) if n else 0.0
    input_max = float(xs.max()) if n else 0.0

    enough = n >= _MIN_FIT_POINTS
    has_spread = n > 1 and float(xs.std()) > 0.0
    if not (enough and has_spread):
        return ModelFit(
            model=model, n=n, method="mean",
            intercept=output_mean, slope=0.0, r2=0.0,
            resid_std=output_std,
            output_mean=output_mean, output_std=output_std,
            input_mean=input_mean, input_min=input_min, input_max=input_max,
        )

    slope, intercept = np.polyfit(xs, ys, 1)
    pred = intercept + slope * xs
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # Residual standard error: ss_res spread over the degrees of freedom left
    # after estimating slope and intercept. This is the band emit-trace samples.
    resid_std = float(np.sqrt(ss_res / (n - 2))) if n > 2 else 0.0

    return ModelFit(
        model=model, n=n, method="ols",
        intercept=float(intercept), slope=float(slope), r2=r2,
        resid_std=resid_std,
        output_mean=output_mean, output_std=output_std,
        input_mean=input_mean, input_min=input_min, input_max=input_max,
    )


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


def fit(records: List[Record]) -> Model:
    """Fit the output-length model from a recorded trace.

    Only successful calls feed the per-model regressions (see Record.succeeded);
    every call, success or not, contributes its structure to the run templates,
    because a failed-but-billed call is part of the call graph emit-trace should
    reproduce.
    """
    by_model: Dict[str, tuple[list, list]] = {}
    for r in records:
        if not r.succeeded():
            continue
        xs, ys = by_model.setdefault(r.model, ([], []))
        xs.append(r.input_tokens)
        ys.append(r.output_tokens)

    fits: Dict[str, ModelFit] = {}
    for model, (xs, ys) in by_model.items():
        fits[model] = _fit_one(model, np.asarray(xs, dtype=float),
                               np.asarray(ys, dtype=float))

    return Model(
        version=1,
        n_records=len(records),
        fits=fits,
        templates=_templates(records),
    )
