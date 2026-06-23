"""Synthesising a predicted trace from a fitted model.

This is how the sidecar pays for itself: given a model learned from one recorded
run, produce a trace for inputs you have NOT executed — a bigger prompt, more
runs, a context-growth scenario — without spending a token. The synthetic trace
is ordinary JSONL, so ``augur aggregate | project | gate`` consume it with no
knowledge that a model, not a proxy, wrote it. That is the loose coupling the
SPEC asks for: the file is the whole interface.

Determinism is deliberate. emit-trace seeds a numpy RNG (default fixed) so the
same model and flags reproduce the same trace — the record/replay ethos of the
Go side carried into the predictive path. Vary ``--seed`` to draw a different
sample from the same learned distribution.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .model import Model
from .trace import Record

# A fixed wall-clock stamp for synthetic rows. They were not observed at any real
# time, and labelling them so keeps a predicted trace honest and reproducible
# (no clock dependence, matching the Go replay path's stable run-ids).
_SYNTHETIC_TS = "1970-01-01T00:00:00Z"


def emit(
    model: Model,
    runs: int,
    input_scale: float = 1.0,
    seed: int = 0,
    scenario_filter: Optional[str] = None,
    run_prefix: str = "pred",
) -> List[Record]:
    """Generate ``runs`` synthetic runs by resampling and rescaling templates.

    For each synthetic run we take an observed template (cycling through them in
    order, so the scenario mix is preserved), scale every call's prompt by
    ``input_scale``, then predict each call's output from the model: the point
    prediction plus Gaussian noise at the fit's residual spread, clipped at zero
    and rounded. Sampling the spread — not just the mean — is what keeps the
    predicted *distribution* (and therefore the p95 the gate cares about) honest
    rather than collapsing every run to the average.

    ``input_scale`` is the predictive analogue of the Go ``--context-growth``
    knob, but it does more: it also feeds the larger prompt back through the
    output model, so a bigger ask predicts a longer answer instead of only a
    costlier prompt.
    """
    if runs <= 0:
        return []
    if input_scale <= 0:
        raise ValueError("input_scale must be positive")

    templates = model.templates
    if scenario_filter is not None:
        templates = [t for t in templates if t.scenario_id == scenario_filter]
    if not templates:
        raise ValueError("no run templates to sample from"
                         + (f" for scenario {scenario_filter!r}" if scenario_filter else ""))

    rng = np.random.default_rng(seed)
    out: List[Record] = []
    width = max(4, len(str(runs)))

    for i in range(runs):
        template = templates[i % len(templates)]
        run_id = f"{run_prefix}-{i:0{width}d}"
        for call in template.calls:
            fit = model.fit_for(call.model)
            scaled_input = int(round(call.input_tokens * input_scale))
            # Cached tokens are a subset of input; scale them with it and never
            # let them exceed the (possibly rounded) input, which would make the
            # row fail cost.Usage.Validate on the Go side.
            scaled_cached = min(scaled_input, int(round(call.cached_tokens * input_scale)))

            if fit is None:
                # A model present in the templates but with no successful calls
                # to learn from: keep the observed output as the best we have.
                predicted = call_output_fallback(call)
            else:
                mean = fit.predict(scaled_input)
                noisy = mean + rng.normal(0.0, fit.resid_std) if fit.resid_std > 0 else mean
                predicted = int(round(max(0.0, noisy)))

            out.append(Record(
                scenario_id=template.scenario_id,
                run_id=run_id,
                seq=call.seq,
                model=call.model,
                input_tokens=scaled_input,
                output_tokens=predicted,
                cached_tokens=scaled_cached,
                latency_ms=0,
                ts=_SYNTHETIC_TS,
                endpoint=call.endpoint,
                status=200,
            ))
    return out


def call_output_fallback(call) -> int:
    """Output for a call whose model has no fit — there is nothing to predict
    from, so emit zero and let the (rare) case be visibly conservative."""
    return 0
