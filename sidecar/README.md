# augur-predict — the predictive output-length sidecar

> Estimate an agent's cost for inputs you **haven't run**, by learning completion
> length from a trace Augur already recorded.

This is the one piece of Augur the [SPEC](../SPEC.md) deliberately writes in
Python instead of Go (Hito 5): a small **predictive output-length model**. It is
optional — delete this directory and Augur's pure-Go v1 is untouched.

## Why it exists

Augur's core measures *real* cost by running your agent through a recording
proxy. But running spends tokens, and you can't run every hypothetical. The one
quantity you genuinely cannot guess for an agent is **completion length** — and,
per Augur's thesis, it's where the cost surprise lives (long outputs in the
tail). So we learn output length from a recorded trace, then use it to project
cost for inputs we never executed: a bigger prompt, a context-growth scenario,
more runs.

This is the PreflightLLMCost direction — an honest linear baseline, not a deep
model — and it's where a second language earns its place: the analytical fit
(numpy OLS, residual spread, prediction bands) is more natural in Python, while
the systems core stays in Go.

## Loose coupling: the trace file is the only contract

The sidecar never calls the Go binary and the Go binary never calls the sidecar.
They share exactly one thing: the JSONL cost-trace schema. `fit` reads a recorded
trace; `emit-trace` writes a synthetic one that `augur aggregate | project |
gate` price with no idea a model — not a proxy — produced it. No RPC, no shared
library, no import in either direction.

```
trace.jsonl ──► augur-predict fit ──► model.json
                                          │
                       augur-predict emit-trace --input-scale 1.5
                                          │
                                          ▼
                              predicted-trace.jsonl ──► augur aggregate | project | gate
```

## Install

Needs Python ≥ 3.10 and numpy. No install required to run:

```sh
cd sidecar
python -m augur_predict --help          # run in place
# or:
pip install -e .                        # exposes the `augur-predict` command
pip install -e '.[dev]' && pytest       # with the test suite
```

## Commands

### `fit` — learn the model from a recorded trace

```sh
python -m augur_predict fit --trace trace.jsonl --out model.json
```

Fits, **per billed model**, `output_tokens ≈ intercept + slope · input_tokens`,
and captures each observed `(scenario, run)` as a *run template* (its call-graph:
which models, what input sizes, in what order). Only successful calls feed the
fit; a failed-but-billed call (a 429 that burned input and produced nothing)
would distort an output-length fit, so it's excluded — but it still contributes
its structure to the templates, because it's part of the call graph `emit-trace`
replays.

`--dist` chooses how the output spread is modelled:

| `--dist` | what it fits | use it when |
|---|---|---|
| `gaussian` *(default)* | OLS mean line + a **symmetric** residual band | the honest baseline; output is roughly symmetric around the trend |
| `quantile` | a grid of **conditional quantiles** by quantile regression | you care about the **tail** — output is right-skewed, and the p95 the gate uses is what matters |

See [Tail accuracy](#tail-accuracy-quantile-mode--run-correlation) below for why
the default isn't always enough.

### `report` — is the fit worth trusting?

```sh
python -m augur_predict report --model model.json
```

```
Output-length model  (source: trace.jsonl)
  100 calls, 50 run templates, 2 model(s)

  model                 n  method      out~in     R2   ±resid
  ----------------------------------------------------------
  gpt-4o               50   ols    65+0.299·in   0.97       30
  gpt-4o-mini          50   ols    22+0.017·in   0.60        5
```

Honesty is a first-class output. With too few points or no spread in the inputs,
a slope is noise: the model degrades to predicting the **mean** output
(`method=mean`) and says so. A low R² is flagged. This mirrors the Go side
reporting confidence intervals instead of bare point estimates.

A quantile model reports the **median and p95 lines** side by side (the two the
gate cares about) plus the Koenker–Machado pseudo-R¹, and flags any model that
fell back to the empirical marginal quantiles (too few points to regress).

### `predict` — one input size

```sh
python -m augur_predict predict --model model.json \
    --model-name gpt-4o --input-tokens 3000 --price-out 10.0
```

```
model=gpt-4o input_tokens=3000.0
  predicted output tokens: 962  (~95% band 903-1021, method=ols)
  est. output cost @ $10.0/Mtok: $0.009622  (p95 $0.010214)
```

A gaussian model prints a point prediction and a symmetric `~95%` band from the
residual spread; a quantile model prints the **p50 and p95** directly. `--price-out`
is optional and prices only the *completion* side — the unknown the sidecar
models; the prompt cost is already exact from the input you supplied. Full
per-call pricing lives in the Go `aggregate` stage, which `emit-trace` feeds.

### `emit-trace` — a predicted trace for the Go gate

```sh
python -m augur_predict emit-trace --model model.json \
    --out predicted-trace.jsonl --runs 50 --input-scale 1.5 --seed 7
```

Resamples the run templates, scales every prompt by `--input-scale`, and draws
each call's output from the fit — sampling the *spread*, not just the mean, so
the predicted distribution (and the p95 the gate cares about) stays honest
instead of collapsing to the average. A gaussian model samples its symmetric
band; a quantile model inverts its conditional CDF, preserving the skew. The
result is an ordinary trace:

```sh
augur aggregate --trace predicted-trace.jsonl
augur gate --trace predicted-trace.jsonl --traffic traffic.yaml --budget budget.yaml
```

`--input-scale` is the predictive analogue of the Go `--context-growth` knob, but
it does more: it feeds the larger prompt *back through the output model*, so a
bigger ask predicts a longer answer — not just a costlier prompt.
`--run-correlation ρ` (0–1, default 0) shares a *verbosity* draw across a run's
calls (see below). Emission is **deterministic per `--seed`** (same seed → same
trace), carrying the Go side's record/replay determinism into the predictive
path.

## Worked example

```sh
augur run --scenarios scenarios.yaml --upstream https://api.openai.com   # record once
python -m augur_predict fit --trace trace.jsonl --out model.json
# "What if prompts grow 1.5× as conversations lengthen?" — no agent re-run:
python -m augur_predict emit-trace --model model.json --out grown.jsonl --input-scale 1.5
augur gate --trace grown.jsonl --traffic traffic.yaml --budget budget.yaml
```

In Augur's own test of this chain, the recorded trace passed the budget
(~$15.3k/month) while the 1.5× predicted trace failed it (~$22.4k/month, over a
$20k cap) — the cost regression caught *before* anyone ran the bigger workload.

## Tail accuracy: quantile mode & run correlation

The whole tool gates on **p95**, because the cost surprise lives in the tail. The
gaussian default has two assumptions that *understate* that tail, and `quantile`
mode plus `--run-correlation` are the opt-in fixes:

1. **Skew.** A symmetric Gaussian band is thin-tailed; real completion lengths
   are right-skewed (rambles, near-`max_tokens` runs), so the gaussian p95 sits
   too low. **`--dist quantile`** fits the conditional quantiles directly — it
   targets the p95 the gate uses and represents the asymmetry instead of a
   symmetric ± band. Quantile regression is solved exactly as a linear program
   (`scipy.optimize.linprog`, HiGHS); crossing quantiles are repaired by
   monotone rearrangement at prediction time.

2. **Run-level correlation.** Sampling each call independently understates the
   variance of the *per-run* total — but the gate aggregates to per-run cost and
   then takes its p95, so that spread is exactly what's gated. Real runs
   correlate (a verbose run is verbose throughout). **`--run-correlation ρ`**
   models it with a Gaussian copula: each run draws one latent `z_run`, each call
   blends it as `z = √ρ·z_run + √(1-ρ)·z_idio`. ρ=0 is independent (the default);
   higher ρ widens the per-run p95.

In a heavy-tailed test against the recorded ground truth (p95 ≈ \$0.0130/req),
the **quantile** projection (\$0.0126, 95% CI [\$0.0117, \$0.0138]) covered the
true value, while the **gaussian** projection (\$0.0121, CI [\$0.0115, \$0.0124])
sat lower with a falsely tight interval that *missed* it — the precise failure
mode the modes exist to fix.

Both stay opt-in: `gaussian`/`ρ=0` is the conservative default so you never get
unmodelled assumptions you didn't ask for.

## Honest limitations

- **Still a one-feature model.** Both modes use only `input_tokens` as the
  predictor. When prompt length doesn't drive output length (low R²/R¹), the fit
  falls back — to the mean (gaussian) or the marginal quantiles (quantile) — and
  the report flags it. Don't read more into a prediction than its goodness-of-fit
  supports.
- **It can't invent structure it never saw.** `emit-trace` resamples observed run
  templates; it cannot predict a call path the agent never took in the recorded
  trace. Representativeness of the original run still bounds everything.
- **The tail is data-bound.** A p95 from a few dozen runs is intrinsically noisy
  no matter the model, which is why the grid stops at 0.95 (a p99 would be false
  precision) and why fit quality is always reported. Quantile mode sharpens the
  *shape*; it does not manufacture tail data you didn't record.

## Tests

```sh
cd sidecar && pytest -q
```

57 cases covering the trace round-trip and Go-schema contract; OLS recovery of a
known slope/intercept and the mean fallback; quantile-mode ordering (p05 ≤ p50 ≤
p95), asymmetric bands on skewed data, quantile p95 > gaussian p95, and the
empirical-quantile fallback; that `--run-correlation` widens the per-run total
variance; and `emit-trace` determinism + validity (cached ≤ input, non-negative
tokens) so emitted rows never fail the Go cost validator.
