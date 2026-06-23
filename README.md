# Augur

> A cost-first FinOps gate for AI agents. **Know the bill before you ship.**

Augur runs your AI agent against a representative workload *before* production,
projects its **unit economics at scale** (cost per request / per user / per
tenant / per month), and **fails the build** if it blows your budget.

The thesis: today you find out an agent is financially unsustainable *after* you
ship and the bill arrives. Augur catches it in CI.

```
$ augur gate
# Augur cost report
**Verdict: ❌ FAIL** — 1 of 3 budget check(s) over limit

| check          | limit      | projected  | status |
|----------------|-----------:|-----------:|:------:|
| $/request p95  | $0.030000  | $0.043125  |   ❌   |
| $/tenant/month | $500.00    | $360.34    |   ✅   |
| monthly bill   | $20000.00  | $18016.88  |   ✅   |

$ echo $?
1   # the build fails
```

---

## Honest positioning (read this before judging it)

This is **not** a virgin space, and pretending otherwise would be dishonest.
What already exists:

- **Token calculators** (LiteLLM's calculator, dozens of tiktoken-based ones).
  You type *average* tokens and volume by hand. Useless for agents — you can't
  guess an agent's token usage, because one user request fans out into many LLM
  calls with retries and tool loops.
- **Eval / CI platforms** (Braintrust, Confident AI, Maxim, Promptfoo, Langfuse,
  …). They run your agent against a dataset in CI and gate the merge, and
  several attach token cost per call. **But cost is a second-class citizen** —
  they are *quality* gates (right tool? hallucination? grounded?) with a cost
  number bolted on the side.
- **Predictive research** (e.g. PreflightLLMCost) — narrow, focused on
  predicting output length. Not a productized gate.

### What makes Augur different (the defensible wedge)

1. **Cost-first, not cost-as-byproduct.** The primary output is unit economics,
   not a quality score.
2. **Projects to production scale.** Not "this run cost $0.04" — "at your
   projected volume this is $X/tenant/month, which breaks your budget."
3. **Models agentic cost drivers from *real* runs.** It runs the agent N times
   and captures the cost *distribution* (retries, tool-loop fan-out, variance),
   not a single naive token estimate.
4. **Complements eval gates; doesn't replace them.** Quality gate: "is it good?"
   Augur: "can we afford it?"

**Honest reality check:** as a *business* this gets crushed by any eval
incumbent adding a cost tab — there's no commercial moat. This is built as a
**portfolio piece**: a clean, pure-Go systems tool (recording proxy, empirical
distributions, bootstrap projection, CI gate) that shows real depth.

---

## How it works

```
scenarios.yaml ─┐
                ├─► augur run ──drives──► your agent ──LLM calls──► augur proxy ──► provider
                │                              (sets X-Augur-* headers)  │
                │                                              writes cost trace
                │                                                        ▼
                │                                                  trace.jsonl
pricing.yaml ───┤                                                        │
traffic.yaml ───┼────────────────────────────► augur project ◄───────────┘
budget.yaml  ───┘                                     │  (distributions × traffic)
                                                       ▼
                                              augur gate ──► report.md / report.json + exit 0/1
```

The proxy captures calls at the **HTTP layer** (you point your agent's
`base_url` at it), so Augur is framework- and language-agnostic and records the
*real* call graph — retries and fan-out included — without you rewriting the
agent.

---

## Quickstart

Build:

```sh
go build -o augur .
```

### 1. Make your agent honour the contract

`augur run` injects four environment variables per invocation. Your agent's only
job is to read them — typically one line of OpenAI-client config:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url=os.environ["AUGUR_BASE_URL"],            # point at the proxy
    default_headers={
        "X-Augur-Scenario-Id": os.environ["AUGUR_SCENARIO_ID"],
        "X-Augur-Run-Id":      os.environ["AUGUR_RUN_ID"],
    },
)
prompt = os.environ["AUGUR_INPUT"]                    # the scenario input
# ... run your agent as usual ...
```

### 2. Describe your scenarios, traffic, and budget

See [`examples/`](examples/) for documented templates:

- **`scenarios.yaml`** — representative inputs + the agent command + how many
  times to run each (the projection is only as good as these inputs).
- **`traffic.yaml`** — production assumptions (users, requests/user/day, tenants,
  scenario mix).
- **`budget.yaml`** — the thresholds that fail the build.
- **`pricing.yaml`** — a dated price snapshot (ships with the repo).

### 3. Run the pipeline

```sh
# Drive the agent N times per scenario through the recording proxy.
augur run --scenarios scenarios.yaml --upstream https://api.openai.com

# Eyeball the observed per-scenario cost distribution.
augur aggregate --trace trace.jsonl

# Project to production scale with confidence intervals.
augur project --traffic traffic.yaml

# Gate it: writes report.md / report.json, exits non-zero if over budget.
augur gate --traffic traffic.yaml --budget budget.yaml
```

`augur gate` is the one you wire into CI.

### Record once, replay for free

Running the agent against the real provider on every CI push spends real tokens.
Record the responses once, then replay them — `augur run --replay` re-executes
the agent against the recorded responses and regenerates the trace **without
contacting the provider**:

```sh
# Once (locally or nightly), against the real provider:
augur run --scenarios scenarios.yaml --record cassette.jsonl

# In CI, on every push — zero tokens spent:
augur run --scenarios scenarios.yaml --replay cassette.jsonl --trace trace.jsonl
augur gate --traffic traffic.yaml --budget budget.yaml
```

Because replay re-runs the *agent* (not just the trace), a cost regression from
an agent code change still surfaces — exercised against the old responses, for
free. Commit `cassette.jsonl` alongside your scenarios. (A call the agent makes
that wasn't recorded is a replay miss — reported loudly, so divergence from the
recording never passes silently.)

### What-if sensitivity analysis

`project` and `gate` take what-if multipliers that re-cost the *recorded* trace
under hypothetical agentic cost drivers — no agent re-run, no tokens:

```sh
# "What if retries climb 30%, sub-agent fan-out adds 50% more calls, and
#  context grows to 2× as conversations lengthen?"
augur project --retry-rate 0.3 --fanout 1.5 --context-growth 2

# Gate against a pessimistic scenario, not just today's happy path:
augur gate --context-growth 1.5 --budget budget.yaml
```

Retries and fan-out scale the whole call (more calls); context growth inflates
only the prompt side, not the completion — so the model reflects *which* driver
moved, not a flat fudge factor.

### Predict cost without running (Python sidecar)

The what-if knobs above re-cost a recorded trace, but they hold output length
fixed. The one thing you genuinely can't guess for an agent is **completion
length** — and it's where the cost surprise lives. The optional
[`sidecar/`](sidecar/) (Python) learns output length from a recorded trace, then
projects cost for inputs you never ran:

```sh
# Learn output_tokens ≈ a + b·input_tokens per model, from one recorded run.
python -m augur_predict fit --trace trace.jsonl --out model.json

# "What if prompts grow 1.5× as conversations lengthen?" — predict, don't run.
python -m augur_predict emit-trace --model model.json --out grown.jsonl --input-scale 1.5

# The predicted trace flows through the normal gate — no tokens spent.
augur gate --trace grown.jsonl --traffic traffic.yaml --budget budget.yaml
```

Coupling is the trace file and nothing else — no RPC, no shared library. Unlike
`--context-growth`, the sidecar feeds the larger prompt *back through the output
model*, so a bigger ask predicts a longer answer, not just a costlier prompt.
The fit is an honest linear baseline (it reports R² and falls back to the mean
when the signal is weak). See [`sidecar/README.md`](sidecar/README.md).

### Self-hosted models (TCO)

For a model you run yourself there's no per-token API price — you pay for an
instance by the hour. Describe the deployment in `tco.yaml` (instance $/hr,
serving throughput, utilization) and Augur derives the effective $/Mtok:

```sh
augur tco --tco tco.yaml          # show the derived effective $/Mtok
augur gate --tco tco.yaml ...     # cost the trace against self-hosted pricing
```

`--tco` is accepted by `aggregate`, `project`, and `gate` as an alternative to
`--pricing`. Utilization is the honest part: you pay for the box 24/7 but it's
rarely saturated, and a half-idle instance doubles the effective token price.

### GitHub Action

Augur ships as a composite action that runs the gate on a pull request, posts
the report as a comment, and fails the check if the projection is over budget.
Combined with a committed cassette, the PR check spends no tokens:

```yaml
# .github/workflows/cost-gate.yml
name: Cost gate
on: pull_request
jobs:
  augur:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write   # to post the report comment
    steps:
      - uses: actions/checkout@v4
      - uses: your-org/augur@v1
        with:
          cassette: cassette.jsonl   # replay → zero tokens
          traffic: traffic.yaml
          budget: budget.yaml
```

See [`action.yml`](action.yml) for all inputs and
[`examples/github-workflow.yml`](examples/github-workflow.yml) for a fuller
example.

---

## Design decisions

- **Gate on p95, not the mean.** The cost surprise lives in the tail — long
  outputs, retry storms, tool loops. The mean hides exactly the failure mode
  Augur exists to catch. Conservative by design.
- **Observation via a proxy, not an SDK wrapper.** Works with any agent that can
  set a base URL; captures the real call graph.
- **The trace records tokens, not dollars.** Cost is computed downstream, so the
  same trace can be re-priced against a different snapshot without re-running the
  agent.
- **Pricing is an updatable, dated snapshot.** Provider prices drift constantly;
  `pricing.yaml` says when it was captured. Do not assume it's current.

## Honest limitations

- **Streaming usage** is captured exactly only when the client sets
  `stream_options.include_usage` (the provider then emits a usage block). Without
  it, Augur records a zero-token row rather than tokenizing on the proxy side —
  proxy-side tokenization is fragile (per-model tokenizers) and `include_usage`
  is the correct, exact path.
- **Multipliers.** Augur reports *calls per run* — the multiplier it can observe
  truthfully. Classifying those calls into retries vs sub-agent fan-out needs
  labeling the trace does not yet carry.
- **Running the agent in CI spends real tokens.** Keep the scenario set small and
  `runs` modest, or record once and replay (`--record`/`--replay`) so CI pushes
  spend nothing.

## Status

**v1 complete** — all milestones implemented and tested (pure Go, only external
dependency is `gopkg.in/yaml.v3`):

| | |
|---|---|
| Hito 0 | single-call cost computation |
| Hito 1 | OpenAI-compatible recording proxy (streaming + non-streaming) |
| Hito 2 | scenario runner + per-scenario aggregation |
| Hito 3 | projection engine with bootstrap confidence intervals |
| Hito 4 | budget gate + Markdown/JSON report + CI exit codes |
| Hito 5 | record/replay cassette (`--record`/`--replay`), what-if knobs (`--retry-rate`/`--fanout`/`--context-growth`), self-hosted TCO mode (`augur tco`, `--tco`), GitHub Action (`action.yml`), Python output-length prediction sidecar ([`sidecar/`](sidecar/)) |

Every SPEC milestone and stretch is implemented. The Go core is pure Go (only
external dependency `gopkg.in/yaml.v3`); the optional prediction sidecar is
Python (numpy), coupled to the core through the trace file alone.

See [`SPEC.md`](SPEC.md) for the full design.

## Naming

**Augur** — a seer who reads omens to foretell what's coming; here, an agent's
production cost before it ships. (There is an unrelated crypto project of the
same name; this is a personal portfolio repo in a different domain.)

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
