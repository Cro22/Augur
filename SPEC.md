# Augur — Spec: Shift-Left FinOps Cost Gate for AI Agents

> **Name:** Augur — *a cost-first FinOps gate for AI agents. Know the bill before you ship.*
> **Status:** Proposed — for review by Jesús before Claude Code starts Hito 0
> **Owner of architecture:** Jesús. **Executor:** Claude Code.
> **Relationship:** Sibling to CloudOracle. CloudOracle = FinOps for *cloud* (runtime, post-hoc). This = FinOps for *AI agents* (pre-prod, predictive). Linked by narrative and philosophy, **not** by shared code.

---

## 1. What this is (one line)

A **cost-first, shift-left FinOps gate** for AI-agent applications: run your agent against a representative workload *before* production, project the **unit economics at scale** (cost per request / per user / per tenant), and **fail the build** if it blows your budget.

The thesis: today you find out an agent is financially unsustainable *after* you ship and the bill arrives. This catches it in CI.

---

## 2. Honest positioning (read before pitching this)

This is **not** a virgin space, and the README must say so. Pretending "no tool does pre-prod cost" is false and will get torn apart in an interview. What actually exists:

- **Token calculators** (LiteLLM Pricing Calculator, dozens of tiktoken-based ones). You type *average* tokens and volume by hand and get a number. Useless for agents — you can't guess an agent's token usage, because one user request fans out into 5–10 LLM calls with retries and tool loops.
- **Eval / CI platforms** (Braintrust, Confident AI, Maxim, Promptfoo, FutureAGI, Langfuse, etc.). These run your agent against a dataset in CI and gate the merge. Several attach token cost per call/tool-call. **But cost is a second-class citizen** — they are *quality* gates (right tool? hallucination? grounded?) and cost is a number bolted on the side.
- **Predictive research** (e.g. PreflightLLMCost) — narrow, focused on predicting output length. Not a productized gate.
- **Academic readiness harnesses** — produce "cost–utility frontiers" for deploy decisions, but research-grade and bundle cost into a multi-objective Pareto thing.

### What makes this different (the defensible wedge)
1. **Cost-first, not cost-as-byproduct.** The primary output is unit economics, not a quality score.
2. **Projects to production scale.** Not "this run cost $0.04" — "at your projected volume this is $X/tenant/month, which breaks your budget."
3. **Models the agentic cost drivers from real runs.** Retries, tool-loop fan-out, sub-agent fan-out, context/history growth — measured by running the agent N times and capturing the cost *distribution*, not a single naive estimate. This is the systems/control-flow angle, not token arithmetic.
4. **Complements eval gates; does not replace them.** Quality gate answers "is it good?"; this answers "can we afford it?".

**Pitch line:** *"A cost-first FinOps gate that models agentic cost drivers, because the eval tools treat cost as an afterthought."* True, narrow, defensible.

**Reality check for CV vs business:** as a *business* this gets crushed by any eval incumbent adding a cost tab — there is no commercial moat. As a *portfolio piece* it's strong: it compounds the CloudOracle FinOps narrative and shows real systems depth (proxy, distributions, projection). Build it as a portfolio asset, not a startup.

---

## 3. Scope

### In scope (v1)
- OpenAI-compatible **recording proxy** to capture real LLM calls from any agent.
- **Scenario runner** that executes the agent against representative inputs N times and aggregates a cost distribution.
- **Projection engine** that turns the distribution + a traffic profile into unit economics with confidence intervals.
- **Gate** + report: pass/fail against a budget, with a human-readable artifact for the PR.
- CLI-first.

### Out of scope (v1) / explicit non-goals
- Quality / correctness evaluation (that's what eval tools are for — we link to them, we don't compete).
- Runtime / production observability (that's CloudOracle's lane, conceptually).
- Auto-optimization ("switch to a cheaper model for you"). We *report*, we don't *act* — same philosophy as CloudOracle's deterministic, no-surprise behavior.
- Anything that requires the agent to be rewritten to adopt the tool.

---

## 4. Core concepts

| Concept | What it is |
|---|---|
| **Scenario** | A representative input / persona / conversation flow the agent will face (`scenarios.yaml`). The user supplies these — quality of projection depends on representativeness. |
| **Run** | One execution of the agent against one scenario. Agents are non-deterministic and output length varies, so we run each scenario N times to get a **cost distribution**, not a point estimate. |
| **Cost trace** | The recorded ledger of every LLM call in a run: model, input/output/cached tokens, latency, scenario_id, run_id, call ordering. Persisted by the proxy (JSONL or SQLite). |
| **Traffic profile** | Production assumptions (`traffic.yaml`): users, requests/user/day, tenants, peak factor, and assumptions for drivers we can't observe directly (e.g. expected retry rate in prod). |
| **Projection** | Distribution × traffic profile → $/request (p50, p95), $/user/day, $/tenant/month, projected monthly bill, with CIs and a breakdown by step/model. |
| **Budget** | Thresholds (`budget.yaml`): max $/request (p95), max $/tenant/month, max projected monthly. |
| **Gate** | Compares projection (p95) to budget. Over → exit non-zero, fail the build. |

---

## 5. Architecture

### 5.1 Data flow

```
scenarios.yaml ─┐
                ├─► [Go] Scenario Runner ──triggers──► Agent under test
traffic.yaml ──┐│                                          │
budget.yaml ──┐││                                  LLM calls (base_url ──►)
              │││                                          ▼
              │││                                 [Go] Recording Proxy ──► provider
              │││                                          │
              │││                                  writes cost trace
              │││                                          ▼
              │││                                  trace.jsonl / trace.db
              │││                                          │
              │└┴────────────────────────────────► [Go] Projection Engine
              │                                            │
              │                                  report.json + report.md
              │                                            ▼
              └──────────────────────────────────► [Go] Gate ──► exit 0/1 + PR comment
```

### 5.2 Language: Go-first (single language for v1)
- **v1 is entirely Go** — proxy, CLI, runner, projection/stats, gate, CI integration, all in one static binary. The v1 projection is descriptive statistics (mean, p50, p95, stddev, bootstrap CIs), which Go handles fine (`gonum/stat`); there is no heavy ML in v1, so no need to drag in Python and a second toolchain.
- **Why not polyglot from day 0:** this is a parallel side-project — fewer moving parts means it actually ships and stays maintained. It also keeps Augur distinct from CloudOracle, which is *already* the Go-CLI-plus-Python-agent showcase; a clean pure-Go systems tool is a second, different artifact and showcases Go (the differentiator) harder toward the Inference/Infra layer.
- **Where Python earns its place (Hito 5, optional):** the predictive output-length model. Introduce it then as an optional sidecar, communicating via the cost-trace file (loose coupling, no RPC). Polyglot becomes earned, not forced.

---

## 6. Key design decisions (mini-ADRs)

### D1 — Observation via OpenAI-compatible **proxy** (not an SDK wrapper)
- **Decision:** Capture calls by having the agent point its `base_url` at our local proxy.
- **Pros:** framework- and language-agnostic (works with any agent that can set a base URL), captures the *real call graph* including retries/fan-out, more product-like, demonstrates systems chops.
- **Cons:** agent must support a custom base URL (almost all do); streaming responses need careful token accounting.
- **Alternative (documented fallback):** a callback/SDK wrapper for frameworks where the proxy is awkward. Not v1.

### D2 — Gate on **p95**, not mean
- **Decision:** Budget comparisons use the p95 of the projected cost, not the average.
- **Why:** the cost surprise lives in the tail — long outputs, retry storms, tool loops. The mean hides exactly the failure mode this tool exists to catch. Conservative by design.

### D3 — **Run the real agent N times** vs static token math
- **Decision:** Execute the agent against scenarios N times (default ~20) and build an empirical distribution.
- **Why:** this is the entire differentiator vs calculators — real token counts, real multipliers, real variance.
- **Trade-off:** running the agent in CI costs real tokens. Mitigations (some are stretch): keep the scenario set small, support a **record-once / replay** mode (Hito 5) so reruns don't re-spend, and let the user set N.

### D4 — Pricing data as updatable config
- **Decision:** ship `pricing.yaml` (per model: input / output / cached $ per Mtok) as a snapshot, clearly dated, with a note that provider prices drift constantly.
- **Stretch:** fetch from a maintained source.

---

## 7. Component breakdown

- **`proxy/`** (Go) — OpenAI-compatible HTTP proxy. Forwards requests to the real provider, parses `usage` from responses (and tokenizes/counts for streaming), and appends a trace row tagged with scenario_id + run_id (passed via header). Non-streaming first, streaming second.
- **`runner/`** (Go) — reads `scenarios.yaml`, drives the agent against each scenario ×N (via a user-provided command/entrypoint, or the user runs their own harness and only needs to set the scenario/run headers).
- **`cost/`** (Go) — single-call cost computation from `pricing.yaml`. Pure, fully unit-tested. The foundation.
- **`projection/`** (Go) — reads the trace, builds per-scenario distributions, decomposes by step/model, computes observed multipliers, applies the traffic profile, emits `report.json` + `report.md` with CIs (descriptive stats via `gonum/stat`).
- **`gate/`** (Go) — reads report + `budget.yaml`, evaluates p95 thresholds, sets exit code, prints failing lines.
- **`ci/`** — GitHub Action wrapper (Hito 5).
- **Tests** — Go `testing` throughout. (CloudOracle had 469+ tests; hold the same bar.)

---

## 8. Milestone plan (hitos — each independently verifiable)

> Work one hito at a time. **Jesús reviews each checkpoint before the next starts.** This is how he owns the architecture without writing every line.

### Hito 0 — Foundation
- Repo scaffold (single Go module), `pricing.yaml`, `cost/` single-call cost computation.
- **✅ Checkpoint:** unit test — given (model, input_tokens, output_tokens, cached_tokens), returns correct cost. Edge cases: unknown model, zero tokens, cached pricing.

### Hito 1 — Recording proxy
- OpenAI-compatible passthrough proxy that logs model + tokens + latency + scenario_id + run_id to `trace.jsonl`/`trace.db`. Non-streaming first, then streaming.
- **✅ Checkpoint:** point a trivial agent (or a curl loop) at the proxy; confirm trace rows match the provider's reported `usage` exactly. Streaming totals reconcile with non-streaming.

### Hito 2 — Scenario runner + aggregation
- Run the agent against `scenarios.yaml` ×N; aggregate per-scenario cost distribution (mean / p50 / p95 / stdev), decompose by step/model, surface observed multipliers (calls/request, retries, fan-out).
- **✅ Checkpoint:** distribution table for a known scenario set; numbers reconcile against the raw trace by hand.

### Hito 3 — Projection engine (Go)
- Distribution + `traffic.yaml` → $/request (p50, p95), $/user/day, $/tenant/month, projected monthly bill, each with a confidence interval, plus a cost breakdown by step/model.
- **✅ Checkpoint:** feed a synthetic distribution + traffic profile; assert projected numbers against a hand calculation.

### Hito 4 — Gate + report
- `budget.yaml`, p95-based pass/fail, exit codes, `report.md` + `report.json`.
- **✅ Checkpoint:** one budget that passes and one that fails → correct exit codes and a clean, readable report.

### Hito 5 — Stretch (each independently shippable)
- GitHub Action (fail PR + post report comment).
- Explicit multiplier **what-if knobs** (retry rate / fan-out / context-growth) for sensitivity analysis without re-running.
- **Record-once / replay** mode to cut CI token cost.
- Self-hosted **TCO mode** (throughput tokens/sec → instance $/hr → effective $/Mtok), following the standard benchmark-then-size approach.
- Simple predictive output-length model to estimate cost without running (the PreflightLLMCost direction) — **this is where Python enters**, as an optional sidecar (the analytical/ML piece that earns the second language).

---

## 9. Tech stack

- **Go (everything in v1):** standard library + a CLI lib (e.g. cobra); `gonum/stat` for percentiles/stats; SQLite (pure-Go driver) or plain JSONL for the trace.
- **Python (Hito 5 only, optional):** the predictive output-length model (e.g. scikit-learn) as a sidecar — not part of v1.
- **Config:** YAML (`scenarios.yaml`, `traffic.yaml`, `budget.yaml`, `pricing.yaml`).
- **Tests:** Go `testing`. CI on every push.

---

## 10. Open questions for Jesús (decide before/at Hito 0)

1. **First dogfood target.** Strong candidate: **CloudOracle's own Insights Agent** (LangGraph multi-agent + RAG = real, messy cost drivers, and it ties the two repos together in the story). Alternatives: a toy LangGraph agent, or Despachito if it gains an LLM feature. → *Recommend CloudOracle's Insights Agent.*
2. **Pricing data:** ship a dated snapshot only (v1), or attempt a live fetch?
3. **CI token-cost tolerance:** how much real spend per CI run is acceptable? This decides whether record/replay is v1 or stretch.
4. **Proxy vs harness for the runner:** does Jesús want the tool to *drive* the agent (needs an entrypoint contract), or just provide the proxy + headers and let the user run their own harness? (Lighter v1 = the latter.)

---

## 11. Naming

**Chosen: Augur.** A seer who reads omens to foretell what's coming — here, the agent's production cost before it ships. Thematically a sibling to CloudOracle (both divination) without repeating "Oracle".

*Note:* there is a known crypto project also called Augur (decentralized prediction markets). For a personal portfolio repo this is fine — different domain — but the name is taken if this were ever productized or trademarked.
