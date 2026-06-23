# Dogfooding Augur on CloudOracle's Insights Agent

> Augur measuring a *real* agent, not a synthetic trace: the Insights Agent from
> [**CloudOracle**](https://github.com/Cro22/CloudOracle) — a LangGraph supervisor
> multi-agent with tool calls and guardrails.

CloudOracle is Augur's sibling project: FinOps for *cloud* (runtime, post-hoc),
where Augur is FinOps for *AI agents* (pre-prod, predictive). Dogfooding Augur on
CloudOracle's own agent closes the loop between the two — and is the first test of
Augur against an agent it didn't author.

This is the first dogfood: point Augur at an agent it didn't author and see
whether the cost picture holds up. It does — and it surfaced real findings,
which is the point of dogfooding.

## The integration: a callback shim, not the proxy

Augur's recording proxy is OpenAI-compatible, but the Insights Agent talks to its
model **natively** through LangChain (no OpenAI `base_url` to redirect). So we use
the capture path the [SPEC](../../SPEC.md) documents as the fallback "for
frameworks where the proxy is awkward" (ADR D1): a **LangChain callback** that
writes Augur's exact `trace.jsonl` schema from the token usage every call reports.

The agent reuses one chat model across the supervisor, every specialist's ReAct
loop, and the synthesizer, so a callback on that model sees the **entire call
graph** of a run — the real agentic fan-out. Coupling is one direction only:
[`augur_dogfood.py`](augur_dogfood.py) *imports* the agent's public building
blocks (`build_supervisor_graph`, `build_tools`, `run_guarded`) and builds them
with our model; **CloudOracle's source is never edited.** Delete this folder and
both repos are untouched.

## Files

| file | what |
|---|---|
| [`augur_dogfood.py`](augur_dogfood.py) | the harness: builds the agent graph with our model + trace callback, runs the scenarios |
| [`scenarios.yaml`](scenarios.yaml) | representative FinOps questions, spread across the agent's three specialists |
| [`traffic.yaml`](traffic.yaml) | production volume assumptions for the projection |
| [`budget.yaml`](budget.yaml) | the thresholds the gate enforces |
| [`pricing-gemini.yaml`](pricing-gemini.yaml) | a Gemini price snapshot (for the `--provider gemini` path; the Claude run uses Augur's default `pricing.yaml`, which already lists Claude) |

## Running it

Prereqs — bring up CloudOracle's stack so the agent's tools have data to fetch:

```sh
cd /path/to/CloudOracle
docker compose up -d postgres                       # pgvector
go build -o oracle.exe ./cmd/oracle
# DB defaults (oracle/oracle_dev/cloudoracle) match docker-compose; seed + serve:
CLOUDORACLE_PROVIDER=synthetic LOG_LEVEL=info ./oracle.exe seed --count 120
CLOUDORACLE_PROVIDER=synthetic LOG_LEVEL=info ./oracle.exe serve --port 8080 &
```

Then run the agent under the Augur callback (uses CloudOracle's venv + `.env`):

```sh
set -a; source insights-agent/.env; set +a       # GEMINI/ANTHROPIC + CLOUDORACLE keys
unset DATABASE_URL                               # RAG off for a clean first run (see caveats)

insights-agent/.venv/Scripts/python.exe \
  /path/to/Augur/examples/cloudoracle/augur_dogfood.py \
  --scenarios /path/to/Augur/examples/cloudoracle/scenarios.yaml \
  --out /path/to/Augur/cloudoracle-trace.jsonl \
  --provider anthropic --runs 5 --no-judge
```

Pipe the real trace through Augur (Claude prices ship in the default snapshot):

```sh
cd /path/to/Augur
./augur gate --trace cloudoracle-trace.jsonl \
    --traffic examples/cloudoracle/traffic.yaml \
    --budget  examples/cloudoracle/budget.yaml \
    --pricing pricing.yaml

# And learn the output-length model from the real run:
cd sidecar && python -m augur_predict fit --trace ../cloudoracle-trace.jsonl \
    --out ../cloudoracle-model.json --dist quantile
python -m augur_predict report --model ../cloudoracle-model.json
```

`--provider gemini` runs the same harness on Gemini (add `--rps 0.15` to stay under
the free-tier rate limit); `--provider anthropic` (default) uses Claude, whose
higher limits make a clean multi-call run practical.

## What Augur measured (Claude Haiku 4.5, 20 runs)

```
scenario "find-savings"  — 5 run(s)
  metric     mean      p50       p95       stdev     min       max
  $/run      0.022428  0.016974  0.038833  0.012192  0.016830  0.044236
  calls/run  6.60      5.00      11.40     3.58      5         13      <-- the tail
```

| scenario | calls/run (p50→p95) | $/run p95 | note |
|---|---|---|---|
| cost-breakdown | 4 → 4 | $0.0069 | input-heavy, cheap output |
| concept-rightsizing | 4 → 4 | $0.0105 | stable |
| full-review | 5 → 5 | $0.0197 | the deliberate multi-specialist path |
| **find-savings** | **5 → 11.4 (max 13)** | **$0.0388** | **the savings specialist sometimes loops** |

**Gate verdict: ✅ PASS** — projected `$/request p95 = $0.0198` (budget $0.02 — it
*just* fits), `$39.72/tenant/month`, `$794/month` at the assumed volume.

The headline is `find-savings`: its **p95 cost is 2.3× its median**, driven by a
call-count tail (5 → 13 calls/run) when the savings specialist's ReAct loop keeps
going. That is precisely the agentic cost driver Augur exists to catch — observed
on a real agent, not assumed. Gating on the mean would have hidden it; gating on
p95 (and the wide p95 CI `[$0.019, $0.044]`) surfaces it.

## Findings the dogfood surfaced

Dogfooding earns its keep by what breaks:

- **The agent isn't provider-portable.** Built for Gemini, its LLM-judge layer
  sends a *system-only* message — which Gemini tolerates but Anthropic rejects
  (`messages: at least one message is required`). `--no-judge` works around it for
  the Claude run; the fix belongs in CloudOracle, and Augur found it. (This is a
  bug in the agent, not in Augur.)
- **Claude Haiku hallucinates figures on this agent.** Most `find-savings` /
  `full-review` runs hit the agent's *deterministic grounding* fallback ("answer
  states $320.00 … not found in any tool result"). A quality signal, orthogonal
  to cost — Augur still measured the real token spend of those runs.
- **The sidecar's output model is weak here (R¹ ≈ 0.03).** Output length isn't
  driven by input length for this agent; it's driven by the call's *role* (a
  router turn is terse, a synthesis turn is long). The sidecar reports the weak
  fit honestly rather than faking a trend — and it motivates the documented next
  step: segment the output model by call role / seq, not just `input_tokens`.

## Caveats / not captured

- **RAG embeddings.** The callback rides the *chat* model; the agent's embeddings
  go through a separate Gemini embeddings client, so RAG retrieval calls aren't in
  this trace. `DATABASE_URL` is unset here to keep the run clean. Capturing them
  would mean instrumenting `GeminiEmbeddingsProvider` too.
- **Synthetic backend.** CloudOracle serves synthetic resources (`seed`), so the
  *tool* outputs are representative-shaped, not a real cloud bill. The *agent's*
  token usage — what Augur measures — is real.
- **Small N.** 20 runs is enough to see the find-savings tail but thin for a
  precise p95; the gate's wide CI says so.
