"""Dogfood harness: run CloudOracle's Insights Agent under Augur.

This is the **callback-shim** capture path the SPEC documents as the fallback
to the proxy "for frameworks where the proxy is awkward" (ADR D1). The Insights
Agent talks to Gemini natively through `langchain_google_genai` (no OpenAI
`base_url` to point at Augur's proxy), so instead of a proxy we attach a
LangChain callback to the one chat model the whole graph shares and write the
exact Augur cost-trace schema from the usage every call reports.

Why this is loose coupling, not a CloudOracle change: we import the agent's
public runtime, build it as the CLI does, then set `.callbacks` on its shared
`_chat` model before running. Nothing in CloudOracle is edited. Delete this file
and both repos are untouched. The output is an ordinary `trace.jsonl` that
`augur aggregate | project | gate` and the sidecar consume with no idea a
callback (not the proxy) produced it.

Run it with CloudOracle's venv, from the CloudOracle repo so its `.env` and
`src/` are found:

    cd F:/JetBrains/GoProjects/CloudOracle
    insights-agent/.venv/Scripts/python.exe \
        F:/JetBrains/GoProjects/Augur/examples/cloudoracle/augur_dogfood.py \
        --scenarios F:/JetBrains/GoProjects/Augur/examples/cloudoracle/scenarios.yaml \
        --out F:/JetBrains/GoProjects/Augur/trace.jsonl --runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

# Make the agent importable when run from the CloudOracle repo root.
_AGENT_SRC = Path.cwd() / "insights-agent" / "src"
if _AGENT_SRC.is_dir():
    sys.path.insert(0, str(_AGENT_SRC))

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except Exception as e:  # pragma: no cover - environment guard
    sys.exit(f"augur_dogfood: LangChain not importable ({e}); run with the "
             f"insights-agent venv from the CloudOracle repo root.")


class AugurTraceCallback(BaseCallbackHandler):
    """Writes one Augur trace row per LLM call from the usage LangChain reports.

    The agent reuses a single chat model across the supervisor, every
    specialist's ReAct loop, the synthesizer, and the LLM judge, so a callback
    on that model sees the *entire* call graph of a run — the real agentic
    fan-out Augur exists to measure. `on_llm_end` is the hook: chat models
    populate `usage_metadata` on the returned message (and `llm_output` as a
    fallback), which mirrors the provider's own token report.

    The (scenario_id, run_id, seq) tagging matches what the proxy would stamp
    from request headers; here we set them directly per run. Seq is the call's
    order within the run, preserving the call-graph ordering downstream tools
    mine for multipliers.
    """

    def __init__(self, writer: "TraceWriter", model_fallback: str) -> None:
        self._writer = writer
        self._model_fallback = model_fallback
        self._lock = threading.Lock()
        self.scenario_id = ""
        self.run_id = ""
        self._seq = 0

    def begin_run(self, scenario_id: str, run_id: str) -> None:
        with self._lock:
            self.scenario_id = scenario_id
            self.run_id = run_id
            self._seq = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = _extract_usage(response)
        if usage is None:
            # No usage reported (rare): still record the call with zero tokens
            # so the call graph's shape (fan-out count) stays truthful.
            usage = {"input": 0, "output": 0, "cached": 0}
        model = _extract_model(response) or self._model_fallback
        with self._lock:
            seq = self._seq
            self._seq += 1
            scenario_id, run_id = self.scenario_id, self.run_id
        self._writer.write({
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "scenario_id": scenario_id,
            "run_id": run_id,
            "seq": seq,
            "model": model,
            "input_tokens": usage["input"],
            "output_tokens": usage["output"],
            "cached_tokens": usage["cached"],
            "latency_ms": 0,
            "endpoint": "/v1/chat/completions",
            "status": 200,
        })


def _extract_usage(response: LLMResult) -> dict[str, int] | None:
    """Pull (input, output, cached) tokens from an LLMResult, provider-agnostic.

    Prefers the per-message `usage_metadata` LangChain normalises across
    providers; falls back to the raw `llm_output` usage block. Cached prompt
    tokens, when reported, live in `input_token_details.cache_read` and are a
    SUBSET of input (Augur's convention), so we clamp them to input.
    """
    for gens in response.generations:
        for gen in gens:
            msg = getattr(gen, "message", None)
            um = getattr(msg, "usage_metadata", None) if msg is not None else None
            if um:
                inp = int(um.get("input_tokens", 0) or 0)
                out = int(um.get("output_tokens", 0) or 0)
                details = um.get("input_token_details") or {}
                cached = int(details.get("cache_read", 0) or 0)
                return {"input": inp, "output": out, "cached": min(cached, inp)}

    out_meta = response.llm_output or {}
    usage = out_meta.get("usage_metadata") or out_meta.get("token_usage") or {}
    if usage:
        inp = int(usage.get("input_tokens", usage.get("prompt_token_count", 0)) or 0)
        out = int(usage.get("output_tokens", usage.get("candidates_token_count", 0)) or 0)
        cached = int(usage.get("cached_content_token_count", 0) or 0)
        return {"input": inp, "output": out, "cached": min(cached, inp)}
    return None


def _normalize_model(name: str) -> str:
    """Match Augur's pricing keys to what providers echo back.

    Gemini returns 'models/gemini-2.5-flash'; Anthropic returns a dated id like
    'claude-haiku-4-5-20251001'. Augur's pricing.yaml keys on the bare family
    name, so strip the 'models/' prefix and a trailing -YYYYMMDD date.
    """
    import re
    name = name.strip()
    if name.startswith("models/"):
        name = name[len("models/"):]
    name = re.sub(r"-\d{8}$", "", name)
    return name


def _extract_model(response: LLMResult) -> str | None:
    meta = response.llm_output or {}
    name = meta.get("model_name") or meta.get("model")
    if name:
        return _normalize_model(str(name))
    for gens in response.generations:
        for gen in gens:
            rmeta = getattr(getattr(gen, "message", None), "response_metadata", {}) or {}
            if rmeta.get("model_name"):
                return _normalize_model(str(rmeta["model_name"]))
    return None


class TraceWriter:
    """Append-only JSONL writer, thread-safe for concurrent LLM callbacks."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def _load_scenarios(path: str) -> tuple[int, str, list[dict[str, str]]]:
    """Minimal YAML reader for the dogfood scenarios file.

    We avoid a yaml dependency: the file is a tiny, fixed shape (runs, an
    optional default model, and a list of {id, input}). Falls back to PyYAML if
    the simple parse misses anything.
    """
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        runs = int(data.get("runs", 3))
        model = str(data.get("model", "gemini-2.5-flash"))
        scenarios = [{"id": s["id"], "input": s["input"]} for s in data["scenarios"]]
        return runs, model, scenarios
    except ImportError:
        pass

    runs, model, scenarios = 3, "gemini-2.5-flash", []
    cur: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("runs:"):
            runs = int(line.split(":", 1)[1].strip())
        elif line.startswith("model:"):
            model = line.split(":", 1)[1].strip().strip('"')
        elif line.lstrip().startswith("- id:"):
            if cur:
                scenarios.append(cur)
            cur = {"id": line.split("id:", 1)[1].strip().strip('"')}
        elif line.lstrip().startswith("input:"):
            cur["input"] = line.split("input:", 1)[1].strip().strip('"')
    if cur:
        scenarios.append(cur)
    return runs, model, scenarios


# Provider -> default model when --model is not given. Anthropic has far higher
# rate limits than Gemini's free tier, so it is the default for a clean run.
_DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "gemini": "gemini-2.5-flash",
}


def _build_model(provider: str, model_name: str, callback: Any, rps: float) -> Any:
    """Build the chat model the whole graph runs on, with the callback attached.

    The callback rides on the model, so every graph call (supervisor, each
    specialist's ReAct loop, synthesizer, LLM judge) writes a trace row. A rate
    limiter is optional — needed for Gemini's free tier, rarely for Anthropic.
    """
    kwargs: dict[str, Any] = {"temperature": 0.2, "callbacks": [callback], "max_retries": 6}
    if rps > 0:
        try:
            from langchain_core.rate_limiters import InMemoryRateLimiter
            kwargs["rate_limiter"] = InMemoryRateLimiter(
                requests_per_second=rps, check_every_n_seconds=0.1, max_bucket_size=2)
        except Exception as e:  # pragma: no cover
            print(f"augur_dogfood: rate limiter unavailable ({e})", file=sys.stderr)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            sys.exit("augur_dogfood: ANTHROPIC_API_KEY not set (load CloudOracle's .env).")
        # max_tokens bounds the completion; generous enough for routing + synthesis.
        return ChatAnthropic(model=model_name, api_key=key, max_tokens=2048, **kwargs)

    from langchain_google_genai import ChatGoogleGenerativeAI
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("augur_dogfood: GEMINI_API_KEY not set (load CloudOracle's .env).")
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=key, **kwargs)


async def _run(args: argparse.Namespace) -> int:
    runs, file_model, scenarios = _load_scenarios(args.scenarios)
    if args.runs is not None:
        runs = args.runs
    model_name = args.model or _DEFAULT_MODELS.get(args.provider, file_model)

    # Build the agent's graph from its own public pieces, but with OUR model so
    # we can pick the provider. CloudOracle is imported, never edited.
    from insights_agent.config import Settings
    from insights_agent.graph.supervisor import build_supervisor_graph
    from insights_agent.guardrails.runner import run_guarded
    from insights_agent.logging import get_logger, setup
    from insights_agent.tools.cloudoracle import CloudOracleClient, build_tools

    # The agent's Settings mandates GEMINI_API_KEY even when we drive it with
    # Anthropic (GeminiProvider is simply never constructed here). Satisfy the
    # required field with a placeholder so a Gemini key isn't needed for a
    # Claude run.
    if args.provider != "gemini":
        os.environ.setdefault("GEMINI_API_KEY", "unused-for-anthropic-run")

    settings = Settings()  # type: ignore[call-arg]  # env-populated
    setup(level="WARNING", fmt="text")  # quiet: the agent's own logs to stderr
    get_logger("augur_dogfood")

    writer = TraceWriter(args.out)
    callback = AugurTraceCallback(writer, _normalize_model(model_name))
    model = _build_model(args.provider, model_name, callback, args.rps)

    client = CloudOracleClient(
        base_url=settings.cloudoracle_base_url,
        api_key=settings.cloudoracle_api_key,
        timeout_seconds=settings.http_timeout_seconds,
    )
    tools = list(build_tools(client))  # RAG knowledge tool omitted (DATABASE_URL ignored)
    graph = build_supervisor_graph(model, tools, settings.run_limits)
    # The agent's LLM judge sends a system-only message, which Anthropic rejects
    # ("at least one message is required") though Gemini tolerates it — a
    # provider-portability bug the dogfood surfaced in the agent. --no-judge
    # skips that layer so a Claude run completes; deterministic grounding still
    # runs. (On Gemini the judge works and can stay on.)
    judge = model if (settings.enable_llm_judge and not args.no_judge) else None

    total = 0
    print(f"augur_dogfood: provider={args.provider} model={model_name}; "
          f"{len(scenarios)} scenario(s) x {runs} run(s) -> {args.out}", file=sys.stderr)
    try:
        for scenario in scenarios:
            sid = scenario["id"]
            for i in range(runs):
                callback.begin_run(sid, f"{sid}-{i:03d}")
                try:
                    result = await run_guarded(
                        graph, scenario["input"],
                        validate=settings.enable_answer_validation, judge_model=judge)
                    if result.fallback_used:
                        why = result.error or (
                            result.validation.reason if result.validation else "?")
                        status = f"fallback ({why})"
                    else:
                        status = "ok"
                except Exception as e:  # keep going; a failed run is data too
                    status = f"error:{e}"
                calls = callback._seq
                print(f"  [{sid} {i+1}/{runs}] {calls} LLM call(s) ({status})",
                      file=sys.stderr)
                total += calls
                if args.delay > 0:
                    await asyncio.sleep(args.delay)
    finally:
        await client.aclose()
        writer.close()

    print(f"augur_dogfood: wrote {total} trace row(s) -> {args.out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="augur_dogfood",
        description="Run CloudOracle's Insights Agent under an Augur trace callback.",
    )
    p.add_argument("--scenarios", required=True, help="dogfood scenarios.yaml")
    p.add_argument("--out", default="trace.jsonl", help="trace file to append to")
    p.add_argument("--provider", choices=["anthropic", "gemini"], default="anthropic",
                   help="which LLM the agent's graph runs on (default: anthropic, "
                        "for its higher rate limits)")
    p.add_argument("--model", default=None,
                   help="model id (default: provider's default; trace is tagged "
                        "with the family name to match Augur's pricing.yaml)")
    p.add_argument("--runs", type=int, default=None,
                   help="override repetitions per scenario (default: from the file)")
    p.add_argument("--no-judge", action="store_true",
                   help="skip the LLM judge layer (needed on Anthropic: the "
                        "agent's judge sends a system-only message Claude rejects)")
    p.add_argument("--delay", type=float, default=0.0,
                   help="seconds to sleep between runs (ease provider rate limits)")
    p.add_argument("--rps", type=float, default=0.0,
                   help="requests/sec cap on the shared chat model (e.g. 0.15 "
                        "for a 10 RPM free tier). 0 disables the limiter.")
    args = p.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
