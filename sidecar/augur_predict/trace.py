"""Loading and writing the Augur cost trace from Python.

The trace is JSON Lines: one LLM call per line, the exact schema the Go proxy
emits (see trace/trace.go — ``trace.Record``). Field names are the contract, so
they are mirrored here verbatim; anything we add on the Python side must round-
trip back to a row the Go aggregator will accept.

We keep a small dataclass rather than leaning on pandas: the trace is the only
data structure the sidecar touches, the files are small (a representative run,
not production telemetry), and a stdlib-only loader keeps the second toolchain
as light as the SPEC wants it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator, List


@dataclass
class Record:
    """One LLM call, mirroring trace.Record on the Go side.

    Only the fields the sidecar reads or writes are typed explicitly; the JSON
    keys match the Go ``json:"..."`` tags so a Record marshals straight back
    into a row ``augur aggregate`` can price.
    """

    scenario_id: str
    run_id: str
    seq: int
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    latency_ms: int = 0
    ts: str = ""
    endpoint: str = ""
    status: int = 0

    @classmethod
    def from_json(cls, obj: dict) -> "Record":
        """Build a Record from a parsed trace line, tolerating absent optionals.

        The proxy omits empty optionals (``omitempty`` on the Go side), so a
        line may carry only the required token-accounting fields. Unknown keys
        are ignored rather than raising: the trace schema may grow, and the
        sidecar should not break on a field it does not model.
        """
        return cls(
            scenario_id=obj.get("scenario_id", ""),
            run_id=obj.get("run_id", ""),
            seq=int(obj.get("seq", 0)),
            model=obj.get("model", ""),
            input_tokens=int(obj.get("input_tokens", 0)),
            output_tokens=int(obj.get("output_tokens", 0)),
            cached_tokens=int(obj.get("cached_tokens", 0)),
            latency_ms=int(obj.get("latency_ms", 0)),
            ts=obj.get("ts", ""),
            endpoint=obj.get("endpoint", ""),
            status=int(obj.get("status", 0)),
        )

    def to_json(self) -> dict:
        """Render to a dict with the Go JSON keys, dropping empty optionals.

        We mirror the proxy's ``omitempty`` behaviour for the optional fields so
        an emitted trace is byte-comparable in spirit to a recorded one and does
        not carry noise the Go reader would just ignore.
        """
        out = {
            "ts": self.ts,
            "scenario_id": self.scenario_id,
            "run_id": self.run_id,
            "seq": self.seq,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_tokens": self.cached_tokens,
            "latency_ms": self.latency_ms,
        }
        if self.endpoint:
            out["endpoint"] = self.endpoint
        if self.status:
            out["status"] = self.status
        return out

    def succeeded(self) -> bool:
        """Whether this call is a normal success for output-length modelling.

        A status of 0 means the proxy left it unset (the common case for a
        clean call); an explicit 2xx is also a success. Non-2xx rows are kept in
        the trace on purpose (they burned input tokens) but they distort an
        output-length fit — a 429 produced no completion — so the model excludes
        them. They remain a *retry/fan-out* phenomenon, which the Go what-if
        knobs already cover.
        """
        return self.status == 0 or 200 <= self.status < 300


def parse_lines(lines: Iterable[str]) -> Iterator[Record]:
    """Parse an iterable of JSONL strings into Records, skipping blank lines.

    A malformed line is a hard error with its 1-based number, matching the Go
    reader's stance: a corrupt trace must not silently shrink the dataset the
    model learns from.
    """
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"trace: parsing line {i}: {e}") from e
        yield Record.from_json(obj)


def load_trace(path: str) -> List[Record]:
    """Read a JSONL trace file into a list of Records."""
    with open(path, "r", encoding="utf-8") as f:
        return list(parse_lines(f))


def write_trace(path: str, records: Iterable[Record]) -> int:
    """Append records to a JSONL trace file, returning how many were written.

    Append (not truncate) mirrors the proxy: a trace is a ledger you add to. The
    caller decides whether to point this at a fresh file or an existing one.
    """
    n = 0
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_json()) + "\n")
            n += 1
    return n
