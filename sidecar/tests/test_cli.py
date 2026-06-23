import json

import numpy as np
import pytest

from augur_predict.cli import main
from augur_predict.trace import Record, load_trace, write_trace


def _write_trace(path, n=40, noise=8.0, seed=5):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        x = 100 + i * 40
        y = 40 + 0.25 * x + rng.normal(0, noise)
        recs.append(Record("s1", f"r{i}", 0, "m", int(x), int(round(y)),
                            cached_tokens=int(x * 0.1)))
    write_trace(str(path), recs)


def test_fit_then_report_then_predict(tmp_path, capsys):
    trace = tmp_path / "trace.jsonl"
    model = tmp_path / "model.json"
    _write_trace(trace)

    assert main(["fit", "--trace", str(trace), "--out", str(model)]) == 0
    assert model.exists()
    obj = json.loads(model.read_text())
    assert "m" in obj["models"]
    assert obj["models"]["m"]["method"] == "ols"

    assert main(["report", "--model", str(model)]) == 0
    out = capsys.readouterr().out
    assert "Output-length model" in out
    assert "m" in out

    assert main(["predict", "--model", str(model),
                 "--model-name", "m", "--input-tokens", "1000"]) == 0
    out = capsys.readouterr().out
    assert "predicted output tokens" in out


def test_predict_with_price_prints_cost(tmp_path, capsys):
    trace = tmp_path / "trace.jsonl"
    model = tmp_path / "model.json"
    _write_trace(trace)
    main(["fit", "--trace", str(trace), "--out", str(model)])
    capsys.readouterr()

    assert main(["predict", "--model", str(model), "--model-name", "m",
                 "--input-tokens", "1000", "--price-out", "0.6"]) == 0
    out = capsys.readouterr().out
    assert "output cost" in out


def test_predict_unknown_model_fails(tmp_path, capsys):
    trace = tmp_path / "trace.jsonl"
    model = tmp_path / "model.json"
    _write_trace(trace)
    main(["fit", "--trace", str(trace), "--out", str(model)])
    capsys.readouterr()

    rc = main(["predict", "--model", str(model), "--model-name", "ghost",
               "--input-tokens", "100"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no fit" in err


def test_fit_empty_trace_fails(tmp_path, capsys):
    trace = tmp_path / "empty.jsonl"
    trace.write_text("")
    rc = main(["fit", "--trace", str(trace), "--out", str(tmp_path / "m.json")])
    assert rc == 1
    assert "empty" in capsys.readouterr().err


def test_emit_trace_produces_consumable_jsonl(tmp_path, capsys):
    trace = tmp_path / "trace.jsonl"
    model = tmp_path / "model.json"
    out = tmp_path / "pred.jsonl"
    _write_trace(trace)
    main(["fit", "--trace", str(trace), "--out", str(model)])
    capsys.readouterr()

    rc = main(["emit-trace", "--model", str(model), "--out", str(out),
               "--runs", "15", "--input-scale", "1.5", "--seed", "9"])
    assert rc == 0
    recs = load_trace(str(out))
    assert len({r.run_id for r in recs}) == 15
    for r in recs:
        assert r.cached_tokens <= r.input_tokens
        assert r.output_tokens >= 0


def test_emit_trace_round_trips_through_trace_loader(tmp_path):
    """The emitted file must parse back as a valid trace (Go schema contract)."""
    trace = tmp_path / "trace.jsonl"
    model = tmp_path / "model.json"
    out = tmp_path / "pred.jsonl"
    _write_trace(trace)
    main(["fit", "--trace", str(trace), "--out", str(model)])
    main(["emit-trace", "--model", str(model), "--out", str(out), "--runs", "5"])

    for line in out.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        # required keys the Go aggregator reads
        for key in ("scenario_id", "run_id", "seq", "model",
                    "input_tokens", "output_tokens", "cached_tokens"):
            assert key in obj


def test_no_subcommand_errors(capsys):
    with pytest.raises(SystemExit):
        main([])
