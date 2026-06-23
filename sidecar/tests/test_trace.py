import json

import pytest

from augur_predict.trace import Record, parse_lines, load_trace, write_trace


def test_from_json_required_and_optional_fields():
    line = {
        "ts": "2026-06-22T10:00:00Z",
        "scenario_id": "s1", "run_id": "r1", "seq": 2,
        "model": "gpt-4o-mini",
        "input_tokens": 1200, "output_tokens": 300, "cached_tokens": 400,
        "latency_ms": 850, "endpoint": "/v1/chat/completions", "status": 200,
    }
    r = Record.from_json(line)
    assert r.scenario_id == "s1"
    assert r.seq == 2
    assert r.input_tokens == 1200
    assert r.cached_tokens == 400
    assert r.status == 200


def test_from_json_tolerates_missing_optionals_and_unknown_keys():
    r = Record.from_json({
        "scenario_id": "s", "run_id": "r", "seq": 0, "model": "m",
        "input_tokens": 10, "output_tokens": 5,
        "some_future_field": "ignored",
    })
    assert r.cached_tokens == 0
    assert r.status == 0
    assert r.endpoint == ""


def test_to_json_drops_empty_optionals_like_go_omitempty():
    r = Record("s", "r", 0, "m", 10, 5)
    obj = r.to_json()
    assert "endpoint" not in obj
    assert "status" not in obj
    # required token accounting always present
    assert obj["input_tokens"] == 10
    assert obj["output_tokens"] == 5
    assert obj["cached_tokens"] == 0


def test_to_json_keeps_set_optionals():
    r = Record("s", "r", 0, "m", 10, 5, status=429, endpoint="/v1/x")
    obj = r.to_json()
    assert obj["status"] == 429
    assert obj["endpoint"] == "/v1/x"


def test_succeeded_classification():
    assert Record("s", "r", 0, "m", 1, 1, status=0).succeeded()
    assert Record("s", "r", 0, "m", 1, 1, status=200).succeeded()
    assert Record("s", "r", 0, "m", 1, 1, status=299).succeeded()
    assert not Record("s", "r", 0, "m", 1, 1, status=429).succeeded()
    assert not Record("s", "r", 0, "m", 1, 1, status=500).succeeded()


def test_parse_lines_skips_blanks():
    lines = ['{"scenario_id":"s","run_id":"r","seq":0,"model":"m","input_tokens":1,"output_tokens":2}',
             "", "  "]
    recs = list(parse_lines(lines))
    assert len(recs) == 1


def test_parse_lines_malformed_is_hard_error_with_line_number():
    with pytest.raises(ValueError, match="line 2"):
        list(parse_lines(["{}", "not json"]))


def test_load_and_write_round_trip(tmp_path):
    src = [
        Record("s1", "r1", 0, "m", 100, 50, cached_tokens=10),
        Record("s1", "r1", 1, "m", 200, 80),
    ]
    path = tmp_path / "t.jsonl"
    n = write_trace(str(path), src)
    assert n == 2
    back = load_trace(str(path))
    assert len(back) == 2
    assert back[0].input_tokens == 100
    assert back[0].cached_tokens == 10
    assert back[1].seq == 1


def test_write_appends_not_truncates(tmp_path):
    path = tmp_path / "t.jsonl"
    write_trace(str(path), [Record("s", "r", 0, "m", 1, 1)])
    write_trace(str(path), [Record("s", "r", 1, "m", 1, 1)])
    assert len(load_trace(str(path))) == 2
