package runner

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"augur/aggregate"
	"augur/cost"
	"augur/proxy"
	"augur/trace"
)

// TestPipelineEndToEnd exercises the whole Hito 0–2 chain wired together:
// the runner drives a (fake) agent N times, the agent honours the header
// contract and calls through the recording proxy, the proxy forwards to a fake
// provider and writes the trace, and aggregate prices it into a distribution.
//
// The "agent" is an in-process Exec that reads the AUGUR_* contract from the
// environment and POSTs a chat-completions request to AUGUR_BASE_URL with the
// scenario/run headers — exactly what a real agent's OpenAI client would do.
func TestPipelineEndToEnd(t *testing.T) {
	// Fake provider: always returns the same usage so the math is checkable.
	const usage = `{"model":"gpt-4o-2024-08-06","usage":{"prompt_tokens":1000,"completion_tokens":500,"total_tokens":1500,"prompt_tokens_details":{"cached_tokens":0}}}`
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		io.WriteString(w, usage)
	}))
	defer upstream.Close()

	// Recording proxy writing into an in-memory trace.
	upURL, _ := url.Parse(upstream.URL)
	var traceBuf bytes.Buffer
	proxySrv := httptest.NewServer(proxy.New(upURL, trace.NewWriter(&traceBuf), nil))
	defer proxySrv.Close()

	// The fake agent: honour the contract, make one LLM call per invocation.
	agent := func(_ context.Context, _, env []string, _, _ io.Writer) error {
		e := envMap(env)
		body := `{"model":"gpt-4o","messages":[{"role":"user","content":` + jsonString(e[EnvInput]) + `}]}`
		req, err := http.NewRequest(http.MethodPost, e[EnvBaseURL]+"/v1/chat/completions", strings.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set(proxy.HeaderScenarioID, e[EnvScenarioID])
		req.Header.Set(proxy.HeaderRunID, e[EnvRunID])
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return err
		}
		io.Copy(io.Discard, resp.Body)
		return resp.Body.Close()
	}

	cfg := Config{
		Runs:    4,
		Command: []string{"fake-agent"},
		Scenarios: []Scenario{
			{ID: "checkout", Input: "return order #1234"},
			{ID: "faq", Input: "do you ship to Mexico?"},
		},
	}
	opts := Options{
		BaseURL: proxySrv.URL,
		BaseEnv: []string{}, // hermetic
		Stdout:  io.Discard, Stderr: io.Discard,
		Exec: agent,
	}

	sum, err := Run(context.Background(), cfg, opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if sum.Total != 8 || sum.Failed != 0 { // 2 scenarios × 4 runs
		t.Fatalf("Summary = %+v, want Total 8 Failed 0", sum)
	}

	// The trace must hold one row per invocation, all tagged correctly.
	records, err := trace.ReadAll(&traceBuf)
	if err != nil {
		t.Fatalf("ReadAll trace: %v", err)
	}
	if len(records) != 8 {
		t.Fatalf("got %d trace rows, want 8", len(records))
	}
	for _, rec := range records {
		if rec.Model != "gpt-4o" {
			t.Errorf("row model = %q, want gpt-4o (request model)", rec.Model)
		}
		if rec.InputTokens != 1000 || rec.OutputTokens != 500 {
			t.Errorf("row usage = in %d out %d, want 1000/500", rec.InputTokens, rec.OutputTokens)
		}
		if rec.ScenarioID == "" || rec.RunID == "" {
			t.Errorf("row not tagged: %+v", rec)
		}
	}

	// Aggregate prices it: each call is 1000/1e6*2.50 + 500/1e6*10.00 = 0.0075,
	// one call per run, so every run costs exactly 0.0075.
	pricing := cost.Pricing{
		SnapshotDate: "2026-06-21",
		Models:       map[string]cost.ModelPrice{"gpt-4o": {Input: 2.50, Output: 10.00, CachedInput: 1.25}},
	}
	res, err := aggregate.Aggregate(records, pricing)
	if err != nil {
		t.Fatalf("Aggregate: %v", err)
	}
	if len(res.Scenarios) != 2 {
		t.Fatalf("got %d scenarios, want 2", len(res.Scenarios))
	}
	for _, s := range res.Scenarios {
		if s.Runs != 4 {
			t.Errorf("scenario %q runs = %d, want 4", s.ScenarioID, s.Runs)
		}
		// All runs identical → mean == p95 == 0.0075, stdev 0.
		if d := s.CostPerRun; !approxEq(d.Mean, 0.0075) || !approxEq(d.P95, 0.0075) || !approxEq(d.Stdev, 0) {
			t.Errorf("scenario %q CostPerRun = %+v, want mean/p95 0.0075 stdev 0", s.ScenarioID, d)
		}
		if !approxEq(s.CallsPerRun.Mean, 1) {
			t.Errorf("scenario %q calls/run = %v, want 1", s.ScenarioID, s.CallsPerRun.Mean)
		}
	}
}

func approxEq(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-9
}

// jsonString quotes s as a JSON string literal (enough for test inputs).
func jsonString(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"', '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}
