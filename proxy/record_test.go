package proxy

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"augur/cassette"
	"augur/trace"
)

// doTagged POSTs body through proxyURL with scenario/run headers and returns the
// response body.
func doTagged(t *testing.T, proxyURL, scenario, run, body string) []byte {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, proxyURL+"/v1/chat/completions", strings.NewReader(body))
	req.Header.Set(HeaderScenarioID, scenario)
	req.Header.Set(HeaderRunID, run)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	return out
}

// TestRecordThenReplay records two calls against a fake provider, then replays
// the cassette with the provider shut down — proving replay spends nothing and
// regenerates an identical trace.
func TestRecordThenReplay(t *testing.T) {
	dir := t.TempDir()
	cassettePath := filepath.Join(dir, "cassette.jsonl")
	fixed := time.Date(2026, 6, 21, 18, 0, 0, 0, time.UTC)

	// --- Record phase ---
	var upstreamHits int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamHits++
		io.Copy(io.Discard, r.Body)
		io.WriteString(w, chatResponse) // from proxy_test.go: usage 1234/567/200
	}))
	upURL, _ := url.Parse(upstream.URL)

	cRec, err := cassette.Create(cassettePath)
	if err != nil {
		t.Fatalf("Create cassette: %v", err)
	}
	var recTrace bytes.Buffer
	recSrv := New(upURL, trace.NewWriter(&recTrace), upstream.Client())
	recSrv.now = func() time.Time { return fixed }
	recSrv.Record(cRec)
	recProxy := httptest.NewServer(recSrv)

	doTagged(t, recProxy.URL, "checkout", "checkout-000", `{"model":"gpt-4o"}`)
	doTagged(t, recProxy.URL, "checkout", "checkout-001", `{"model":"gpt-4o"}`)

	recProxy.Close()
	cRec.Close()
	upstream.Close() // provider is GONE for replay

	if upstreamHits != 2 {
		t.Fatalf("record phase hit upstream %d times, want 2", upstreamHits)
	}

	recRows, err := trace.ReadAll(&recTrace)
	if err != nil {
		t.Fatalf("read record trace: %v", err)
	}
	if len(recRows) != 2 {
		t.Fatalf("record trace has %d rows, want 2", len(recRows))
	}

	// --- Replay phase --- (no upstream; would panic on a real call)
	cPlay, err := cassette.Load(cassettePath)
	if err != nil {
		t.Fatalf("Load cassette: %v", err)
	}
	var playTrace bytes.Buffer
	// A client that fails any request, to prove replay never calls out.
	failClient := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		t.Fatal("replay must not contact the provider")
		return nil, nil
	})}
	playSrv := New(upURL, trace.NewWriter(&playTrace), failClient)
	playSrv.now = func() time.Time { return fixed }
	playSrv.Replay(cPlay)
	playProxy := httptest.NewServer(playSrv)
	defer playProxy.Close()

	gotBody := doTagged(t, playProxy.URL, "checkout", "checkout-000", `{"model":"gpt-4o"}`)
	doTagged(t, playProxy.URL, "checkout", "checkout-001", `{"model":"gpt-4o"}`)

	// The agent gets the recorded body back.
	if string(gotBody) != chatResponse {
		t.Errorf("replay body mismatch:\n got %q", gotBody)
	}

	playRows, err := trace.ReadAll(&playTrace)
	if err != nil {
		t.Fatalf("read replay trace: %v", err)
	}
	if len(playRows) != 2 {
		t.Fatalf("replay trace has %d rows, want 2", len(playRows))
	}
	// Replay regenerates an identical trace (same tokens, model, latency).
	for i := range recRows {
		if recRows[i] != playRows[i] {
			t.Errorf("row %d diverged:\n record %+v\n replay %+v", i, recRows[i], playRows[i])
		}
	}
}

// A replay miss (a call that wasn't recorded) must fail loudly with 502.
func TestReplayMiss(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "c.jsonl")
	c, _ := cassette.Create(path)
	c.Record(cassette.Entry{ScenarioID: "a", RunID: "a-000", Seq: 0, Status: 200,
		ContentType: "application/json", Body: chatResponse})
	c.Close()

	loaded, _ := cassette.Load(path)
	var tr bytes.Buffer
	srv := New(&url.URL{}, trace.NewWriter(&tr), nil)
	srv.Replay(loaded)
	proxy := httptest.NewServer(srv)
	defer proxy.Close()

	// seq 0 hits; the agent's second call (seq 1) was never recorded -> 502.
	req1, _ := http.NewRequest(http.MethodPost, proxy.URL+"/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o"}`))
	req1.Header.Set(HeaderScenarioID, "a")
	req1.Header.Set(HeaderRunID, "a-000")
	resp1, _ := http.DefaultClient.Do(req1)
	resp1.Body.Close()
	if resp1.StatusCode != 200 {
		t.Errorf("first replayed call status = %d, want 200", resp1.StatusCode)
	}

	req2, _ := http.NewRequest(http.MethodPost, proxy.URL+"/v1/chat/completions", strings.NewReader(`{"model":"gpt-4o"}`))
	req2.Header.Set(HeaderScenarioID, "a")
	req2.Header.Set(HeaderRunID, "a-000")
	resp2, _ := http.DefaultClient.Do(req2)
	resp2.Body.Close()
	if resp2.StatusCode != http.StatusBadGateway {
		t.Errorf("replay miss status = %d, want 502", resp2.StatusCode)
	}
}

// Recording a streaming response must capture the SSE body and replay it with
// the same usage accounting.
func TestRecordReplayStreaming(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "stream.jsonl")
	fixed := time.Date(2026, 6, 21, 18, 0, 0, 0, time.UTC)

	upstream := newStreamingUpstream(streamChunks) // usage 1234/567/200
	upURL, _ := url.Parse(upstream.URL)

	cRec, _ := cassette.Create(path)
	var recTrace bytes.Buffer
	recSrv := New(upURL, trace.NewWriter(&recTrace), nil)
	recSrv.now = func() time.Time { return fixed }
	recSrv.Record(cRec)
	recProxy := httptest.NewServer(recSrv)

	doTagged(t, recProxy.URL, "s", "s-000", `{"model":"gpt-4o","stream":true}`)
	recProxy.Close()
	cRec.Close()
	upstream.Close()

	recRows, _ := trace.ReadAll(&recTrace)
	if len(recRows) != 1 || recRows[0].InputTokens != 1234 || recRows[0].OutputTokens != 567 {
		t.Fatalf("record trace wrong: %+v", recRows)
	}

	cPlay, _ := cassette.Load(path)
	var playTrace bytes.Buffer
	playSrv := New(upURL, trace.NewWriter(&playTrace), nil)
	playSrv.now = func() time.Time { return fixed }
	playSrv.Replay(cPlay)
	playProxy := httptest.NewServer(playSrv)
	defer playProxy.Close()

	body := doTagged(t, playProxy.URL, "s", "s-000", `{"model":"gpt-4o","stream":true}`)
	if string(body) != streamChunks {
		t.Errorf("replayed SSE body mismatch:\n got %q", body)
	}
	playRows, _ := trace.ReadAll(&playTrace)
	if len(playRows) != 1 || playRows[0] != recRows[0] {
		t.Errorf("streaming replay row diverged:\n record %+v\n replay %+v", recRows[0], playRows)
	}
}

// roundTripFunc adapts a function to http.RoundTripper.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
