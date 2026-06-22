package proxy

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"augur/trace"
)

// streamChunks is an OpenAI-compatible SSE response whose final chunk carries a
// usage block (as it does when the request sets stream_options.include_usage).
// Its totals deliberately equal the non-streaming chatResponse (1234/567/200)
// so the reconciliation test can assert the two paths agree.
const streamChunks = "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Hel\"}}]}\n\n" +
	"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n" +
	"data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o-2024-08-06\",\"choices\":[],\"usage\":{\"prompt_tokens\":1234,\"completion_tokens\":567,\"total_tokens\":1801,\"prompt_tokens_details\":{\"cached_tokens\":200}}}\n\n" +
	"data: [DONE]\n\n"

// newStreamingUpstream serves body as an SSE stream, flushing between writes so
// the proxy genuinely exercises its incremental relay path.
func newStreamingUpstream(body string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		rc := http.NewResponseController(w)
		for _, line := range strings.SplitAfter(body, "\n") {
			if line == "" {
				continue
			}
			_, _ = io.WriteString(w, line)
			_ = rc.Flush()
		}
	}))
}

func newStreamTestProxy(t *testing.T, upstreamURL string, buf *bytes.Buffer) *httptest.Server {
	t.Helper()
	u, err := url.Parse(upstreamURL)
	if err != nil {
		t.Fatalf("parse upstream url: %v", err)
	}
	s := New(u, trace.NewWriter(buf), nil)
	fixed := time.Date(2026, 6, 21, 18, 0, 0, 0, time.UTC)
	s.now = func() time.Time { return fixed }
	return httptest.NewServer(s)
}

func TestProxyStreamingRecordsUsage(t *testing.T) {
	up := newStreamingUpstream(streamChunks)
	defer up.Close()

	var buf bytes.Buffer
	srv := newStreamTestProxy(t, up.URL, &buf)
	defer srv.Close()

	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4o","stream":true,"stream_options":{"include_usage":true}}`))
	req.Header.Set(HeaderScenarioID, "checkout")
	req.Header.Set(HeaderRunID, "run-1")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	gotBody, _ := io.ReadAll(resp.Body)
	resp.Body.Close()

	// The agent receives the provider's SSE stream byte-for-byte.
	if string(gotBody) != streamChunks {
		t.Errorf("streamed body not relayed verbatim:\n got %q\nwant %q", gotBody, streamChunks)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}

	recs, err := trace.ReadAll(&buf)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if len(recs) != 1 {
		t.Fatalf("got %d trace rows, want 1", len(recs))
	}
	got := recs[0]
	if got.InputTokens != 1234 || got.OutputTokens != 567 || got.CachedTokens != 200 {
		t.Errorf("usage = (in %d, out %d, cached %d), want (1234, 567, 200)",
			got.InputTokens, got.OutputTokens, got.CachedTokens)
	}
	if got.Model != "gpt-4o" {
		t.Errorf("model = %q, want gpt-4o (request model preferred)", got.Model)
	}
	if got.Status != 200 {
		t.Errorf("status = %d, want 200", got.Status)
	}
}

// The Hito 1 checkpoint: streaming totals must reconcile with non-streaming.
// We run the SAME logical response through both paths and assert the trace rows
// carry identical token accounting.
func TestStreamingReconcilesWithNonStreaming(t *testing.T) {
	// Non-streaming row.
	nonStreamUp := newFakeUpstream()
	defer nonStreamUp.close()
	nonStreamUp.respBody = chatResponse
	var nsBuf bytes.Buffer
	ns := newTestProxy(t, nonStreamUp, &nsBuf)
	nsSrv := httptest.NewServer(ns)
	defer nsSrv.Close()

	doReq(t, nsSrv.URL, `{"model":"gpt-4o"}`)
	nsRecs, err := trace.ReadAll(&nsBuf)
	if err != nil || len(nsRecs) != 1 {
		t.Fatalf("non-streaming trace: err=%v rows=%d", err, len(nsRecs))
	}

	// Streaming row.
	streamUp := newStreamingUpstream(streamChunks)
	defer streamUp.Close()
	var sBuf bytes.Buffer
	sSrv := newStreamTestProxy(t, streamUp.URL, &sBuf)
	defer sSrv.Close()

	doReq(t, sSrv.URL, `{"model":"gpt-4o","stream":true}`)
	sRecs, err := trace.ReadAll(&sBuf)
	if err != nil || len(sRecs) != 1 {
		t.Fatalf("streaming trace: err=%v rows=%d", err, len(sRecs))
	}

	ns0, s0 := nsRecs[0], sRecs[0]
	if ns0.InputTokens != s0.InputTokens ||
		ns0.OutputTokens != s0.OutputTokens ||
		ns0.CachedTokens != s0.CachedTokens {
		t.Errorf("streaming/non-streaming usage diverged:\n non-stream %+v\n stream     %+v", ns0, s0)
	}
}

// A stream WITHOUT a usage block (client didn't set include_usage) still relays
// cleanly and records a zero-token row rather than crashing or dropping it.
func TestProxyStreamingNoUsageBlock(t *testing.T) {
	noUsage := "data: {\"model\":\"gpt-4o-2024-08-06\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: [DONE]\n\n"
	up := newStreamingUpstream(noUsage)
	defer up.Close()

	var buf bytes.Buffer
	srv := newStreamTestProxy(t, up.URL, &buf)
	defer srv.Close()

	resp := doReq(t, srv.URL, `{"model":"gpt-4o","stream":true}`)
	if string(resp) != noUsage {
		t.Errorf("body not relayed verbatim:\n got %q\nwant %q", resp, noUsage)
	}

	recs, err := trace.ReadAll(&buf)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if len(recs) != 1 {
		t.Fatalf("got %d rows, want 1", len(recs))
	}
	if recs[0].InputTokens != 0 || recs[0].OutputTokens != 0 {
		t.Errorf("no-usage stream should record zero tokens, got %+v", recs[0])
	}
}

func TestIsEventStream(t *testing.T) {
	cases := []struct {
		ct   string
		want bool
	}{
		{"text/event-stream", true},
		{"text/event-stream; charset=utf-8", true},
		{"TEXT/EVENT-STREAM", true},
		{"application/json", false},
		{"", false},
	}
	for _, c := range cases {
		h := http.Header{}
		if c.ct != "" {
			h.Set("Content-Type", c.ct)
		}
		if got := isEventStream(h); got != c.want {
			t.Errorf("isEventStream(%q) = %v, want %v", c.ct, got, c.want)
		}
	}
}

func TestParseSSEChunk(t *testing.T) {
	// Final chunk with usage.
	u, has, model := parseSSEChunk([]byte(`data: {"model":"gpt-4o-x","usage":{"prompt_tokens":10,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":3}}}`))
	if !has || u.InputTokens != 10 || u.OutputTokens != 5 || u.CachedTokens != 3 || model != "gpt-4o-x" {
		t.Errorf("usage chunk: has=%v u=%+v model=%q", has, u, model)
	}

	// Content chunk: model present, no usage.
	_, has, model = parseSSEChunk([]byte(`data: {"model":"gpt-4o-x","choices":[{"delta":{"content":"hi"}}]}`))
	if has {
		t.Error("content chunk reported hasUsage=true")
	}
	if model != "gpt-4o-x" {
		t.Errorf("content chunk model = %q, want gpt-4o-x", model)
	}

	// [DONE] and non-data lines.
	if _, has, _ := parseSSEChunk([]byte("data: [DONE]")); has {
		t.Error("[DONE] reported hasUsage=true")
	}
	if _, has, _ := parseSSEChunk([]byte(": this is an SSE comment")); has {
		t.Error("comment line reported hasUsage=true")
	}
	if _, has, _ := parseSSEChunk([]byte("\n")); has {
		t.Error("blank line reported hasUsage=true")
	}
}

// doReq POSTs body to a proxy and returns the response body. Scenario/run
// headers are set so the trace row is well-formed.
func doReq(t *testing.T, proxyURL, body string) []byte {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, proxyURL+"/v1/chat/completions", strings.NewReader(body))
	req.Header.Set(HeaderScenarioID, "s")
	req.Header.Set(HeaderRunID, "r")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	defer resp.Body.Close()
	out, _ := io.ReadAll(resp.Body)
	return out
}
