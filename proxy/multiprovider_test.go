package proxy

import (
	"bytes"
	"net/http/httptest"
	"strings"
	"testing"

	"augur/trace"
)

// Anthropic's non-streaming Messages response: input_tokens EXCLUDES the cache
// buckets, so the recorded InputTokens must be their sum.
func TestParseUsageAnthropic(t *testing.T) {
	body := `{"type":"message","model":"claude-haiku-4-5","usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":20,"cache_creation_input_tokens":10}}`
	u, has, model := parseUsageJSON([]byte(body))
	if !has {
		t.Fatal("hasUsage = false, want true")
	}
	if model != "claude-haiku-4-5" {
		t.Errorf("model = %q, want claude-haiku-4-5", model)
	}
	// total prompt = 100 + 20 read + 10 write = 130
	want := tokenUsage{InputTokens: 130, OutputTokens: 50, CachedTokens: 20, CacheWriteTokens: 10}
	if u != want {
		t.Errorf("usage = %+v, want %+v", u, want)
	}
}

// Anthropic streams usage across two events: message_start carries the input and
// cache buckets (with a placeholder output_tokens), message_delta carries the
// real output_tokens. The SSE merge must combine them, not overwrite.
func TestAnthropicStreamingMerge(t *testing.T) {
	sse := "event: message_start\n" +
		`data: {"type":"message_start","message":{"model":"claude-haiku-4-5","usage":{"input_tokens":100,"cache_read_input_tokens":20,"cache_creation_input_tokens":10,"output_tokens":1}}}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","usage":{"output_tokens":50}}` + "\n\n"

	u, model := usageFromSSE([]byte(sse))
	want := tokenUsage{InputTokens: 130, OutputTokens: 50, CachedTokens: 20, CacheWriteTokens: 10}
	if u != want {
		t.Errorf("merged usage = %+v, want %+v", u, want)
	}
	if model != "claude-haiku-4-5" {
		t.Errorf("model = %q, want claude-haiku-4-5", model)
	}
}

// Gemini reports usage under usageMetadata with camelCase counts; cached content
// is a subset of the prompt count.
func TestParseUsageGemini(t *testing.T) {
	body := `{"usageMetadata":{"promptTokenCount":200,"candidatesTokenCount":80,"cachedContentTokenCount":30,"totalTokenCount":280}}`
	u, has, _ := parseUsageJSON([]byte(body))
	if !has {
		t.Fatal("hasUsage = false, want true")
	}
	want := tokenUsage{InputTokens: 200, OutputTokens: 80, CachedTokens: 30}
	if u != want {
		t.Errorf("usage = %+v, want %+v", u, want)
	}
}

func TestModelFromPath(t *testing.T) {
	cases := []struct{ path, want string }{
		{"/v1beta/models/gemini-2.0-flash:generateContent", "gemini-2.0-flash"},
		{"/v1beta/models/gemini-1.5-pro:streamGenerateContent", "gemini-1.5-pro"},
		{"/v1/chat/completions", ""}, // OpenAI names the model in the body
		{"/v1/messages", ""},         // Anthropic too
	}
	for _, c := range cases {
		if got := modelFromPath(c.path); got != c.want {
			t.Errorf("modelFromPath(%q) = %q, want %q", c.path, got, c.want)
		}
	}
}

func TestMaybeInjectIncludeUsage(t *testing.T) {
	// A streaming OpenAI request without the option gets it injected.
	in := `{"model":"gpt-4o","stream":true,"messages":[]}`
	out := maybeInjectIncludeUsage("/v1/chat/completions", []byte(in))
	if !strings.Contains(string(out), `"include_usage":true`) {
		t.Errorf("expected include_usage injected, got %s", out)
	}
	if !strings.Contains(string(out), `"model":"gpt-4o"`) {
		t.Errorf("injection dropped other fields: %s", out)
	}

	// Non-streaming request is left byte-for-byte unchanged.
	ns := `{"model":"gpt-4o","messages":[]}`
	if got := maybeInjectIncludeUsage("/v1/chat/completions", []byte(ns)); string(got) != ns {
		t.Errorf("non-streaming body mutated: %s", got)
	}

	// Already opted in → unchanged.
	opted := `{"model":"gpt-4o","stream":true,"stream_options":{"include_usage":true}}`
	if got := maybeInjectIncludeUsage("/v1/chat/completions", []byte(opted)); string(got) != opted {
		t.Errorf("opted-in body mutated: %s", got)
	}

	// Non-OpenAI path (Anthropic) → never injected, even when streaming.
	anthropic := `{"model":"claude-haiku-4-5","stream":true}`
	if got := maybeInjectIncludeUsage("/v1/messages", []byte(anthropic)); string(got) != anthropic {
		t.Errorf("Anthropic body mutated: %s", got)
	}

	// Malformed body → returned unchanged (never break the request).
	bad := `not json`
	if got := maybeInjectIncludeUsage("/v1/chat/completions", []byte(bad)); string(got) != bad {
		t.Errorf("malformed body mutated: %s", got)
	}
}

// The proxy classifies a byte-identical repeat within a run as a retry, distinct
// bodies as initial calls, and resets per run.
func TestRetryClassification(t *testing.T) {
	up := newFakeUpstream()
	defer up.close()
	up.respBody = chatResponse

	var buf bytes.Buffer
	s := newTestProxy(t, up, &buf)
	srv := httptest.NewServer(s)
	defer srv.Close()

	same := `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	other := `{"model":"gpt-4o","messages":[{"role":"user","content":"different"}]}`

	doTagged(t, srv.URL, "s", "run-1", same)  // initial
	doTagged(t, srv.URL, "s", "run-1", same)  // retry (identical body)
	doTagged(t, srv.URL, "s", "run-1", other) // initial (new body)
	doTagged(t, srv.URL, "s", "run-2", same)  // initial (new run resets)

	recs, err := trace.ReadAll(&buf)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	want := []string{trace.KindInitial, trace.KindRetry, trace.KindInitial, trace.KindInitial}
	if len(recs) != len(want) {
		t.Fatalf("got %d rows, want %d", len(recs), len(want))
	}
	for i, w := range want {
		if recs[i].Kind != w {
			t.Errorf("row %d Kind = %q, want %q", i, recs[i].Kind, w)
		}
	}
}

// End-to-end: an Anthropic-shaped response flows through the proxy and lands in
// the trace with the cache buckets split correctly.
func TestProxyRecordsAnthropicUsage(t *testing.T) {
	up := newFakeUpstream()
	defer up.close()
	up.respBody = `{"type":"message","model":"claude-haiku-4-5","usage":{"input_tokens":100,"output_tokens":50,"cache_read_input_tokens":20,"cache_creation_input_tokens":10}}`

	var buf bytes.Buffer
	s := newTestProxy(t, up, &buf)
	srv := httptest.NewServer(s)
	defer srv.Close()

	doTagged(t, srv.URL, "s", "run-1", `{"model":"claude-haiku-4-5","messages":[]}`)

	recs, err := trace.ReadAll(&buf)
	if err != nil || len(recs) != 1 {
		t.Fatalf("trace: err=%v rows=%d", err, len(recs))
	}
	r := recs[0]
	if r.Model != "claude-haiku-4-5" {
		t.Errorf("model = %q", r.Model)
	}
	if r.InputTokens != 130 || r.OutputTokens != 50 || r.CachedTokens != 20 || r.CacheWriteTokens != 10 {
		t.Errorf("tokens = in %d out %d cached %d write %d, want 130/50/20/10",
			r.InputTokens, r.OutputTokens, r.CachedTokens, r.CacheWriteTokens)
	}
}
