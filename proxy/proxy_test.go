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

// fakeUpstream stands in for the real provider. It records the requests it
// received so tests can assert what the proxy forwarded, and replies with a
// canned OpenAI-compatible body.
type fakeUpstream struct {
	srv      *httptest.Server
	gotReqs  []recordedReq
	respBody string
	respCode int
	respHdr  map[string]string
}

type recordedReq struct {
	method string
	path   string
	query  string
	header http.Header
	body   string
}

func newFakeUpstream() *fakeUpstream {
	f := &fakeUpstream{respCode: 200}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		f.gotReqs = append(f.gotReqs, recordedReq{
			method: r.Method, path: r.URL.Path, query: r.URL.RawQuery,
			header: r.Header.Clone(), body: string(body),
		})
		for k, v := range f.respHdr {
			w.Header().Set(k, v)
		}
		w.WriteHeader(f.respCode)
		_, _ = io.WriteString(w, f.respBody)
	}))
	return f
}

func (f *fakeUpstream) close() { f.srv.Close() }

// newTestProxy wires a Server pointing at the fake upstream, writing trace rows
// into buf, with a fixed clock so timestamps are deterministic.
func newTestProxy(t *testing.T, up *fakeUpstream, buf *bytes.Buffer) *Server {
	t.Helper()
	u, err := url.Parse(up.srv.URL)
	if err != nil {
		t.Fatalf("parse upstream url: %v", err)
	}
	s := New(u, trace.NewWriter(buf), up.srv.Client())
	fixed := time.Date(2026, 6, 21, 18, 0, 0, 0, time.UTC)
	s.now = func() time.Time { return fixed }
	return s
}

const chatResponse = `{
  "id": "chatcmpl-abc",
  "object": "chat.completion",
  "model": "gpt-4o-2024-08-06",
  "choices": [{"index":0,"message":{"role":"assistant","content":"hi"}}],
  "usage": {
    "prompt_tokens": 1234,
    "completion_tokens": 567,
    "total_tokens": 1801,
    "prompt_tokens_details": {"cached_tokens": 200}
  }
}`

func TestProxyRecordsUsage(t *testing.T) {
	up := newFakeUpstream()
	defer up.close()
	up.respBody = chatResponse

	var buf bytes.Buffer
	s := newTestProxy(t, up, &buf)
	srv := httptest.NewServer(s)
	defer srv.Close()

	reqBody := `{"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}`
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/completions", strings.NewReader(reqBody))
	req.Header.Set("Authorization", "Bearer sk-test")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderScenarioID, "checkout")
	req.Header.Set(HeaderRunID, "run-7")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	gotBody, _ := io.ReadAll(resp.Body)
	resp.Body.Close()

	// 1. The agent gets the upstream response back, unchanged.
	if resp.StatusCode != 200 {
		t.Errorf("status = %d, want 200", resp.StatusCode)
	}
	if string(gotBody) != chatResponse {
		t.Errorf("response body was not passed through unchanged:\n%s", gotBody)
	}

	// 2. The trace row matches the provider's reported usage exactly (the Hito 1
	//    checkpoint).
	recs, err := trace.ReadAll(&buf)
	if err != nil {
		t.Fatalf("ReadAll trace: %v", err)
	}
	if len(recs) != 1 {
		t.Fatalf("got %d trace rows, want 1", len(recs))
	}
	got := recs[0]
	want := trace.Record{
		Timestamp:    "2026-06-21T18:00:00Z",
		ScenarioID:   "checkout",
		RunID:        "run-7",
		Seq:          0,
		Model:        "gpt-4o", // request model, not the resolved gpt-4o-2024-08-06
		InputTokens:  1234,
		OutputTokens: 567,
		CachedTokens: 200,
		LatencyMs:    0, // fixed clock → zero elapsed
		Endpoint:     "/v1/chat/completions",
		Status:       200,
	}
	if got != want {
		t.Errorf("trace row mismatch:\n got %+v\nwant %+v", got, want)
	}

	// 3. The upstream saw the forwarded request: same path, body, auth header;
	//    Augur's own headers stripped.
	if len(up.gotReqs) != 1 {
		t.Fatalf("upstream got %d requests, want 1", len(up.gotReqs))
	}
	fwd := up.gotReqs[0]
	if fwd.path != "/v1/chat/completions" {
		t.Errorf("forwarded path = %q, want /v1/chat/completions", fwd.path)
	}
	if fwd.body != reqBody {
		t.Errorf("forwarded body = %q, want %q", fwd.body, reqBody)
	}
	if fwd.header.Get("Authorization") != "Bearer sk-test" {
		t.Errorf("Authorization not forwarded: %q", fwd.header.Get("Authorization"))
	}
	if fwd.header.Get(HeaderScenarioID) != "" || fwd.header.Get(HeaderRunID) != "" {
		t.Errorf("Augur headers leaked upstream: scenario=%q run=%q",
			fwd.header.Get(HeaderScenarioID), fwd.header.Get(HeaderRunID))
	}
}

func TestProxySeqIncrementsPerRun(t *testing.T) {
	up := newFakeUpstream()
	defer up.close()
	up.respBody = chatResponse

	var buf bytes.Buffer
	s := newTestProxy(t, up, &buf)
	srv := httptest.NewServer(s)
	defer srv.Close()

	do := func(scenario, run string) {
		req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/completions",
			strings.NewReader(`{"model":"gpt-4o"}`))
		req.Header.Set(HeaderScenarioID, scenario)
		req.Header.Set(HeaderRunID, run)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("request: %v", err)
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}

	do("checkout", "run-1") // seq 0
	do("checkout", "run-1") // seq 1
	do("checkout", "run-2") // seq 0 (new run resets)
	do("refund", "run-1")   // seq 0 (new scenario)

	recs, err := trace.ReadAll(&buf)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	type key struct {
		scenario, run string
		seq           int
	}
	got := make([]key, len(recs))
	for i, r := range recs {
		got[i] = key{r.ScenarioID, r.RunID, r.Seq}
	}
	want := []key{
		{"checkout", "run-1", 0},
		{"checkout", "run-1", 1},
		{"checkout", "run-2", 0},
		{"refund", "run-1", 0},
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("row %d seq = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// A non-2xx upstream response (e.g. a rate-limit error) must still produce a
// trace row and be passed through: failed-but-attempted calls matter.
func TestProxyRecordsErrorResponse(t *testing.T) {
	up := newFakeUpstream()
	defer up.close()
	up.respCode = 429
	up.respBody = `{"error":{"message":"rate limited","type":"rate_limit_error"}}`

	var buf bytes.Buffer
	s := newTestProxy(t, up, &buf)
	srv := httptest.NewServer(s)
	defer srv.Close()

	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/completions",
		strings.NewReader(`{"model":"gpt-4o"}`))
	req.Header.Set(HeaderScenarioID, "checkout")
	req.Header.Set(HeaderRunID, "run-1")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	resp.Body.Close()

	if resp.StatusCode != 429 {
		t.Errorf("status = %d, want 429 (passed through)", resp.StatusCode)
	}
	recs, err := trace.ReadAll(&buf)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if len(recs) != 1 {
		t.Fatalf("got %d trace rows, want 1", len(recs))
	}
	r := recs[0]
	if r.Status != 429 {
		t.Errorf("trace Status = %d, want 429", r.Status)
	}
	if r.InputTokens != 0 || r.OutputTokens != 0 {
		t.Errorf("error row should have zero tokens, got in=%d out=%d", r.InputTokens, r.OutputTokens)
	}
	if r.Model != "gpt-4o" {
		t.Errorf("model = %q, want gpt-4o (from request)", r.Model)
	}
}

func TestUsageFromResponse(t *testing.T) {
	u, model := usageFromResponse([]byte(chatResponse))
	if model != "gpt-4o-2024-08-06" {
		t.Errorf("model = %q, want gpt-4o-2024-08-06", model)
	}
	if u.InputTokens != 1234 || u.OutputTokens != 567 || u.CachedTokens != 200 {
		t.Errorf("usage = %+v, want {1234 567 200}", u)
	}

	// Garbage / no usage → zero, no panic.
	z, m := usageFromResponse([]byte(`not json`))
	if z != (oaiUsage{}) || m != "" {
		t.Errorf("garbage body = (%+v, %q), want zero", z, m)
	}
}

func TestSingleJoiningSlash(t *testing.T) {
	cases := []struct{ a, b, want string }{
		{"", "/v1/chat", "/v1/chat"},
		{"/", "/v1/chat", "/v1/chat"},
		{"/proxy", "/v1/chat", "/proxy/v1/chat"},
		{"/proxy/", "/v1/chat", "/proxy/v1/chat"},
		{"/proxy", "v1/chat", "/proxy/v1/chat"},
	}
	for _, c := range cases {
		if got := singleJoiningSlash(c.a, c.b); got != c.want {
			t.Errorf("singleJoiningSlash(%q,%q) = %q, want %q", c.a, c.b, got, c.want)
		}
	}
}

// modelFromRequest must be robust to a missing/garbled model field.
func TestModelFromRequest(t *testing.T) {
	if got := modelFromRequest([]byte(`{"model":"gpt-4o-mini"}`)); got != "gpt-4o-mini" {
		t.Errorf("got %q, want gpt-4o-mini", got)
	}
	if got := modelFromRequest([]byte(`{}`)); got != "" {
		t.Errorf("got %q, want empty", got)
	}
	if got := modelFromRequest([]byte(`nonsense`)); got != "" {
		t.Errorf("got %q, want empty", got)
	}
}

// Guard that a JSON request round-trips through the proxy unchanged at the byte
// level (no re-encoding that could reorder keys or change formatting).
func TestProxyForwardsBodyByteForByte(t *testing.T) {
	up := newFakeUpstream()
	defer up.close()
	up.respBody = chatResponse

	var buf bytes.Buffer
	s := newTestProxy(t, up, &buf)
	srv := httptest.NewServer(s)
	defer srv.Close()

	// Deliberately odd formatting/key order.
	body := "{\n  \"messages\": [],\n  \"model\":   \"gpt-4o\"\n}"
	req, _ := http.NewRequest(http.MethodPost, srv.URL+"/v1/chat/completions", strings.NewReader(body))
	req.Header.Set(HeaderScenarioID, "s")
	req.Header.Set(HeaderRunID, "r")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("request: %v", err)
	}
	resp.Body.Close()

	if up.gotReqs[0].body != body {
		t.Errorf("body mutated in transit:\n got %q\nwant %q", up.gotReqs[0].body, body)
	}
}
