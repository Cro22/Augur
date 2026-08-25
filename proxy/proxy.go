// Package proxy is the OpenAI-compatible recording proxy: the agent under test
// points its base_url at this server, and every LLM call it makes is forwarded
// verbatim to the real provider while a trace row (model + token usage +
// latency, tagged with scenario/run) is appended to the ledger.
//
// Capturing at the HTTP layer — rather than wrapping an SDK — is decision D1 in
// SPEC.md: it is framework- and language-agnostic and records the real call
// graph (retries, tool-loop fan-out) exactly as the agent emits it.
//
// This file implements the non-streaming path and the shared request plumbing;
// the streaming (SSE) path lives in proxy_stream.go. Both converge on
// parseUsageJSON for token accounting, so streaming and non-streaming totals
// reconcile by construction.
package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"augur/cassette"
	"augur/trace"
)

// Header names the runner sets so the proxy can tag each call with the scenario
// it exercised and the specific repetition (run). They are stripped before the
// request is forwarded upstream — they are Augur's, not the provider's.
const (
	HeaderScenarioID = "X-Augur-Scenario-Id"
	HeaderRunID      = "X-Augur-Run-Id"
)

// nowFunc returns the current time. It is a field on Server (defaulting to
// time.Now) so tests can stamp deterministic timestamps.
type nowFunc func() time.Time

// Mode selects how the proxy obtains responses.
type Mode int

const (
	// ModeLive forwards to the real provider (the default).
	ModeLive Mode = iota
	// ModeRecord forwards to the provider AND saves each response to a cassette.
	ModeRecord
	// ModeReplay serves responses from a cassette without calling the provider.
	ModeReplay
)

// Server is the recording proxy. Construct it with New and mount it with
// http.ListenAndServe; it implements http.Handler.
type Server struct {
	upstream *url.URL
	tracer   *trace.Writer
	client   *http.Client
	now      nowFunc

	// InjectUsage, when true (the default from New), rewrites streaming OpenAI
	// chat-completion requests to set stream_options.include_usage=true so the
	// provider emits an exact usage block even when the agent didn't ask for it.
	// Set false to forward the request byte-for-byte.
	InjectUsage bool

	mode     Mode
	cassette *cassette.Cassette

	mu   sync.Mutex
	seq  map[string]int             // per (scenario|run) next call ordinal
	seen map[string]map[uint64]bool // per (scenario|run) request-body hashes seen (retry detection)
}

// New returns a Server that forwards to upstream (e.g. https://api.openai.com)
// and appends trace rows via tracer. A nil client uses a sensible default. The
// server starts in ModeLive with usage injection on; call Record or Replay to
// change mode.
func New(upstream *url.URL, tracer *trace.Writer, client *http.Client) *Server {
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Minute}
	}
	return &Server{
		upstream:    upstream,
		tracer:      tracer,
		client:      client,
		now:         time.Now,
		InjectUsage: true,
		seq:         make(map[string]int),
		seen:        make(map[string]map[uint64]bool),
	}
}

// Record switches the server to ModeRecord: responses are still fetched from
// the provider and relayed, but each is also saved to c.
func (s *Server) Record(c *cassette.Cassette) {
	s.mode = ModeRecord
	s.cassette = c
}

// Replay switches the server to ModeReplay: responses are served from c and the
// provider is never contacted (no tokens spent). The cost trace is regenerated
// from the recorded responses.
func (s *Server) Replay(c *cassette.Cassette) {
	s.mode = ModeReplay
	s.cassette = c
}

// nextSeq returns and increments the call ordinal for a (scenario, run) pair.
func (s *Server) nextSeq(scenario, run string) int {
	key := scenario + "|" + run
	s.mu.Lock()
	defer s.mu.Unlock()
	n := s.seq[key]
	s.seq[key] = n + 1
	return n
}

// classify labels a call by whether its request body has been seen before in
// the same (scenario, run): a repeat is the observable signature of a client-
// library retry, a first sighting is an initial call. This is the honest
// classification the proxy can make — it deliberately does not try to separate
// fan-out from sequential tool-loop steps (both are initial calls with distinct
// bodies), because call concurrency is not visible at the HTTP layer.
func (s *Server) classify(scenario, run string, body []byte) string {
	h := fnv.New64a()
	_, _ = h.Write(body)
	sum := h.Sum64()

	key := scenario + "|" + run
	s.mu.Lock()
	defer s.mu.Unlock()
	seen := s.seen[key]
	if seen == nil {
		seen = make(map[uint64]bool)
		s.seen[key] = seen
	}
	if seen[sum] {
		return trace.KindRetry
	}
	seen[sum] = true
	return trace.KindInitial
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	scenario := r.Header.Get(HeaderScenarioID)
	run := r.Header.Get(HeaderRunID)

	// Read the whole request body: we need it both to forward and to learn which
	// model the call targets (the request model matches pricing.yaml keys; the
	// response echoes a resolved, dated variant).
	reqBody, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "augur proxy: reading request body: "+err.Error(), http.StatusBadGateway)
		return
	}
	_ = r.Body.Close()
	reqModel := modelFromRequest(reqBody)
	if reqModel == "" {
		// Gemini names the model in the URL path, not the body.
		reqModel = modelFromPath(r.URL.Path)
	}

	// Assign the call's ordinal once, up front: it is both the trace's seq and
	// the cassette key, so record and replay must compute it identically.
	seq := s.nextSeq(scenario, run)
	// Classify from the ORIGINAL request body (before any usage injection), so a
	// retry — a byte-identical repeat — is recognised regardless of injection,
	// and record/replay classify identically.
	kind := s.classify(scenario, run, reqBody)

	if s.mode == ModeReplay {
		s.replay(w, scenario, run, seq, reqModel, r.URL.Path, kind)
		return
	}

	// Forward the (possibly usage-injected) body upstream; classification and
	// the cassette are unaffected because both key off the original request.
	fwdBody := reqBody
	if s.InjectUsage {
		fwdBody = maybeInjectIncludeUsage(r.URL.Path, reqBody)
	}
	outReq, err := s.buildUpstreamRequest(r, fwdBody)
	if err != nil {
		http.Error(w, "augur proxy: building upstream request: "+err.Error(), http.StatusBadGateway)
		return
	}

	start := s.now()
	resp, err := s.client.Do(outReq)
	if err != nil {
		http.Error(w, "augur proxy: upstream request failed: "+err.Error(), http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()

	// Live mode streams an SSE body back chunk-by-chunk so the agent sees tokens
	// in real time. Record mode (and every non-streaming response) goes through
	// the buffered path so the full body can be captured for the cassette.
	if s.mode == ModeLive && isEventStream(resp.Header) {
		usage, respModel := s.streamResponse(w, resp)
		latency := s.now().Sub(start)
		s.recordTrace(start, scenario, run, seq, pickModel(reqModel, respModel), usage, latency.Milliseconds(), r.URL.Path, resp.StatusCode, kind)
		return
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "augur proxy: reading upstream response: "+err.Error(), http.StatusBadGateway)
		return
	}
	latency := s.now().Sub(start)
	contentType := resp.Header.Get("Content-Type")

	copyHeader(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(body)

	if s.mode == ModeRecord && s.cassette != nil {
		if err := s.cassette.Record(cassette.Entry{
			ScenarioID: scenario, RunID: run, Seq: seq,
			Status: resp.StatusCode, ContentType: contentType,
			LatencyMs: latency.Milliseconds(), Body: string(body),
		}); err != nil {
			fmt.Printf("augur proxy: WARNING cassette write failed: %v\n", err)
		}
	}

	usage, respModel := extractUsage(contentType, body)
	s.recordTrace(start, scenario, run, seq, pickModel(reqModel, respModel), usage, latency.Milliseconds(), r.URL.Path, resp.StatusCode, kind)
}

// replay serves a previously recorded response from the cassette without
// contacting the provider, and regenerates the call's trace row from it. A miss
// means the agent made a call that was not recorded — surfaced as a 502 so the
// divergence is loud rather than silently mis-costed.
func (s *Server) replay(w http.ResponseWriter, scenario, run string, seq int, reqModel, path, kind string) {
	e, ok := s.cassette.Lookup(scenario, run, seq)
	if !ok {
		http.Error(w, fmt.Sprintf("augur proxy: replay miss for scenario %q run %q seq %d (agent diverged from the recording?)",
			scenario, run, seq), http.StatusBadGateway)
		return
	}
	body := []byte(e.Body)
	if e.ContentType != "" {
		w.Header().Set("Content-Type", e.ContentType)
	}
	w.WriteHeader(e.Status)
	_, _ = w.Write(body)

	usage, respModel := extractUsage(e.ContentType, body)
	s.recordTrace(s.now(), scenario, run, seq, pickModel(reqModel, respModel), usage, e.LatencyMs, path, e.Status, kind)
}

// recordTrace writes one trace row. A write failure must not corrupt the
// agent's response but must be loud: a dropped row means an under-counted bill.
func (s *Server) recordTrace(ts time.Time, scenario, run string, seq int, model string, u tokenUsage, latencyMs int64, path string, status int, kind string) {
	rec := trace.Record{
		Timestamp:        ts.UTC().Format(time.RFC3339Nano),
		ScenarioID:       scenario,
		RunID:            run,
		Seq:              seq,
		Model:            model,
		InputTokens:      u.InputTokens,
		OutputTokens:     u.OutputTokens,
		CachedTokens:     u.CachedTokens,
		CacheWriteTokens: u.CacheWriteTokens,
		LatencyMs:        latencyMs,
		Endpoint:         path,
		Status:           status,
		Kind:             kind,
	}
	if err := s.tracer.Write(rec); err != nil {
		fmt.Printf("augur proxy: WARNING trace write failed: %v\n", err)
	}
}

// pickModel prefers the request's model (it matches pricing.yaml keys) and falls
// back to the model echoed in the response.
func pickModel(reqModel, respModel string) string {
	if reqModel != "" {
		return reqModel
	}
	return respModel
}

// buildUpstreamRequest clones the inbound request onto the upstream base URL,
// preserving method, path, query, and headers (including Authorization) while
// stripping Augur's own headers and Accept-Encoding (so Go's transport handles
// compression transparently and we get a decoded body to parse usage from).
func (s *Server) buildUpstreamRequest(r *http.Request, body []byte) (*http.Request, error) {
	out := *s.upstream
	out.Path = singleJoiningSlash(s.upstream.Path, r.URL.Path)
	out.RawQuery = r.URL.RawQuery

	req, err := http.NewRequestWithContext(r.Context(), r.Method, out.String(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	copyHeader(req.Header, r.Header)
	stripHopByHop(req.Header)
	req.Header.Del(HeaderScenarioID)
	req.Header.Del(HeaderRunID)
	// Let the Go transport negotiate and transparently decode compression so the
	// response body we read (and parse usage from) is the decoded JSON.
	req.Header.Del("Accept-Encoding")
	// ContentLength must reflect the body we actually send.
	req.ContentLength = int64(len(body))
	return req, nil
}

// tokenUsage is the provider-neutral token accounting Augur records for one LLM
// call. CachedTokens and CacheWriteTokens are subsets of InputTokens (see
// package cost). parseUsageJSON maps the OpenAI, Anthropic, and Gemini wire
// shapes onto it.
type tokenUsage struct {
	InputTokens      int
	OutputTokens     int
	CachedTokens     int
	CacheWriteTokens int
}

// merge folds a newly parsed chunk's usage into u, taking each field's latest
// non-zero value. Streaming responses spread usage across chunks differently by
// provider — OpenAI emits one final block, Anthropic splits input (message_start)
// from output (message_delta), Gemini repeats a cumulative block — and
// last-non-zero-wins reconciles all three without provider-specific stream state.
func (u *tokenUsage) merge(n tokenUsage) {
	if n.InputTokens != 0 {
		u.InputTokens = n.InputTokens
	}
	if n.OutputTokens != 0 {
		u.OutputTokens = n.OutputTokens
	}
	if n.CachedTokens != 0 {
		u.CachedTokens = n.CachedTokens
	}
	if n.CacheWriteTokens != 0 {
		u.CacheWriteTokens = n.CacheWriteTokens
	}
}

// wireUsage is the union of the OpenAI and Anthropic per-call usage blocks (both
// carried under a "usage" key). Pointer fields distinguish "key absent" from a
// genuine zero, so parseUsageJSON can tell which provider's shape it holds.
type wireUsage struct {
	// OpenAI
	PromptTokens        *int `json:"prompt_tokens"`
	CompletionTokens    *int `json:"completion_tokens"`
	PromptTokensDetails *struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
	// Anthropic
	InputTokens              *int `json:"input_tokens"`
	OutputTokens             *int `json:"output_tokens"`
	CacheReadInputTokens     int  `json:"cache_read_input_tokens"`
	CacheCreationInputTokens int  `json:"cache_creation_input_tokens"`
}

// parseUsageJSON extracts token usage and the resolved model from one OpenAI-,
// Anthropic-, or Gemini-compatible JSON object: a full non-streaming response
// body or a single streamed chunk's payload. hasUsage is false when the object
// carries no usage block at all (e.g. an OpenAI content chunk), letting callers
// tell "no usage here" from "usage that is genuinely zero". Every path funnels
// through here so the providers cannot drift apart in accounting.
func parseUsageJSON(data []byte) (u tokenUsage, hasUsage bool, model string) {
	var parsed struct {
		Model   string     `json:"model"`
		Usage   *wireUsage `json:"usage"`
		Message *struct { // Anthropic streaming message_start nests usage + model
			Model string     `json:"model"`
			Usage *wireUsage `json:"usage"`
		} `json:"message"`
		UsageMetadata *struct { // Gemini
			PromptTokenCount        int `json:"promptTokenCount"`
			CandidatesTokenCount    int `json:"candidatesTokenCount"`
			CachedContentTokenCount int `json:"cachedContentTokenCount"`
		} `json:"usageMetadata"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return tokenUsage{}, false, ""
	}

	model = parsed.Model
	if parsed.Message != nil && parsed.Message.Model != "" {
		model = parsed.Message.Model
	}

	// Gemini: usageMetadata with camelCase counts (cached is a subset of prompt).
	if g := parsed.UsageMetadata; g != nil {
		return tokenUsage{
			InputTokens:  g.PromptTokenCount,
			OutputTokens: g.CandidatesTokenCount,
			CachedTokens: g.CachedContentTokenCount,
		}, true, model
	}

	// OpenAI / Anthropic: a "usage" block, possibly nested under "message".
	w := parsed.Usage
	if w == nil && parsed.Message != nil {
		w = parsed.Message.Usage
	}
	if w == nil {
		return tokenUsage{}, false, model
	}

	switch {
	case w.PromptTokens != nil || w.CompletionTokens != nil || w.PromptTokensDetails != nil:
		// OpenAI: cached_tokens is a subset of prompt_tokens (already the total).
		u = tokenUsage{InputTokens: deref(w.PromptTokens), OutputTokens: deref(w.CompletionTokens)}
		if w.PromptTokensDetails != nil {
			u.CachedTokens = w.PromptTokensDetails.CachedTokens
		}
		return u, true, model
	case w.InputTokens != nil || w.OutputTokens != nil || w.CacheReadInputTokens != 0 || w.CacheCreationInputTokens != 0:
		// Anthropic: input_tokens EXCLUDES the cache buckets, so the total prompt
		// is the sum. A message_delta carries only output_tokens (input nil) —
		// the stream merge folds it into the message_start usage.
		return tokenUsage{
			InputTokens:      deref(w.InputTokens) + w.CacheReadInputTokens + w.CacheCreationInputTokens,
			OutputTokens:     deref(w.OutputTokens),
			CachedTokens:     w.CacheReadInputTokens,
			CacheWriteTokens: w.CacheCreationInputTokens,
		}, true, model
	default:
		return tokenUsage{}, false, model
	}
}

func deref(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

// usageFromResponse is a thin wrapper over parseUsageJSON for the non-streaming
// path and tests.
func usageFromResponse(body []byte) (tokenUsage, string) {
	u, _, model := parseUsageJSON(body)
	return u, model
}

// modelFromRequest reads the "model" field from an OpenAI- or Anthropic-
// compatible request body. Returns "" if absent or unparseable (Gemini names the
// model in the URL path instead — see modelFromPath).
func modelFromRequest(body []byte) string {
	var parsed struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return ""
	}
	return parsed.Model
}

// modelFromPath extracts the model from a Gemini-style request path such as
// /v1beta/models/gemini-2.0-flash:generateContent. Returns "" for paths that do
// not carry a "/models/<name>" segment (OpenAI/Anthropic name it in the body).
func modelFromPath(path string) string {
	const marker = "/models/"
	i := strings.Index(path, marker)
	if i < 0 {
		return ""
	}
	rest := path[i+len(marker):]
	// Strip the ":method" suffix (generateContent, streamGenerateContent, …).
	if c := strings.IndexByte(rest, ':'); c >= 0 {
		rest = rest[:c]
	}
	// And any trailing path segment.
	if sl := strings.IndexByte(rest, '/'); sl >= 0 {
		rest = rest[:sl]
	}
	return rest
}

// maybeInjectIncludeUsage returns body with stream_options.include_usage set to
// true when it is an OpenAI chat-completions streaming request that hasn't opted
// in — so the provider emits an exact usage block Augur can record instead of a
// zero-token row. It is a no-op (returns body unchanged) for non-matching paths,
// non-streaming requests, requests that already set the option, and bodies that
// don't parse — the request must never break because of injection. Restricting
// to the OpenAI chat path keeps the extra field away from Anthropic/Gemini
// (which report usage without opting in and would reject an unknown parameter).
func maybeInjectIncludeUsage(path string, body []byte) []byte {
	if !strings.Contains(path, "/chat/completions") {
		return body
	}
	var probe struct {
		Stream        *bool `json:"stream"`
		StreamOptions *struct {
			IncludeUsage *bool `json:"include_usage"`
		} `json:"stream_options"`
	}
	if err := json.Unmarshal(body, &probe); err != nil {
		return body
	}
	if probe.Stream == nil || !*probe.Stream {
		return body // not a streaming request
	}
	if probe.StreamOptions != nil && probe.StreamOptions.IncludeUsage != nil && *probe.StreamOptions.IncludeUsage {
		return body // already opted in
	}

	// Merge into a generic map so every other field the agent set is preserved
	// (key order is not, which is immaterial to the provider).
	var m map[string]json.RawMessage
	if err := json.Unmarshal(body, &m); err != nil {
		return body
	}
	opts := map[string]json.RawMessage{}
	if raw, ok := m["stream_options"]; ok {
		_ = json.Unmarshal(raw, &opts) // best-effort; overwrite include_usage below
	}
	opts["include_usage"] = json.RawMessage("true")
	optsRaw, err := json.Marshal(opts)
	if err != nil {
		return body
	}
	m["stream_options"] = optsRaw
	out, err := json.Marshal(m)
	if err != nil {
		return body
	}
	return out
}

// hopByHopHeaders are connection-specific headers that must not be forwarded by
// a proxy (RFC 7230 §6.1).
var hopByHopHeaders = []string{
	"Connection", "Proxy-Connection", "Keep-Alive", "Proxy-Authenticate",
	"Proxy-Authorization", "Te", "Trailer", "Transfer-Encoding", "Upgrade",
}

func stripHopByHop(h http.Header) {
	for _, k := range hopByHopHeaders {
		h.Del(k)
	}
}

func copyHeader(dst, src http.Header) {
	for k, vs := range src {
		if isHopByHop(k) {
			continue
		}
		for _, v := range vs {
			dst.Add(k, v)
		}
	}
}

func isHopByHop(key string) bool {
	for _, h := range hopByHopHeaders {
		if strings.EqualFold(h, key) {
			return true
		}
	}
	return false
}

// singleJoiningSlash joins two URL path segments with exactly one slash,
// matching the behavior of httputil.NewSingleHostReverseProxy.
func singleJoiningSlash(a, b string) string {
	aslash := strings.HasSuffix(a, "/")
	bslash := strings.HasPrefix(b, "/")
	switch {
	case aslash && bslash:
		return a + b[1:]
	case !aslash && !bslash:
		return a + "/" + b
	}
	return a + b
}
