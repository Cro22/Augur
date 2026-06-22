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
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

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

// Server is the recording proxy. Construct it with New and mount it with
// http.ListenAndServe; it implements http.Handler.
type Server struct {
	upstream *url.URL
	tracer   *trace.Writer
	client   *http.Client
	now      nowFunc

	mu  sync.Mutex
	seq map[string]int // per (scenario|run) next call ordinal
}

// New returns a Server that forwards to upstream (e.g. https://api.openai.com)
// and appends trace rows via tracer. A nil client uses a sensible default.
func New(upstream *url.URL, tracer *trace.Writer, client *http.Client) *Server {
	if client == nil {
		client = &http.Client{Timeout: 10 * time.Minute}
	}
	return &Server{
		upstream: upstream,
		tracer:   tracer,
		client:   client,
		now:      time.Now,
		seq:      make(map[string]int),
	}
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

	outReq, err := s.buildUpstreamRequest(r, reqBody)
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

	// Dispatch on the response shape. A streamed (SSE) body must be relayed
	// chunk-by-chunk so the agent sees tokens in real time; a regular JSON body
	// is read in full. Both extract usage the same way, so totals reconcile.
	var (
		usage     oaiUsage
		respModel string
		ok        bool
	)
	if isEventStream(resp.Header) {
		usage, respModel = s.streamResponse(w, resp)
	} else if usage, respModel, ok = s.bufferResponse(w, resp); !ok {
		return // bufferResponse already wrote an error to the client
	}
	latency := s.now().Sub(start)

	// Record the call. Parse failures (a non-JSON body, an error response with
	// no usage) yield zero tokens but the row is still written — a burned-but-
	// failed call is precisely the surprise the trace exists to surface.
	model := reqModel
	if model == "" {
		model = respModel
	}
	rec := trace.Record{
		Timestamp:    start.UTC().Format(time.RFC3339Nano),
		ScenarioID:   scenario,
		RunID:        run,
		Seq:          s.nextSeq(scenario, run),
		Model:        model,
		InputTokens:  usage.InputTokens,
		OutputTokens: usage.OutputTokens,
		CachedTokens: usage.CachedTokens,
		LatencyMs:    latency.Milliseconds(),
		Endpoint:     r.URL.Path,
		Status:       resp.StatusCode,
	}
	if err := s.tracer.Write(rec); err != nil {
		// A trace write failure must not corrupt the agent's response, but it
		// must be loud: a silently dropped row means an under-counted bill.
		fmt.Printf("augur proxy: WARNING trace write failed: %v\n", err)
	}
}

// bufferResponse relays a non-streaming response: it reads the whole body,
// copies headers/status, writes the body through unchanged, and parses usage.
// It returns ok=false (after writing a 502) only if the upstream body cannot be
// read — in which case the caller must not record a row.
func (s *Server) bufferResponse(w http.ResponseWriter, resp *http.Response) (oaiUsage, string, bool) {
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "augur proxy: reading upstream response: "+err.Error(), http.StatusBadGateway)
		return oaiUsage{}, "", false
	}
	copyHeader(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(body)

	usage, _, model := parseUsageJSON(body)
	return usage, model, true
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

// oaiUsage is the token accounting block of an OpenAI-compatible response.
type oaiUsage struct {
	InputTokens  int
	OutputTokens int
	CachedTokens int
}

// parseUsageJSON extracts token usage and the resolved model from one
// OpenAI-compatible JSON object — a full non-streaming response body or a single
// streamed chunk's payload. usage is a pointer in the wire shape so we can tell
// "no usage block" (every streamed chunk before the last) from "usage with zero
// tokens": hasUsage is false in the former. Both the buffered and streaming
// paths funnel through here so their accounting cannot drift apart.
func parseUsageJSON(data []byte) (u oaiUsage, hasUsage bool, model string) {
	var parsed struct {
		Model string `json:"model"`
		Usage *struct {
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			PromptTokensDetails struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(data, &parsed); err != nil {
		return oaiUsage{}, false, ""
	}
	if parsed.Usage == nil {
		return oaiUsage{}, false, parsed.Model
	}
	return oaiUsage{
		InputTokens:  parsed.Usage.PromptTokens,
		OutputTokens: parsed.Usage.CompletionTokens,
		CachedTokens: parsed.Usage.PromptTokensDetails.CachedTokens,
	}, true, parsed.Model
}

// usageFromResponse is a thin wrapper over parseUsageJSON for the non-streaming
// path and tests.
func usageFromResponse(body []byte) (oaiUsage, string) {
	u, _, model := parseUsageJSON(body)
	return u, model
}

// modelFromRequest reads the "model" field from an OpenAI-compatible request
// body. Returns "" if absent or unparseable.
func modelFromRequest(body []byte) string {
	var parsed struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return ""
	}
	return parsed.Model
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
