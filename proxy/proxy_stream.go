package proxy

import (
	"bufio"
	"bytes"
	"net/http"
	"strings"
)

// isEventStream reports whether a response is a Server-Sent Events stream (the
// shape an OpenAI-compatible provider returns for stream:true). We branch on the
// response's Content-Type rather than the request's stream flag because the
// response is the source of truth for how the body must be relayed.
func isEventStream(h http.Header) bool {
	ct := h.Get("Content-Type")
	return strings.HasPrefix(strings.TrimSpace(strings.ToLower(ct)), "text/event-stream")
}

// streamResponse relays an SSE body to the client chunk-by-chunk, flushing after
// each line so the agent receives tokens in real time, while scanning for the
// usage block that providers emit in the final chunk when the request set
// stream_options.include_usage. It returns the captured usage (zero if the
// stream carried none) and the resolved model.
//
// Relaying is byte-exact: each line is read with its delimiter intact and
// written straight back, so the client sees precisely the provider's stream.
func (s *Server) streamResponse(w http.ResponseWriter, resp *http.Response) (oaiUsage, string) {
	copyHeader(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)

	rc := http.NewResponseController(w)
	_ = rc.Flush() // flush headers so the client opens the stream immediately

	var (
		usage oaiUsage
		model string
	)
	br := bufio.NewReader(resp.Body)
	for {
		// ReadBytes keeps the trailing '\n', so what we write is byte-identical
		// to what the provider sent (including the blank lines between events).
		chunk, err := br.ReadBytes('\n')
		if len(chunk) > 0 {
			_, _ = w.Write(chunk)
			_ = rc.Flush()
			if u, has, m := parseSSEChunk(chunk); m != "" || has {
				if m != "" {
					model = m
				}
				if has {
					usage = u
				}
			}
		}
		if err != nil {
			// io.EOF is the clean end of stream; any other error means the
			// upstream connection broke mid-stream. We have already written the
			// status and partial body, so we cannot change the response — record
			// whatever usage we captured and stop.
			break
		}
	}
	return usage, model
}

// dataPrefix is the SSE field that carries each chunk's JSON payload.
var dataPrefix = []byte("data:")

// doneMarker terminates an OpenAI SSE stream.
var doneMarker = []byte("[DONE]")

// parseSSEChunk pulls usage/model out of a single SSE line. Non-data lines,
// comments, the [DONE] marker, and chunks without a usage block return
// hasUsage=false (model may still be set).
func parseSSEChunk(line []byte) (u oaiUsage, hasUsage bool, model string) {
	t := bytes.TrimSpace(line)
	if !bytes.HasPrefix(t, dataPrefix) {
		return oaiUsage{}, false, ""
	}
	payload := bytes.TrimSpace(t[len(dataPrefix):])
	if len(payload) == 0 || bytes.Equal(payload, doneMarker) {
		return oaiUsage{}, false, ""
	}
	return parseUsageJSON(payload)
}
