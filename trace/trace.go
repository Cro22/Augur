// Package trace defines the cost-trace record — the ledger row the recording
// proxy appends for every LLM call — and a concurrency-safe JSONL writer for
// it.
//
// A trace deliberately records token usage, NOT dollar cost. Cost is a
// downstream computation (see package cost + the projection engine): the same
// trace can be re-priced against a different pricing snapshot without re-running
// the agent. Keeping the proxy ignorant of pricing is the whole point of the
// split.
//
// The on-disk format is JSON Lines (one JSON object per line): trivially
// appendable while a run is in flight, streamable when reading back, and
// greppable by hand for the Hito 1 checkpoint (reconcile rows against the
// provider's reported usage).
package trace

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
)

// Record is one LLM call as observed by the proxy. Field names in JSON are
// short and stable: a trace file is a data artifact other tools (and humans)
// read, so the schema is part of the contract.
type Record struct {
	// Timestamp is when the proxy received the response, RFC3339 with
	// nanoseconds. The proxy stamps it so the writer stays pure I/O.
	Timestamp string `json:"ts"`
	// ScenarioID and RunID tie the call back to the scenario it exercised and
	// the specific repetition (run) it belonged to. The proxy reads them from
	// request headers set by the runner.
	ScenarioID string `json:"scenario_id"`
	RunID      string `json:"run_id"`
	// Seq is the call's ordinal within its (scenario, run): 0 for the first
	// LLM call the agent makes, 1 for the next, and so on. This preserves the
	// call graph's ordering — the fan-out/retry structure the projection engine
	// later mines for multipliers.
	Seq int `json:"seq"`
	// Model is the model the call was billed against. The proxy prefers the
	// model named in the request (it matches pricing.yaml keys), falling back
	// to the model echoed in the response.
	Model string `json:"model"`
	// Token accounting, mirroring the provider's report. CachedTokens is the
	// cached SUBSET of InputTokens, not an additional bucket (see package cost).
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
	CachedTokens int `json:"cached_tokens"`
	// LatencyMs is the wall-clock time the upstream provider took to respond.
	LatencyMs int64 `json:"latency_ms"`
	// Endpoint is the request path (e.g. /v1/chat/completions), kept so a trace
	// can distinguish chat from embeddings or other endpoints.
	Endpoint string `json:"endpoint,omitempty"`
	// Status is the upstream HTTP status code. Non-2xx rows are kept on purpose:
	// failed calls that still burned tokens are exactly the cost surprise Augur
	// hunts for, and a row with zero tokens still records that an attempt was
	// made.
	Status int `json:"status,omitempty"`
}

// Writer appends Records to an io.Writer as JSON Lines. It is safe for
// concurrent use: the proxy serves requests in parallel, so Write is guarded by
// a mutex to keep lines from interleaving.
type Writer struct {
	mu     sync.Mutex
	w      io.Writer
	closer io.Closer // non-nil only when the writer owns a file
}

// NewWriter returns a Writer that appends to w. The caller owns w's lifecycle;
// Close is a no-op for the underlying writer. Use OpenFile when you want the
// Writer to own a file on disk.
func NewWriter(w io.Writer) *Writer {
	return &Writer{w: w}
}

// OpenFile opens (creating if needed) path for appending and returns a Writer
// that owns it. Existing trace rows are preserved — a run appends to the
// ledger, it does not truncate it. Close flushes and closes the file.
func OpenFile(path string) (*Writer, error) {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("trace: opening trace file: %w", err)
	}
	return &Writer{w: f, closer: f}, nil
}

// Write appends one record as a single JSON line. It is atomic with respect to
// other concurrent Writes: each call marshals to a buffer first, then writes
// the whole line under the lock, so records never interleave.
func (w *Writer) Write(r Record) error {
	line, err := json.Marshal(r)
	if err != nil {
		return fmt.Errorf("trace: marshaling record: %w", err)
	}
	line = append(line, '\n')

	w.mu.Lock()
	defer w.mu.Unlock()
	if _, err := w.w.Write(line); err != nil {
		return fmt.Errorf("trace: writing record: %w", err)
	}
	return nil
}

// Close closes the underlying file if this Writer owns one. It is safe to call
// on a Writer created with NewWriter (it does nothing).
func (w *Writer) Close() error {
	if w.closer == nil {
		return nil
	}
	return w.closer.Close()
}

// ReadAll parses a JSONL trace stream into records. It is used by tests and by
// downstream consumers (aggregation, projection) that need the whole ledger in
// memory. Blank lines are skipped; a malformed line is a hard error with its
// line number, because a corrupt trace must not silently shrink the bill.
func ReadAll(r io.Reader) ([]Record, error) {
	var records []Record
	sc := bufio.NewScanner(r)
	// Trace lines can be long (large requests echo no body here, but be
	// generous); raise the scanner's max token size well above the default 64K.
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)

	line := 0
	for sc.Scan() {
		line++
		b := sc.Bytes()
		if len(b) == 0 {
			continue
		}
		var rec Record
		if err := json.Unmarshal(b, &rec); err != nil {
			return nil, fmt.Errorf("trace: parsing line %d: %w", line, err)
		}
		records = append(records, rec)
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("trace: reading trace: %w", err)
	}
	return records, nil
}
