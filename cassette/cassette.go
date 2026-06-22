// Package cassette is the record/replay store for the proxy: a recording of
// every LLM response keyed by the call's identity (scenario, run, seq).
//
// It exists to cut the token cost of re-running the gate (SPEC D3 / Hito 5).
// Record once against the real provider; thereafter replay the cassette so CI
// re-runs the agent against the recorded responses and regenerates the cost
// trace without spending a cent. Because replay re-executes the AGENT (not just
// the trace), a cost regression introduced by an agent code change still shows
// up — it is exercised against the old LLM responses, for free.
//
// The on-disk format is JSON Lines, one Entry per line, so a cassette is
// diffable and can be committed to the repo alongside the scenarios.
package cassette

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sync"
)

// Entry is one recorded response. Body holds the raw response bytes as a string
// (OpenAI-compatible responses are UTF-8 text — JSON or an SSE stream), so the
// cassette stays human-readable.
type Entry struct {
	ScenarioID  string `json:"scenario_id"`
	RunID       string `json:"run_id"`
	Seq         int    `json:"seq"`
	Status      int    `json:"status"`
	ContentType string `json:"content_type,omitempty"`
	LatencyMs   int64  `json:"latency_ms"`
	Body        string `json:"body"`
}

type key struct {
	scenario, run string
	seq           int
}

// Cassette records entries (in record mode) or serves them (in replay mode).
// A given Cassette is used for one or the other, not both. It is safe for
// concurrent use.
type Cassette struct {
	mu      sync.Mutex
	entries map[key]Entry // populated for replay
	w       io.Writer     // set for record
	closer  io.Closer
}

// Create opens path for recording, truncating any existing cassette — a
// recording is a fresh capture, not an append.
func Create(path string) (*Cassette, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("cassette: creating cassette: %w", err)
	}
	return &Cassette{w: f, closer: f}, nil
}

// Load reads a cassette from path into memory for replay.
func Load(path string) (*Cassette, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("cassette: opening cassette: %w", err)
	}
	defer f.Close()

	entries := make(map[key]Entry)
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	line := 0
	for sc.Scan() {
		line++
		b := sc.Bytes()
		if len(b) == 0 {
			continue
		}
		var e Entry
		if err := json.Unmarshal(b, &e); err != nil {
			return nil, fmt.Errorf("cassette: parsing line %d: %w", line, err)
		}
		entries[key{e.ScenarioID, e.RunID, e.Seq}] = e
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("cassette: reading cassette: %w", err)
	}
	return &Cassette{entries: entries}, nil
}

// Record appends one entry to the cassette. Concurrent calls are serialized so
// lines never interleave.
func (c *Cassette) Record(e Entry) error {
	b, err := json.Marshal(e)
	if err != nil {
		return fmt.Errorf("cassette: marshaling entry: %w", err)
	}
	b = append(b, '\n')

	c.mu.Lock()
	defer c.mu.Unlock()
	if _, err := c.w.Write(b); err != nil {
		return fmt.Errorf("cassette: writing entry: %w", err)
	}
	return nil
}

// Lookup returns the recorded entry for a call, or ok=false on a replay miss
// (the agent made a call that was not recorded — a sign its behavior diverged
// from the recording).
func (c *Cassette) Lookup(scenario, run string, seq int) (Entry, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[key{scenario, run, seq}]
	return e, ok
}

// Len reports how many entries a loaded cassette holds.
func (c *Cassette) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.entries)
}

// Close closes the underlying file when the cassette owns one (record mode).
func (c *Cassette) Close() error {
	if c.closer == nil {
		return nil
	}
	return c.closer.Close()
}
