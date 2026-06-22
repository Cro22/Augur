package trace

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func TestWriteReadRoundTrip(t *testing.T) {
	want := []Record{
		{
			Timestamp: "2026-06-21T18:00:00.5Z", ScenarioID: "checkout", RunID: "run-1",
			Seq: 0, Model: "gpt-4o", InputTokens: 1234, OutputTokens: 567,
			CachedTokens: 200, LatencyMs: 842, Endpoint: "/v1/chat/completions", Status: 200,
		},
		{
			Timestamp: "2026-06-21T18:00:01Z", ScenarioID: "checkout", RunID: "run-1",
			Seq: 1, Model: "gpt-4o-mini", InputTokens: 50, OutputTokens: 10,
			LatencyMs: 120, Endpoint: "/v1/chat/completions", Status: 200,
		},
	}

	var buf bytes.Buffer
	w := NewWriter(&buf)
	for _, r := range want {
		if err := w.Write(r); err != nil {
			t.Fatalf("Write: %v", err)
		}
	}

	got, err := ReadAll(&buf)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("got %d records, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("record %d:\n got %+v\nwant %+v", i, got[i], want[i])
		}
	}
}

func TestWriteOneRecordPerLine(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)
	for i := range 3 {
		if err := w.Write(Record{Seq: i, Model: "gpt-4o"}); err != nil {
			t.Fatalf("Write: %v", err)
		}
	}
	lines := strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
	if len(lines) != 3 {
		t.Fatalf("got %d lines, want 3 (one JSON object per line):\n%s", len(lines), buf.String())
	}
	for _, l := range lines {
		if !strings.HasPrefix(l, "{") || !strings.HasSuffix(l, "}") {
			t.Errorf("line is not a standalone JSON object: %q", l)
		}
	}
}

func TestReadAllSkipsBlankLines(t *testing.T) {
	in := "\n" + `{"seq":0,"model":"gpt-4o"}` + "\n\n" + `{"seq":1,"model":"gpt-4o-mini"}` + "\n"
	got, err := ReadAll(strings.NewReader(in))
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d records, want 2", len(got))
	}
}

func TestReadAllRejectsMalformed(t *testing.T) {
	in := `{"seq":0,"model":"gpt-4o"}` + "\n" + `{not json}` + "\n"
	_, err := ReadAll(strings.NewReader(in))
	if err == nil {
		t.Fatal("expected error for malformed line, got nil")
	}
	if !strings.Contains(err.Error(), "line 2") {
		t.Errorf("error should name the bad line number, got: %v", err)
	}
}

// Concurrent writes must not interleave: every line must remain a parseable
// JSON object. This guards the mutex in Write — the proxy serves requests in
// parallel.
func TestWriteConcurrent(t *testing.T) {
	var buf bytes.Buffer
	w := NewWriter(&buf)

	const n = 200
	var wg sync.WaitGroup
	wg.Add(n)
	for i := range n {
		go func(i int) {
			defer wg.Done()
			if err := w.Write(Record{Seq: i, Model: "gpt-4o", RunID: "r"}); err != nil {
				t.Errorf("Write: %v", err)
			}
		}(i)
	}
	wg.Wait()

	got, err := ReadAll(&buf)
	if err != nil {
		t.Fatalf("ReadAll after concurrent writes (lines interleaved?): %v", err)
	}
	if len(got) != n {
		t.Fatalf("got %d records, want %d", len(got), n)
	}
}

func TestOpenFileAppends(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trace.jsonl")

	w1, err := OpenFile(path)
	if err != nil {
		t.Fatalf("OpenFile: %v", err)
	}
	if err := w1.Write(Record{Seq: 0, Model: "gpt-4o"}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if err := w1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Reopening the same path must preserve the first row, not truncate it.
	w2, err := OpenFile(path)
	if err != nil {
		t.Fatalf("OpenFile (reopen): %v", err)
	}
	if err := w2.Write(Record{Seq: 1, Model: "gpt-4o-mini"}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if err := w2.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	got := mustReadFile(t, path)
	if len(got) != 2 {
		t.Fatalf("got %d records after reopen+append, want 2", len(got))
	}
	if got[0].Seq != 0 || got[1].Seq != 1 {
		t.Errorf("append order wrong: %+v", got)
	}
}

func mustReadFile(t *testing.T, path string) []Record {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open for read: %v", err)
	}
	defer f.Close()
	recs, err := ReadAll(f)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	return recs
}
