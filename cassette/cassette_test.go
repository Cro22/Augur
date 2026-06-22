package cassette

import (
	"os"
	"path/filepath"
	"testing"
)

// appendLine appends a raw line to a file, for corrupting a cassette in tests.
func appendLine(path, line string) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.WriteString(line + "\n")
	return err
}

func TestRecordThenLoad(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cassette.jsonl")

	c, err := Create(path)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	entries := []Entry{
		{ScenarioID: "checkout", RunID: "checkout-000", Seq: 0, Status: 200,
			ContentType: "application/json", LatencyMs: 120, Body: `{"usage":{"prompt_tokens":10}}`},
		{ScenarioID: "checkout", RunID: "checkout-000", Seq: 1, Status: 200,
			ContentType: "text/event-stream", LatencyMs: 80, Body: "data: [DONE]\n\n"},
	}
	for _, e := range entries {
		if err := c.Record(e); err != nil {
			t.Fatalf("Record: %v", err)
		}
	}
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if loaded.Len() != 2 {
		t.Fatalf("Len = %d, want 2", loaded.Len())
	}

	got, ok := loaded.Lookup("checkout", "checkout-000", 0)
	if !ok {
		t.Fatal("lookup seq 0 missed")
	}
	if got != entries[0] {
		t.Errorf("entry 0 = %+v, want %+v", got, entries[0])
	}
	if _, ok := loaded.Lookup("checkout", "checkout-000", 1); !ok {
		t.Error("lookup seq 1 missed")
	}
}

func TestLookupMiss(t *testing.T) {
	path := filepath.Join(t.TempDir(), "c.jsonl")
	c, _ := Create(path)
	c.Record(Entry{ScenarioID: "a", RunID: "a-000", Seq: 0, Body: "{}"})
	c.Close()

	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if _, ok := loaded.Lookup("a", "a-000", 1); ok {
		t.Error("expected miss for unrecorded seq")
	}
	if _, ok := loaded.Lookup("b", "b-000", 0); ok {
		t.Error("expected miss for unrecorded scenario")
	}
}

func TestCreateTruncates(t *testing.T) {
	path := filepath.Join(t.TempDir(), "c.jsonl")
	c1, _ := Create(path)
	c1.Record(Entry{ScenarioID: "a", RunID: "r", Seq: 0, Body: "{}"})
	c1.Close()

	// Re-creating must start fresh, not append.
	c2, _ := Create(path)
	c2.Record(Entry{ScenarioID: "b", RunID: "r", Seq: 0, Body: "{}"})
	c2.Close()

	loaded, _ := Load(path)
	if loaded.Len() != 1 {
		t.Errorf("Len = %d, want 1 (Create should truncate)", loaded.Len())
	}
	if _, ok := loaded.Lookup("b", "r", 0); !ok {
		t.Error("expected the second recording to be the only entry")
	}
}

func TestLoadRejectsMalformed(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.jsonl")
	c, _ := Create(path)
	c.Record(Entry{ScenarioID: "a", RunID: "r", Seq: 0, Body: "{}"})
	c.Close()
	// Append a malformed line.
	if err := appendLine(path, "{not json"); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Error("expected error loading malformed cassette")
	}
}
