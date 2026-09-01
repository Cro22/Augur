package runner

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestRunParallelRunsEverything checks the parallel path executes every
// invocation and reports a Summary assembled in task order regardless of
// completion order.
func TestRunParallelRunsEverything(t *testing.T) {
	var mu sync.Mutex
	var n int
	opts := Options{
		Concurrency: 4,
		Stdout:      io.Discard, Stderr: io.Discard,
		Exec: func(_ context.Context, _, _ []string, _, _ io.Writer) error {
			mu.Lock()
			n++
			mu.Unlock()
			return nil
		},
	}
	sum, err := Run(context.Background(), twoScenarios(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if n != 6 || sum.Total != 6 || sum.Failed != 0 {
		t.Fatalf("n=%d Summary=%+v, want 6 invocations, Total 6 Failed 0", n, sum)
	}

	// Summary order must be the deterministic task order (scenario, then index),
	// not whatever order the goroutines happened to finish in.
	want := []string{
		"checkout-000", "checkout-001", "checkout-002",
		"faq-000", "faq-001", "faq-002",
	}
	if len(sum.Invocations) != len(want) {
		t.Fatalf("got %d invocations, want %d", len(sum.Invocations), len(want))
	}
	for i, inv := range sum.Invocations {
		if inv.RunID != want[i] {
			t.Errorf("Invocations[%d].RunID = %q, want %q", i, inv.RunID, want[i])
		}
	}
}

// TestRunParallelIsActuallyConcurrent proves invocations really overlap: with
// concurrency N and N tasks, all N must be in flight at once before any is
// released. If the runner were still sequential this blocks and the test times
// out.
func TestRunParallelIsActuallyConcurrent(t *testing.T) {
	const n = 3
	cfg := Config{Runs: n, Command: []string{"x"}, Scenarios: []Scenario{{ID: "a", Input: "i"}}}

	var mu sync.Mutex
	inFlight, peak := 0, 0
	var arrived sync.WaitGroup
	arrived.Add(n)
	gate := make(chan struct{})

	opts := Options{
		Concurrency: n,
		Stdout:      io.Discard, Stderr: io.Discard,
		Exec: func(_ context.Context, _, _ []string, _, _ io.Writer) error {
			mu.Lock()
			inFlight++
			if inFlight > peak {
				peak = inFlight
			}
			mu.Unlock()
			arrived.Done()
			<-gate // hold until every invocation has arrived
			mu.Lock()
			inFlight--
			mu.Unlock()
			return nil
		},
	}

	done := make(chan Summary, 1)
	go func() {
		sum, _ := Run(context.Background(), cfg, opts)
		done <- sum
	}()

	allArrived := make(chan struct{})
	go func() { arrived.Wait(); close(allArrived) }()
	select {
	case <-allArrived:
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for concurrent invocations; runner is not parallel")
	}
	close(gate)

	sum := <-done
	if sum.Total != n {
		t.Errorf("Total = %d, want %d", sum.Total, n)
	}
	if peak != n {
		t.Errorf("peak concurrency = %d, want %d", peak, n)
	}
}

// TestRunParallelOutputNotInterleaved checks each invocation's output is flushed
// as one contiguous block even when many run at once.
func TestRunParallelOutputNotInterleaved(t *testing.T) {
	const runs = 20
	cfg := Config{Runs: runs, Command: []string{"x"}, Scenarios: []Scenario{{ID: "a", Input: "i"}}}

	var out bytes.Buffer // runner serializes flushes into this shared writer
	opts := Options{
		Concurrency: 8,
		Stdout:      &out, Stderr: io.Discard,
		Exec: func(_ context.Context, _, env []string, stdout, _ io.Writer) error {
			id := envMap(env)[EnvRunID]
			// Two separate writes: if flushing weren't buffered per invocation,
			// another goroutine's block could land between them.
			fmt.Fprintf(stdout, "<%s>", id)
			fmt.Fprintf(stdout, "</%s>", id)
			return nil
		},
	}
	if _, err := Run(context.Background(), cfg, opts); err != nil {
		t.Fatalf("Run: %v", err)
	}

	got := out.String()
	for i := range runs {
		block := fmt.Sprintf("<a-%03d></a-%03d>", i, i)
		if !strings.Contains(got, block) {
			t.Errorf("invocation %d output not contiguous; missing %q in:\n%s", i, block, got)
		}
	}
}

// TestRunParallelAbortsOnError checks that without ContinueOnError the first
// genuine failure aborts the run and is the error returned.
func TestRunParallelAbortsOnError(t *testing.T) {
	cfg := Config{Runs: 1, Command: []string{"x"}, Scenarios: []Scenario{
		{ID: "boom", Input: "i"},
	}}
	opts := Options{
		Concurrency: 2,
		Stdout:      io.Discard, Stderr: io.Discard,
		Exec: func(_ context.Context, _, _ []string, _, _ io.Writer) error {
			return errors.New("agent crashed")
		},
	}
	sum, err := Run(context.Background(), cfg, opts)
	if err == nil {
		t.Fatal("expected the failure to be returned")
	}
	if !strings.Contains(err.Error(), "agent crashed") {
		t.Errorf("error = %v, want it to wrap the agent failure", err)
	}
	if sum.Failed == 0 {
		t.Errorf("Failed = 0, want >= 1")
	}
}

// TestRunParallelContinueOnError checks every invocation runs and failures are
// counted, not aborted, under concurrency.
func TestRunParallelContinueOnError(t *testing.T) {
	var mu sync.Mutex
	var n int
	opts := Options{
		Concurrency:     4,
		ContinueOnError: true,
		Stdout:          io.Discard, Stderr: io.Discard,
		Exec: func(_ context.Context, _, env []string, _, _ io.Writer) error {
			mu.Lock()
			n++
			mu.Unlock()
			// Fail exactly the third repetition of each scenario.
			if strings.HasSuffix(envMap(env)[EnvRunID], "-002") {
				return errors.New("flaky")
			}
			return nil
		},
	}
	sum, err := Run(context.Background(), twoScenarios(), opts)
	if err != nil {
		t.Fatalf("ContinueOnError should not return error: %v", err)
	}
	if n != 6 || sum.Total != 6 {
		t.Errorf("n=%d Total=%d, want 6 (kept going)", n, sum.Total)
	}
	if sum.Failed != 2 { // checkout-002 and faq-002
		t.Errorf("Failed = %d, want 2", sum.Failed)
	}
}
