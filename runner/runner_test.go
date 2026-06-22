package runner

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"
)

// envMap turns a KEY=VALUE slice into a map for assertions.
func envMap(env []string) map[string]string {
	m := make(map[string]string, len(env))
	for _, kv := range env {
		if k, v, ok := strings.Cut(kv, "="); ok {
			m[k] = v
		}
	}
	return m
}

func twoScenarios() Config {
	return Config{
		Runs:    3,
		Command: []string{"./agent", "--prompt", "{{input}}"},
		Scenarios: []Scenario{
			{ID: "checkout", Input: "return order"},
			{ID: "faq", Input: "shipping?"},
		},
	}
}

func TestRunInjectsContract(t *testing.T) {
	type capture struct {
		args []string
		env  map[string]string
	}
	var caps []capture

	opts := Options{
		BaseURL: "http://localhost:9999",
		BaseEnv: []string{"PATH=/usr/bin", "AUGUR_RUN_ID=stale"}, // stale AUGUR_* must be dropped
		Stdout:  io.Discard,
		Stderr:  io.Discard,
		Exec: func(_ context.Context, args, env []string, _, _ io.Writer) error {
			caps = append(caps, capture{args: args, env: envMap(env)})
			return nil
		},
	}

	sum, err := Run(context.Background(), twoScenarios(), opts)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	// 2 scenarios × 3 runs = 6 invocations.
	if sum.Total != 6 || sum.Failed != 0 {
		t.Fatalf("Summary = %+v, want Total 6 Failed 0", sum)
	}
	if len(caps) != 6 {
		t.Fatalf("got %d invocations, want 6", len(caps))
	}

	first := caps[0]
	if first.env[EnvScenarioID] != "checkout" {
		t.Errorf("scenario id = %q, want checkout", first.env[EnvScenarioID])
	}
	if first.env[EnvRunID] != "checkout-000" {
		t.Errorf("run id = %q, want checkout-000", first.env[EnvRunID])
	}
	if first.env[EnvInput] != "return order" {
		t.Errorf("input env = %q, want 'return order'", first.env[EnvInput])
	}
	if first.env[EnvBaseURL] != "http://localhost:9999" {
		t.Errorf("base url = %q", first.env[EnvBaseURL])
	}
	// Stale AUGUR_RUN_ID from BaseEnv must not survive.
	if first.env[EnvRunID] == "stale" {
		t.Error("stale AUGUR_RUN_ID leaked from BaseEnv")
	}
	if first.env["PATH"] != "/usr/bin" {
		t.Errorf("PATH not inherited: %q", first.env["PATH"])
	}
	// {{input}} substituted into argv.
	if first.args[2] != "return order" {
		t.Errorf("argv input substitution = %q, want 'return order'", first.args[2])
	}

	// Run ids are unique and grouped per scenario.
	seen := map[string]bool{}
	for _, c := range caps {
		id := c.env[EnvRunID]
		if seen[id] {
			t.Errorf("duplicate run id %q", id)
		}
		seen[id] = true
	}
	if !seen["checkout-002"] || !seen["faq-000"] || !seen["faq-002"] {
		t.Errorf("expected per-scenario indexed run ids, got %v", seen)
	}
}

func TestRunSessionPrefixesRunID(t *testing.T) {
	var ids []string
	opts := Options{
		Session: "20260621-1800",
		Stdout:  io.Discard, Stderr: io.Discard,
		Exec: func(_ context.Context, _, env []string, _, _ io.Writer) error {
			ids = append(ids, envMap(env)[EnvRunID])
			return nil
		},
	}
	cfg := Config{Runs: 1, Command: []string{"x"}, Scenarios: []Scenario{{ID: "a", Input: "i"}}}
	if _, err := Run(context.Background(), cfg, opts); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if ids[0] != "20260621-1800-a-000" {
		t.Errorf("run id = %q, want 20260621-1800-a-000", ids[0])
	}
}

func TestRunOverrideRuns(t *testing.T) {
	var n int
	opts := Options{
		Runs:   2, // override config's 3
		Stdout: io.Discard, Stderr: io.Discard,
		Exec: func(_ context.Context, _, _ []string, _, _ io.Writer) error {
			n++
			return nil
		},
	}
	if _, err := Run(context.Background(), twoScenarios(), opts); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if n != 4 { // 2 scenarios × 2 runs
		t.Errorf("invocations = %d, want 4", n)
	}
}

func TestRunAbortsOnError(t *testing.T) {
	var n int
	opts := Options{
		Stdout: io.Discard, Stderr: io.Discard,
		Exec: func(_ context.Context, _, _ []string, _, _ io.Writer) error {
			n++
			if n == 2 {
				return errors.New("agent crashed")
			}
			return nil
		},
	}
	sum, err := Run(context.Background(), twoScenarios(), opts)
	if err == nil {
		t.Fatal("expected error to abort the run")
	}
	if n != 2 {
		t.Errorf("ran %d invocations, want abort at 2", n)
	}
	if sum.Failed != 1 {
		t.Errorf("Failed = %d, want 1", sum.Failed)
	}
}

func TestRunContinueOnError(t *testing.T) {
	var n int
	opts := Options{
		ContinueOnError: true,
		Stdout:          io.Discard, Stderr: io.Discard,
		Exec: func(_ context.Context, _, _ []string, _, _ io.Writer) error {
			n++
			if n%2 == 0 {
				return errors.New("flaky")
			}
			return nil
		},
	}
	sum, err := Run(context.Background(), twoScenarios(), opts)
	if err != nil {
		t.Fatalf("ContinueOnError should not return error: %v", err)
	}
	if sum.Total != 6 {
		t.Errorf("Total = %d, want 6 (kept going)", sum.Total)
	}
	if sum.Failed != 3 {
		t.Errorf("Failed = %d, want 3", sum.Failed)
	}
}

func TestRunCancelledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled
	opts := Options{
		Stdout: io.Discard, Stderr: io.Discard,
		Exec: func(_ context.Context, _, _ []string, _, _ io.Writer) error {
			t.Fatal("Exec should not run with a cancelled context")
			return nil
		},
	}
	if _, err := Run(ctx, twoScenarios(), opts); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}
