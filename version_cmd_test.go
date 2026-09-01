package main

import (
	"slices"
	"strings"
	"testing"
)

// TestVersionCommandRegistered guards the dispatch wiring: `augur version` must
// resolve to runVersion and appear in the usage order.
func TestVersionCommandRegistered(t *testing.T) {
	if _, ok := commands["version"]; !ok {
		t.Fatal("version command not registered in the dispatch table")
	}
	if !slices.Contains(order, "version") {
		t.Error("version missing from the usage order slice")
	}
}

// TestVersionDefault checks a from-source build reports itself as "dev" (the
// release build overrides this via -ldflags).
func TestVersionDefault(t *testing.T) {
	if version != "dev" {
		t.Errorf("default version = %q, want dev", version)
	}
}

// TestRunVersion checks the command runs cleanly with any args.
func TestRunVersion(t *testing.T) {
	if err := runVersion(nil); err != nil {
		t.Fatalf("runVersion: %v", err)
	}
	if err := runVersion([]string{"ignored"}); err != nil {
		t.Fatalf("runVersion with args: %v", err)
	}
}

// TestVersionSummaryNoTabs keeps the usage table aligned: the summary must be a
// single line.
func TestVersionSummaryNoTabs(t *testing.T) {
	if s := commands["version"].summary; strings.ContainsAny(s, "\n\t") {
		t.Errorf("version summary has newline/tab: %q", s)
	}
}
