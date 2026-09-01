package cost

import (
	"errors"
	"testing"
)

func fallbackPricing() Pricing {
	return Pricing{
		SnapshotDate: "2026-08-31",
		Models: map[string]ModelPrice{
			"gpt-4o":      {Input: 2.5, Output: 10},
			"gpt-4o-mini": {Input: 0.15, Output: 0.6},
		},
	}
}

func TestResolveExactMatch(t *testing.T) {
	p := fallbackPricing()
	canonical, mp, ok := p.Resolve("gpt-4o")
	if !ok || canonical != "gpt-4o" || mp.Input != 2.5 {
		t.Fatalf("Resolve(gpt-4o) = %q,%+v,%v; want gpt-4o with Input 2.5", canonical, mp, ok)
	}
}

func TestResolveFallbackOffByDefault(t *testing.T) {
	p := fallbackPricing()
	if _, _, ok := p.Resolve("gpt-4o-2024-08-06"); ok {
		t.Error("Resolve should not fall back when PrefixFallback is off")
	}
	if _, err := p.Cost("gpt-4o-2024-08-06", Usage{InputTokens: 1000}); !errors.Is(err, ErrUnknownModel) {
		t.Errorf("Cost error = %v, want ErrUnknownModel with fallback off", err)
	}
}

func TestResolvePrefixFallback(t *testing.T) {
	p := fallbackPricing()
	p.PrefixFallback = true

	// Dated variant falls back to its base entry.
	canonical, mp, ok := p.Resolve("gpt-4o-2024-08-06")
	if !ok || canonical != "gpt-4o" || mp.Input != 2.5 {
		t.Errorf("Resolve(gpt-4o-2024-08-06) = %q,%+v,%v; want gpt-4o", canonical, mp, ok)
	}

	// Longest prefix wins: the mini variant must not collapse to gpt-4o.
	canonical, mp, ok = p.Resolve("gpt-4o-mini-2024-07-18")
	if !ok || canonical != "gpt-4o-mini" || mp.Input != 0.15 {
		t.Errorf("Resolve(gpt-4o-mini-2024-07-18) = %q,%+v,%v; want gpt-4o-mini", canonical, mp, ok)
	}

	// Cost now succeeds via the fallback price.
	got, err := p.Cost("gpt-4o-2024-08-06", Usage{InputTokens: 1_000_000})
	if err != nil {
		t.Fatalf("Cost with fallback: %v", err)
	}
	if got != 2.5 {
		t.Errorf("Cost = %v, want 2.5 (gpt-4o input price)", got)
	}
}

func TestResolveDashBoundary(t *testing.T) {
	p := fallbackPricing()
	p.PrefixFallback = true
	// "gpt-4omini" shares a leading substring with "gpt-4o" but not at a dash
	// boundary — it must NOT match, or a base name would swallow unrelated ones.
	if canonical, _, ok := p.Resolve("gpt-4omini"); ok {
		t.Errorf("Resolve(gpt-4omini) matched %q; want no match (dash boundary)", canonical)
	}
}

func TestResolveNoMatchStillFails(t *testing.T) {
	p := fallbackPricing()
	p.PrefixFallback = true
	if _, _, ok := p.Resolve("claude-3-5-sonnet"); ok {
		t.Error("Resolve matched an unrelated model family; want no match")
	}
}
