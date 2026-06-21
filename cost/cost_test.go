package cost

import (
	"errors"
	"math"
	"testing"
)

// approxEqual compares dollar amounts with a tolerance that swallows float
// rounding but is far tighter than any meaningful fraction of a cent.
func approxEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

// gpt4o mirrors the pricing.yaml snapshot: $2.50 / $10.00 / $1.25 per Mtok.
var gpt4o = ModelPrice{Input: 2.50, Output: 10.00, CachedInput: 1.25}

func TestModelPriceCost(t *testing.T) {
	tests := []struct {
		name  string
		price ModelPrice
		usage Usage
		want  float64
	}{
		{
			// 1M input @ $2.50 + 1M output @ $10.00, no cache.
			name:  "round million tokens",
			price: gpt4o,
			usage: Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000},
			want:  12.50,
		},
		{
			// Hand calc: 500k full input @2.50 = 1.25; 500k cached @1.25 = 0.625;
			// 200k output @10.00 = 2.00. Total = 3.875.
			name:  "cached portion billed at cached rate",
			price: gpt4o,
			usage: Usage{InputTokens: 1_000_000, OutputTokens: 200_000, CachedTokens: 500_000},
			want:  3.875,
		},
		{
			// Everything cached: 1M @ $1.25.
			name:  "fully cached prompt",
			price: gpt4o,
			usage: Usage{InputTokens: 1_000_000, CachedTokens: 1_000_000},
			want:  1.25,
		},
		{
			name:  "zero tokens is free",
			price: gpt4o,
			usage: Usage{},
			want:  0,
		},
		{
			// A realistic small call: 1,234 in / 567 out, no cache.
			// 1234/1e6*2.50 + 567/1e6*10.00 = 0.003085 + 0.00567 = 0.008755.
			name:  "sub-cent realistic call",
			price: gpt4o,
			usage: Usage{InputTokens: 1234, OutputTokens: 567},
			want:  0.008755,
		},
		{
			// CachedInput defaulted to Input means cached tokens cost the same.
			name:  "no cache discount",
			price: ModelPrice{Input: 3.0, Output: 15.0, CachedInput: 3.0},
			usage: Usage{InputTokens: 1_000_000, CachedTokens: 400_000},
			want:  3.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.price.Cost(tt.usage)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !approxEqual(got, tt.want) {
				t.Errorf("Cost(%+v) = %.10f, want %.10f", tt.usage, got, tt.want)
			}
		})
	}
}

func TestCostInvalidUsage(t *testing.T) {
	tests := []struct {
		name  string
		usage Usage
	}{
		{"negative input", Usage{InputTokens: -1}},
		{"negative output", Usage{OutputTokens: -5}},
		{"negative cached", Usage{InputTokens: 10, CachedTokens: -1}},
		{"cached exceeds input", Usage{InputTokens: 100, CachedTokens: 101}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := gpt4o.Cost(tt.usage); err == nil {
				t.Errorf("Cost(%+v) = nil error, want validation error", tt.usage)
			}
		})
	}
}

func TestPricingCost(t *testing.T) {
	p := Pricing{
		SnapshotDate: "2026-06-21",
		Models:       map[string]ModelPrice{"gpt-4o": gpt4o},
	}

	got, err := p.Cost("gpt-4o", Usage{InputTokens: 1_000_000, OutputTokens: 1_000_000})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !approxEqual(got, 12.50) {
		t.Errorf("Cost = %.6f, want 12.50", got)
	}
}

func TestPricingUnknownModel(t *testing.T) {
	p := Pricing{Models: map[string]ModelPrice{"gpt-4o": gpt4o}}

	_, err := p.Cost("does-not-exist", Usage{InputTokens: 10})
	if err == nil {
		t.Fatal("expected error for unknown model, got nil")
	}
	if !errors.Is(err, ErrUnknownModel) {
		t.Errorf("error = %v, want errors.Is ErrUnknownModel", err)
	}
}

func TestPricePresence(t *testing.T) {
	p := Pricing{Models: map[string]ModelPrice{"gpt-4o": gpt4o}}

	if _, ok := p.Price("gpt-4o"); !ok {
		t.Error("Price(gpt-4o) ok = false, want true")
	}
	if _, ok := p.Price("nope"); ok {
		t.Error("Price(nope) ok = true, want false")
	}
}
