package cost

import "testing"

func TestParsePricing(t *testing.T) {
	data := []byte(`
version: 1
snapshot_date: "2026-06-21"
currency: USD
unit: per_mtok
models:
  gpt-4o:
    input: 2.50
    output: 10.00
    cached_input: 1.25
  no-cache-model:
    input: 3.00
    output: 15.00
`)

	p, err := ParsePricing(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.SnapshotDate != "2026-06-21" {
		t.Errorf("SnapshotDate = %q, want 2026-06-21", p.SnapshotDate)
	}

	mp, ok := p.Price("gpt-4o")
	if !ok {
		t.Fatal("gpt-4o missing from parsed pricing")
	}
	if mp.Input != 2.50 || mp.Output != 10.00 || mp.CachedInput != 1.25 {
		t.Errorf("gpt-4o = %+v, want {2.50 10.00 1.25}", mp)
	}

	// Omitted cached_input must default to the input rate, not zero.
	nc, ok := p.Price("no-cache-model")
	if !ok {
		t.Fatal("no-cache-model missing from parsed pricing")
	}
	if nc.CachedInput != nc.Input {
		t.Errorf("no-cache-model CachedInput = %v, want %v (defaults to input)", nc.CachedInput, nc.Input)
	}
}

func TestParsePricingRejectsBadUnit(t *testing.T) {
	data := []byte(`
unit: per_ktok
models:
  gpt-4o:
    input: 2.50
    output: 10.00
`)
	if _, err := ParsePricing(data); err == nil {
		t.Error("expected error for non per_mtok unit, got nil")
	}
}

func TestParsePricingRejectsEmpty(t *testing.T) {
	if _, err := ParsePricing([]byte(`unit: per_mtok`)); err == nil {
		t.Error("expected error for snapshot with no models, got nil")
	}
}

func TestLoadPricingSnapshot(t *testing.T) {
	// The committed snapshot must parse and price a known call. Guards against a
	// future edit breaking the shipped pricing.yaml.
	p, err := LoadPricing("../pricing.yaml")
	if err != nil {
		t.Fatalf("loading committed pricing.yaml: %v", err)
	}
	if _, ok := p.Price("gpt-4o"); !ok {
		t.Error("committed pricing.yaml missing gpt-4o")
	}
	if _, err := p.Cost("gpt-4o", Usage{InputTokens: 1000, OutputTokens: 500}); err != nil {
		t.Errorf("pricing committed gpt-4o call: %v", err)
	}
}
