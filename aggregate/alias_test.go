package aggregate

import (
	"errors"
	"testing"

	"augur/cost"
	"augur/trace"
)

// TestAggregateUnknownModelStillErrors confirms the default (no fallback)
// behavior is unchanged: an un-priced model is a hard error.
func TestAggregateUnknownModelStillErrors(t *testing.T) {
	records := []trace.Record{rec("s", "r", 0, "gpt-4o-2024-08-06", 1000, 500, 0)}
	_, err := Aggregate(records, testPricing())
	if !errors.Is(err, cost.ErrUnknownModel) {
		t.Fatalf("err = %v, want ErrUnknownModel", err)
	}
}

// TestAggregateModelAliasesSurfaced checks that with PrefixFallback on, a dated
// model is priced via its base entry AND reported in Result.ModelAliases so the
// caller can warn.
func TestAggregateModelAliasesSurfaced(t *testing.T) {
	pricing := testPricing()
	pricing.PrefixFallback = true

	records := []trace.Record{
		rec("checkout", "run-1", 0, "gpt-4o-2024-08-06", 1_000_000, 0, 0), // -> gpt-4o
		rec("checkout", "run-1", 1, "gpt-4o", 0, 0, 0),                    // exact, no alias
	}
	res, err := AggregateWithKnobs(records, pricing, Knobs{})
	if err != nil {
		t.Fatalf("AggregateWithKnobs: %v", err)
	}

	if got := res.ModelAliases["gpt-4o-2024-08-06"]; got != "gpt-4o" {
		t.Errorf("ModelAliases[gpt-4o-2024-08-06] = %q, want gpt-4o", got)
	}
	if _, aliased := res.ModelAliases["gpt-4o"]; aliased {
		t.Errorf("exact match gpt-4o must not appear in ModelAliases: %v", res.ModelAliases)
	}

	// The dated call was priced at gpt-4o's input rate (2.50 / Mtok on 1M tokens).
	var total float64
	for _, r := range res.Runs {
		total += r.CostUSD
	}
	if total != 2.50 {
		t.Errorf("total cost = %v, want 2.50 (billed via gpt-4o fallback)", total)
	}
}

// TestAggregateNoAliasesWhenAllExact checks ModelAliases stays nil when nothing
// was normalized, so the warning path is a genuine no-op on the common case.
func TestAggregateNoAliasesWhenAllExact(t *testing.T) {
	pricing := testPricing()
	pricing.PrefixFallback = true
	records := []trace.Record{rec("s", "r", 0, "gpt-4o", 1000, 500, 0)}
	res, err := Aggregate(records, pricing)
	if err != nil {
		t.Fatalf("Aggregate: %v", err)
	}
	if res.ModelAliases != nil {
		t.Errorf("ModelAliases = %v, want nil when no fallback happened", res.ModelAliases)
	}
}
