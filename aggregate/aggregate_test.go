package aggregate

import (
	"errors"
	"math"
	"testing"

	"augur/cost"
	"augur/trace"
)

// testPricing mirrors the gpt-4o / gpt-4o-mini lines from pricing.yaml.
func testPricing() cost.Pricing {
	return cost.Pricing{
		SnapshotDate: "2026-06-21",
		Models: map[string]cost.ModelPrice{
			"gpt-4o":      {Input: 2.50, Output: 10.00, CachedInput: 1.25},
			"gpt-4o-mini": {Input: 0.15, Output: 0.60, CachedInput: 0.075},
		},
	}
}

func rec(scenario, run string, seq int, model string, in, out, cached int) trace.Record {
	return trace.Record{
		ScenarioID: scenario, RunID: run, Seq: seq, Model: model,
		InputTokens: in, OutputTokens: out, CachedTokens: cached,
	}
}

// TestAggregateKnownSet is the Hito 2 checkpoint: a known scenario set whose
// numbers are reconciled against the trace BY HAND in the comments below.
func TestAggregateKnownSet(t *testing.T) {
	// Scenario "checkout", gpt-4o (in 2.50, out 10.00 per Mtok):
	//   run-1, call0: in 1000, out 500  -> 0.0025 + 0.0050 = 0.0075
	//   run-1, call1: in 2000, out 1000 -> 0.0050 + 0.0100 = 0.0150  => run-1 = 0.0225, calls 2
    //   run-2, call0: in 1000, out 500  -> 0.0075                     => run-2 = 0.0075, calls 1
	records := []trace.Record{
		rec("checkout", "run-1", 0, "gpt-4o", 1000, 500, 0),
		rec("checkout", "run-1", 1, "gpt-4o", 2000, 1000, 0),
		rec("checkout", "run-2", 0, "gpt-4o", 1000, 500, 0),
	}

	res, err := Aggregate(records, testPricing())
	if err != nil {
		t.Fatalf("Aggregate: %v", err)
	}
	if res.SnapshotDate != "2026-06-21" {
		t.Errorf("SnapshotDate = %q", res.SnapshotDate)
	}
	if len(res.Scenarios) != 1 {
		t.Fatalf("got %d scenarios, want 1", len(res.Scenarios))
	}
	s := res.Scenarios[0]

	if s.Runs != 2 {
		t.Errorf("Runs = %d, want 2", s.Runs)
	}
	// CostPerRun over {0.0225, 0.0075}:
	//   mean 0.0150; p50 0.0150; p95 0.0075+0.95*0.0150 = 0.02175
	//   stdev = sqrt(2 * 0.0075^2 / 1) = 0.0075*sqrt(2)
	wantCost := Distribution{
		N: 2, Mean: 0.015, P50: 0.015, P95: 0.02175,
		Stdev: 0.0075 * math.Sqrt2, Min: 0.0075, Max: 0.0225,
	}
	assertDist(t, "CostPerRun", s.CostPerRun, wantCost)

	// CallsPerRun over {2, 1}: mean 1.5, p50 1.5, p95 1.95, stdev sqrt(0.5).
	wantCalls := Distribution{
		N: 2, Mean: 1.5, P50: 1.5, P95: 1.95,
		Stdev: math.Sqrt(0.5), Min: 1, Max: 2,
	}
	assertDist(t, "CallsPerRun", s.CallsPerRun, wantCalls)

	if !approx(s.TotalCost, 0.03) {
		t.Errorf("TotalCost = %v, want 0.03", s.TotalCost)
	}

	// ByModel: single model gpt-4o, calls 3, in 4000, out 2000, cost 0.03.
	if len(s.ByModel) != 1 {
		t.Fatalf("ByModel len = %d, want 1", len(s.ByModel))
	}
	m := s.ByModel[0]
	if m.Model != "gpt-4o" || m.Calls != 3 || m.InputTokens != 4000 || m.OutputTokens != 2000 {
		t.Errorf("ByModel[0] = %+v", m)
	}
	if !approx(m.CostUSD, 0.03) {
		t.Errorf("ByModel cost = %v, want 0.03", m.CostUSD)
	}

	// Per-run rows present for hand reconciliation.
	if len(res.Runs) != 2 {
		t.Fatalf("Runs len = %d, want 2", len(res.Runs))
	}
	if !approx(res.Runs[0].CostUSD, 0.0225) || res.Runs[0].Calls != 2 {
		t.Errorf("run-1 = %+v, want cost 0.0225 calls 2", res.Runs[0])
	}
	if !approx(res.Runs[1].CostUSD, 0.0075) || res.Runs[1].Calls != 1 {
		t.Errorf("run-2 = %+v, want cost 0.0075 calls 1", res.Runs[1])
	}
}

// ByModel must be sorted by cost descending and decompose a mixed-model run.
func TestAggregateByModelSorted(t *testing.T) {
	// One run, two models. gpt-4o call dominates cost; gpt-4o-mini is cheaper.
	//   gpt-4o:      in 1000, out 1000 -> 0.0025 + 0.0100 = 0.0125
	//   gpt-4o-mini: in 1000, out 1000 -> 0.00015 + 0.0006 = 0.00075
	records := []trace.Record{
		rec("s", "r1", 0, "gpt-4o-mini", 1000, 1000, 0),
		rec("s", "r1", 1, "gpt-4o", 1000, 1000, 0),
	}
	res, err := Aggregate(records, testPricing())
	if err != nil {
		t.Fatalf("Aggregate: %v", err)
	}
	bm := res.Scenarios[0].ByModel
	if len(bm) != 2 {
		t.Fatalf("ByModel len = %d, want 2", len(bm))
	}
	if bm[0].Model != "gpt-4o" {
		t.Errorf("ByModel[0] = %q, want gpt-4o (highest cost first)", bm[0].Model)
	}
	if bm[1].Model != "gpt-4o-mini" {
		t.Errorf("ByModel[1] = %q, want gpt-4o-mini", bm[1].Model)
	}
}

func TestAggregateUnknownModelIsHardError(t *testing.T) {
	records := []trace.Record{rec("s", "r", 0, "mystery-model", 10, 10, 0)}
	_, err := Aggregate(records, testPricing())
	if err == nil {
		t.Fatal("expected error for un-priceable model, got nil")
	}
	if !errors.Is(err, cost.ErrUnknownModel) {
		t.Errorf("error = %v, want errors.Is cost.ErrUnknownModel", err)
	}
}

func TestAggregateWithKnobs(t *testing.T) {
	// One run, one call: gpt-4o, 1000 in (200 cached), 500 out.
	//   prompt = 800@2.50 + 200@1.25 = 0.0020 + 0.00025 = 0.00225
	//   output = 500@10.00 = 0.0050
	records := []trace.Record{rec("s", "r", 0, "gpt-4o", 1000, 500, 200)}

	// Identity knobs reproduce the observed cost (0.00725).
	base, err := AggregateWithKnobs(records, testPricing(), Knobs{})
	if err != nil {
		t.Fatalf("Aggregate identity: %v", err)
	}
	if !approx(base.Runs[0].CostUSD, 0.00725) {
		t.Errorf("identity run cost = %v, want 0.00725", base.Runs[0].CostUSD)
	}

	// Knobs: retry +50%, fanout ×2, context ×3.
	//   callMult = 1.5 * 2 = 3
	//   cost' = 3 * (3*0.00225 + 0.0050) = 3 * 0.01175 = 0.03525
	knobs := Knobs{RetryRate: 0.5, FanoutFactor: 2, ContextGrowth: 3}
	res, err := AggregateWithKnobs(records, testPricing(), knobs)
	if err != nil {
		t.Fatalf("Aggregate with knobs: %v", err)
	}
	if !approx(res.Runs[0].CostUSD, 0.03525) {
		t.Errorf("knob run cost = %v, want 0.03525", res.Runs[0].CostUSD)
	}
	// The scenario distribution and per-model totals scale identically.
	if !approx(res.Scenarios[0].TotalCost, 0.03525) {
		t.Errorf("scenario total = %v, want 0.03525", res.Scenarios[0].TotalCost)
	}
	if !approx(res.Scenarios[0].ByModel[0].CostUSD, 0.03525) {
		t.Errorf("by-model cost = %v, want 0.03525", res.Scenarios[0].ByModel[0].CostUSD)
	}
}

func TestKnobsIdentity(t *testing.T) {
	cases := []struct {
		k    Knobs
		want bool
	}{
		{Knobs{}, true},
		{Knobs{FanoutFactor: 1, ContextGrowth: 1}, true},
		{Knobs{RetryRate: 0.1}, false},
		{Knobs{FanoutFactor: 2}, false},
		{Knobs{ContextGrowth: 1.5}, false},
	}
	for _, c := range cases {
		if got := c.k.IsIdentity(); got != c.want {
			t.Errorf("%+v IsIdentity = %v, want %v", c.k, got, c.want)
		}
	}
}

// Context growth must inflate ONLY the prompt side, not the completion.
func TestKnobsContextGrowthPromptOnly(t *testing.T) {
	// Pure-output call (0 input): context growth must leave it unchanged.
	outOnly := []trace.Record{rec("s", "r", 0, "gpt-4o", 0, 1000, 0)} // 1000@10 = 0.01
	res, err := AggregateWithKnobs(outOnly, testPricing(), Knobs{ContextGrowth: 5})
	if err != nil {
		t.Fatalf("Aggregate: %v", err)
	}
	if !approx(res.Runs[0].CostUSD, 0.01) {
		t.Errorf("output-only cost under context growth = %v, want 0.01 (unchanged)", res.Runs[0].CostUSD)
	}
}

func TestAggregateEmpty(t *testing.T) {
	res, err := Aggregate(nil, testPricing())
	if err != nil {
		t.Fatalf("Aggregate(nil): %v", err)
	}
	if len(res.Scenarios) != 0 || len(res.Runs) != 0 {
		t.Errorf("empty trace produced %d scenarios / %d runs, want 0/0",
			len(res.Scenarios), len(res.Runs))
	}
}

func TestAggregateScenariosSorted(t *testing.T) {
	records := []trace.Record{
		rec("zebra", "r", 0, "gpt-4o", 10, 10, 0),
		rec("alpha", "r", 0, "gpt-4o", 10, 10, 0),
		rec("mango", "r", 0, "gpt-4o", 10, 10, 0),
	}
	res, err := Aggregate(records, testPricing())
	if err != nil {
		t.Fatalf("Aggregate: %v", err)
	}
	want := []string{"alpha", "mango", "zebra"}
	for i, s := range res.Scenarios {
		if s.ScenarioID != want[i] {
			t.Errorf("scenario %d = %q, want %q", i, s.ScenarioID, want[i])
		}
	}
}

func assertDist(t *testing.T, name string, got, want Distribution) {
	t.Helper()
	if got.N != want.N {
		t.Errorf("%s.N = %d, want %d", name, got.N, want.N)
	}
	for _, f := range []struct {
		field      string
		got, want  float64
	}{
		{"Mean", got.Mean, want.Mean},
		{"P50", got.P50, want.P50},
		{"P95", got.P95, want.P95},
		{"Stdev", got.Stdev, want.Stdev},
		{"Min", got.Min, want.Min},
		{"Max", got.Max, want.Max},
	} {
		if !approx(f.got, f.want) {
			t.Errorf("%s.%s = %v, want %v", name, f.field, f.got, f.want)
		}
	}
}
