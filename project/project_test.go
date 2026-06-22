package project

import (
	"math"
	"testing"

	"augur/aggregate"
)

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

// singleScenarioResult: scenario "a", 4 runs costing 0.01/0.02/0.03/0.04, all on
// gpt-4o (per-scenario model total 0.10).
func singleScenarioResult() aggregate.Result {
	return aggregate.Result{
		SnapshotDate: "2026-06-21",
		Scenarios: []aggregate.Scenario{
			{ScenarioID: "a", Runs: 4, ByModel: []aggregate.ModelUsage{{Model: "gpt-4o", CostUSD: 0.10}}},
		},
		Runs: []aggregate.Run{
			{ScenarioID: "a", CostUSD: 0.01},
			{ScenarioID: "a", CostUSD: 0.02},
			{ScenarioID: "a", CostUSD: 0.03},
			{ScenarioID: "a", CostUSD: 0.04},
		},
	}
}

// TestProjectHandCalc is the Hito 3 checkpoint: a synthetic distribution +
// traffic profile with every projected number reconciled by hand.
func TestProjectHandCalc(t *testing.T) {
	res := singleScenarioResult()
	traffic := Traffic{
		Users:                 1000,
		RequestsPerUserPerDay: 2,
		Tenants:               10,
		DaysPerMonth:          30,
		ScenarioMix:           map[string]float64{"a": 1},
	}

	p, err := Project(res, traffic, Options{})
	if err != nil {
		t.Fatalf("Project: %v", err)
	}

	// Per-request mixture over {0.01,0.02,0.03,0.04}, each weight 0.25:
	//   mean = 0.025
	//   p50: cum weight hits 0.5 at 0.02  -> 0.02
	//   p95: cum weight reaches 0.95 at 0.04 -> 0.04
	if !approx(p.RequestMean.Value, 0.025) {
		t.Errorf("RequestMean = %v, want 0.025", p.RequestMean.Value)
	}
	if !approx(p.RequestP50, 0.02) {
		t.Errorf("RequestP50 = %v, want 0.02", p.RequestP50)
	}
	if !approx(p.RequestP95.Value, 0.04) {
		t.Errorf("RequestP95 = %v, want 0.04", p.RequestP95.Value)
	}

	// Aggregates scale from mean 0.025:
	//   $/user/day        = 0.025 * 2          = 0.05
	//   requests/month    = 1000 * 2 * 30       = 60000
	//   monthly bill      = 0.025 * 60000       = 1500
	//   $/tenant/month    = 1500 / 10           = 150
	if !approx(p.PerUserPerDay.Value, 0.05) {
		t.Errorf("PerUserPerDay = %v, want 0.05", p.PerUserPerDay.Value)
	}
	if !approx(p.MonthlyBill.Value, 1500) {
		t.Errorf("MonthlyBill = %v, want 1500", p.MonthlyBill.Value)
	}
	if !approx(p.PerTenantPerMonth.Value, 150) {
		t.Errorf("PerTenantPerMonth = %v, want 150", p.PerTenantPerMonth.Value)
	}

	// CI sanity: bounds bracket the point estimate, and the CI scales linearly
	// into the aggregates.
	if !(p.RequestMean.Lo <= p.RequestMean.Value && p.RequestMean.Value <= p.RequestMean.Hi) {
		t.Errorf("mean CI does not bracket value: %+v", p.RequestMean)
	}
	if !approx(p.MonthlyBill.Lo, p.RequestMean.Lo*60000) || !approx(p.MonthlyBill.Hi, p.RequestMean.Hi*60000) {
		t.Errorf("monthly bill CI did not scale from mean CI: %+v", p.MonthlyBill)
	}

	// ByModel: single model accounts for 100% of the request cost, equal to mean.
	if len(p.ByModel) != 1 {
		t.Fatalf("ByModel len = %d, want 1", len(p.ByModel))
	}
	if p.ByModel[0].Model != "gpt-4o" || !approx(p.ByModel[0].PerRequestUSD, 0.025) || !approx(p.ByModel[0].Pct, 100) {
		t.Errorf("ByModel[0] = %+v, want gpt-4o 0.025 100%%", p.ByModel[0])
	}
}

// TestProjectWeightedMixture hand-checks the weighting across two scenarios with
// different sample sizes.
func TestProjectWeightedMixture(t *testing.T) {
	// a: {0.01, 0.03}, weight 0.25 → each obs weight 0.125
	// b: {0.05},       weight 0.75 → obs weight 0.75
	res := aggregate.Result{
		SnapshotDate: "2026-06-21",
		Scenarios: []aggregate.Scenario{
			{ScenarioID: "a", Runs: 2},
			{ScenarioID: "b", Runs: 1},
		},
		Runs: []aggregate.Run{
			{ScenarioID: "a", CostUSD: 0.01},
			{ScenarioID: "a", CostUSD: 0.03},
			{ScenarioID: "b", CostUSD: 0.05},
		},
	}
	traffic := Traffic{
		Users: 1, RequestsPerUserPerDay: 1, Tenants: 1, DaysPerMonth: 30,
		ScenarioMix: map[string]float64{"a": 0.25, "b": 0.75},
	}
	p, err := Project(res, traffic, Options{})
	if err != nil {
		t.Fatalf("Project: %v", err)
	}
	// mean = 0.01*0.125 + 0.03*0.125 + 0.05*0.75 = 0.0425
	if !approx(p.RequestMean.Value, 0.0425) {
		t.Errorf("mean = %v, want 0.0425", p.RequestMean.Value)
	}
	// sorted cum weights: 0.01->0.125, 0.03->0.25, 0.05->1.0
	//   p50 target 0.5 -> 0.05 ; p95 target 0.95 -> 0.05
	if !approx(p.RequestP50, 0.05) {
		t.Errorf("p50 = %v, want 0.05", p.RequestP50)
	}
	if !approx(p.RequestP95.Value, 0.05) {
		t.Errorf("p95 = %v, want 0.05", p.RequestP95.Value)
	}
}

// A degenerate sample (all runs identical) must collapse the bootstrap CI onto
// the point estimate exactly — a clean deterministic CI check.
func TestProjectDegenerateCICollapses(t *testing.T) {
	res := aggregate.Result{
		SnapshotDate: "2026-06-21",
		Scenarios:    []aggregate.Scenario{{ScenarioID: "a", Runs: 3}},
		Runs: []aggregate.Run{
			{ScenarioID: "a", CostUSD: 0.02},
			{ScenarioID: "a", CostUSD: 0.02},
			{ScenarioID: "a", CostUSD: 0.02},
		},
	}
	traffic := Traffic{Users: 1, RequestsPerUserPerDay: 1, Tenants: 1, DaysPerMonth: 30}
	p, err := Project(res, traffic, Options{})
	if err != nil {
		t.Fatalf("Project: %v", err)
	}
	m := p.RequestMean
	if !approx(m.Value, 0.02) || !approx(m.Lo, 0.02) || !approx(m.Hi, 0.02) {
		t.Errorf("degenerate mean estimate = %+v, want all 0.02", m)
	}
	if !approx(p.RequestP95.Lo, 0.02) || !approx(p.RequestP95.Hi, 0.02) {
		t.Errorf("degenerate p95 CI = [%v, %v], want [0.02, 0.02]", p.RequestP95.Lo, p.RequestP95.Hi)
	}
}

// The bootstrap is seeded, so a projection is reproducible bit-for-bit.
func TestProjectReproducible(t *testing.T) {
	res := singleScenarioResult()
	traffic := Traffic{Users: 1, RequestsPerUserPerDay: 1, Tenants: 1, DaysPerMonth: 30,
		ScenarioMix: map[string]float64{"a": 1}}
	p1, _ := Project(res, traffic, Options{Seed: 42})
	p2, _ := Project(res, traffic, Options{Seed: 42})
	if p1.RequestMean != p2.RequestMean || p1.RequestP95 != p2.RequestP95 {
		t.Errorf("same seed gave different CIs:\n%+v\n%+v", p1, p2)
	}
}

func TestProjectDefaultEqualWeights(t *testing.T) {
	// Two scenarios, no mix → equal weight 0.5 each.
	// a mean 0.02, b mean 0.04 → request mean = 0.5*0.02 + 0.5*0.04 = 0.03
	res := aggregate.Result{
		SnapshotDate: "2026-06-21",
		Scenarios: []aggregate.Scenario{
			{ScenarioID: "a", Runs: 1},
			{ScenarioID: "b", Runs: 1},
		},
		Runs: []aggregate.Run{
			{ScenarioID: "a", CostUSD: 0.02},
			{ScenarioID: "b", CostUSD: 0.04},
		},
	}
	traffic := Traffic{Users: 1, RequestsPerUserPerDay: 1, Tenants: 1, DaysPerMonth: 30}
	p, err := Project(res, traffic, Options{})
	if err != nil {
		t.Fatalf("Project: %v", err)
	}
	if !approx(p.RequestMean.Value, 0.03) {
		t.Errorf("mean = %v, want 0.03 (equal weights)", p.RequestMean.Value)
	}
}

func TestProjectUnknownScenarioInMix(t *testing.T) {
	res := singleScenarioResult()
	traffic := Traffic{Users: 1, RequestsPerUserPerDay: 1, Tenants: 1, DaysPerMonth: 30,
		ScenarioMix: map[string]float64{"a": 0.5, "ghost": 0.5}}
	if _, err := Project(res, traffic, Options{}); err == nil {
		t.Fatal("expected error for mix referencing a scenario not in the trace")
	}
}

func TestProjectZeroTenantsSafe(t *testing.T) {
	res := singleScenarioResult()
	traffic := Traffic{Users: 1, RequestsPerUserPerDay: 1, Tenants: 0, DaysPerMonth: 30,
		ScenarioMix: map[string]float64{"a": 1}}
	p, err := Project(res, traffic, Options{})
	if err != nil {
		t.Fatalf("Project: %v", err)
	}
	if p.PerTenantPerMonth != (Estimate{}) {
		t.Errorf("zero tenants should leave PerTenantPerMonth zero, got %+v", p.PerTenantPerMonth)
	}
}
