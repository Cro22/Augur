package gate

import (
	"strings"
	"testing"

	"augur/project"
)

func ptr(f float64) *float64 { return &f }

// sampleProjection is a fixed projection used across gate tests:
//   $/request p95 = 0.04, $/tenant/month = 150, monthly bill = 1500.
func sampleProjection() project.Projection {
	return project.Projection{
		SnapshotDate:      "2026-06-21",
		CILevel:           0.95,
		BootstrapSamples:  2000,
		Seed:              1,
		RequestMean:       project.Estimate{Value: 0.025, Lo: 0.02, Hi: 0.03},
		RequestP50:        0.02,
		RequestP95:        project.Estimate{Value: 0.04, Lo: 0.035, Hi: 0.045},
		PerUserPerDay:     project.Estimate{Value: 0.05, Lo: 0.04, Hi: 0.06},
		PerTenantPerMonth: project.Estimate{Value: 150, Lo: 120, Hi: 180},
		MonthlyBill:       project.Estimate{Value: 1500, Lo: 1200, Hi: 1800},
		ByModel:           []project.ModelShare{{Model: "gpt-4o", PerRequestUSD: 0.025, Pct: 100}},
		Traffic: project.Traffic{
			Users: 1000, RequestsPerUserPerDay: 2, Tenants: 10, DaysPerMonth: 30,
		},
	}
}

func TestEvaluatePasses(t *testing.T) {
	// All limits comfortably above the projection.
	b := Budget{
		MaxRequestP95USD:     ptr(0.05),
		MaxTenantPerMonthUSD: ptr(500),
		MaxMonthlyBillUSD:    ptr(20000),
	}
	res := Evaluate(sampleProjection(), b)
	if !res.Pass {
		t.Fatalf("expected pass, got %+v", res)
	}
	if len(res.Checks) != 3 {
		t.Errorf("got %d checks, want 3", len(res.Checks))
	}
	for _, c := range res.Checks {
		if !c.Pass {
			t.Errorf("check %q should pass: limit %g actual %g", c.Name, c.Limit, c.Actual)
		}
	}
}

func TestEvaluateFailsOnRequestP95(t *testing.T) {
	// p95 actual 0.04 exceeds a 0.03 limit; the others pass.
	b := Budget{
		MaxRequestP95USD:     ptr(0.03),
		MaxTenantPerMonthUSD: ptr(500),
		MaxMonthlyBillUSD:    ptr(20000),
	}
	res := Evaluate(sampleProjection(), b)
	if res.Pass {
		t.Fatal("expected overall fail")
	}
	var p95 Check
	for _, c := range res.Checks {
		if c.Name == "$/request p95" {
			p95 = c
		}
	}
	if p95.Pass {
		t.Errorf("p95 check should fail: actual %g limit %g", p95.Actual, p95.Limit)
	}
	if p95.Actual != 0.04 {
		t.Errorf("p95 check uses p95 value: actual %g, want 0.04", p95.Actual)
	}
}

func TestEvaluateBoundaryIsInclusive(t *testing.T) {
	// actual exactly equal to the limit passes (<=).
	b := Budget{MaxRequestP95USD: ptr(0.04)}
	res := Evaluate(sampleProjection(), b)
	if !res.Pass {
		t.Errorf("actual == limit should pass, got %+v", res.Checks)
	}
}

func TestEvaluateOnlyChecksSetThresholds(t *testing.T) {
	b := Budget{MaxMonthlyBillUSD: ptr(1000)} // only one threshold, and it fails (1500 > 1000)
	res := Evaluate(sampleProjection(), b)
	if len(res.Checks) != 1 {
		t.Fatalf("got %d checks, want 1", len(res.Checks))
	}
	if res.Checks[0].Name != "monthly bill" || res.Pass {
		t.Errorf("expected single failing monthly-bill check, got %+v", res)
	}
}

func TestParseBudget(t *testing.T) {
	b, err := ParseBudget([]byte(`
version: 1
max_request_p95_usd: 0.05
max_monthly_bill_usd: 20000
`))
	if err != nil {
		t.Fatalf("ParseBudget: %v", err)
	}
	if b.MaxRequestP95USD == nil || *b.MaxRequestP95USD != 0.05 {
		t.Errorf("MaxRequestP95USD = %v, want 0.05", b.MaxRequestP95USD)
	}
	if b.MaxTenantPerMonthUSD != nil {
		t.Errorf("omitted tenant threshold should be nil, got %v", *b.MaxTenantPerMonthUSD)
	}
	if b.MaxMonthlyBillUSD == nil || *b.MaxMonthlyBillUSD != 20000 {
		t.Errorf("MaxMonthlyBillUSD = %v, want 20000", b.MaxMonthlyBillUSD)
	}
}

func TestParseBudgetRejectsEmpty(t *testing.T) {
	if _, err := ParseBudget([]byte(`version: 1`)); err == nil {
		t.Error("budget with no thresholds should be rejected")
	}
}

func TestParseBudgetRejectsNegative(t *testing.T) {
	if _, err := ParseBudget([]byte(`max_request_p95_usd: -0.01`)); err == nil {
		t.Error("negative threshold should be rejected")
	}
}

// Zero is a valid (very strict) threshold, distinct from omitted.
func TestParseBudgetZeroThreshold(t *testing.T) {
	b, err := ParseBudget([]byte(`max_request_p95_usd: 0`))
	if err != nil {
		t.Fatalf("ParseBudget: %v", err)
	}
	if b.MaxRequestP95USD == nil || *b.MaxRequestP95USD != 0 {
		t.Errorf("zero threshold should parse as 0, got %v", b.MaxRequestP95USD)
	}
	// Any positive p95 fails a zero budget.
	if Evaluate(sampleProjection(), b).Pass {
		t.Error("positive p95 should fail a zero budget")
	}
}

func TestReportMarkdownReadable(t *testing.T) {
	b := Budget{MaxRequestP95USD: ptr(0.03)} // fails
	res := Evaluate(sampleProjection(), b)
	rep := NewReport(sampleProjection(), res)

	var sb strings.Builder
	if err := rep.WriteMarkdown(&sb); err != nil {
		t.Fatalf("WriteMarkdown: %v", err)
	}
	md := sb.String()
	for _, want := range []string{
		"# Augur cost report",
		"FAIL",
		"Budget checks",
		"$/request p95",
		"Projected unit economics",
		"gpt-4o",
	} {
		if !strings.Contains(md, want) {
			t.Errorf("markdown report missing %q:\n%s", want, md)
		}
	}
}

func TestReportJSONRoundTrips(t *testing.T) {
	res := Evaluate(sampleProjection(), Budget{MaxMonthlyBillUSD: ptr(20000)})
	rep := NewReport(sampleProjection(), res)
	var sb strings.Builder
	if err := rep.WriteJSON(&sb); err != nil {
		t.Fatalf("WriteJSON: %v", err)
	}
	out := sb.String()
	if !strings.Contains(out, `"pass": true`) {
		t.Errorf("json should report pass=true:\n%s", out)
	}
	if !strings.Contains(out, `"projection"`) || !strings.Contains(out, `"gate"`) {
		t.Errorf("json should embed projection and gate:\n%s", out)
	}
}
