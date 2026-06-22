// Package gate is the budget check: it compares a projection against the
// thresholds in budget.yaml and decides pass/fail. Per SPEC decision D2 the
// per-request comparison is on the p95, not the mean — the cost surprise lives
// in the tail, and gating on the average hides exactly the failure mode Augur
// exists to catch.
//
// A failing gate is what fails the CI build (a non-zero exit code), so the
// evaluation is deliberately simple and explicit: every checked threshold
// becomes a Check with its limit, the projected actual, and a verdict.
package gate

import (
	"fmt"
	"os"

	"augur/project"

	"gopkg.in/yaml.v3"
)

// Budget is a parsed budget.yaml. Each threshold is optional (a nil pointer
// means "don't check this dimension"), so a project can gate on just the
// per-request tail, or add tenant/monthly ceilings as it matures.
type Budget struct {
	// MaxRequestP95USD caps the p95 cost of a single request.
	MaxRequestP95USD *float64
	// MaxTenantPerMonthUSD caps the projected cost per tenant per month.
	MaxTenantPerMonthUSD *float64
	// MaxMonthlyBillUSD caps the projected total monthly bill.
	MaxMonthlyBillUSD *float64
}

type yamlBudget struct {
	Version              int      `yaml:"version"`
	MaxRequestP95USD     *float64 `yaml:"max_request_p95_usd"`
	MaxTenantPerMonthUSD *float64 `yaml:"max_tenant_per_month_usd"`
	MaxMonthlyBillUSD    *float64 `yaml:"max_monthly_bill_usd"`
}

// LoadBudget reads and validates budget.yaml from path.
func LoadBudget(path string) (Budget, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Budget{}, fmt.Errorf("gate: reading budget file: %w", err)
	}
	return ParseBudget(raw)
}

// ParseBudget parses and validates a budget from YAML bytes. A budget with no
// thresholds at all is an error: it would gate nothing and silently pass every
// build, defeating the point.
func ParseBudget(data []byte) (Budget, error) {
	var yb yamlBudget
	if err := yaml.Unmarshal(data, &yb); err != nil {
		return Budget{}, fmt.Errorf("gate: parsing budget yaml: %w", err)
	}

	b := Budget{
		MaxRequestP95USD:     yb.MaxRequestP95USD,
		MaxTenantPerMonthUSD: yb.MaxTenantPerMonthUSD,
		MaxMonthlyBillUSD:    yb.MaxMonthlyBillUSD,
	}
	if b.MaxRequestP95USD == nil && b.MaxTenantPerMonthUSD == nil && b.MaxMonthlyBillUSD == nil {
		return Budget{}, fmt.Errorf("gate: budget has no thresholds (set at least one of max_request_p95_usd, max_tenant_per_month_usd, max_monthly_bill_usd)")
	}
	for name, v := range map[string]*float64{
		"max_request_p95_usd":      b.MaxRequestP95USD,
		"max_tenant_per_month_usd": b.MaxTenantPerMonthUSD,
		"max_monthly_bill_usd":     b.MaxMonthlyBillUSD,
	} {
		if v != nil && *v < 0 {
			return Budget{}, fmt.Errorf("gate: %s must be >= 0, got %g", name, *v)
		}
	}
	return b, nil
}

// Check is one threshold comparison.
type Check struct {
	Name   string  `json:"name"`
	Limit  float64 `json:"limit_usd"`
	Actual float64 `json:"actual_usd"`
	Pass   bool    `json:"pass"`
}

// Result is the gate verdict: the overall pass/fail plus every individual
// check, in a stable order.
type Result struct {
	Pass   bool    `json:"pass"`
	Checks []Check `json:"checks"`
}

// Evaluate compares a projection against the budget. A check passes when the
// projected actual is at or below the limit. The overall result passes only if
// every checked dimension passes. The per-request check uses the p95 (D2).
func Evaluate(proj project.Projection, budget Budget) Result {
	var checks []Check
	add := func(name string, limit *float64, actual float64) {
		if limit == nil {
			return
		}
		checks = append(checks, Check{
			Name:   name,
			Limit:  *limit,
			Actual: actual,
			Pass:   actual <= *limit,
		})
	}

	add("$/request p95", budget.MaxRequestP95USD, proj.RequestP95.Value)
	add("$/tenant/month", budget.MaxTenantPerMonthUSD, proj.PerTenantPerMonth.Value)
	add("monthly bill", budget.MaxMonthlyBillUSD, proj.MonthlyBill.Value)

	pass := true
	for _, c := range checks {
		if !c.Pass {
			pass = false
		}
	}
	return Result{Pass: pass, Checks: checks}
}
