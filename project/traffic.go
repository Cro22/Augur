// Package project turns observed per-scenario cost distributions (Hito 2) into
// projected production unit economics (Hito 3): $/request (p50, p95), $/user/day,
// $/tenant/month, and the projected monthly bill, each with a bootstrap
// confidence interval, plus a per-model breakdown.
//
// A production "request" is modelled as a mixture over scenarios weighted by the
// traffic profile's scenario_mix. The per-request cost distribution is the
// weighted empirical mixture of the observed run costs — deterministic and
// reproducible by hand (the Hito 3 checkpoint). Confidence intervals come from a
// seeded bootstrap, so they are reproducible too. The day/month/tenant figures
// scale linearly from the mean per-request cost, so their CIs scale with it.
package project

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// defaultDaysPerMonth is used when traffic.yaml omits days_per_month.
const defaultDaysPerMonth = 30

// Traffic is a parsed traffic profile: the production assumptions a projection
// is computed against.
type Traffic struct {
	// Users is the total number of users in the projected population.
	Users int
	// RequestsPerUserPerDay is how many agent requests one user makes per day
	// (may be fractional).
	RequestsPerUserPerDay float64
	// Tenants is the number of tenants, for per-tenant economics.
	Tenants int
	// DaysPerMonth is the billing horizon (defaults to 30).
	DaysPerMonth int
	// ScenarioMix maps scenario id -> weight (the probability a request is that
	// scenario). Weights are normalized to sum to 1 on load. Empty means "equal
	// weight across every scenario present in the aggregate", resolved later.
	ScenarioMix map[string]float64
}

type yamlTraffic struct {
	Version               int                `yaml:"version"`
	Users                 int                `yaml:"users"`
	RequestsPerUserPerDay float64            `yaml:"requests_per_user_per_day"`
	Tenants               int                `yaml:"tenants"`
	DaysPerMonth          int                `yaml:"days_per_month"`
	ScenarioMix           map[string]float64 `yaml:"scenario_mix"`
}

// LoadTraffic reads and validates a traffic profile from path.
func LoadTraffic(path string) (Traffic, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Traffic{}, fmt.Errorf("project: reading traffic file: %w", err)
	}
	return ParseTraffic(raw)
}

// ParseTraffic parses and validates a traffic profile from YAML bytes.
func ParseTraffic(data []byte) (Traffic, error) {
	var yt yamlTraffic
	if err := yaml.Unmarshal(data, &yt); err != nil {
		return Traffic{}, fmt.Errorf("project: parsing traffic yaml: %w", err)
	}

	if yt.Users < 0 {
		return Traffic{}, fmt.Errorf("project: users must be >= 0, got %d", yt.Users)
	}
	if yt.RequestsPerUserPerDay < 0 {
		return Traffic{}, fmt.Errorf("project: requests_per_user_per_day must be >= 0, got %g", yt.RequestsPerUserPerDay)
	}
	if yt.Tenants < 0 {
		return Traffic{}, fmt.Errorf("project: tenants must be >= 0, got %d", yt.Tenants)
	}
	if yt.DaysPerMonth < 0 {
		return Traffic{}, fmt.Errorf("project: days_per_month must be >= 0, got %d", yt.DaysPerMonth)
	}

	days := yt.DaysPerMonth
	if days == 0 {
		days = defaultDaysPerMonth
	}

	mix, err := normalizeMix(yt.ScenarioMix)
	if err != nil {
		return Traffic{}, err
	}

	return Traffic{
		Users:                 yt.Users,
		RequestsPerUserPerDay: yt.RequestsPerUserPerDay,
		Tenants:               yt.Tenants,
		DaysPerMonth:          days,
		ScenarioMix:           mix,
	}, nil
}

// normalizeMix validates weights are non-negative and rescales them to sum to
// 1. A nil/empty mix passes through as nil (resolved to equal weights against
// the aggregate at projection time).
func normalizeMix(mix map[string]float64) (map[string]float64, error) {
	if len(mix) == 0 {
		return nil, nil
	}
	var sum float64
	for id, w := range mix {
		if w < 0 {
			return nil, fmt.Errorf("project: scenario_mix weight for %q is negative (%g)", id, w)
		}
		sum += w
	}
	if sum == 0 {
		return nil, fmt.Errorf("project: scenario_mix weights sum to zero")
	}
	out := make(map[string]float64, len(mix))
	for id, w := range mix {
		out[id] = w / sum
	}
	return out, nil
}
