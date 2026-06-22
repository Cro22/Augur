package project

import (
	"math"
	"testing"
)

func TestParseTraffic(t *testing.T) {
	data := []byte(`
version: 1
users: 10000
requests_per_user_per_day: 5
tenants: 50
days_per_month: 30
scenario_mix:
  a: 1
  b: 3
`)
	tr, err := ParseTraffic(data)
	if err != nil {
		t.Fatalf("ParseTraffic: %v", err)
	}
	if tr.Users != 10000 || tr.RequestsPerUserPerDay != 5 || tr.Tenants != 50 || tr.DaysPerMonth != 30 {
		t.Errorf("traffic = %+v", tr)
	}
	// Weights 1:3 normalize to 0.25 / 0.75.
	if math.Abs(tr.ScenarioMix["a"]-0.25) > 1e-9 || math.Abs(tr.ScenarioMix["b"]-0.75) > 1e-9 {
		t.Errorf("normalized mix = %v, want a:0.25 b:0.75", tr.ScenarioMix)
	}
}

func TestParseTrafficDefaultsDays(t *testing.T) {
	tr, err := ParseTraffic([]byte(`users: 1`))
	if err != nil {
		t.Fatalf("ParseTraffic: %v", err)
	}
	if tr.DaysPerMonth != defaultDaysPerMonth {
		t.Errorf("DaysPerMonth = %d, want %d", tr.DaysPerMonth, defaultDaysPerMonth)
	}
	if tr.ScenarioMix != nil {
		t.Errorf("omitted mix should be nil, got %v", tr.ScenarioMix)
	}
}

func TestParseTrafficRejects(t *testing.T) {
	cases := map[string]string{
		"negative users":    "users: -1",
		"negative requests": "requests_per_user_per_day: -2",
		"negative tenants":  "tenants: -3",
		"negative days":     "days_per_month: -4",
		"negative weight":   "scenario_mix:\n  a: -1",
		"zero-sum weights":  "scenario_mix:\n  a: 0\n  b: 0",
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseTraffic([]byte(data)); err == nil {
				t.Errorf("%s: expected error, got nil", name)
			}
		})
	}
}
