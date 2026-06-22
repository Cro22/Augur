package tco

import (
	"math"
	"testing"
)

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-6 }

func TestEffectivePerMtok(t *testing.T) {
	// $36/hr, 2500 tok/s, 100% util:
	//   tokens/hr = 2500 * 3600 = 9,000,000 = 9 Mtok/hr
	//   $/Mtok = 36 / 9 = 4.00
	d := Deployment{InstanceCostPerHour: 36, TokensPerSec: 2500, Utilization: 1}
	if !approx(d.EffectivePerMtok(), 4.0) {
		t.Errorf("effective = %v, want 4.00", d.EffectivePerMtok())
	}

	// At 50% utilization the effective price doubles: $8.00/Mtok.
	d.Utilization = 0.5
	if !approx(d.EffectivePerMtok(), 8.0) {
		t.Errorf("effective at 50%% util = %v, want 8.00", d.EffectivePerMtok())
	}
}

func TestParseTCO(t *testing.T) {
	data := []byte(`
version: 1
deployments:
  llama-70b:
    instance_cost_per_hour: 36
    tokens_per_sec: 2500
    utilization: 0.5
  mixtral:
    instance_cost_per_hour: 12
    tokens_per_sec: 3000
`)
	tc, err := ParseTCO(data)
	if err != nil {
		t.Fatalf("ParseTCO: %v", err)
	}
	if len(tc.Deployments) != 2 {
		t.Fatalf("got %d deployments, want 2", len(tc.Deployments))
	}
	// Omitted utilization defaults to 1.0.
	if tc.Deployments["mixtral"].Utilization != 1.0 {
		t.Errorf("mixtral util = %v, want 1.0 default", tc.Deployments["mixtral"].Utilization)
	}
	if !approx(tc.Deployments["llama-70b"].EffectivePerMtok(), 8.0) {
		t.Errorf("llama-70b effective = %v, want 8.00", tc.Deployments["llama-70b"].EffectivePerMtok())
	}
}

func TestParseTCORejects(t *testing.T) {
	cases := map[string]string{
		"no deployments": `version: 1`,
		"zero throughput": `
deployments:
  m:
    instance_cost_per_hour: 10
    tokens_per_sec: 0
`,
		"negative cost": `
deployments:
  m:
    instance_cost_per_hour: -1
    tokens_per_sec: 100
`,
		"util over 1": `
deployments:
  m:
    instance_cost_per_hour: 10
    tokens_per_sec: 100
    utilization: 1.5
`,
		"util zero": `
deployments:
  m:
    instance_cost_per_hour: 10
    tokens_per_sec: 100
    utilization: 0
`,
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseTCO([]byte(data)); err == nil {
				t.Errorf("%s: expected error, got nil", name)
			}
		})
	}
}

func TestPricingFromTCO(t *testing.T) {
	tc := TCO{Deployments: map[string]Deployment{
		"llama-70b": {InstanceCostPerHour: 36, TokensPerSec: 2500, Utilization: 1},
	}}
	p := tc.Pricing("2026-06-21")
	if p.SnapshotDate != "2026-06-21" {
		t.Errorf("SnapshotDate = %q", p.SnapshotDate)
	}
	mp, ok := p.Price("llama-70b")
	if !ok {
		t.Fatal("llama-70b missing from derived pricing")
	}
	// Self-hosted prices input/output/cached identically at the effective rate.
	if !approx(mp.Input, 4.0) || !approx(mp.Output, 4.0) || !approx(mp.CachedInput, 4.0) {
		t.Errorf("derived price = %+v, want all 4.00", mp)
	}
}
