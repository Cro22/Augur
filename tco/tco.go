// Package tco derives an effective $/Mtok price for self-hosted models from
// their total cost of ownership, so Augur can gate self-hosted deployments with
// the same machinery it uses for API pricing (SPEC Hito 5).
//
// For a hosted API you are quoted $/Mtok directly. For a model you run
// yourself, you instead pay for an instance by the hour and get some serving
// throughput; the effective token price is the standard benchmark-then-size
// calculation:
//
//	$/Mtok = instance_$/hr × 1e6 / (tokens_per_sec × 3600 × utilization)
//
// Utilization is the fraction of paid time the instance is actually serving
// tokens. It matters: you pay for the box 24/7, but it is rarely saturated, and
// a half-idle instance doubles the effective token price. Modelling it is the
// honest part — ignoring it is how self-hosting looks cheaper than it is.
//
// A self-hosted price is token-type-agnostic: it is the same compute whether the
// tokens are prompt, completion, or cached, so input/output/cached all carry the
// effective rate.
package tco

import (
	"fmt"
	"os"

	"augur/cost"

	"gopkg.in/yaml.v3"
)

// secondsPerHour and tokensPerMtok convert between the units a deployment is
// described in and the $/Mtok the pricing pipeline expects.
const (
	secondsPerHour = 3600.0
	tokensPerMtok  = 1_000_000.0
)

// Deployment is one self-hosted model: what its instance costs and how fast it
// serves. The deployment name (the map key in tco.yaml) must match the model id
// the agent sends, so the derived price lines up with the trace.
type Deployment struct {
	// InstanceCostPerHour is the all-in hourly cost of the serving instance(s).
	InstanceCostPerHour float64
	// TokensPerSec is the measured aggregate serving throughput (prompt +
	// completion tokens processed per second) at your batch size.
	TokensPerSec float64
	// Utilization is the fraction of paid time the instance is serving (0..1].
	// Defaults to 1.0 (always busy) when omitted.
	Utilization float64
}

// EffectivePerMtok is the derived $/Mtok for the deployment.
func (d Deployment) EffectivePerMtok() float64 {
	return d.InstanceCostPerHour * tokensPerMtok / (d.TokensPerSec * secondsPerHour * d.Utilization)
}

// TCO is a parsed tco.yaml: a set of named self-hosted deployments.
type TCO struct {
	Deployments map[string]Deployment
}

type yamlTCO struct {
	Version     int `yaml:"version"`
	Deployments map[string]struct {
		InstanceCostPerHour float64  `yaml:"instance_cost_per_hour"`
		TokensPerSec        float64  `yaml:"tokens_per_sec"`
		Utilization         *float64 `yaml:"utilization"`
	} `yaml:"deployments"`
}

// LoadTCO reads and validates a tco.yaml from path.
func LoadTCO(path string) (TCO, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return TCO{}, fmt.Errorf("tco: reading tco file: %w", err)
	}
	return ParseTCO(raw)
}

// ParseTCO parses and validates a TCO config from YAML bytes.
func ParseTCO(data []byte) (TCO, error) {
	var yt yamlTCO
	if err := yaml.Unmarshal(data, &yt); err != nil {
		return TCO{}, fmt.Errorf("tco: parsing tco yaml: %w", err)
	}
	if len(yt.Deployments) == 0 {
		return TCO{}, fmt.Errorf("tco: config has no deployments")
	}

	deployments := make(map[string]Deployment, len(yt.Deployments))
	for name, d := range yt.Deployments {
		util := 1.0
		if d.Utilization != nil {
			util = *d.Utilization
		}
		switch {
		case d.InstanceCostPerHour < 0:
			return TCO{}, fmt.Errorf("tco: %q instance_cost_per_hour must be >= 0, got %g", name, d.InstanceCostPerHour)
		case d.TokensPerSec <= 0:
			return TCO{}, fmt.Errorf("tco: %q tokens_per_sec must be > 0, got %g", name, d.TokensPerSec)
		case util <= 0 || util > 1:
			return TCO{}, fmt.Errorf("tco: %q utilization must be in (0, 1], got %g", name, util)
		}
		deployments[name] = Deployment{
			InstanceCostPerHour: d.InstanceCostPerHour,
			TokensPerSec:        d.TokensPerSec,
			Utilization:         util,
		}
	}
	return TCO{Deployments: deployments}, nil
}

// Pricing converts the deployments into a cost.Pricing: each deployment becomes
// a model priced at its effective $/Mtok for input, output, and cached tokens
// alike (self-hosted compute does not distinguish token types). snapshotDate is
// stamped onto the pricing for reporting.
func (t TCO) Pricing(snapshotDate string) cost.Pricing {
	models := make(map[string]cost.ModelPrice, len(t.Deployments))
	for name, d := range t.Deployments {
		eff := d.EffectivePerMtok()
		models[name] = cost.ModelPrice{Input: eff, Output: eff, CachedInput: eff}
	}
	return cost.Pricing{SnapshotDate: snapshotDate, Models: models}
}
