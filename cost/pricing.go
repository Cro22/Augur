package cost

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// yamlPricing is the on-disk shape of pricing.yaml. It is intentionally
// separate from the in-memory Pricing type so the YAML schema can evolve
// independently of the computation types.
type yamlPricing struct {
	Version      int    `yaml:"version"`
	SnapshotDate string `yaml:"snapshot_date"`
	Currency     string `yaml:"currency"`
	Unit         string `yaml:"unit"`
	Models       map[string]struct {
		Input  float64 `yaml:"input"`
		Output float64 `yaml:"output"`
		// CachedInput is a pointer so we can distinguish "omitted" (no cache
		// discount → bill cached tokens at the full input rate) from an
		// explicit 0.0 (a genuinely free cache).
		CachedInput *float64 `yaml:"cached_input"`
	} `yaml:"models"`
}

// LoadPricing reads and parses a pricing snapshot from a YAML file.
func LoadPricing(path string) (Pricing, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Pricing{}, fmt.Errorf("cost: reading pricing file: %w", err)
	}
	return ParsePricing(raw)
}

// ParsePricing parses a pricing snapshot from YAML bytes. It is exported (and
// I/O-free) so tests and other callers can supply pricing without a file.
func ParsePricing(data []byte) (Pricing, error) {
	var yp yamlPricing
	if err := yaml.Unmarshal(data, &yp); err != nil {
		return Pricing{}, fmt.Errorf("cost: parsing pricing yaml: %w", err)
	}

	// Augur quotes every price per Mtok. Guard against a snapshot authored in a
	// different unit silently producing bills off by a factor of a million.
	if yp.Unit != "" && yp.Unit != "per_mtok" {
		return Pricing{}, fmt.Errorf("cost: unsupported pricing unit %q (want per_mtok)", yp.Unit)
	}
	if len(yp.Models) == 0 {
		return Pricing{}, fmt.Errorf("cost: pricing snapshot has no models")
	}

	models := make(map[string]ModelPrice, len(yp.Models))
	for name, m := range yp.Models {
		cached := m.Input // default: no cache discount
		if m.CachedInput != nil {
			cached = *m.CachedInput
		}
		models[name] = ModelPrice{
			Input:       m.Input,
			Output:      m.Output,
			CachedInput: cached,
		}
	}

	return Pricing{
		SnapshotDate: yp.SnapshotDate,
		Models:       models,
	}, nil
}
