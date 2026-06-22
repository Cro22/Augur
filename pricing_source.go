package main

import (
	"fmt"

	"augur/cost"
	"augur/tco"
)

// resolvePricing loads the pricing the cost pipeline should use. When tcoPath is
// set, prices are derived from a self-hosted TCO config (effective $/Mtok from
// instance cost + throughput); otherwise they come from the pricing snapshot.
// The two are alternatives — tco takes precedence when both are provided.
func resolvePricing(pricingPath, tcoPath string) (cost.Pricing, error) {
	if tcoPath != "" {
		tc, err := tco.LoadTCO(tcoPath)
		if err != nil {
			return cost.Pricing{}, err
		}
		return tc.Pricing(fmt.Sprintf("tco (%s)", tcoPath)), nil
	}
	return cost.LoadPricing(pricingPath)
}
