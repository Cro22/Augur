package main

import (
	"fmt"
	"io"
	"sort"

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

// warnModelAliases prints, to w, one warning per model that was priced via the
// snapshot's prefix fallback (see aggregate.Result.ModelAliases). A fallback is
// a convenience that can mis-bill if the base entry's price differs from the
// requested variant's, so it is never silent. Keys are sorted for stable output.
func warnModelAliases(w io.Writer, aliases map[string]string) {
	if len(aliases) == 0 {
		return
	}
	requested := make([]string, 0, len(aliases))
	for r := range aliases {
		requested = append(requested, r)
	}
	sort.Strings(requested)
	for _, r := range requested {
		fmt.Fprintf(w, "augur: WARNING model %q not in pricing snapshot; billed at %q via prefix fallback\n", r, aliases[r])
	}
}
