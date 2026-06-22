package tco

import (
	"fmt"
	"io"
	"sort"
	"text/tabwriter"
)

// WriteTable renders the derived effective $/Mtok for each deployment, with the
// inputs so the number can be checked at a glance.
func (t TCO) WriteTable(w io.Writer) error {
	names := make([]string, 0, len(t.Deployments))
	for name := range t.Deployments {
		names = append(names, name)
	}
	sort.Strings(names)

	if _, err := fmt.Fprintln(w, "Augur — self-hosted effective pricing (TCO)"); err != nil {
		return err
	}
	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintln(tw, "  deployment\t$/hr\ttok/s\tutil\teffective $/Mtok")
	for _, name := range names {
		d := t.Deployments[name]
		fmt.Fprintf(tw, "  %s\t%.2f\t%.0f\t%.0f%%\t$%.4f\n",
			name, d.InstanceCostPerHour, d.TokensPerSec, d.Utilization*100, d.EffectivePerMtok())
	}
	return tw.Flush()
}
