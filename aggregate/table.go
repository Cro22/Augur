package aggregate

import (
	"fmt"
	"io"
	"text/tabwriter"
)

// WriteTable renders a human-readable distribution table for eyeballing and the
// Hito 2 checkpoint. It is intentionally plain text (not the Hito 4 report):
// enough to reconcile against the raw trace at a glance.
func (r Result) WriteTable(w io.Writer) error {
	if _, err := fmt.Fprintf(w, "Augur — observed cost (pricing snapshot %s)\n\n", r.SnapshotDate); err != nil {
		return err
	}
	if len(r.Scenarios) == 0 {
		_, err := fmt.Fprintln(w, "(no scenarios in trace)")
		return err
	}

	for _, s := range r.Scenarios {
		if _, err := fmt.Fprintf(w, "scenario %q — %d run(s), total $%.6f\n", s.ScenarioID, s.Runs, s.TotalCost); err != nil {
			return err
		}

		tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
		fmt.Fprintln(tw, "  metric\tmean\tp50\tp95\tstdev\tmin\tmax")
		fmt.Fprintf(tw, "  $/run\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
			s.CostPerRun.Mean, s.CostPerRun.P50, s.CostPerRun.P95, s.CostPerRun.Stdev, s.CostPerRun.Min, s.CostPerRun.Max)
		fmt.Fprintf(tw, "  calls/run\t%.2f\t%.2f\t%.2f\t%.2f\t%.0f\t%.0f\n",
			s.CallsPerRun.Mean, s.CallsPerRun.P50, s.CallsPerRun.P95, s.CallsPerRun.Stdev, s.CallsPerRun.Min, s.CallsPerRun.Max)
		if err := tw.Flush(); err != nil {
			return err
		}

		mw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
		fmt.Fprintln(mw, "  by model\tcalls\tin tok\tout tok\tcached\tcost $")
		for _, m := range s.ByModel {
			fmt.Fprintf(mw, "  %s\t%d\t%d\t%d\t%d\t%.6f\n",
				m.Model, m.Calls, m.InputTokens, m.OutputTokens, m.CachedTokens, m.CostUSD)
		}
		if err := mw.Flush(); err != nil {
			return err
		}
		if _, err := fmt.Fprintln(w); err != nil {
			return err
		}
	}
	return nil
}
