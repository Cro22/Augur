package project

import (
	"fmt"
	"io"
	"text/tabwriter"
)

// WriteTable renders the projection as a human-readable table. The full
// report.md artifact (with pass/fail) arrives with the gate in Hito 4; this is
// the eyeball view for the Hito 3 checkpoint.
func (p Projection) WriteTable(w io.Writer) error {
	tr := p.Traffic
	if _, err := fmt.Fprintf(w, "Augur — projected unit economics (pricing snapshot %s)\n", p.SnapshotDate); err != nil {
		return err
	}
	fmt.Fprintf(w, "traffic: %d users x %g req/user/day, %d tenants, %d days/month\n",
		tr.Users, tr.RequestsPerUserPerDay, tr.Tenants, tr.DaysPerMonth)
	fmt.Fprintf(w, "bootstrap: %d samples, %.0f%% CI, seed %d\n", p.BootstrapSamples, p.CILevel*100, p.Seed)
	if p.WhatIf != "" {
		fmt.Fprintf(w, "%s\n", p.WhatIf)
	}
	fmt.Fprintln(w)

	tw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
	fmt.Fprintf(tw, "  metric\tvalue\t%.0f%% CI\n", p.CILevel*100)
	fmt.Fprintf(tw, "  $/request p50\t%s\t%s\n", money(p.RequestP50), "—")
	fmt.Fprintf(tw, "  $/request p95\t%s\t%s\n", money(p.RequestP95.Value), ci(p.RequestP95))
	fmt.Fprintf(tw, "  $/request mean\t%s\t%s\n", money(p.RequestMean.Value), ci(p.RequestMean))
	fmt.Fprintf(tw, "  $/user/day\t%s\t%s\n", money(p.PerUserPerDay.Value), ci(p.PerUserPerDay))
	fmt.Fprintf(tw, "  $/tenant/month\t%s\t%s\n", money(p.PerTenantPerMonth.Value), ci(p.PerTenantPerMonth))
	fmt.Fprintf(tw, "  monthly bill\t%s\t%s\n", money(p.MonthlyBill.Value), ci(p.MonthlyBill))
	if err := tw.Flush(); err != nil {
		return err
	}

	if len(p.ByModel) > 0 {
		fmt.Fprintln(w, "\n  by model (avg request):")
		mw := tabwriter.NewWriter(w, 0, 0, 2, ' ', 0)
		for _, m := range p.ByModel {
			fmt.Fprintf(mw, "  %s\t%s\t%.1f%%\n", m.Model, money(m.PerRequestUSD), m.Pct)
		}
		if err := mw.Flush(); err != nil {
			return err
		}
	}
	_, err := fmt.Fprintln(w)
	return err
}

// money formats a dollar amount with six decimals (per-request costs are
// routinely sub-cent).
func money(v float64) string { return fmt.Sprintf("$%.6f", v) }

// ci formats an estimate's confidence interval.
func ci(e Estimate) string { return fmt.Sprintf("[%s, %s]", money(e.Lo), money(e.Hi)) }
