package gate

import (
	"encoding/json"
	"fmt"
	"io"

	"augur/project"
)

// Report is the machine-readable artifact (report.json): the full projection
// plus the gate verdict. It is what a CI job archives and what a PR comment is
// rendered from.
type Report struct {
	Pass       bool               `json:"pass"`
	Gate       Result             `json:"gate"`
	Projection project.Projection `json:"projection"`
}

// NewReport bundles a projection and its gate result.
func NewReport(proj project.Projection, res Result) Report {
	return Report{Pass: res.Pass, Gate: res, Projection: proj}
}

// WriteJSON writes the report as indented JSON (report.json).
func (r Report) WriteJSON(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(r)
}

// WriteMarkdown writes a human-readable report (report.md) suitable for a PR
// comment: the verdict up top, the budget checks as a table, then the projected
// unit economics for context.
func (r Report) WriteMarkdown(w io.Writer) error {
	verdict := "✅ PASS"
	if !r.Pass {
		verdict = "❌ FAIL"
	}
	p := r.Projection
	tr := p.Traffic

	bw := &bufErr{w: w}
	bw.printf("# Augur cost report\n\n")
	bw.printf("**Verdict: %s** — %s\n\n", verdict, checkSummary(r.Gate))
	bw.printf("- Pricing snapshot: `%s`\n", p.SnapshotDate)
	bw.printf("- Traffic: %d users × %g req/user/day, %d tenants, %d days/month\n",
		tr.Users, tr.RequestsPerUserPerDay, tr.Tenants, tr.DaysPerMonth)
	bw.printf("- Bootstrap: %d samples, %.0f%% CI, seed %d\n\n", p.BootstrapSamples, p.CILevel*100, p.Seed)

	bw.printf("## Budget checks\n\n")
	bw.printf("| check | limit | projected | status |\n")
	bw.printf("|---|---:|---:|:---:|\n")
	for _, c := range r.Gate.Checks {
		status := "✅"
		if !c.Pass {
			status = "❌"
		}
		bw.printf("| %s | %s | %s | %s |\n", c.Name, money(c.Limit), money(c.Actual), status)
	}

	bw.printf("\n## Projected unit economics\n\n")
	bw.printf("| metric | value | %.0f%% CI |\n", p.CILevel*100)
	bw.printf("|---|---:|---:|\n")
	bw.printf("| $/request p50 | %s | — |\n", money(p.RequestP50))
	bw.printf("| $/request p95 | %s | %s |\n", money(p.RequestP95.Value), ciMd(p.RequestP95))
	bw.printf("| $/request mean | %s | %s |\n", money(p.RequestMean.Value), ciMd(p.RequestMean))
	bw.printf("| $/user/day | %s | %s |\n", money(p.PerUserPerDay.Value), ciMd(p.PerUserPerDay))
	bw.printf("| $/tenant/month | %s | %s |\n", money(p.PerTenantPerMonth.Value), ciMd(p.PerTenantPerMonth))
	bw.printf("| monthly bill | %s | %s |\n", money(p.MonthlyBill.Value), ciMd(p.MonthlyBill))

	if len(p.ByModel) > 0 {
		bw.printf("\n## Cost by model (average request)\n\n")
		bw.printf("| model | $/request | share |\n")
		bw.printf("|---|---:|---:|\n")
		for _, m := range p.ByModel {
			bw.printf("| %s | %s | %.1f%% |\n", m.Model, money(m.PerRequestUSD), m.Pct)
		}
	}
	return bw.err
}

// checkSummary describes the verdict in one phrase.
func checkSummary(res Result) string {
	over := 0
	for _, c := range res.Checks {
		if !c.Pass {
			over++
		}
	}
	if over == 0 {
		return fmt.Sprintf("all %d budget check(s) within limit", len(res.Checks))
	}
	return fmt.Sprintf("%d of %d budget check(s) over limit", over, len(res.Checks))
}

func money(v float64) string  { return fmt.Sprintf("$%.6f", v) }
func ciMd(e project.Estimate) string {
	return fmt.Sprintf("[%s, %s]", money(e.Lo), money(e.Hi))
}

// bufErr accumulates the first write error so the long sequence of printfs in
// WriteMarkdown doesn't need an `if err` after each line.
type bufErr struct {
	w   io.Writer
	err error
}

func (b *bufErr) printf(format string, args ...any) {
	if b.err != nil {
		return
	}
	_, b.err = fmt.Fprintf(b.w, format, args...)
}
