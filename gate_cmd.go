package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"augur/aggregate"
	"augur/gate"
	"augur/project"
	"augur/trace"
)

// runGate is the full v1 pipeline ending in a pass/fail verdict: trace +
// pricing + traffic + budget → projection → gate. It writes report.md and
// report.json, prints the markdown to stdout, and returns an *exitErr{1} when
// the projection is over budget so the CI build fails.
func runGate(args []string) error {
	fs := flag.NewFlagSet("gate", flag.ContinueOnError)
	tracePath := fs.String("trace", "trace.jsonl", "path to the cost trace (JSONL)")
	pricingPath := fs.String("pricing", "pricing.yaml", "path to the pricing snapshot")
	tcoPath := fs.String("tco", "", "derive pricing from a self-hosted TCO config instead of -pricing")
	trafficPath := fs.String("traffic", "traffic.yaml", "path to the traffic profile")
	budgetPath := fs.String("budget", "budget.yaml", "path to the budget thresholds")
	reportMD := fs.String("report-md", "report.md", "path to write the Markdown report (empty to skip)")
	reportJSON := fs.String("report-json", "report.json", "path to write the JSON report (empty to skip)")
	ciLevel := fs.Float64("ci", 0.95, "confidence level for bootstrap intervals (0..1)")
	bootstrap := fs.Int("bootstrap", 2000, "number of bootstrap resamples")
	seed := fs.Uint64("seed", 1, "PRNG seed for reproducible bootstrap intervals")
	knobFs := addKnobFlags(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}
	knobs := knobFs.knobs()

	records, err := readTrace(*tracePath)
	if err != nil {
		return err
	}
	pricing, err := resolvePricing(*pricingPath, *tcoPath)
	if err != nil {
		return err
	}
	traffic, err := project.LoadTraffic(*trafficPath)
	if err != nil {
		return err
	}
	budget, err := gate.LoadBudget(*budgetPath)
	if err != nil {
		return err
	}

	res, err := aggregate.AggregateWithKnobs(records, pricing, knobs)
	if err != nil {
		return err
	}
	proj, err := project.Project(res, traffic, project.Options{
		CILevel: *ciLevel, BootstrapSamples: *bootstrap, Seed: *seed,
	})
	if err != nil {
		return err
	}
	proj.WhatIf = describeKnobs(knobs)

	report := gate.NewReport(proj, gate.Evaluate(proj, budget))

	if *reportMD != "" {
		if err := writeReportFile(*reportMD, report.WriteMarkdown); err != nil {
			return err
		}
	}
	if *reportJSON != "" {
		if err := writeReportFile(*reportJSON, report.WriteJSON); err != nil {
			return err
		}
	}

	// Print the human report to stdout so the verdict is visible in CI logs.
	if err := report.WriteMarkdown(os.Stdout); err != nil {
		return err
	}

	if !report.Pass {
		fmt.Fprintln(os.Stderr, "augur gate: projection is OVER budget")
		return &exitErr{code: 1}
	}
	fmt.Fprintln(os.Stderr, "augur gate: within budget")
	return nil
}

// readTrace opens and parses a JSONL trace file.
func readTrace(path string) ([]trace.Record, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("opening trace: %w", err)
	}
	defer f.Close()
	return trace.ReadAll(f)
}

// writeReportFile writes a report to path via the given render function,
// creating/truncating the file.
func writeReportFile(path string, render func(w io.Writer) error) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating report %s: %w", path, err)
	}
	if err := render(f); err != nil {
		f.Close()
		return fmt.Errorf("writing report %s: %w", path, err)
	}
	return f.Close()
}
