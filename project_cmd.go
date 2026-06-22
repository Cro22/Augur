package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"augur/aggregate"
	"augur/project"
	"augur/trace"
)

// runProject reads a trace + pricing + traffic profile, aggregates, projects the
// production unit economics, and prints them — as a table by default, or JSON
// with -json (the form the gate will consume in Hito 4).
func runProject(args []string) error {
	fs := flag.NewFlagSet("project", flag.ContinueOnError)
	tracePath := fs.String("trace", "trace.jsonl", "path to the cost trace (JSONL)")
	pricingPath := fs.String("pricing", "pricing.yaml", "path to the pricing snapshot")
	tcoPath := fs.String("tco", "", "derive pricing from a self-hosted TCO config instead of -pricing")
	trafficPath := fs.String("traffic", "traffic.yaml", "path to the traffic profile")
	asJSON := fs.Bool("json", false, "emit the projection as JSON instead of a table")
	ciLevel := fs.Float64("ci", 0.95, "confidence level for bootstrap intervals (0..1)")
	bootstrap := fs.Int("bootstrap", 2000, "number of bootstrap resamples")
	seed := fs.Uint64("seed", 1, "PRNG seed for reproducible bootstrap intervals")
	knobFs := addKnobFlags(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}
	knobs := knobFs.knobs()

	f, err := os.Open(*tracePath)
	if err != nil {
		return fmt.Errorf("opening trace: %w", err)
	}
	defer f.Close()
	records, err := trace.ReadAll(f)
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

	res, err := aggregate.AggregateWithKnobs(records, pricing, knobs)
	if err != nil {
		return err
	}
	proj, err := project.Project(res, traffic, project.Options{
		CILevel:          *ciLevel,
		BootstrapSamples: *bootstrap,
		Seed:             *seed,
	})
	if err != nil {
		return err
	}
	proj.WhatIf = describeKnobs(knobs)

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(proj)
	}
	return proj.WriteTable(os.Stdout)
}
