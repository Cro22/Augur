package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"

	"augur/aggregate"
	"augur/trace"
)

// runAggregate reads a cost trace and a pricing snapshot, builds per-scenario
// cost distributions, and prints them — as a table by default, or JSON with
// -json (the machine-readable form the projection engine will consume).
func runAggregate(args []string) error {
	fs := flag.NewFlagSet("aggregate", flag.ContinueOnError)
	tracePath := fs.String("trace", "trace.jsonl", "path to the cost trace (JSONL) to aggregate")
	pricingPath := fs.String("pricing", "pricing.yaml", "path to the pricing snapshot")
	tcoPath := fs.String("tco", "", "derive pricing from a self-hosted TCO config instead of -pricing")
	asJSON := fs.Bool("json", false, "emit the aggregation as JSON instead of a table")
	if err := fs.Parse(args); err != nil {
		return err
	}

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

	res, err := aggregate.Aggregate(records, pricing)
	if err != nil {
		return err
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(res)
	}
	return res.WriteTable(os.Stdout)
}
