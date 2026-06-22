package main

import (
	"encoding/json"
	"flag"
	"os"

	"augur/tco"
)

// runTCO shows the effective $/Mtok derived from a self-hosted TCO config. Feed
// the same config to project/gate via their -tco flag to cost a trace against
// it.
func runTCO(args []string) error {
	fs := flag.NewFlagSet("tco", flag.ContinueOnError)
	tcoPath := fs.String("tco", "tco.yaml", "path to the TCO config")
	asJSON := fs.Bool("json", false, "emit the derived pricing as JSON instead of a table")
	if err := fs.Parse(args); err != nil {
		return err
	}

	tc, err := tco.LoadTCO(*tcoPath)
	if err != nil {
		return err
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(tc.Pricing("tco (" + *tcoPath + ")"))
	}
	return tc.WriteTable(os.Stdout)
}
