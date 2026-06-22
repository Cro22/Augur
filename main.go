// Command augur is the CLI entrypoint for the Augur cost gate.
//
// Subcommands land one hito at a time (see SPEC.md). Implemented so far:
//
//	augur proxy — OpenAI-compatible recording proxy (Hito 1)
//	augur run — drive the agent against scenarios.yaml ×N through the proxy (Hito 2)
//	augur aggregate — trace + pricing → per-scenario cost distribution (Hito 2)
//	augur project — aggregate + traffic → projected unit economics with CIs (Hito 3)
//
// Still to come: gate (Hito 4).
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
)

// command is one augur subcommand: a run function plus the one-line summary
// shown in usage.
type command struct {
	run     func(args []string) error
	summary string
}

// exitErr lets a subcommand request a specific process exit code without main
// printing an "augur <cmd>:" error line. The gate uses it to fail the build
// after it has already written its report — a budget overrun is a normal,
// expected outcome, not an internal error.
type exitErr struct{ code int }

func (e *exitErr) Error() string { return fmt.Sprintf("exit status %d", e.code) }

// commands is the subcommand dispatch table. Adding a hito's command is one
// entry here plus its run<Name> function in its own file.
var commands = map[string]command{
	"proxy":     {runProxy, "run the OpenAI-compatible recording proxy"},
	"run":       {runRun, "drive the agent against scenarios.yaml ×N through the proxy"},
	"aggregate": {runAggregate, "summarize a cost trace into per-scenario distributions"},
	"project":   {runProject, "project a trace to production unit economics with CIs"},
	"gate":      {runGate, "check a projection against budget.yaml (exit 1 if over)"},
}

// order fixes the usage listing (maps don't iterate deterministically).
var order = []string{"proxy", "run", "aggregate", "project", "gate"}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	name, args := os.Args[1], os.Args[2:]
	switch name {
	case "-h", "--help", "help":
		usage()
		return
	}

	cmd, ok := commands[name]
	if !ok {
		fmt.Fprintf(os.Stderr, "augur: unknown command %q\n\n", name)
		usage()
		os.Exit(2)
	}

	if err := cmd.run(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return // flag already printed usage to stderr
		}
		var ee *exitErr
		if errors.As(err, &ee) {
			os.Exit(ee.code) // command already reported; just set the code
		}
		fmt.Fprintf(os.Stderr, "augur %s: %v\n", name, err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "augur — cost-first FinOps gate for AI agents")
	fmt.Fprintln(os.Stderr, "\nusage:")
	for _, name := range order {
		fmt.Fprintf(os.Stderr, "  augur %-11s %s\n", name, commands[name].summary)
	}
	fmt.Fprintln(os.Stderr, "\nrun \"augur <command> -h\" for command flags.")
}
