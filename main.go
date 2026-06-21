// Command augur is the CLI entrypoint for the Augur cost gate.
//
// Subcommands land one hito at a time (see SPEC.md). Implemented so far:
//
//	augur proxy — OpenAI-compatible recording proxy (Hito 1)
//	augur aggregate — trace + pricing → per-scenario cost distribution (Hito 2)
//
// Still to come: runner (Hito 2), project (Hito 3), gate (Hito 4).
package main

import (
	"errors"
	"flag"
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}

	cmd, args := os.Args[1], os.Args[2:]
	switch cmd {
	case "proxy":
		if err := runProxy(args); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return // flag already printed usage
			}
			fmt.Fprintln(os.Stderr, "augur proxy:", err)
			os.Exit(1)
		}
	case "aggregate":
		if err := runAggregate(args); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return
			}
			fmt.Fprintln(os.Stderr, "augur aggregate:", err)
			os.Exit(1)
		}
	case "-h", "--help", "help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "augur: unknown command %q\n\n", cmd)
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `augur — cost-first FinOps gate for AI agents

usage:
  augur proxy [flags]       run the OpenAI-compatible recording proxy
  augur aggregate [flags]   summarize a cost trace into per-scenario distributions

run "augur <command> -h" for command flags.
`)
}
