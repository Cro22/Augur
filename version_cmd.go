package main

import (
	"fmt"
	"runtime"
)

// Build metadata, injected at release time via -ldflags (see .goreleaser.yaml):
//
//	-X main.version=... -X main.commit=... -X main.date=...
//
// A plain `go build` leaves them at these defaults, so a from-source binary
// honestly reports itself as "dev".
var (
	version = "dev"
	commit  = ""
	date    = ""
)

// runVersion prints the build version and, when the release build injected them,
// the commit and date, plus the Go toolchain and target platform. It is the
// handle the GitHub Action uses to confirm a downloaded prebuilt binary runs.
func runVersion(_ []string) error {
	fmt.Printf("augur %s\n", version)
	if commit != "" {
		fmt.Printf("  commit: %s\n", commit)
	}
	if date != "" {
		fmt.Printf("  built:  %s\n", date)
	}
	fmt.Printf("  go:     %s %s/%s\n", runtime.Version(), runtime.GOOS, runtime.GOARCH)
	return nil
}
