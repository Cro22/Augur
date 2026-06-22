package main

import (
	"flag"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"augur/proxy"
	"augur/trace"
)

// runProxy parses flags and starts the recording proxy. It blocks until the
// server exits.
func runProxy(args []string) error {
	fs := flag.NewFlagSet("proxy", flag.ContinueOnError)
	listen := fs.String("listen", ":8080", "address to listen on")
	upstream := fs.String("upstream", "https://api.openai.com", "base URL of the real OpenAI-compatible provider")
	tracePath := fs.String("trace", "trace.jsonl", "path to append the cost trace to (JSONL)")
	if err := fs.Parse(args); err != nil {
		return err
	}

	up, err := url.Parse(strings.TrimRight(*upstream, "/"))
	if err != nil {
		return fmt.Errorf("invalid -upstream %q: %w", *upstream, err)
	}
	if up.Scheme == "" || up.Host == "" {
		return fmt.Errorf("invalid -upstream %q: need scheme and host (e.g. https://api.openai.com)", *upstream)
	}

	tracer, err := trace.OpenFile(*tracePath)
	if err != nil {
		return err
	}
	defer tracer.Close()

	srv := proxy.New(up, tracer, nil)

	fmt.Printf("augur proxy: listening on %s → forwarding to %s, tracing to %s\n",
		*listen, up.String(), *tracePath)
	fmt.Printf("augur proxy: point your agent's base_url at http://localhost%s and set headers %s / %s\n",
		*listen, proxy.HeaderScenarioID, proxy.HeaderRunID)

	return http.ListenAndServe(*listen, srv)
}
