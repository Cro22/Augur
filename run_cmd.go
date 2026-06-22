package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"time"

	"augur/proxy"
	"augur/runner"
	"augur/trace"
)

// runRun is the all-in-one Hito 2 command: it starts the recording proxy
// in-process, drives the agent against scenarios.yaml N times through it, and
// leaves a complete cost trace ready for `augur aggregate`.
func runRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	scenariosPath := fs.String("scenarios", "scenarios.yaml", "path to the scenarios file")
	upstream := fs.String("upstream", "https://api.openai.com", "base URL of the real OpenAI-compatible provider")
	tracePath := fs.String("trace", "trace.jsonl", "path to append the cost trace to (JSONL)")
	listen := fs.String("listen", "127.0.0.1:0", "address for the in-process proxy (default: random local port)")
	runs := fs.Int("runs", 0, "override the repetitions per scenario (0 = use scenarios.yaml)")
	session := fs.String("session", "", "run-id prefix to keep repeated invocations distinct (default: timestamp)")
	continueOnError := fs.Bool("continue-on-error", false, "keep going after an agent invocation fails")
	if err := fs.Parse(args); err != nil {
		return err
	}

	cfg, err := runner.LoadConfig(*scenariosPath)
	if err != nil {
		return err
	}

	up, err := url.Parse(strings.TrimRight(*upstream, "/"))
	if err != nil || up.Scheme == "" || up.Host == "" {
		return fmt.Errorf("invalid -upstream %q: need scheme and host (e.g. https://api.openai.com)", *upstream)
	}

	tracer, err := trace.OpenFile(*tracePath)
	if err != nil {
		return err
	}
	defer tracer.Close()

	// Bind the proxy listener up front so we know the real address (handles a
	// :0 random port) before pointing the agent at it.
	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		return fmt.Errorf("binding proxy listener: %w", err)
	}
	baseURL := "http://" + ln.Addr().String()

	srv := &http.Server{Handler: proxy.New(up, tracer, nil)}
	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.Serve(ln) }()

	// Cancel the run on Ctrl-C so a long scenario sweep can be interrupted
	// cleanly (the trace written so far is still valid).
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	sess := *session
	if sess == "" {
		sess = time.Now().Format("20060102-150405")
	}

	fmt.Printf("augur run: proxy on %s → %s, tracing to %s\n", baseURL, up.String(), *tracePath)
	fmt.Printf("augur run: %d scenario(s), %d run(s) each, session %q\n",
		len(cfg.Scenarios), effectiveRuns(cfg.Runs, *runs), sess)

	sum, runErr := runner.Run(ctx, cfg, runner.Options{
		BaseURL:         baseURL,
		Runs:            *runs,
		Session:         sess,
		ContinueOnError: *continueOnError,
	})

	// Shut the proxy down and surface any serve error that isn't the expected
	// "closed" signal.
	_ = srv.Close()
	if err := <-serveErr; err != nil && err != http.ErrServerClosed {
		fmt.Fprintln(os.Stderr, "augur run: proxy serve error:", err)
	}

	fmt.Printf("augur run: %d invocation(s), %d failed\n", sum.Total, sum.Failed)
	if runErr != nil {
		return runErr
	}
	fmt.Printf("augur run: trace written to %s — summarize it with: augur aggregate -trace %s\n", *tracePath, *tracePath)
	return nil
}

// effectiveRuns reports the run count that will actually be used, for the
// startup banner.
func effectiveRuns(configRuns, override int) int {
	if override > 0 {
		return override
	}
	return configRuns
}
