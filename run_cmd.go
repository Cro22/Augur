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

	"augur/cassette"
	"augur/proxy"
	"augur/runner"
	"augur/trace"
)

// runRun is the all-in-one Hito 2 command: it starts the recording proxy
// in-process, drives the agent against scenarios.yaml N times through it, and
// leaves a complete cost trace ready for `augur aggregate`.
//
// With -record it also saves every response to a cassette; with -replay it
// serves responses from a cassette and never contacts the provider (Hito 5),
// so re-running the gate in CI costs no tokens.
func runRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	scenariosPath := fs.String("scenarios", "scenarios.yaml", "path to the scenarios file")
	upstream := fs.String("upstream", "https://api.openai.com", "base URL of the real OpenAI-compatible provider")
	tracePath := fs.String("trace", "trace.jsonl", "path to append the cost trace to (JSONL)")
	listen := fs.String("listen", "127.0.0.1:0", "address for the in-process proxy (default: random local port)")
	runs := fs.Int("runs", 0, "override the repetitions per scenario (0 = use scenarios.yaml)")
	session := fs.String("session", "", "run-id prefix (default: timestamp live, empty for record/replay so ids are stable)")
	continueOnError := fs.Bool("continue-on-error", false, "keep going after an agent invocation fails")
	record := fs.String("record", "", "record every response to this cassette file (real provider calls)")
	replay := fs.String("replay", "", "replay responses from this cassette file (no provider calls, no tokens)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *record != "" && *replay != "" {
		return fmt.Errorf("-record and -replay are mutually exclusive")
	}
	replaying := *replay != ""

	cfg, err := runner.LoadConfig(*scenariosPath)
	if err != nil {
		return err
	}

	// In replay mode the provider is never contacted, so the upstream URL is
	// irrelevant; otherwise it must be valid.
	up, err := url.Parse(strings.TrimRight(*upstream, "/"))
	if err != nil || up.Scheme == "" || up.Host == "" {
		if !replaying {
			return fmt.Errorf("invalid -upstream %q: need scheme and host (e.g. https://api.openai.com)", *upstream)
		}
		up = &url.URL{}
	}

	tracer, err := trace.OpenFile(*tracePath)
	if err != nil {
		return err
	}
	defer tracer.Close()

	pxy := proxy.New(up, tracer, nil)
	cass, err := configureCassette(pxy, *record, *replay)
	if err != nil {
		return err
	}
	if cass != nil {
		defer cass.Close()
	}

	// Bind the proxy listener up front so we know the real address (handles a
	// :0 random port) before pointing the agent at it.
	ln, err := net.Listen("tcp", *listen)
	if err != nil {
		return fmt.Errorf("binding proxy listener: %w", err)
	}
	baseURL := "http://" + ln.Addr().String()

	srv := &http.Server{Handler: pxy}
	serveErr := make(chan error, 1)
	go func() { serveErr <- srv.Serve(ln) }()

	// Cancel the run on Ctrl-C so a long scenario sweep can be interrupted
	// cleanly (the trace written so far is still valid).
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	// Stable run ids matter for record/replay (the cassette is keyed by them), so
	// default the session to empty there; live runs default to a timestamp to
	// keep repeated invocations distinct in a shared trace.
	sess := *session
	if sess == "" && *record == "" && !replaying {
		sess = time.Now().Format("20060102-150405")
	}

	fmt.Printf("augur run: proxy on %s (%s), tracing to %s\n", baseURL, modeLabel(*record, *replay, up), *tracePath)
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
	if *record != "" {
		fmt.Printf("augur run: cassette written to %s — replay it for free with: augur run -replay %s\n", *record, *record)
	}
	fmt.Printf("augur run: trace written to %s — summarize it with: augur aggregate -trace %s\n", *tracePath, *tracePath)
	return nil
}

// configureCassette puts the proxy into record or replay mode, opening the
// cassette file. It returns the cassette (for the caller to Close) or nil in
// live mode.
func configureCassette(pxy *proxy.Server, record, replay string) (*cassette.Cassette, error) {
	switch {
	case record != "":
		c, err := cassette.Create(record)
		if err != nil {
			return nil, err
		}
		pxy.Record(c)
		return c, nil
	case replay != "":
		c, err := cassette.Load(replay)
		if err != nil {
			return nil, err
		}
		pxy.Replay(c)
		return c, nil
	default:
		return nil, nil
	}
}

// modeLabel describes the proxy's mode for the startup banner.
func modeLabel(record, replay string, up *url.URL) string {
	switch {
	case record != "":
		return "RECORD → " + up.String() + ", cassette " + record
	case replay != "":
		return "REPLAY from " + replay + " (no provider calls)"
	default:
		return "LIVE → " + up.String()
	}
}

// effectiveRuns reports the run count that will actually be used, for the
// startup banner.
func effectiveRuns(configRuns, override int) int {
	if override > 0 {
		return override
	}
	return configRuns
}
