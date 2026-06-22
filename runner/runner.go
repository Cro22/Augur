package runner

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"slices"
	"strings"
)

// Environment-variable names the runner injects per invocation. See the package
// doc for the agent's side of the contract.
const (
	EnvScenarioID = "AUGUR_SCENARIO_ID"
	EnvRunID      = "AUGUR_RUN_ID"
	EnvInput      = "AUGUR_INPUT"
	EnvBaseURL    = "AUGUR_BASE_URL"
)

// inputPlaceholder is replaced in command arguments with the scenario input.
const inputPlaceholder = "{{input}}"

// ExecFunc runs one agent invocation. It is a field on Options so tests can
// substitute a fake that records the environment instead of spawning a process.
type ExecFunc func(ctx context.Context, args, env []string, stdout, stderr io.Writer) error

// defaultExec runs the command as a real subprocess.
func defaultExec(ctx context.Context, args, env []string, stdout, stderr io.Writer) error {
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.Env = env
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	return cmd.Run()
}

// Options configures a Run. Zero values are sensible: Exec defaults to spawning
// a real subprocess, BaseEnv to the current process environment, and the output
// writers to os.Stdout/os.Stderr.
type Options struct {
	// BaseURL is the proxy URL handed to the agent via AUGUR_BASE_URL.
	BaseURL string
	// Runs overrides Config.Runs when > 0.
	Runs int
	// Session, when non-empty, prefixes every run id so repeated invocations of
	// augur against the same trace file don't collide.
	Session string
	// BaseEnv is the environment each invocation inherits before the AUGUR_*
	// variables are layered on. nil means os.Environ().
	BaseEnv []string
	// ContinueOnError keeps running after an invocation fails instead of
	// aborting at the first failure.
	ContinueOnError bool

	Stdout io.Writer
	Stderr io.Writer
	Exec   ExecFunc
}

// Invocation is the outcome of one agent execution.
type Invocation struct {
	ScenarioID string
	RunID      string
	Index      int
	Err        error
}

// Summary reports what Run did.
type Summary struct {
	Total       int
	Failed      int
	Invocations []Invocation
}

// Run executes every scenario Config.Runs (or Options.Runs) times, injecting
// the AUGUR_* contract into each invocation's environment. It returns a Summary
// of all invocations. Unless ContinueOnError is set, the first failing
// invocation aborts the run and is returned as the error.
func Run(ctx context.Context, cfg Config, opts Options) (Summary, error) {
	runs := cfg.Runs
	if opts.Runs > 0 {
		runs = opts.Runs
	}
	baseEnv := opts.BaseEnv
	if baseEnv == nil {
		baseEnv = os.Environ()
	}
	execFn := opts.Exec
	if execFn == nil {
		execFn = defaultExec
	}
	stdout := opts.Stdout
	if stdout == nil {
		stdout = os.Stdout
	}
	stderr := opts.Stderr
	if stderr == nil {
		stderr = os.Stderr
	}

	var sum Summary
	for _, sc := range cfg.Scenarios {
		for i := range runs {
			if err := ctx.Err(); err != nil {
				return sum, err
			}

			runID := makeRunID(opts.Session, sc.ID, i)
			env := buildEnv(baseEnv, sc, runID, opts.BaseURL)
			args := substituteInput(cfg.Command, sc.Input)

			err := execFn(ctx, args, env, stdout, stderr)
			sum.Total++
			if err != nil {
				sum.Failed++
				err = fmt.Errorf("scenario %q run %q (index %d): %w", sc.ID, runID, i, err)
			}
			sum.Invocations = append(sum.Invocations, Invocation{
				ScenarioID: sc.ID, RunID: runID, Index: i, Err: err,
			})
			if err != nil && !opts.ContinueOnError {
				return sum, err
			}
		}
	}
	return sum, nil
}

// makeRunID builds a per-invocation run id. With a session it is
// "<session>-<scenario>-<index>", otherwise "<scenario>-<index>", zero-padded
// so lexical and numeric ordering agree for up to 1000 runs.
func makeRunID(session, scenarioID string, index int) string {
	if session != "" {
		return fmt.Sprintf("%s-%s-%03d", session, scenarioID, index)
	}
	return fmt.Sprintf("%s-%03d", scenarioID, index)
}

// buildEnv layers the AUGUR_* contract onto baseEnv, removing any pre-existing
// AUGUR_* entries so the runner's values are the only ones the agent sees.
func buildEnv(baseEnv []string, sc Scenario, runID, baseURL string) []string {
	out := make([]string, 0, len(baseEnv)+4)
	for _, kv := range baseEnv {
		if !isAugurVar(kv) {
			out = append(out, kv)
		}
	}
	out = append(out,
		EnvScenarioID+"="+sc.ID,
		EnvRunID+"="+runID,
		EnvInput+"="+sc.Input,
		EnvBaseURL+"="+baseURL,
	)
	return out
}

// augurVars are the environment keys the runner owns.
var augurVars = []string{EnvScenarioID, EnvRunID, EnvInput, EnvBaseURL}

func isAugurVar(kv string) bool {
	eq := strings.IndexByte(kv, '=')
	if eq < 0 {
		return false
	}
	return slices.Contains(augurVars, kv[:eq])
}

// substituteInput replaces the {{input}} placeholder in each command argument
// with the scenario input. Arguments without the placeholder are unchanged; the
// input is always also available via AUGUR_INPUT.
func substituteInput(command []string, input string) []string {
	out := make([]string, len(command))
	for i, arg := range command {
		out[i] = strings.ReplaceAll(arg, inputPlaceholder, input)
	}
	return out
}
