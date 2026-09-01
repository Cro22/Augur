package runner

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"slices"
	"strings"
	"sync"
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
	// Concurrency is how many invocations run in parallel. Values <= 1 (the
	// default) preserve the original strictly-sequential behavior, streaming each
	// invocation's output directly. With >1, invocations run through a worker pool
	// and each one's stdout/stderr is buffered and flushed as a block so parallel
	// output never interleaves. The proxy and trace writer are concurrency-safe,
	// so parallel invocations record correctly; the Summary is always assembled
	// in scenario/run order regardless of completion order.
	Concurrency int

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

	// Flatten scenarios × repetitions into an ordered task list. The Summary is
	// always assembled in this order, so parallel completion order never changes
	// the output.
	var tasks []task
	for _, sc := range cfg.Scenarios {
		for i := range runs {
			tasks = append(tasks, task{sc: sc, index: i})
		}
	}

	concurrency := max(opts.Concurrency, 1)

	r := invoker{
		cfg: cfg, opts: opts, baseEnv: baseEnv,
		execFn: execFn, stdout: stdout, stderr: stderr,
	}
	if concurrency == 1 {
		return r.runSequential(ctx, tasks)
	}
	return r.runParallel(ctx, tasks, concurrency)
}

// task is one scheduled invocation: a scenario and which repetition (index) of
// it to run.
type task struct {
	sc    Scenario
	index int
}

// invoker holds the resolved per-Run configuration so the sequential and
// parallel paths share one place that builds each invocation.
type invoker struct {
	cfg     Config
	opts    Options
	baseEnv []string
	execFn  ExecFunc
	stdout  io.Writer
	stderr  io.Writer
}

// invoke runs one task, writing the agent's output to the given writers, and
// returns the Invocation (with any exec error already wrapped).
func (r invoker) invoke(ctx context.Context, t task, stdout, stderr io.Writer) Invocation {
	runID := makeRunID(r.opts.Session, t.sc.ID, t.index)
	env := buildEnv(r.baseEnv, t.sc, runID, r.opts.BaseURL)
	args := substituteInput(r.cfg.Command, t.sc.Input)

	err := r.execFn(ctx, args, env, stdout, stderr)
	if err != nil {
		err = fmt.Errorf("scenario %q run %q (index %d): %w", t.sc.ID, runID, t.index, err)
	}
	return Invocation{ScenarioID: t.sc.ID, RunID: runID, Index: t.index, Err: err}
}

// runSequential executes tasks one at a time, streaming each invocation's output
// directly. This is the original behavior and the default (Concurrency <= 1):
// unless ContinueOnError is set, the first failing invocation aborts the run.
func (r invoker) runSequential(ctx context.Context, tasks []task) (Summary, error) {
	var sum Summary
	for _, t := range tasks {
		if err := ctx.Err(); err != nil {
			return sum, err
		}
		inv := r.invoke(ctx, t, r.stdout, r.stderr)
		sum.Total++
		if inv.Err != nil {
			sum.Failed++
		}
		sum.Invocations = append(sum.Invocations, inv)
		if inv.Err != nil && !r.opts.ContinueOnError {
			return sum, inv.Err
		}
	}
	return sum, nil
}

// runParallel executes up to `concurrency` tasks at once. Each invocation's
// stdout/stderr is buffered and flushed as a single block under a mutex, so
// parallel output never interleaves. Without ContinueOnError, the first genuine
// failure cancels the run's context: in-flight invocations are signalled to
// stop and no further tasks are launched, and that triggering error is returned.
func (r invoker) runParallel(ctx context.Context, tasks []task, concurrency int) (Summary, error) {
	results := make([]Invocation, len(tasks))
	ran := make([]bool, len(tasks))

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup
	var flushMu sync.Mutex // serializes output flushes
	var errOnce sync.Once
	var triggerErr error // the failure that aborted the run (if any)

	for idx, t := range tasks {
		if runCtx.Err() != nil {
			break // aborted or cancelled: stop launching new work
		}
		wg.Add(1)
		sem <- struct{}{}
		go func(idx int, t task) {
			defer wg.Done()
			defer func() { <-sem }()

			var ob, eb bytes.Buffer
			inv := r.invoke(runCtx, t, &ob, &eb)

			flushMu.Lock()
			_, _ = io.Copy(r.stdout, &ob)
			_, _ = io.Copy(r.stderr, &eb)
			flushMu.Unlock()

			// Distinct index per goroutine: writes to different slice elements
			// don't race, and the wg.Wait below establishes the read barrier.
			results[idx] = inv
			ran[idx] = true
			if inv.Err != nil && !r.opts.ContinueOnError {
				errOnce.Do(func() { triggerErr = inv.Err })
				cancel()
			}
		}(idx, t)
	}
	wg.Wait()

	sum := summarize(tasks, results, ran)
	if triggerErr != nil {
		return sum, triggerErr
	}
	if err := ctx.Err(); err != nil {
		return sum, err
	}
	return sum, nil
}

// summarize assembles the Summary in task order from the parallel results,
// including only invocations that actually ran.
func summarize(tasks []task, results []Invocation, ran []bool) Summary {
	var sum Summary
	for idx := range tasks {
		if !ran[idx] {
			continue
		}
		inv := results[idx]
		sum.Total++
		if inv.Err != nil {
			sum.Failed++
		}
		sum.Invocations = append(sum.Invocations, inv)
	}
	return sum
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
