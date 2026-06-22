// Package runner drives the agent under test against a set of scenarios N
// times, so the proxy records a cost distribution rather than a single
// estimate (SPEC decision D3). It is the "scenario runner" half of Hito 2.
//
// The runner does not modify or wrap the agent. For each repetition it executes
// the user's agent command as a subprocess, injecting a small, documented
// contract via environment variables:
//
//	AUGUR_SCENARIO_ID  the scenario being exercised
//	AUGUR_RUN_ID       this specific repetition (unique per invocation)
//	AUGUR_INPUT        the scenario's representative input
//	AUGUR_BASE_URL     the recording proxy's base URL
//
// The agent's one obligation (its side of the contract) is to point its OpenAI
// base_url at AUGUR_BASE_URL and copy AUGUR_SCENARIO_ID / AUGUR_RUN_ID onto its
// LLM calls as the X-Augur-Scenario-Id / X-Augur-Run-Id headers the proxy reads.
// That is typically one line of default-headers config.
package runner

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// defaultRuns is the repetition count used when scenarios.yaml omits it. ~20 is
// enough to see the cost spread without spending excessively in CI (SPEC D3).
const defaultRuns = 20

// Scenario is one representative input the agent will face.
type Scenario struct {
	ID    string
	Input string
}

// Config is a parsed scenarios.yaml: the agent command, how many times to run
// each scenario, and the scenarios themselves.
type Config struct {
	// Runs is the number of repetitions per scenario.
	Runs int
	// Command is the agent entrypoint, argv-style. If any argument contains the
	// placeholder {{input}} it is replaced with the scenario's input; the input
	// is always also available via the AUGUR_INPUT environment variable.
	Command []string
	// Scenarios is the ordered list of scenarios to run.
	Scenarios []Scenario
}

type yamlConfig struct {
	Version   int      `yaml:"version"`
	Runs      int      `yaml:"runs"`
	Command   []string `yaml:"command"`
	Scenarios []struct {
		ID    string `yaml:"id"`
		Input string `yaml:"input"`
	} `yaml:"scenarios"`
}

// LoadConfig reads and validates scenarios.yaml from path.
func LoadConfig(path string) (Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("runner: reading scenarios file: %w", err)
	}
	return ParseConfig(raw)
}

// ParseConfig parses and validates scenarios YAML bytes. It is I/O-free so
// tests and other callers can supply config without a file.
func ParseConfig(data []byte) (Config, error) {
	var yc yamlConfig
	if err := yaml.Unmarshal(data, &yc); err != nil {
		return Config{}, fmt.Errorf("runner: parsing scenarios yaml: %w", err)
	}

	if len(yc.Command) == 0 {
		return Config{}, fmt.Errorf("runner: scenarios file has no command (the agent entrypoint)")
	}
	if len(yc.Scenarios) == 0 {
		return Config{}, fmt.Errorf("runner: scenarios file has no scenarios")
	}
	if yc.Runs < 0 {
		return Config{}, fmt.Errorf("runner: runs must be >= 0, got %d", yc.Runs)
	}

	runs := yc.Runs
	if runs == 0 {
		runs = defaultRuns
	}

	scenarios := make([]Scenario, 0, len(yc.Scenarios))
	seen := make(map[string]bool, len(yc.Scenarios))
	for i, s := range yc.Scenarios {
		if s.ID == "" {
			return Config{}, fmt.Errorf("runner: scenario %d has no id", i)
		}
		if seen[s.ID] {
			return Config{}, fmt.Errorf("runner: duplicate scenario id %q", s.ID)
		}
		seen[s.ID] = true
		scenarios = append(scenarios, Scenario{ID: s.ID, Input: s.Input})
	}

	return Config{
		Runs:      runs,
		Command:   yc.Command,
		Scenarios: scenarios,
	}, nil
}
