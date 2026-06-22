package runner

import (
	"strings"
	"testing"
)

func TestParseConfig(t *testing.T) {
	data := []byte(`
version: 1
runs: 5
command: ["python", "agent.py", "--prompt", "{{input}}"]
scenarios:
  - id: checkout
    input: "return order #1234"
  - id: faq
    input: "do you ship to Mexico?"
`)
	cfg, err := ParseConfig(data)
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.Runs != 5 {
		t.Errorf("Runs = %d, want 5", cfg.Runs)
	}
	if len(cfg.Command) != 4 || cfg.Command[0] != "python" {
		t.Errorf("Command = %v", cfg.Command)
	}
	if len(cfg.Scenarios) != 2 {
		t.Fatalf("got %d scenarios, want 2", len(cfg.Scenarios))
	}
	if cfg.Scenarios[0].ID != "checkout" || cfg.Scenarios[0].Input != "return order #1234" {
		t.Errorf("scenario[0] = %+v", cfg.Scenarios[0])
	}
}

func TestParseConfigDefaultsRuns(t *testing.T) {
	cfg, err := ParseConfig([]byte(`
command: ["./agent"]
scenarios:
  - id: a
    input: x
`))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.Runs != defaultRuns {
		t.Errorf("Runs = %d, want default %d", cfg.Runs, defaultRuns)
	}
}

func TestParseConfigRejects(t *testing.T) {
	cases := map[string]string{
		"no command": `
scenarios:
  - id: a
    input: x
`,
		"no scenarios": `
command: ["./agent"]
`,
		"duplicate id": `
command: ["./agent"]
scenarios:
  - id: a
    input: x
  - id: a
    input: y
`,
		"empty id": `
command: ["./agent"]
scenarios:
  - input: x
`,
		"negative runs": `
runs: -1
command: ["./agent"]
scenarios:
  - id: a
    input: x
`,
	}
	for name, data := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseConfig([]byte(data)); err == nil {
				t.Errorf("%s: expected error, got nil", name)
			}
		})
	}
}

func TestParseConfigRejectsMalformedYAML(t *testing.T) {
	if _, err := ParseConfig([]byte("\tnot: [valid")); err == nil {
		t.Error("expected error for malformed yaml")
	}
}

// Sanity: the placeholder constant is what the docs claim.
func TestInputPlaceholder(t *testing.T) {
	if !strings.Contains("--prompt {{input}}", inputPlaceholder) {
		t.Errorf("inputPlaceholder %q not the documented token", inputPlaceholder)
	}
}
