package aggregate

import (
	"fmt"
	"sort"

	"augur/cost"
	"augur/trace"
)

// ModelUsage is the total calls, tokens, and cost attributed to one model
// within a scenario — the "decompose by model" view that shows which model
// drives a scenario's bill.
type ModelUsage struct {
	Model        string  `json:"model"`
	Calls        int     `json:"calls"`
	InputTokens  int     `json:"input_tokens"`
	OutputTokens int     `json:"output_tokens"`
	CachedTokens int     `json:"cached_tokens"`
	CostUSD      float64 `json:"cost_usd"`
}

// Run is the cost of a single (scenario, run): the sum over every LLM call the
// agent made during that one execution. Kept individually so the checkpoint can
// reconcile per-run totals against the raw trace by hand.
type Run struct {
	ScenarioID string  `json:"scenario_id"`
	RunID      string  `json:"run_id"`
	Calls      int     `json:"calls"`
	CostUSD    float64 `json:"cost_usd"`
}

// Scenario aggregates all runs of one scenario into the cost distribution and
// the observed call multiplier (calls per run), plus the per-model breakdown.
type Scenario struct {
	ScenarioID string `json:"scenario_id"`
	Runs       int    `json:"runs"`
	// CostPerRun is the distribution of per-run dollar cost — the quantity the
	// projection engine and the gate ultimately care about.
	CostPerRun Distribution `json:"cost_per_run_usd"`
	// CallsPerRun is the distribution of how many LLM calls one run made — the
	// primary OBSERVED agentic multiplier. Retry/fan-out classification needs
	// labeling the trace does not yet carry; calls-per-run is what we can state
	// truthfully from the data.
	CallsPerRun Distribution `json:"calls_per_run"`
	// ByModel is the per-model breakdown, sorted by cost descending.
	ByModel []ModelUsage `json:"by_model"`
	// TotalCost is the summed cost of every run of this scenario.
	TotalCost float64 `json:"total_cost_usd"`
}

// Result is the full aggregation of a trace against a pricing snapshot.
type Result struct {
	// SnapshotDate echoes the pricing snapshot the costs were computed against,
	// so a report can state which prices it used.
	SnapshotDate string `json:"snapshot_date"`
	// Scenarios is sorted by scenario id for stable output.
	Scenarios []Scenario `json:"scenarios"`
	// Runs lists every run (sorted by scenario then run id) for hand
	// reconciliation against the raw trace.
	Runs []Run `json:"runs"`
}

// Knobs are what-if multipliers applied to every call's cost, for sensitivity
// analysis that reuses an existing trace instead of re-running the agent. The
// zero value is the identity (the observed cost). They model the agentic cost
// drivers:
//
//	RetryRate     extra fraction of calls retried (0.2 = 20% more calls)
//	FanoutFactor  multiplier on call count from sub-agent/tool fan-out
//	ContextGrowth multiplier on prompt (input + cached) cost from history growth
//
// Retries and fan-out add whole calls, so they scale the entire call cost;
// context growth inflates the prompt only, not the completion. Per call:
//
//	cost' = (1 + RetryRate) * FanoutFactor * (ContextGrowth*prompt + output)
type Knobs struct {
	RetryRate     float64
	FanoutFactor  float64
	ContextGrowth float64
}

// normalized fills in identity defaults: FanoutFactor and ContextGrowth default
// to 1 (no change) when left at zero, RetryRate stays 0.
func (k Knobs) normalized() Knobs {
	if k.FanoutFactor == 0 {
		k.FanoutFactor = 1
	}
	if k.ContextGrowth == 0 {
		k.ContextGrowth = 1
	}
	return k
}

// IsIdentity reports whether the knobs leave cost unchanged.
func (k Knobs) IsIdentity() bool {
	n := k.normalized()
	return n.RetryRate == 0 && n.FanoutFactor == 1 && n.ContextGrowth == 1
}

// apply returns the what-if cost of a single call given its component breakdown.
func (k Knobs) apply(b cost.Breakdown) float64 {
	n := k.normalized()
	callMult := (1 + n.RetryRate) * n.FanoutFactor
	return callMult * (n.ContextGrowth*b.PromptUSD() + b.OutputUSD)
}

// Aggregate prices every record against pricing and builds per-scenario cost
// distributions. An un-priceable model is a hard error (wrapping
// cost.ErrUnknownModel): an un-priced call means an unknowable bill, the exact
// surprise Augur exists to prevent, so we refuse to silently omit it.
//
// An empty trace yields a zero Result and no error.
func Aggregate(records []trace.Record, pricing cost.Pricing) (Result, error) {
	return AggregateWithKnobs(records, pricing, Knobs{})
}

// AggregateWithKnobs is Aggregate with what-if multipliers applied to every
// call's cost (see Knobs). With the zero Knobs it is identical to Aggregate.
func AggregateWithKnobs(records []trace.Record, pricing cost.Pricing, knobs Knobs) (Result, error) {
	type runKey struct{ scenario, run string }

	runs := make(map[runKey]*Run)
	var runOrder []runKey
	// scenario -> model -> accumulating usage
	scenarioModels := make(map[string]map[string]*ModelUsage)

	for _, rec := range records {
		u := cost.Usage{
			InputTokens:  rec.InputTokens,
			OutputTokens: rec.OutputTokens,
			CachedTokens: rec.CachedTokens,
		}
		b, err := pricing.Breakdown(rec.Model, u)
		if err != nil {
			return Result{}, fmt.Errorf("aggregate: scenario %q run %q seq %d: %w",
				rec.ScenarioID, rec.RunID, rec.Seq, err)
		}
		c := knobs.apply(b)

		k := runKey{rec.ScenarioID, rec.RunID}
		r, ok := runs[k]
		if !ok {
			r = &Run{ScenarioID: rec.ScenarioID, RunID: rec.RunID}
			runs[k] = r
			runOrder = append(runOrder, k)
		}
		r.Calls++
		r.CostUSD += c

		models := scenarioModels[rec.ScenarioID]
		if models == nil {
			models = make(map[string]*ModelUsage)
			scenarioModels[rec.ScenarioID] = models
		}
		mu := models[rec.Model]
		if mu == nil {
			mu = &ModelUsage{Model: rec.Model}
			models[rec.Model] = mu
		}
		mu.Calls++
		mu.InputTokens += rec.InputTokens
		mu.OutputTokens += rec.OutputTokens
		mu.CachedTokens += rec.CachedTokens
		mu.CostUSD += c
	}

	// Group runs by scenario.
	scenarioRuns := make(map[string][]*Run)
	for _, k := range runOrder {
		r := runs[k]
		scenarioRuns[r.ScenarioID] = append(scenarioRuns[r.ScenarioID], r)
	}

	scenarios := make([]Scenario, 0, len(scenarioRuns))
	for id, rs := range scenarioRuns {
		costs := make([]float64, len(rs))
		calls := make([]float64, len(rs))
		var total float64
		for i, r := range rs {
			costs[i] = r.CostUSD
			calls[i] = float64(r.Calls)
			total += r.CostUSD
		}
		scenarios = append(scenarios, Scenario{
			ScenarioID:  id,
			Runs:        len(rs),
			CostPerRun:  Summarize(costs),
			CallsPerRun: Summarize(calls),
			ByModel:     sortedModels(scenarioModels[id]),
			TotalCost:   total,
		})
	}
	sort.Slice(scenarios, func(i, j int) bool {
		return scenarios[i].ScenarioID < scenarios[j].ScenarioID
	})

	allRuns := make([]Run, 0, len(runOrder))
	for _, k := range runOrder {
		allRuns = append(allRuns, *runs[k])
	}
	sort.Slice(allRuns, func(i, j int) bool {
		if allRuns[i].ScenarioID != allRuns[j].ScenarioID {
			return allRuns[i].ScenarioID < allRuns[j].ScenarioID
		}
		return allRuns[i].RunID < allRuns[j].RunID
	})

	return Result{
		SnapshotDate: pricing.SnapshotDate,
		Scenarios:    scenarios,
		Runs:         allRuns,
	}, nil
}

// sortedModels flattens the per-model map into a slice ordered by cost
// descending (ties broken by model name for determinism).
func sortedModels(m map[string]*ModelUsage) []ModelUsage {
	out := make([]ModelUsage, 0, len(m))
	for _, mu := range m {
		out = append(out, *mu)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].CostUSD != out[j].CostUSD {
			return out[i].CostUSD > out[j].CostUSD
		}
		return out[i].Model < out[j].Model
	})
	return out
}
