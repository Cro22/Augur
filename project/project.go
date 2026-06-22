package project

import (
	"fmt"
	"math/rand/v2"
	"sort"

	"augur/aggregate"
)

// Default projection knobs.
const (
	defaultCILevel          = 0.95
	defaultBootstrapSamples = 2000
	defaultSeed             = 1
)

// Estimate is a projected value with a confidence interval. Lo and Hi are the
// bootstrap CI bounds at the projection's CILevel; for a degenerate sample
// (all runs identical) Lo == Value == Hi.
type Estimate struct {
	Value float64 `json:"value"`
	Lo    float64 `json:"ci_lo"`
	Hi    float64 `json:"ci_hi"`
}

// ModelShare is one model's contribution to the cost of an average request.
type ModelShare struct {
	Model         string  `json:"model"`
	PerRequestUSD float64 `json:"per_request_usd"`
	Pct           float64 `json:"pct"`
}

// Projection is the full set of projected unit economics.
type Projection struct {
	SnapshotDate     string  `json:"snapshot_date"`
	CILevel          float64 `json:"ci_level"`
	BootstrapSamples int     `json:"bootstrap_samples"`
	Seed             uint64  `json:"seed"`

	// Per single request, drawn from the scenario mix.
	RequestMean Estimate `json:"request_cost_mean_usd"`
	RequestP50  float64  `json:"request_cost_p50_usd"`
	RequestP95  Estimate `json:"request_cost_p95_usd"` // the tail the gate checks (D2)

	// Aggregates, scaling linearly from the mean per-request cost.
	PerUserPerDay     Estimate `json:"per_user_per_day_usd"`
	PerTenantPerMonth Estimate `json:"per_tenant_per_month_usd"`
	MonthlyBill       Estimate `json:"monthly_bill_usd"`

	// ByModel decomposes the average request's cost by model, sorted by cost.
	ByModel []ModelShare `json:"by_model"`

	// WhatIf, when non-empty, describes the what-if knobs applied to the
	// underlying aggregation (set by the caller). Empty means the projection is
	// of the observed cost as-is.
	WhatIf string `json:"what_if,omitempty"`

	Traffic Traffic `json:"-"`
}

// Options tunes the bootstrap. The zero value uses sensible defaults.
type Options struct {
	CILevel          float64 // e.g. 0.95; <=0 or >=1 → default
	BootstrapSamples int     // <=0 → default
	Seed             uint64  // 0 → default (deterministic)
}

func (o Options) withDefaults() Options {
	if o.CILevel <= 0 || o.CILevel >= 1 {
		o.CILevel = defaultCILevel
	}
	if o.BootstrapSamples <= 0 {
		o.BootstrapSamples = defaultBootstrapSamples
	}
	if o.Seed == 0 {
		o.Seed = defaultSeed
	}
	return o
}

// Project turns an aggregation and a traffic profile into projected unit
// economics. It returns an error if the scenario mix references a scenario that
// is not present in the aggregate (an un-grounded weight would silently distort
// the projection).
func Project(res aggregate.Result, traffic Traffic, opts Options) (Projection, error) {
	opts = opts.withDefaults()

	samples := runCostsByScenario(res)
	weights, err := resolveWeights(traffic.ScenarioMix, res)
	if err != nil {
		return Projection{}, err
	}

	// Point estimates over the observed mixture.
	mean, p50, p95 := mixtureStats(samples, weights)

	// Bootstrap CIs for the mean and the p95.
	meanLo, meanHi, p95Lo, p95Hi := bootstrapCI(samples, weights, opts)
	requestMean := Estimate{Value: mean, Lo: meanLo, Hi: meanHi}
	requestP95 := Estimate{Value: p95, Lo: p95Lo, Hi: p95Hi}

	// Aggregates scale linearly from the mean per-request cost.
	reqPerMonth := float64(traffic.Users) * traffic.RequestsPerUserPerDay * float64(traffic.DaysPerMonth)
	perUserPerDay := scale(requestMean, traffic.RequestsPerUserPerDay)
	monthlyBill := scale(requestMean, reqPerMonth)
	perTenantPerMonth := Estimate{}
	if traffic.Tenants > 0 {
		perTenantPerMonth = scale(monthlyBill, 1/float64(traffic.Tenants))
	}

	return Projection{
		SnapshotDate:      res.SnapshotDate,
		CILevel:           opts.CILevel,
		BootstrapSamples:  opts.BootstrapSamples,
		Seed:              opts.Seed,
		RequestMean:       requestMean,
		RequestP50:        p50,
		RequestP95:        requestP95,
		PerUserPerDay:     perUserPerDay,
		PerTenantPerMonth: perTenantPerMonth,
		MonthlyBill:       monthlyBill,
		ByModel:           modelShares(res, weights, mean),
		Traffic:           traffic,
	}, nil
}

// runCostsByScenario extracts the per-run cost sample for each scenario from the
// aggregate's per-run rows.
func runCostsByScenario(res aggregate.Result) map[string][]float64 {
	m := make(map[string][]float64)
	for _, r := range res.Runs {
		m[r.ScenarioID] = append(m[r.ScenarioID], r.CostUSD)
	}
	return m
}

// resolveWeights returns the normalized scenario weights to project with. An
// explicit mix must reference only scenarios present in the aggregate. An empty
// mix defaults to equal weight across every scenario that has at least one run.
func resolveWeights(mix map[string]float64, res aggregate.Result) (map[string]float64, error) {
	present := make(map[string]bool, len(res.Scenarios))
	for _, s := range res.Scenarios {
		present[s.ScenarioID] = true
	}

	if len(mix) == 0 {
		if len(present) == 0 {
			return nil, fmt.Errorf("project: no scenarios in aggregate to project")
		}
		w := 1.0 / float64(len(present))
		out := make(map[string]float64, len(present))
		for id := range present {
			out[id] = w
		}
		return out, nil
	}

	for id := range mix {
		if !present[id] {
			return nil, fmt.Errorf("project: scenario_mix references %q, which is not in the trace", id)
		}
	}
	return mix, nil
}

// wpair is one observation's value and its mixture weight.
type wpair struct {
	value, weight float64
}

// mixtureStats computes the mean, p50, and p95 of the weighted empirical
// mixture: each run cost of scenario s carries weight w_s / n_s, so the scenario
// contributes total weight w_s regardless of how many times it was run.
func mixtureStats(samples map[string][]float64, weights map[string]float64) (mean, p50, p95 float64) {
	var pairs []wpair
	for s, w := range weights {
		xs := samples[s]
		n := len(xs)
		if n == 0 {
			continue
		}
		per := w / float64(n)
		for _, x := range xs {
			pairs = append(pairs, wpair{value: x, weight: per})
			mean += x * per
		}
	}
	if len(pairs) == 0 {
		return 0, 0, 0
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].value < pairs[j].value })
	return mean, weightedQuantileSorted(pairs, 0.50), weightedQuantileSorted(pairs, 0.95)
}

// weightedQuantileSorted returns the inverse of the weighted empirical CDF: the
// smallest observed value whose cumulative weight reaches p of the total. No
// interpolation — the result is always an observed cost, which is what makes a
// projected p95 reconcilable by hand against the trace.
func weightedQuantileSorted(sorted []wpair, p float64) float64 {
	var total float64
	for _, pr := range sorted {
		total += pr.weight
	}
	target := p * total
	var cum float64
	for _, pr := range sorted {
		cum += pr.weight
		// Tolerance absorbs float drift so an exact-boundary p (e.g. cum hits
		// 0.95 precisely) selects the value at the boundary, not the next one.
		if cum >= target-1e-12 {
			return pr.value
		}
	}
	return sorted[len(sorted)-1].value
}

// bootstrapCI resamples runs within each scenario (with replacement) B times,
// recomputes the mixture mean and p95 each time, and returns the CI bounds at
// opts.CILevel. The PRNG is seeded so the interval is reproducible.
func bootstrapCI(samples map[string][]float64, weights map[string]float64, opts Options) (meanLo, meanHi, p95Lo, p95Hi float64) {
	rng := rand.New(rand.NewPCG(opts.Seed, opts.Seed^0x9e3779b97f4a7c15))

	means := make([]float64, opts.BootstrapSamples)
	p95s := make([]float64, opts.BootstrapSamples)

	// Resample only scenarios that carry weight.
	resampled := make(map[string][]float64, len(weights))
	for b := range opts.BootstrapSamples {
		for s := range weights {
			xs := samples[s]
			n := len(xs)
			if n == 0 {
				continue
			}
			rs := resampled[s]
			if cap(rs) < n {
				rs = make([]float64, n)
			}
			rs = rs[:n]
			for i := range rs {
				rs[i] = xs[rng.IntN(n)]
			}
			resampled[s] = rs
		}
		m, _, p := mixtureStats(resampled, weights)
		means[b] = m
		p95s[b] = p
	}

	lo := (1 - opts.CILevel) / 2
	hi := 1 - lo
	sort.Float64s(means)
	sort.Float64s(p95s)
	return percentileSorted(means, lo), percentileSorted(means, hi),
		percentileSorted(p95s, lo), percentileSorted(p95s, hi)
}

// percentileSorted returns the p-quantile (p in 0..1) of a sorted slice using
// R-7 linear interpolation, matching package aggregate's convention.
func percentileSorted(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return 0
	}
	if n == 1 {
		return sorted[0]
	}
	h := float64(n-1) * p
	lo := int(h)
	if float64(lo) == h {
		return sorted[lo]
	}
	frac := h - float64(lo)
	return sorted[lo] + frac*(sorted[lo+1]-sorted[lo])
}

// scale multiplies an estimate (value and both CI bounds) by a constant factor.
// Valid because the aggregates are linear functions of the mean per-request
// cost, so the whole interval scales together.
func scale(e Estimate, factor float64) Estimate {
	return Estimate{Value: e.Value * factor, Lo: e.Lo * factor, Hi: e.Hi * factor}
}

// modelShares decomposes the average request's cost by model: each scenario
// contributes w_s × (its per-run mean cost for that model). Shares sum to the
// mean per-request cost.
func modelShares(res aggregate.Result, weights map[string]float64, requestMean float64) []ModelShare {
	perModel := make(map[string]float64)
	for _, s := range res.Scenarios {
		w, ok := weights[s.ScenarioID]
		if !ok || s.Runs == 0 {
			continue
		}
		for _, m := range s.ByModel {
			perModel[m.Model] += w * (m.CostUSD / float64(s.Runs))
		}
	}

	out := make([]ModelShare, 0, len(perModel))
	for model, cost := range perModel {
		pct := 0.0
		if requestMean > 0 {
			pct = cost / requestMean * 100
		}
		out = append(out, ModelShare{Model: model, PerRequestUSD: cost, Pct: pct})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].PerRequestUSD != out[j].PerRequestUSD {
			return out[i].PerRequestUSD > out[j].PerRequestUSD
		}
		return out[i].Model < out[j].Model
	})
	return out
}
