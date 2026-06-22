// Package aggregate turns a recorded cost trace into per-scenario cost
// distributions: it prices every call against a pricing snapshot, sums calls
// into runs, and summarizes the spread of run costs (mean / p50 / p95 / stdev)
// plus the observed call multiplier. This is Hito 2 — describing what the runs
// actually cost. Projecting that to production scale (× a traffic profile) is
// Hito 3's job and lives in a separate package.
//
// Stats are computed in pure Go rather than via gonum: the v1 summaries are
// elementary, exact, and trivially reconciled by hand (the Hito 2 checkpoint).
// A heavier dependency can be revisited if Hito 3's bootstrap CIs warrant it.
package aggregate

import (
	"math"
	"sort"
)

// Distribution is the descriptive summary of a sample of observations (run
// costs, or calls per run). The percentile convention is linear interpolation
// between closest ranks — the R-7 / NumPy-default method — chosen because it is
// unambiguous to reproduce by hand when reconciling against the raw trace.
type Distribution struct {
	N     int     `json:"n"`
	Mean  float64 `json:"mean"`
	P50   float64 `json:"p50"`
	P95   float64 `json:"p95"`
	Stdev float64 `json:"stdev"` // sample standard deviation (n-1 denominator)
	Min   float64 `json:"min"`
	Max   float64 `json:"max"`
}

// Summarize computes the distribution of xs. An empty sample yields a zero
// Distribution (N == 0); callers treat that as "no data" rather than a value.
func Summarize(xs []float64) Distribution {
	n := len(xs)
	if n == 0 {
		return Distribution{}
	}

	sorted := make([]float64, n)
	copy(sorted, xs)
	sort.Float64s(sorted)

	var sum float64
	for _, x := range sorted {
		sum += x
	}
	mean := sum / float64(n)

	var ss float64
	for _, x := range sorted {
		d := x - mean
		ss += d * d
	}
	stdev := 0.0
	if n > 1 {
		// Sample standard deviation (Bessel's correction): we observe a sample
		// of runs, not the whole population, so divide by n-1.
		stdev = math.Sqrt(ss / float64(n-1))
	}

	return Distribution{
		N:     n,
		Mean:  mean,
		P50:   percentileSorted(sorted, 50),
		P95:   percentileSorted(sorted, 95),
		Stdev: stdev,
		Min:   sorted[0],
		Max:   sorted[n-1],
	}
}

// percentileSorted returns the p-th percentile (0..100) of an already-sorted,
// non-empty slice using R-7 linear interpolation: the rank h = (n-1)·p/100, and
// the result interpolates between the values bracketing h.
func percentileSorted(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 1 {
		return sorted[0]
	}
	h := float64(n-1) * (p / 100)
	lo := int(math.Floor(h))
	hi := int(math.Ceil(h))
	if lo == hi {
		return sorted[lo]
	}
	frac := h - float64(lo)
	return sorted[lo] + frac*(sorted[hi]-sorted[lo])
}
