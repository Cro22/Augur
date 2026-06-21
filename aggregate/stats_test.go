package aggregate

import (
	"math"
	"testing"
)

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestSummarize(t *testing.T) {
	// Sample: 1,2,3,4,5 (n=5).
	//   mean = 3
	//   sample stdev = sqrt(((2^2+1^2+0+1^2+2^2)/4)) = sqrt(10/4) = sqrt(2.5)
	//   p50 (R-7): h=(5-1)*0.5=2 → sorted[2]=3
	//   p95 (R-7): h=(5-1)*0.95=3.8 → sorted[3] + 0.8*(sorted[4]-sorted[3])
	//              = 4 + 0.8*(5-4) = 4.8
	d := Summarize([]float64{3, 1, 4, 5, 2}) // deliberately unsorted
	if d.N != 5 {
		t.Errorf("N = %d, want 5", d.N)
	}
	if !approx(d.Mean, 3) {
		t.Errorf("Mean = %v, want 3", d.Mean)
	}
	if !approx(d.Stdev, math.Sqrt(2.5)) {
		t.Errorf("Stdev = %v, want sqrt(2.5)=%v", d.Stdev, math.Sqrt(2.5))
	}
	if !approx(d.P50, 3) {
		t.Errorf("P50 = %v, want 3", d.P50)
	}
	if !approx(d.P95, 4.8) {
		t.Errorf("P95 = %v, want 4.8", d.P95)
	}
	if d.Min != 1 || d.Max != 5 {
		t.Errorf("Min/Max = %v/%v, want 1/5", d.Min, d.Max)
	}
}

func TestSummarizeEmpty(t *testing.T) {
	d := Summarize(nil)
	if d != (Distribution{}) {
		t.Errorf("empty sample = %+v, want zero Distribution", d)
	}
}

func TestSummarizeSingle(t *testing.T) {
	d := Summarize([]float64{42})
	if d.N != 1 || d.Mean != 42 || d.P50 != 42 || d.P95 != 42 || d.Stdev != 0 || d.Min != 42 || d.Max != 42 {
		t.Errorf("single-value distribution = %+v", d)
	}
}

func TestPercentileInterpolates(t *testing.T) {
	// Even count: 10,20,30,40.
	//   p50 (R-7): h=(4-1)*0.5=1.5 → sorted[1] + 0.5*(sorted[2]-sorted[1]) = 20+0.5*10 = 25
	sorted := []float64{10, 20, 30, 40}
	if got := percentileSorted(sorted, 50); !approx(got, 25) {
		t.Errorf("p50 = %v, want 25", got)
	}
	// p0 and p100 are the extremes.
	if got := percentileSorted(sorted, 0); got != 10 {
		t.Errorf("p0 = %v, want 10", got)
	}
	if got := percentileSorted(sorted, 100); got != 40 {
		t.Errorf("p100 = %v, want 40", got)
	}
}
