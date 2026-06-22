package main

import (
	"flag"
	"fmt"
	"strings"

	"augur/aggregate"
)

// knobFlags registers the shared what-if multiplier flags on fs and returns
// accessors. They let `project` and `gate` re-cost a recorded trace under
// hypothetical agentic cost drivers without re-running the agent.
type knobFlags struct {
	retry  *float64
	fanout *float64
	growth *float64
}

func addKnobFlags(fs *flag.FlagSet) knobFlags {
	return knobFlags{
		retry:  fs.Float64("retry-rate", 0, "what-if: extra fraction of calls retried (0.2 = 20% more calls)"),
		fanout: fs.Float64("fanout", 1, "what-if: multiplier on call count from sub-agent/tool fan-out"),
		growth: fs.Float64("context-growth", 1, "what-if: multiplier on prompt cost from history growth"),
	}
}

func (k knobFlags) knobs() aggregate.Knobs {
	return aggregate.Knobs{
		RetryRate:     *k.retry,
		FanoutFactor:  *k.fanout,
		ContextGrowth: *k.growth,
	}
}

// describeKnobs renders a one-line summary of non-identity knobs (empty for the
// observed-as-is case), for the projection banner and the gate report.
func describeKnobs(k aggregate.Knobs) string {
	if k.IsIdentity() {
		return ""
	}
	var parts []string
	if k.RetryRate != 0 {
		parts = append(parts, fmt.Sprintf("retry +%.0f%%", k.RetryRate*100))
	}
	if k.FanoutFactor != 0 && k.FanoutFactor != 1 {
		parts = append(parts, fmt.Sprintf("fanout ×%g", k.FanoutFactor))
	}
	if k.ContextGrowth != 0 && k.ContextGrowth != 1 {
		parts = append(parts, fmt.Sprintf("context ×%g", k.ContextGrowth))
	}
	return strings.Join(parts, ", ")
}
