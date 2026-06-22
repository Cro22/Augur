// Package cost computes the dollar cost of a single LLM call from a pricing
// snapshot. It is the deterministic foundation the rest of Augur builds on:
// the proxy records token usage, and everything downstream (distributions,
// projections, the budget gate) is just arithmetic on top of Cost.
//
// The package is pure — no I/O, no clock, no global state — so it can be
// exhaustively unit-tested. Loading pricing from disk lives in pricing.go.
package cost

import (
	"errors"
	"fmt"
)

// tokensPerMtok is the denominator that turns a per-million-token price into a
// per-token price. Prices in pricing.yaml are quoted per Mtok.
const tokensPerMtok = 1_000_000.0

// ErrUnknownModel is returned by Pricing.Cost when a call references a model
// that is not present in the loaded pricing snapshot. Callers should treat this
// as a hard failure: an un-priced call means an un-knowable bill, which is
// exactly the surprise Augur exists to prevent.
var ErrUnknownModel = errors.New("cost: unknown model")

// ModelPrice is the per-Mtok price of one model. Zero values are valid (some
// models genuinely cost nothing for a dimension), so absence of a model is
// signalled by Pricing's map, not by a zero ModelPrice.
type ModelPrice struct {
	// Input is USD per Mtok for non-cached prompt tokens.
	Input float64
	// Output is USD per Mtok for completion tokens.
	Output float64
	// CachedInput is USD per Mtok for prompt tokens served from the provider
	// cache. When a model has no cache discount this should equal Input;
	// LoadPricing fills it in that way when the field is omitted.
	CachedInput float64
}

// Usage is the token accounting for a single LLM call, mirroring how providers
// report it. CachedTokens is a SUBSET of InputTokens (the cached portion of the
// prompt), not an additional bucket — this matches OpenAI's
// prompt_tokens / prompt_tokens_details.cached_tokens and Anthropic's
// cache_read_input_tokens. Billing therefore splits InputTokens into a cached
// part and a full-price part.
type Usage struct {
	InputTokens  int // total prompt tokens, INCLUDING the cached portion
	OutputTokens int // completion tokens
	CachedTokens int // cached prompt tokens, billed at the cached rate
}

// Validate reports whether the usage is internally consistent. Negative counts
// are nonsensical, and more cached tokens than input tokens means the trace is
// corrupt — we refuse to invent a number from bad input rather than silently
// under-bill.
func (u Usage) Validate() error {
	switch {
	case u.InputTokens < 0:
		return fmt.Errorf("cost: negative input tokens (%d)", u.InputTokens)
	case u.OutputTokens < 0:
		return fmt.Errorf("cost: negative output tokens (%d)", u.OutputTokens)
	case u.CachedTokens < 0:
		return fmt.Errorf("cost: negative cached tokens (%d)", u.CachedTokens)
	case u.CachedTokens > u.InputTokens:
		return fmt.Errorf("cost: cached tokens (%d) exceed input tokens (%d)",
			u.CachedTokens, u.InputTokens)
	}
	return nil
}

// Breakdown is a single call's cost split into its three components. It exists
// so callers (the what-if knobs) can scale the prompt side independently of the
// completion side — context growth inflates input, not output.
type Breakdown struct {
	// InputUSD is the cost of the non-cached prompt tokens.
	InputUSD float64
	// CachedUSD is the cost of the cached prompt tokens.
	CachedUSD float64
	// OutputUSD is the cost of the completion tokens.
	OutputUSD float64
}

// Total is the full call cost: the three components summed.
func (b Breakdown) Total() float64 { return b.InputUSD + b.CachedUSD + b.OutputUSD }

// PromptUSD is the cost attributable to the prompt (input + cached) — the part
// that scales with context growth.
func (b Breakdown) PromptUSD() float64 { return b.InputUSD + b.CachedUSD }

// Breakdown returns the per-component cost of a single call priced at p. The
// cached portion of the prompt is billed at CachedInput, the remainder at
// Input, and completion tokens at Output. It errors if the usage is invalid.
func (p ModelPrice) Breakdown(u Usage) (Breakdown, error) {
	if err := u.Validate(); err != nil {
		return Breakdown{}, err
	}
	fullInput := u.InputTokens - u.CachedTokens
	return Breakdown{
		InputUSD:  float64(fullInput) / tokensPerMtok * p.Input,
		CachedUSD: float64(u.CachedTokens) / tokensPerMtok * p.CachedInput,
		OutputUSD: float64(u.OutputTokens) / tokensPerMtok * p.Output,
	}, nil
}

// Cost returns the USD cost of a single call priced at p. It returns an error if
// the usage is invalid (see Validate).
func (p ModelPrice) Cost(u Usage) (float64, error) {
	b, err := p.Breakdown(u)
	if err != nil {
		return 0, err
	}
	return b.Total(), nil
}

// Pricing is a loaded pricing snapshot: a set of per-model prices plus the date
// it was captured, kept for reporting so a projection can state which snapshot
// it was computed against.
type Pricing struct {
	SnapshotDate string
	Models       map[string]ModelPrice
}

// Price returns the price for a model and whether it is known.
func (p Pricing) Price(model string) (ModelPrice, bool) {
	mp, ok := p.Models[model]
	return mp, ok
}

// Cost computes the cost of a single call for the named model. It wraps
// ErrUnknownModel (so callers can errors.Is it) when the model is absent.
func (p Pricing) Cost(model string, u Usage) (float64, error) {
	mp, ok := p.Models[model]
	if !ok {
		return 0, fmt.Errorf("%w: %q", ErrUnknownModel, model)
	}
	return mp.Cost(u)
}

// Breakdown computes the per-component cost of a single call for the named
// model, wrapping ErrUnknownModel when the model is absent.
func (p Pricing) Breakdown(model string, u Usage) (Breakdown, error) {
	mp, ok := p.Models[model]
	if !ok {
		return Breakdown{}, fmt.Errorf("%w: %q", ErrUnknownModel, model)
	}
	return mp.Breakdown(u)
}
