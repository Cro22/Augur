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
	"strings"
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
	// CacheWrite is USD per Mtok for prompt tokens WRITTEN to the provider cache
	// (Anthropic's cache_creation_input_tokens), typically a premium over Input.
	// When a provider does not bill cache writes separately this equals Input;
	// LoadPricing fills it in that way when the field is omitted.
	CacheWrite float64
}

// Usage is the token accounting for a single LLM call, mirroring how providers
// report it. CachedTokens and CacheWriteTokens are both SUBSETS of InputTokens
// (portions of the prompt), not additional buckets — this matches OpenAI's
// prompt_tokens / prompt_tokens_details.cached_tokens and Anthropic's
// cache_read_input_tokens / cache_creation_input_tokens. Billing therefore
// splits InputTokens into a cached-read part, a cache-write part, and a
// full-price remainder.
type Usage struct {
	InputTokens      int // total prompt tokens, INCLUDING the cached and cache-write portions
	OutputTokens     int // completion tokens
	CachedTokens     int // cached-read prompt tokens, billed at the cached rate
	CacheWriteTokens int // cache-write prompt tokens, billed at the cache-write rate
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
	case u.CacheWriteTokens < 0:
		return fmt.Errorf("cost: negative cache-write tokens (%d)", u.CacheWriteTokens)
	case u.CachedTokens+u.CacheWriteTokens > u.InputTokens:
		return fmt.Errorf("cost: cached (%d) + cache-write (%d) tokens exceed input tokens (%d)",
			u.CachedTokens, u.CacheWriteTokens, u.InputTokens)
	}
	return nil
}

// Breakdown is a single call's cost split into its three components. It exists
// so callers (the what-if knobs) can scale the prompt side independently of the
// completion side — context growth inflates input, not output.
type Breakdown struct {
	// InputUSD is the cost of the non-cached prompt tokens.
	InputUSD float64
	// CachedUSD is the cost of the cached-read prompt tokens.
	CachedUSD float64
	// CacheWriteUSD is the cost of the cache-write prompt tokens.
	CacheWriteUSD float64
	// OutputUSD is the cost of the completion tokens.
	OutputUSD float64
}

// Total is the full call cost: the four components summed.
func (b Breakdown) Total() float64 {
	return b.InputUSD + b.CachedUSD + b.CacheWriteUSD + b.OutputUSD
}

// PromptUSD is the cost attributable to the prompt (input + cached + cache
// write) — the part that scales with context growth.
func (b Breakdown) PromptUSD() float64 { return b.InputUSD + b.CachedUSD + b.CacheWriteUSD }

// Breakdown returns the per-component cost of a single call priced at p. The
// cached-read portion of the prompt is billed at CachedInput, the cache-write
// portion at CacheWrite, the remainder at Input, and completion tokens at
// Output. It errors if the usage is invalid.
func (p ModelPrice) Breakdown(u Usage) (Breakdown, error) {
	if err := u.Validate(); err != nil {
		return Breakdown{}, err
	}
	fullInput := u.InputTokens - u.CachedTokens - u.CacheWriteTokens
	return Breakdown{
		InputUSD:      float64(fullInput) / tokensPerMtok * p.Input,
		CachedUSD:     float64(u.CachedTokens) / tokensPerMtok * p.CachedInput,
		CacheWriteUSD: float64(u.CacheWriteTokens) / tokensPerMtok * p.CacheWrite,
		OutputUSD:     float64(u.OutputTokens) / tokensPerMtok * p.Output,
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
	// PrefixFallback, when true, lets Resolve match a model that is absent from
	// the snapshot to the longest snapshot key that is a dash-delimited prefix of
	// it — so a dated or suffixed variant (e.g. "gpt-4o-2024-08-06") falls back to
	// its base entry ("gpt-4o") instead of failing. It is OFF by default because a
	// silent fallback can mis-bill; callers that enable it should surface which
	// models were normalized (Resolve reports the matched key so they can warn).
	PrefixFallback bool
}

// Price returns the exact price for a model and whether it is known. It does not
// apply PrefixFallback — use Resolve for that.
func (p Pricing) Price(model string) (ModelPrice, bool) {
	mp, ok := p.Models[model]
	return mp, ok
}

// Resolve looks up the price for a model, returning the snapshot key it matched
// (the canonical model), its price, and whether a match was found. An exact
// match returns the model unchanged. When PrefixFallback is on and there is no
// exact match, it returns the longest snapshot key that is a dash-delimited
// prefix of the model (so callers can detect normalization by comparing the
// returned canonical to the requested model).
func (p Pricing) Resolve(model string) (canonical string, mp ModelPrice, ok bool) {
	if mp, ok := p.Models[model]; ok {
		return model, mp, true
	}
	if !p.PrefixFallback {
		return "", ModelPrice{}, false
	}
	best := ""
	for key := range p.Models {
		if isModelPrefix(model, key) && len(key) > len(best) {
			best = key
		}
	}
	if best == "" {
		return "", ModelPrice{}, false
	}
	return best, p.Models[best], true
}

// isModelPrefix reports whether key is a prefix of model at a dash boundary, so
// "gpt-4o" matches "gpt-4o" and "gpt-4o-2024-08-06" but never "gpt-4omini". The
// boundary rule keeps a base name from swallowing an unrelated one that merely
// shares a leading substring.
func isModelPrefix(model, key string) bool {
	if !strings.HasPrefix(model, key) {
		return false
	}
	return len(model) == len(key) || model[len(key)] == '-'
}

// Cost computes the cost of a single call for the named model. It applies
// PrefixFallback via Resolve and wraps ErrUnknownModel (so callers can
// errors.Is it) when no model matches.
func (p Pricing) Cost(model string, u Usage) (float64, error) {
	_, mp, ok := p.Resolve(model)
	if !ok {
		return 0, fmt.Errorf("%w: %q", ErrUnknownModel, model)
	}
	return mp.Cost(u)
}

// Breakdown computes the per-component cost of a single call for the named
// model, applying PrefixFallback via Resolve and wrapping ErrUnknownModel when
// no model matches.
func (p Pricing) Breakdown(model string, u Usage) (Breakdown, error) {
	_, mp, ok := p.Resolve(model)
	if !ok {
		return Breakdown{}, fmt.Errorf("%w: %q", ErrUnknownModel, model)
	}
	return mp.Breakdown(u)
}
