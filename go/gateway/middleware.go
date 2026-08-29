// Package gateway provides thin-waist HTTP middleware for LLM routing.
//
// Two calls integrate any gateway (LiteLLM, Portkey, custom):
//
//	provider, _ := policy.Select(rng)
//	// forward request to provider
//	policy.RecordOutcome(rng, provider, outcome)
//
// This middleware wraps an http.Handler, selecting a provider per request and
// recording Outcome via thompson.Observer for OTEL/Prometheus.
package gateway

import (
	"math/rand/v2"
	"net/http"
	"time"

	"github.com/wiramahendra/thompson-sampling/go/thompson"
)

// Middleware selects a provider via Policy and records latency/cost/quality.
// `forward` maps provider id -> upstream URL; return 502 if unknown.
type Middleware struct {
	Policy  *thompson.Policy
	Forward func(provider string, w http.ResponseWriter, r *http.Request) (thompson.Outcome, error)
}

// Handler returns an http.Handler that does Select -> Forward -> Record.
func (m *Middleware) Handler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if next != nil && m.Policy.Len() == 0 {
			next.ServeHTTP(w, r)
			return
		}
		rng := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 0))
		provider, err := m.Policy.Select(rng)
		if err != nil {
			http.Error(w, err.Error(), http.StatusServiceUnavailable)
			return
		}
		start := time.Now()
		outcome, fwdErr := m.Forward(provider, w, r)
		// If forwarder already wrote response, just record
		if outcome.LatencyMs == 0 {
			outcome.LatencyMs = float64(time.Since(start).Milliseconds())
		}
		// Record even on failure — failure_is_zero scores 0
		_ = m.Policy.RecordOutcome(rng, provider, outcome)
		if fwdErr != nil {
			// already written? ensure error surface
			if w.Header().Get("Content-Type") == "" {
				http.Error(w, fwdErr.Error(), http.StatusBadGateway)
			}
			return
		}
	})
}

// NewMiddleware creates a Middleware with default Exact sampler.
func NewMiddleware(policy *thompson.Policy, forward func(string, http.ResponseWriter, *http.Request) (thompson.Outcome, error)) *Middleware {
	return &Middleware{Policy: policy, Forward: forward}
}
