package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/wiramahendra/thompson-sampling/go/thompson"
)

// DashboardHandler serves policy stats as JSON for UI.
// Mount at /dashboard or /metrics.
func DashboardHandler(policy *thompson.Policy) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		stats := policy.Stats()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"total_pulls": policy.TotalPulls(),
			"sampler":     policy.SamplerName(),
			"arms":        stats,
		})
	})
}

// MetricsHandler serves Prometheus-style text metrics.
func MetricsHandler(policy *thompson.Policy) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		stats := policy.Stats()
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		for _, s := range stats {
			// e.g. thompson_posterior_mean{arm="openai/gpt-4"} 0.82
			w.Write([]byte(formatMetric("thompson_posterior_mean", s.ID, s.PosteriorMean)))
			w.Write([]byte(formatMetric("thompson_pulls", s.ID, float64(s.Pulls))))
			w.Write([]byte(formatMetric("thompson_credible_width", s.ID, s.CredibleWidth)))
		}
	})
}

func formatMetric(name, arm string, value float64) string {
	// Avoid fmt import overhead per call — use simple string builder
	return name + "{arm=\"" + arm + "\"} " + floatToString(value) + "\n"
}

func floatToString(v float64) string { return fmt.Sprintf("%g", v) }
