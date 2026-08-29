// Package harness — trace replay for thin-waist validation.
package harness

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

// TraceRecord is one JSONL line from an instrumented gateway.
type TraceRecord struct {
	ID        string   `json:"id"`
	T         *int     `json:"t,omitempty"`
	Reward    *float64 `json:"reward,omitempty"`
	LatencyMs *float64 `json:"latency_ms,omitempty"`
	Success   *bool    `json:"success,omitempty"`
	CostUSD   *float64 `json:"cost_usd,omitempty"`
	Quality   *float64 `json:"quality,omitempty"`
}

// LoadTrace reads JSONL from path.
func LoadTrace(path string) ([]TraceRecord, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []TraceRecord
	scanner := bufio.NewScanner(f)
	lineno := 0
	for scanner.Scan() {
		lineno++
		line := scanner.Text()
		if len(line) == 0 {
			continue
		}
		// skip whitespace-only lines
		isSpace := true
		for _, c := range line {
			if c != ' ' && c != '\t' && c != '\r' && c != '\n' {
				isSpace = false
				break
			}
		}
		if isSpace {
			continue
		}
		var rec TraceRecord
		if err := json.Unmarshal([]byte(line), &rec); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, lineno, err)
		}
		out = append(out, rec)
	}
	return out, scanner.Err()
}

// TraceReplay replays rewards deterministically by (t, id).
type TraceReplay struct {
	byRound  []map[string]float64
	scenario Scenario
}

// NewTraceReplay builds a replay from records. Missing t is file order.
func NewTraceReplay(records []TraceRecord) (*TraceReplay, error) {
	if len(records) == 0 {
		return nil, fmt.Errorf("empty trace")
	}
	// assign t if missing
	for i := range records {
		if records[i].T == nil {
			t := i
			records[i].T = &t
		}
		if records[i].Reward != nil && (!isFinite(*records[i].Reward)) {
			return nil, fmt.Errorf("non-finite reward at t=%d id=%s", *records[i].T, records[i].ID)
		}
	}
	maxT := 0
	ids := make(map[string]struct{})
	for _, r := range records {
		if *r.T > maxT {
			maxT = *r.T
		}
		ids[r.ID] = struct{}{}
	}
	byRound := make([]map[string]float64, maxT+1)
	for i := range byRound {
		byRound[i] = make(map[string]float64)
	}
	for _, r := range records {
		v := 0.5
		if r.Reward != nil {
			v = clamp01(*r.Reward)
		}
		byRound[*r.T][r.ID] = v
	}
	arms := make([]ArmSpec, 0, len(ids))
	for id := range ids {
		arms = append(arms, Fixed(id, 0.5))
	}
	scenario := Scenario{
		Name:        "trace",
		Description: "Exact trace replay — reward is file, not draw",
		Arms:        arms,
		Horizon:     len(byRound),
		RewardKind:  Bernoulli,
	}
	return &TraceReplay{byRound: byRound, scenario: scenario}, nil
}

func (tr *TraceReplay) Scenario() *Scenario { return &tr.scenario }

func (tr *TraceReplay) Draw(id string, t int) float64 {
	if t < 0 || t >= len(tr.byRound) {
		return 0
	}
	if v, ok := tr.byRound[t][id]; ok {
		return v
	}
	return 0
}

func (tr *TraceReplay) Horizon() int { return len(tr.byRound) }

func clamp01(v float64) float64 {
	if v != v {
		return 0
	}
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func isFinite(v float64) bool { return v == v && v > -1e308 && v < 1e308 }
