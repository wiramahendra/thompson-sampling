package thompson

import "sync"

// ArmHealth tracks breaker state per arm.
type armHealth struct {
	consecutiveFailures uint32
	trippedUntilRound   *uint64
}

// CircuitBreaker temporarily excludes failing arms.
type CircuitBreaker struct {
	mu               sync.Mutex
	FailureThreshold uint32
	CooldownRounds   uint64
	health           map[string]*armHealth
}

// NewCircuitBreaker creates a breaker with thresholds.
func NewCircuitBreaker(failureThreshold uint32, cooldownRounds uint64) *CircuitBreaker {
	return &CircuitBreaker{
		FailureThreshold: failureThreshold,
		CooldownRounds:   cooldownRounds,
		health:           make(map[string]*armHealth),
	}
}

// DefaultCircuitBreaker is 3 failures, 100 rounds cooldown.
func DefaultCircuitBreaker() *CircuitBreaker { return NewCircuitBreaker(3, 100) }

// Record updates breaker state. Returns true if tripped this call.
func (cb *CircuitBreaker) Record(id string, outcome Outcome, round uint64) bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	h, ok := cb.health[id]
	if !ok {
		h = &armHealth{}
		cb.health[id] = h
	}
	if !outcome.Success {
		h.consecutiveFailures++
		if h.consecutiveFailures >= cb.FailureThreshold {
			until := round + cb.CooldownRounds
			h.trippedUntilRound = &until
			h.consecutiveFailures = 0
			return true
		}
	} else {
		h.consecutiveFailures = 0
	}
	return false
}

// IsTripped reports whether arm is currently tripped.
func (cb *CircuitBreaker) IsTripped(id string, round uint64) bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	h, ok := cb.health[id]
	if !ok || h.trippedUntilRound == nil {
		return false
	}
	return round < *h.trippedUntilRound
}

// Available filters ids to not-tripped, bypassing filter if all tripped.
func (cb *CircuitBreaker) Available(ids []string, round uint64) []string {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	var out []string
	for _, id := range ids {
		h, ok := cb.health[id]
		if ok && h.trippedUntilRound != nil && round < *h.trippedUntilRound {
			continue
		}
		out = append(out, id)
	}
	if len(out) == 0 {
		return ids
	}
	return out
}
