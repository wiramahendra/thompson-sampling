package thompson

import (
	"fmt"
	"math"
	"testing"
)

// Probe: does priorFor pick a deterministic neighbour when means tie?
func TestScratchPriorForDeterminism(t *testing.T) {
	seen := map[string]int{}
	for i := 0; i < 200; i++ {
		p := NewDefault()
		// Two same-provider, same-family arms with identical posterior means
		// but different evidence weight.
		p.AddArmWithPrior("openai/gpt-4-a", InformedPrior{Alpha: 5, Beta: 5})
		p.AddArmWithPrior("openai/gpt-4-b", InformedPrior{Alpha: 50, Beta: 50})
		// Give both real pulls so they are eligible to lend.
		p.arms["openai/gpt-4-a"].Posterior.Pulls = 8
		p.arms["openai/gpt-4-b"].Posterior.Pulls = 98
		prior, _ := p.AddArm("openai/gpt-4.5-turbo")
		seen[fmt.Sprintf("%v/%v", prior.Alpha, prior.Beta)]++
	}
	t.Logf("distinct priors observed: %v", seen)
	if len(seen) > 1 {
		t.Errorf("priorFor is nondeterministic across runs: %v", seen)
	}
}

// Probe: Inf latency / Inf cost through the Go reward.
func TestScratchGoRewardInf(t *testing.T) {
	rp := DefaultRewardPolicy()
	rp.FailureIsZero = false
	inf := math.Inf(1)
	o := NewOutcome(inf, true, inf)
	t.Logf("go reward with +Inf latency and cost: %v", rp.Reward(o))
	t.Logf("go rampDown(+Inf) = %v", rampDown(inf, 500, 10000))
	t.Logf("go reward with NaN quality: %v", rp.Reward(NewOutcome(100, true, 0).WithQuality(math.NaN())))
}
