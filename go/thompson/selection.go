package thompson

import (
	"math"
	"math/rand/v2"
)

// SelectionStrategy is a pluggable arm-selection algorithm.
//
// The built-in [Selection] enum covers the common cases; implement this
// interface to add a custom strategy without forking policy.go and pass it to
// [Policy.SelectWith].
type SelectionStrategy interface {
	Name() string
	Select(rng *rand.Rand, arms map[string]*Arm, order []string, sampler Sampler, totalPulls uint64) string
}

// ThompsonStrategy draws once per arm and takes the argmax.
type ThompsonStrategy struct{}

func (ThompsonStrategy) Name() string { return "thompson" }
func (ThompsonStrategy) Select(rng *rand.Rand, arms map[string]*Arm, order []string, sampler Sampler, _ uint64) string {
	best, bestScore := "", math.Inf(-1)
	for _, id := range order {
		score := sampler.Sample(rng, arms[id].Posterior)
		if best == "" || score > bestScore {
			best, bestScore = id, score
		}
	}
	return best
}

// UCBRegularizedStrategy adds a UCB bonus to under-explored arms.
type UCBRegularizedStrategy struct {
	C          float64
	UntilPulls uint64
}

func (s UCBRegularizedStrategy) Name() string { return "ucb-regularized" }
func (s UCBRegularizedStrategy) Select(rng *rand.Rand, arms map[string]*Arm, order []string, sampler Sampler, totalPulls uint64) string {
	logTotal := math.Log(float64(totalPulls + 1))
	best, bestScore := "", math.Inf(-1)
	for _, id := range order {
		arm := arms[id]
		sample := sampler.Sample(rng, arm.Posterior)
		var score float64
		switch {
		case arm.Posterior.Pulls >= s.UntilPulls:
			score = sample
		case arm.Posterior.Pulls == 0:
			score = math.Inf(1)
		default:
			score = sample + s.C*math.Sqrt(logTotal/float64(arm.Posterior.Pulls))
		}
		if best == "" || score > bestScore {
			best, bestScore = id, score
		}
	}
	return best
}

// PhasedStrategy round-robins to a quota, then Thompson sampling.
type PhasedStrategy struct {
	Bootstrap          uint64
	MinPullsForExploit uint64
}

func (s PhasedStrategy) Name() string { return "phased" }
func (s PhasedStrategy) Select(rng *rand.Rand, arms map[string]*Arm, order []string, sampler Sampler, totalPulls uint64) string {
	quota := s.Bootstrap
	if s.MinPullsForExploit > quota {
		quota = s.MinPullsForExploit
	}
	best, bestPulls := "", uint64(math.MaxUint64)
	for _, id := range order {
		pulls := arms[id].Posterior.Pulls
		if pulls < quota && pulls < bestPulls {
			best, bestPulls = id, pulls
		}
	}
	if best != "" {
		return best
	}
	return ThompsonStrategy{}.Select(rng, arms, order, sampler, totalPulls)
}
