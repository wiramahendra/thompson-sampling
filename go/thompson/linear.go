package thompson

// LinearWeights is a stub for linear contextual bandit.
type LinearWeights struct {
	Weights []float64
}

func NewLinearWeights(dim int) *LinearWeights {
	return &LinearWeights{Weights: make([]float64, dim)}
}

func (lw *LinearWeights) Score(features []float64) float64 {
	sum := 0.0
	for i, w := range lw.Weights {
		if i < len(features) {
			sum += w * features[i]
		}
	}
	if sum < 0 {
		return 0
	}
	if sum > 1 {
		return 1
	}
	return sum
}

// LinearPolicy wraps Posterior with shared linear weights.
type LinearPolicy struct {
	Dim     int
	Weights *LinearWeights
}

func NewLinearPolicy(dim int) *LinearPolicy {
	return &LinearPolicy{Dim: dim, Weights: NewLinearWeights(dim)}
}

func (lp *LinearPolicy) AdjustedMean(p Posterior, features []float64) float64 {
	base := p.Mean()
	ctx := lp.Weights.Score(features)
	v := base*0.7 + ctx*0.3
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
