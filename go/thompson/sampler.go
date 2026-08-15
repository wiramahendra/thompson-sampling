package thompson

import (
	"math"
	"math/rand/v2"
)

func sqrt(v float64) float64 { return math.Sqrt(v) }

// Sampler draws a value in [0, 1] standing in for a sample from a Beta
// posterior. Implementations must return a finite value in range.
type Sampler interface {
	Name() string
	Sample(rng *rand.Rand, p Posterior) float64
}

// ExactSampler draws exactly, via two Gamma variates: X / (X + Y) where
// X ~ Gamma(alpha, 1) and Y ~ Gamma(beta, 1).
//
// Gamma draws use Marsaglia & Tsang (2000), with the boosting identity
// Gamma(k) = Gamma(k+1) * U^(1/k) for shapes below 1. This is the reference
// implementation; every other sampler here is measured against it.
type ExactSampler struct{}

// Name implements Sampler.
func (ExactSampler) Name() string { return "exact" }

// Sample implements Sampler.
func (s ExactSampler) Sample(rng *rand.Rand, p Posterior) float64 {
	x := SampleGamma(rng, p.Alpha)
	y := SampleGamma(rng, p.Beta)
	denom := x + y
	if denom <= 0 || math.IsInf(denom, 0) || math.IsNaN(denom) {
		// Both draws underflowed, which needs both shapes far below 1.
		return p.Mean()
	}
	return x / denom
}

// SampleGamma draws from Gamma(shape, 1).
//
// The transform constant c = 1/sqrt(9d) and the squeeze constant 0.0331 are
// jointly tuned. Changing either without the other silently biases the output
// while leaving it superficially plausible — see docs/FINDINGS.md.
func SampleGamma(rng *rand.Rand, shape float64) float64 {
	if shape <= 0 || math.IsNaN(shape) {
		return 0
	}

	if shape < 1 {
		u := openUnit(rng)
		return SampleGamma(rng, shape+1) * math.Pow(u, 1/shape)
	}

	d := shape - 1.0/3.0
	c := 1 / math.Sqrt(9*d)

	for {
		x := rng.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v3 := v * v * v
		u := openUnit(rng)

		if u < 1-0.0331*x*x*x*x {
			return d * v3
		}
		if math.Log(u) < 0.5*x*x+d*(1-v3+math.Log(v3)) {
			return d * v3
		}
	}
}

// openUnit returns a uniform draw in (0, 1], excluding zero so Log stays finite.
func openUnit(rng *rand.Rand) float64 {
	u := rng.Float64()
	if u <= 0 {
		return math.SmallestNonzeroFloat64
	}
	return u
}

// MeanPlusGaussianSampler returns the posterior mean plus Gaussian noise scaled
// by the posterior standard deviation, clamped to [0, 1].
//
// It matches the Beta's first two moments and does shrink exploration as the
// posterior concentrates, which makes it the most defensible approximation
// here. It misses the Beta's skew, which is worst exactly where it matters:
// Beta(1,1) is flat, but this draws a bell around 0.5 and clamps the tails, so
// a never-tried arm is under-explored.
type MeanPlusGaussianSampler struct{}

// Name implements Sampler.
func (MeanPlusGaussianSampler) Name() string { return "mean+gaussian" }

// Sample implements Sampler.
func (MeanPlusGaussianSampler) Sample(rng *rand.Rand, p Posterior) float64 {
	return clamp01(p.Mean() + rng.NormFloat64()*p.StdDev())
}

// MeanPlusUniformSampler returns the posterior mean plus uniform noise of fixed
// half-width, clamped to [0, 1].
//
// The noise does not depend on the posterior at all. An arm with two
// observations and an arm with two thousand get identical exploration pressure,
// so the policy neither converges nor concentrates its exploration where the
// uncertainty is.
type MeanPlusUniformSampler struct {
	HalfWidth float64
}

// Name implements Sampler.
func (MeanPlusUniformSampler) Name() string { return "mean+uniform" }

// Sample implements Sampler.
func (s MeanPlusUniformSampler) Sample(rng *rand.Rand, p Posterior) float64 {
	w := s.HalfWidth
	if w <= 0 {
		w = 0.1
	}
	return clamp01(p.Mean() + (rng.Float64()-0.5)*2*w)
}

// DeterministicSampler shifts the posterior mean by a fixed multiple of its
// standard deviation, with no randomness at all.
//
// Included because it is what a widely-copied implementation actually does. It
// is a fixed function of (Alpha, Beta, Pulls), so an argmax over it performs no
// exploration whatsoever. It is the null treatment.
type DeterministicSampler struct{}

// Name implements Sampler.
func (DeterministicSampler) Name() string { return "deterministic" }

// Sample implements Sampler.
func (DeterministicSampler) Sample(_ *rand.Rand, p Posterior) float64 {
	factor := 0.1
	switch {
	case p.Pulls < 5:
		factor = 2.0
	case p.Pulls < 20:
		factor = 1.0
	}
	return clamp01(p.Mean() + p.StdDev()*0.1*factor)
}

// ConcentrationSwitchedSampler dispatches between two samplers on posterior
// concentration, reproducing routers that special-case "enough data".
//
// The cheap branch governs precisely the early rounds where selection quality
// determines total regret.
type ConcentrationSwitchedSampler struct {
	Threshold    float64
	Diffuse      Sampler
	Concentrated Sampler
}

// ProductionSwitched returns the configuration observed in production: uniform
// noise of half-width 0.1 below Alpha, Beta = 100, Gaussian above.
func ProductionSwitched() ConcentrationSwitchedSampler {
	return ConcentrationSwitchedSampler{
		Threshold:    100,
		Diffuse:      MeanPlusUniformSampler{HalfWidth: 0.1},
		Concentrated: MeanPlusGaussianSampler{},
	}
}

// Name implements Sampler.
func (ConcentrationSwitchedSampler) Name() string { return "concentration-switched" }

// Sample implements Sampler.
func (s ConcentrationSwitchedSampler) Sample(rng *rand.Rand, p Posterior) float64 {
	if p.Alpha > s.Threshold && p.Beta > s.Threshold {
		return s.Concentrated.Sample(rng, p)
	}
	return s.Diffuse.Sample(rng, p)
}

func clamp01(v float64) float64 {
	if math.IsNaN(v) {
		return 0
	}
	return math.Max(0, math.Min(1, v))
}
