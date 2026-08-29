package thompson

import (
	"fmt"
	"math"
)

// DiscountPolicy controls how posteriors age.
//
// The effective memory is 1/(1-factor) observations; stationary (factor == 0
// or 1) has infinite memory.
type DiscountPolicy interface {
	Factor() float64
	Apply(p *Posterior)
	EffectiveMemory() float64
	Label() string
}

// FixedDiscount is a constant per-round factor in (0,1). Zero or 1 means stationary.
type FixedDiscount struct {
	FactorValue float64
}

// NewFixedDiscount clamps to stationary when outside (0,1).
func NewFixedDiscount(factor float64) FixedDiscount {
	if factor <= 0 || factor >= 1 || math.IsNaN(factor) || math.IsInf(factor, 0) {
		return FixedDiscount{FactorValue: 0}
	}
	return FixedDiscount{FactorValue: factor}
}

func (d FixedDiscount) Factor() float64 { return d.FactorValue }

func (d FixedDiscount) Apply(p *Posterior) {
	if d.FactorValue > 0 && d.FactorValue < 1 {
		p.Discount(d.FactorValue)
	}
}

func (d FixedDiscount) EffectiveMemory() float64 {
	if d.FactorValue <= 0 || d.FactorValue >= 1 {
		return math.Inf(1)
	}
	return 1 / (1 - d.FactorValue)
}

func (d FixedDiscount) Label() string {
	if d.FactorValue <= 0 || d.FactorValue >= 1 {
		return "none"
	}
	return fmt.Sprintf("%g", d.FactorValue)
}
