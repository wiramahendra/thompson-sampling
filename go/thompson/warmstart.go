package thompson

import "strings"

// WarmStartKind selects how a newly-added arm is initialised.
type WarmStartKind int

const (
	// ColdStart always begins at Beta(1, 1).
	ColdStart WarmStartKind = iota
	// FixedPrior always begins from the same prior.
	FixedPrior
	// FamilySimilarity inherits from the best-performing arm in the same model
	// family, falling back to the same provider, then to Fallback.
	FamilySimilarity
)

// InformedPrior is a prior expressed as Beta parameters.
type InformedPrior struct {
	Alpha float64 `json:"alpha"`
	Beta  float64 `json:"beta"`
}

// NewInformedPrior clamps both parameters to at least 1.
func NewInformedPrior(alpha, beta float64) InformedPrior {
	if alpha < 1 {
		alpha = 1
	}
	if beta < 1 {
		beta = 1
	}
	return InformedPrior{Alpha: alpha, Beta: beta}
}

// DefaultPrior is mildly optimistic and weakly held: mean 0.8 with the weight
// of about five observations. Strong enough to get a new arm tried, weak enough
// that a handful of real failures overrides it.
func DefaultPrior() InformedPrior { return InformedPrior{Alpha: 4, Beta: 1} }

// Posterior returns the posterior this prior implies.
func (p InformedPrior) Posterior() Posterior {
	return Posterior{Alpha: p.Alpha, Beta: p.Beta}
}

// WarmStart configures how new arms are initialised.
type WarmStart struct {
	Kind WarmStartKind
	// Fixed is used when Kind is FixedPrior.
	Fixed InformedPrior
	// Discount is the fraction of a neighbour's evidence carried over when Kind
	// is FamilySimilarity.
	//
	// At 1.0 the new arm inherits the neighbour's full confidence, which is
	// wrong: it is a different model. Values near 0.2 transfer the location of
	// the estimate while staying loose enough for real observations to dominate.
	Discount float64
	// Fallback is used when no related arm exists.
	Fallback InformedPrior
}

// DefaultWarmStart returns family-similarity inheritance at a 0.2 discount.
func DefaultWarmStart() WarmStart {
	return WarmStart{
		Kind:     FamilySimilarity,
		Discount: 0.2,
		Fallback: DefaultPrior(),
	}
}

// ModelID is a parsed arm identifier of the form provider/model.
type ModelID struct {
	Provider string
	Model    string
	Family   string
}

// ParseModelID splits provider/model and derives the family.
func ParseModelID(id string) ModelID {
	provider, model, found := strings.Cut(id, "/")
	if !found {
		model = ""
	}
	return ModelID{Provider: provider, Model: model, Family: ModelFamily(model)}
}

// ModelFamily reduces a model name to its family by truncating at the major
// version: the leading run of non-numeric tokens is the name, the first token
// starting with a digit contributes its integer part, anything after is a
// variant.
//
//	gpt-4.5-turbo     -> gpt-4
//	claude-3-5-sonnet -> claude-3
//	claude-sonnet-4-5 -> claude-sonnet-4
//	mistral-large     -> mistral-large
func ModelFamily(model string) string {
	var name []string
	for _, token := range strings.Split(model, "-") {
		if len(token) > 0 && token[0] >= '0' && token[0] <= '9' {
			major := token
			if i := strings.IndexByte(token, '.'); i >= 0 {
				major = token[:i]
			}
			if len(name) == 0 {
				return major
			}
			return strings.Join(name, "-") + "-" + major
		}
		name = append(name, token)
	}
	return strings.Join(name, "-")
}

// priorFor chooses a prior for newID given the arms already present.
//
// order must be the policy's sorted arm IDs. Ranging over the map instead would
// make the choice depend on Go's randomised map iteration order whenever two
// candidates have the same posterior mean — the comparison below is strict, so
// whichever is visited first keeps the slot — and a new arm's prior strength
// would then vary between runs of the same program.
func priorFor(ws WarmStart, newID string, order []string, existing map[string]*Arm) InformedPrior {
	switch ws.Kind {
	case ColdStart:
		return NewInformedPrior(1, 1)
	case FixedPrior:
		return ws.Fixed
	}

	target := ParseModelID(newID)
	var bestFamily, bestProvider *Arm

	for _, id := range order {
		arm, ok := existing[id]
		// An arm with no observations carries no information to lend, and
		// chaining warm starts would let one guess propagate through the set.
		if !ok || arm.Posterior.Pulls == 0 || id == newID {
			continue
		}
		candidate := ParseModelID(id)
		if candidate.Provider != target.Provider {
			continue
		}
		if candidate.Family == target.Family &&
			(bestFamily == nil || arm.Posterior.Mean() > bestFamily.Posterior.Mean()) {
			bestFamily = arm
		}
		if bestProvider == nil || arm.Posterior.Mean() > bestProvider.Posterior.Mean() {
			bestProvider = arm
		}
	}

	neighbour := bestFamily
	if neighbour == nil {
		neighbour = bestProvider
	}
	if neighbour == nil {
		return ws.Fallback
	}

	d := ws.Discount
	if d < 0 {
		d = 0
	}
	if d > 1 {
		d = 1
	}
	return NewInformedPrior(neighbour.Posterior.Alpha*d, neighbour.Posterior.Beta*d)
}
