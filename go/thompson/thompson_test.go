package thompson

import (
	"encoding/json"
	"math"
	"math/rand/v2"
	"sync"
	"testing"
)

func newRNG(seed uint64) *rand.Rand {
	return rand.New(rand.NewPCG(seed, seed^0x9E3779B97F4A7C15))
}

// --- posterior -------------------------------------------------------------

func TestNewPosteriorRejectsBadParameters(t *testing.T) {
	for _, tc := range []struct{ alpha, beta float64 }{
		{0, 1}, {1, -1}, {math.NaN(), 1}, {math.Inf(1), 1}, {1, math.Inf(1)},
	} {
		if _, err := NewPosterior(tc.alpha, tc.beta); err == nil {
			t.Errorf("NewPosterior(%v, %v) should have failed", tc.alpha, tc.beta)
		}
	}
}

func TestPosteriorMomentsMatchClosedForm(t *testing.T) {
	p, err := NewPosterior(2, 3)
	if err != nil {
		t.Fatal(err)
	}
	if got := p.Mean(); math.Abs(got-0.4) > 1e-12 {
		t.Errorf("Mean() = %v, want 0.4", got)
	}
	if got := p.Variance(); math.Abs(got-0.04) > 1e-12 {
		t.Errorf("Variance() = %v, want 0.04", got)
	}
}

func TestBinarizeDiscardsRewardMagnitude(t *testing.T) {
	rng := newRNG(1)
	rule := UpdateRule{Kind: Binarize, Threshold: 0.6}

	low, high := Uninformative(), Uninformative()
	if err := low.Observe(rng, 0.61, rule); err != nil {
		t.Fatal(err)
	}
	if err := high.Observe(rng, 0.99, rule); err != nil {
		t.Fatal(err)
	}
	// The information loss, asserted so it stays visible.
	if low != high {
		t.Errorf("expected 0.61 and 0.99 to be indistinguishable, got %+v and %+v", low, high)
	}
}

func TestBernoulliConvergesToRewardRate(t *testing.T) {
	rng := newRNG(42)
	p := Uninformative()
	for i := 0; i < 20000; i++ {
		if err := p.Observe(rng, 0.3, UpdateRule{Kind: Bernoulli}); err != nil {
			t.Fatal(err)
		}
	}
	if math.Abs(p.Mean()-0.3) > 0.02 {
		t.Errorf("Mean() = %v, want ~0.3", p.Mean())
	}
}

func TestObserveRejectsOutOfRangeReward(t *testing.T) {
	rng := newRNG(1)
	p := Uninformative()
	for _, reward := range []float64{1.5, -0.1, math.NaN()} {
		if err := p.Observe(rng, reward, UpdateRule{Kind: Fractional}); err == nil {
			t.Errorf("Observe(%v) should have failed", reward)
		}
	}
	if p.Pulls != 0 {
		t.Errorf("rejected observations must not count: Pulls = %d", p.Pulls)
	}
}

func TestDiscountNeverFallsBelowThePrior(t *testing.T) {
	p := Posterior{Alpha: 101, Beta: 11}
	p.Discount(0.5)
	if math.Abs(p.Alpha-51) > 1e-12 || math.Abs(p.Beta-6) > 1e-12 {
		t.Fatalf("after one discount: %+v", p)
	}
	for i := 0; i < 1000; i++ {
		p.Discount(0.5)
	}
	if p.Alpha < 1 || p.Beta < 1 {
		t.Errorf("discount drove parameters below the prior: %+v", p)
	}
}

// --- samplers --------------------------------------------------------------

func TestExactGammaMatchesClosedFormMoments(t *testing.T) {
	// Gamma(k, 1) has mean k and variance k.
	for _, k := range []float64{0.3, 1, 2.5, 17, 250} {
		rng := newRNG(7)
		const n = 200000
		var sum, sumSq float64
		for i := 0; i < n; i++ {
			x := SampleGamma(rng, k)
			sum += x
			sumSq += x * x
		}
		mean := sum / n
		variance := sumSq/n - mean*mean

		tol := 0.02 * math.Max(k, 1)
		if math.Abs(mean-k) > tol {
			t.Errorf("Gamma(%v): mean = %v, want %v", k, mean, k)
		}
		if math.Abs(variance-k) > 0.1*math.Max(k, 1) {
			t.Errorf("Gamma(%v): variance = %v, want %v", k, variance, k)
		}
	}
}

func TestExactSamplerMatchesBetaMoments(t *testing.T) {
	for _, tc := range []struct{ alpha, beta float64 }{
		{1, 1}, {2, 3}, {30, 5}, {0.5, 0.5},
	} {
		p := Posterior{Alpha: tc.alpha, Beta: tc.beta}
		rng := newRNG(0xC0FFEE)
		const n = 200000
		var sum, sumSq float64
		for i := 0; i < n; i++ {
			s := ExactSampler{}.Sample(rng, p)
			sum += s
			sumSq += s * s
		}
		mean := sum / n
		variance := sumSq/n - mean*mean

		if math.Abs(mean-p.Mean()) > 0.01 {
			t.Errorf("Beta(%v,%v): mean = %v, want %v", tc.alpha, tc.beta, mean, p.Mean())
		}
		if math.Abs(variance-p.Variance()) > 0.01 {
			t.Errorf("Beta(%v,%v): variance = %v, want %v",
				tc.alpha, tc.beta, variance, p.Variance())
		}
	}
}

func TestExactSamplerExploresFlatPosteriorUniformly(t *testing.T) {
	// Beta(1,1) is uniform. An approximation that draws a bell around the mean
	// will fail this, which is the point.
	rng := newRNG(11)
	p := Uninformative()
	var deciles [10]int
	const n = 100000
	for i := 0; i < n; i++ {
		idx := int(ExactSampler{}.Sample(rng, p) * 10)
		if idx > 9 {
			idx = 9
		}
		deciles[idx]++
	}
	for i, count := range deciles {
		if share := float64(count) / n; math.Abs(share-0.1) > 0.01 {
			t.Errorf("decile %d share = %v, want ~0.1", i, share)
		}
	}
}

func TestDeterministicSamplerNeverVaries(t *testing.T) {
	rng := newRNG(1)
	p := Posterior{Alpha: 30, Beta: 5}
	var sampler Sampler = DeterministicSampler{}
	first := sampler.Sample(rng, p)
	for i := 0; i < 1000; i++ {
		if got := sampler.Sample(rng, p); got != first {
			t.Fatalf("sample %d = %v, want %v", i, got, first)
		}
	}
}

func TestUniformNoiseDoesNotShrinkWithEvidence(t *testing.T) {
	s := MeanPlusUniformSampler{HalfWidth: 0.1}
	spread := func(alpha, beta float64) float64 {
		rng := newRNG(3)
		p := Posterior{Alpha: alpha, Beta: beta}
		const n = 100000
		var sum, sumSq float64
		for i := 0; i < n; i++ {
			v := s.Sample(rng, p)
			sum += v
			sumSq += v * v
		}
		mean := sum / n
		return sumSq/n - mean*mean
	}
	// A correct sampler's variance would fall three orders of magnitude between
	// these posteriors. Here it is flat.
	if ratio := spread(2000, 2000) / spread(2, 2); ratio < 0.9 {
		t.Errorf("exploration noise shrank with evidence: ratio = %v", ratio)
	}
}

func TestAllSamplersStayInRange(t *testing.T) {
	samplers := []Sampler{
		ExactSampler{},
		MeanPlusGaussianSampler{},
		MeanPlusUniformSampler{HalfWidth: 0.1},
		DeterministicSampler{},
		ProductionSwitched(),
	}
	extremes := []Posterior{
		{Alpha: 1e-3, Beta: 1e-3}, {Alpha: 1e6, Beta: 1}, {Alpha: 1, Beta: 1e6}, {Alpha: 1e6, Beta: 1e6},
	}

	rng := newRNG(99)
	for _, s := range samplers {
		for _, p := range extremes {
			for i := 0; i < 2000; i++ {
				v := s.Sample(rng, p)
				if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > 1 {
					t.Fatalf("%s produced %v for Beta(%v,%v)", s.Name(), v, p.Alpha, p.Beta)
				}
			}
		}
	}
}

// --- warm start ------------------------------------------------------------

func TestModelFamily(t *testing.T) {
	for _, tc := range []struct{ in, want string }{
		{"gpt-4.5-turbo", "gpt-4"},
		{"gpt-4o-mini", "gpt-4o"},
		{"claude-3-5-sonnet", "claude-3"},
		{"claude-3-opus", "claude-3"},
		{"claude-sonnet-4-5", "claude-sonnet-4"},
		{"gpt-4", "gpt-4"},
		{"mistral-large", "mistral-large"},
		{"", ""},
	} {
		if got := ModelFamily(tc.in); got != tc.want {
			t.Errorf("ModelFamily(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestWarmStartInheritsFromSameFamily(t *testing.T) {
	p := NewDefault()
	rng := newRNG(5)

	p.AddArmWithPrior("openai/gpt-4", NewInformedPrior(1, 1))
	for i := 0; i < 50; i++ {
		if err := p.Record(rng, "openai/gpt-4", 0.95); err != nil {
			t.Fatal(err)
		}
	}

	prior, added := p.AddArm("openai/gpt-4.5-turbo")
	if !added {
		t.Fatal("arm should be new")
	}
	if prior.Alpha <= 1 {
		t.Errorf("expected an informed prior, got %+v", prior)
	}
	if mean := prior.Posterior().Mean(); mean <= 0.5 {
		t.Errorf("expected to inherit the neighbour's optimism, got mean %v", mean)
	}

	fresh := p.Stats()
	for _, s := range fresh {
		if s.ID == "openai/gpt-4.5-turbo" {
			if !s.WarmStarted {
				t.Error("new arm should be flagged as warm started")
			}
			if s.Pulls != 0 {
				t.Errorf("new arm has %d pulls, want 0", s.Pulls)
			}
		}
	}
}

func TestWarmStartDoesNotInheritFromUnpulledArm(t *testing.T) {
	// A warm-started arm has a concentrated posterior and zero pulls. Letting
	// it seed the next arm would launder a guess into evidence.
	p := NewDefault()
	p.AddArmWithPrior("openai/gpt-4", NewInformedPrior(18, 2))

	prior, _ := p.AddArm("openai/gpt-4.5")
	if prior != DefaultPrior() {
		t.Errorf("prior = %+v, want the fallback %+v", prior, DefaultPrior())
	}
}

// --- policy ----------------------------------------------------------------

func TestSelectOnEmptyPolicy(t *testing.T) {
	p := New(DefaultConfig(), ExactSampler{})
	if _, err := p.Select(newRNG(1)); err != ErrNoArms {
		t.Errorf("err = %v, want %v", err, ErrNoArms)
	}
}

func TestRecordUnknownArm(t *testing.T) {
	p := NewDefault("a")
	if err := p.Record(newRNG(1), "nope", 1.0); err == nil {
		t.Error("recording an unknown arm should fail")
	}
	if p.TotalPulls() != 0 {
		t.Errorf("TotalPulls = %d, want 0", p.TotalPulls())
	}
}

func TestConvergesOnTheBetterArm(t *testing.T) {
	p := NewDefault("good", "bad")
	rng := newRNG(0xBEEF)

	for i := 0; i < 400; i++ {
		id, err := p.Select(rng)
		if err != nil {
			t.Fatal(err)
		}
		reward := 0.1
		if id == "good" {
			reward = 0.9
		}
		if err := p.Record(rng, id, reward); err != nil {
			t.Fatal(err)
		}
	}

	best, ok := p.BestArm(1)
	if !ok || best != "good" {
		t.Errorf("BestArm = %q (%v), want \"good\"", best, ok)
	}

	stats := p.Stats()
	var goodPulls, badPulls uint64
	for _, s := range stats {
		if s.ID == "good" {
			goodPulls = s.Pulls
		} else {
			badPulls = s.Pulls
		}
	}
	if goodPulls <= badPulls*3 {
		t.Errorf("good=%d bad=%d: expected clear preference", goodPulls, badPulls)
	}
	if badPulls == 0 {
		t.Error("Thompson Sampling must not close off an arm entirely")
	}
}

func TestPhasedSelectionExploitsOnceQuotaIsMet(t *testing.T) {
	// Regression guard: gating exploitation on a per-arm threshold while
	// exploiting only among arms past it locks onto the first arm to cross.
	cfg := DefaultConfig()
	cfg.Selection = Selection{Kind: PhasedSelection, Bootstrap: 10, MinPullsForExploit: 10}
	p := New(cfg, ExactSampler{})
	p.AddArmWithPrior("bad", NewInformedPrior(1, 1))
	p.AddArmWithPrior("good", NewInformedPrior(1, 1))

	rng := newRNG(0xBEEF)
	for i := 0; i < 600; i++ {
		id, err := p.Select(rng)
		if err != nil {
			t.Fatal(err)
		}
		reward := 0.05
		if id == "good" {
			reward = 0.95
		}
		if err := p.Record(rng, id, reward); err != nil {
			t.Fatal(err)
		}
	}

	var goodPulls, badPulls uint64
	for _, s := range p.Stats() {
		if s.ID == "good" {
			goodPulls = s.Pulls
		} else {
			badPulls = s.Pulls
		}
	}
	if badPulls < 10 {
		t.Errorf("quota not honoured: bad has %d pulls", badPulls)
	}
	if goodPulls <= badPulls*4 {
		t.Errorf("expected exploitation after quota: good=%d bad=%d", goodPulls, badPulls)
	}
}

func TestUCBSurvivesTheFirstRound(t *testing.T) {
	// Log(totalPulls) at zero pulls is the classic NaN trap.
	cfg := DefaultConfig()
	cfg.Selection = Selection{Kind: UCBRegularized, C: 2, UntilPulls: 30}
	p := New(cfg, ExactSampler{})
	for _, id := range []string{"a", "b", "c"} {
		p.AddArm(id)
	}

	id, err := p.Select(newRNG(1))
	if err != nil {
		t.Fatal(err)
	}
	if id != "a" && id != "b" && id != "c" {
		t.Errorf("Select returned %q", id)
	}
}

func TestDiscountingFollowsARegimeChange(t *testing.T) {
	cfg := DefaultConfig()
	cfg.Discount = 0.99
	p := New(cfg, ExactSampler{})
	for _, id := range []string{"a", "b"} {
		p.AddArm(id)
	}

	rng := newRNG(0xBEEF)
	play := func(rounds int, winner string) {
		for i := 0; i < rounds; i++ {
			id, err := p.Select(rng)
			if err != nil {
				t.Fatal(err)
			}
			reward := 0.1
			if id == winner {
				reward = 0.9
			}
			if err := p.Record(rng, id, reward); err != nil {
				t.Fatal(err)
			}
		}
	}

	play(500, "a")
	if best, _ := p.BestArm(1); best != "a" {
		t.Fatalf("before the switch BestArm = %q, want \"a\"", best)
	}
	play(1500, "b")
	if best, _ := p.BestArm(1); best != "b" {
		t.Errorf("after the switch BestArm = %q, want \"b\"; stats %+v", best, p.Stats())
	}
}

func TestSnapshotRoundTripsThroughJSON(t *testing.T) {
	p := NewDefault("a", "b")
	rng := newRNG(0xBEEF)
	for i := 0; i < 30; i++ {
		id, err := p.Select(rng)
		if err != nil {
			t.Fatal(err)
		}
		if err := p.Record(rng, id, 0.8); err != nil {
			t.Fatal(err)
		}
	}

	encoded, err := json.Marshal(p.Snapshot())
	if err != nil {
		t.Fatal(err)
	}
	var decoded Snapshot
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	restored, err := Restore(decoded, DefaultConfig(), ExactSampler{})
	if err != nil {
		t.Fatal(err)
	}

	if restored.TotalPulls() != p.TotalPulls() {
		t.Errorf("TotalPulls = %d, want %d", restored.TotalPulls(), p.TotalPulls())
	}
	before, after := p.Stats(), restored.Stats()
	if len(before) != len(after) {
		t.Fatalf("arm count changed: %d vs %d", len(before), len(after))
	}
	for i := range before {
		// Tolerance, not equality: JSON float round trips are not bit-exact.
		if before[i].ID != after[i].ID ||
			before[i].Pulls != after[i].Pulls ||
			math.Abs(before[i].Alpha-after[i].Alpha) > 1e-9 {
			t.Errorf("arm %d differs: %+v vs %+v", i, before[i], after[i])
		}
	}
}

func TestRestoreRejectsBadInput(t *testing.T) {
	p := NewDefault("a")
	s := p.Snapshot()

	bad := s
	bad.Version = 999
	if _, err := Restore(bad, DefaultConfig(), ExactSampler{}); err == nil {
		t.Error("expected a version error")
	}

	corrupt := p.Snapshot()
	corrupt.Arms[0].Posterior.Alpha = 0
	if _, err := Restore(corrupt, DefaultConfig(), ExactSampler{}); err == nil {
		t.Error("expected a posterior validation error")
	}
}

func TestConcurrentUseIsSafe(t *testing.T) {
	// Run with -race. The failure mode this guards against is a lock-order
	// inversion between a policy-level lock and per-arm locks: one path takes
	// them policy-first, another arm-first, and the deadlock only appears under
	// real concurrency. A single mutex makes the inversion unrepresentable.
	p := NewDefault("a", "b", "c")

	const goroutines = 16
	const iterations = 500

	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(seed uint64) {
			defer wg.Done()
			rng := newRNG(seed)
			for i := 0; i < iterations; i++ {
				id, err := p.Select(rng)
				if err != nil {
					t.Error(err)
					return
				}
				if err := p.Record(rng, id, rng.Float64()); err != nil {
					t.Error(err)
					return
				}
				// Interleave the readers that would take the locks in the
				// opposite order under a per-arm design.
				_ = p.Stats()
				_, _ = p.BestArm(1)
				_ = p.Snapshot()
				p.AddArm("d")
			}
		}(uint64(g) + 1)
	}
	wg.Wait()

	if want := uint64(goroutines * iterations); p.TotalPulls() != want {
		t.Errorf("TotalPulls = %d, want %d", p.TotalPulls(), want)
	}
}

// --- reward ----------------------------------------------------------------

func TestFailureScoresZero(t *testing.T) {
	rp := DefaultRewardPolicy()
	if got := rp.Reward(NewOutcome(10, false, 0)); got != 0 {
		t.Errorf("Reward = %v, want 0", got)
	}
}

func TestFastFailureOutscoresSlowSuccessWithoutTheGuard(t *testing.T) {
	rp := DefaultRewardPolicy()
	rp.FailureIsZero = false
	// Documents exactly why FailureIsZero defaults to true.
	if got := rp.Reward(NewOutcome(10, false, 0)); got <= 0.2 {
		t.Errorf("a free instant failure scored %v; expected it to look attractive", got)
	}
}

func TestAbsentQualityRedistributesItsWeight(t *testing.T) {
	rp := DefaultRewardPolicy()
	perfect := Outcome{LatencyMs: 0, Success: true, CacheHit: true, CostUSD: 0}
	// Every present component is perfect, so the total must be 1.0 rather than
	// 0.85, which is what scoring absent quality as zero would give.
	if got := rp.Reward(perfect); math.Abs(got-1) > 1e-9 {
		t.Errorf("Reward = %v, want 1.0", got)
	}
}

func TestRewardIsBounded(t *testing.T) {
	rp := DefaultRewardPolicy()
	for _, o := range []Outcome{
		NewOutcome(-5, true, -1),
		NewOutcome(math.MaxFloat64, true, math.MaxFloat64),
		NewOutcome(0, true, 0).WithQuality(2),
		NewOutcome(1, true, 1).WithQuality(-1),
	} {
		if r := rp.Reward(o); r < 0 || r > 1 || math.IsNaN(r) {
			t.Errorf("Reward(%+v) = %v, out of range", o, r)
		}
	}
}
