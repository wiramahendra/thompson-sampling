// Thin-waist Go example: two calls to add to any gateway.
package main

import (
	"fmt"
	"math/rand/v2"

	"github.com/wiramahendra/thompson-sampling/go/thompson"
)

// LoggingObserver implements thompson.Observer for OTEL/Prometheus.
type LoggingObserver struct{}

func (LoggingObserver) OnSelect(chosen string, scores map[string]float64) {
	fmt.Printf("select -> %s scores=%v\n", chosen, scores)
}
func (LoggingObserver) OnRecord(arm string, reward float64, p thompson.Posterior) {
	fmt.Printf("record %s reward=%.3f mean=%.3f\n", arm, reward, p.Mean())
}
func (LoggingObserver) OnArmAdded(id string, warmStarted bool) {
	fmt.Printf("arm added %s warmStarted=%v\n", id, warmStarted)
}
func (LoggingObserver) OnDiscount(f float64) { fmt.Printf("discount %g\n", f) }

func main() {
	rng := rand.New(rand.NewPCG(42, 42))
	policy := thompson.NewDefault("openai/gpt-4", "anthropic/claude-3-opus")
	policy.SetObserver(LoggingObserver{})

	for i := 0; i < 5; i++ {
		provider, err := policy.Select(rng)
		if err != nil {
			panic(err)
		}
		// ... forward request to provider ...

		outcome := thompson.NewOutcome(320, true, 0.0012).WithQuality(0.87)
		if err := policy.RecordOutcome(rng, provider, outcome); err != nil {
			panic(err)
		}
		fmt.Printf("stats: %+v\n", policy.Stats())
	}

	policy.AddArm("openai/gpt-4.5-turbo")
	fmt.Printf("effective_memory: %v\n", policy.EffectiveMemory())

	store := thompson.NewMemoryStore()
	if err := policy.SaveToStore(store); err != nil {
		panic(err)
	}
	snap, _ := store.Load()
	fmt.Printf("snapshot version %d with %d arms\n", snap.Version, len(snap.Arms))
}
