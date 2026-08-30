package thompson

import "log"

// OtelObserver logs OTEL-style spans. Replace log.Printf with otel.Tracer in prod.
type OtelObserver struct {
	Service string
}

func NewOtelObserver(service string) *OtelObserver { return &OtelObserver{Service: service} }

func (o *OtelObserver) OnSelect(chosen string, scores map[string]float64) {
	log.Printf("[otel:%s] thompson.select chosen=%s scores=%v", o.Service, chosen, scores)
}
func (o *OtelObserver) OnRecord(arm string, reward float64, p Posterior) {
	log.Printf("[otel:%s] thompson.record arm=%s reward=%.3f mean=%.3f pulls=%d", o.Service, arm, reward, p.Mean(), p.Pulls)
}
func (o *OtelObserver) OnArmAdded(id string, warmStarted bool) {
	log.Printf("[otel:%s] thompson.arm_added id=%s warmStarted=%v", o.Service, id, warmStarted)
}
func (o *OtelObserver) OnDiscount(factor float64) {
	log.Printf("[otel:%s] thompson.discount factor=%g", o.Service, factor)
}
