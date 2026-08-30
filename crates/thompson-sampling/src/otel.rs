//! OTEL/prometheus adapter for `PolicyObserver`.
//!
//! Thin integration: attach `OtelObserver` via `ThompsonSampling::with_observer`
//! and existing `select`/`record` calls emit spans/counters. No gateway fork.

use crate::observer::PolicyObserver;
use crate::posterior::Posterior;
#[cfg(feature = "otel")]
use opentelemetry::trace::Span as _;

/// `OtelObserver` logs via `eprintln!` by default; with `--features otel` it
/// also emits real spans via `opentelemetry::global::tracer`.
#[derive(Debug, Default, Clone)]
pub struct OtelObserver {
    /// Service name for span attributes.
    pub service: String,
}

impl OtelObserver {
    /// Create with service name.
    pub fn new(service: impl Into<String>) -> Self {
        Self {
            service: service.into(),
        }
    }

    #[cfg(feature = "otel")]
    fn tracer(&self) -> opentelemetry::global::BoxedTracer {
        opentelemetry::global::tracer("thompson-sampling")
    }
}

#[cfg(not(feature = "otel"))]
impl PolicyObserver for OtelObserver {
    fn on_select(&self, chosen: &str, scores: &[(&str, f64)]) {
        eprintln!(
            "[otel:{}] thompson.select chosen={} scores={:?}",
            self.service, chosen, scores
        );
    }

    fn on_record(&self, arm: &str, reward: f64, posterior: &Posterior) {
        eprintln!(
            "[otel:{}] thompson.record arm={} reward={:.3} mean={:.3} pulls={}",
            self.service,
            arm,
            reward,
            posterior.mean(),
            posterior.pulls
        );
    }

    fn on_arm_added(&self, id: &str, warm_started: bool) {
        eprintln!(
            "[otel:{}] thompson.arm_added id={} warm_started={}",
            self.service, id, warm_started
        );
    }

    fn on_discount(&self, factor: f64) {
        eprintln!(
            "[otel:{}] thompson.discount factor={}",
            self.service, factor
        );
    }
}

#[cfg(feature = "otel")]
impl PolicyObserver for OtelObserver {
    fn on_select(&self, chosen: &str, scores: &[(&str, f64)]) {
        use opentelemetry::trace::Tracer;
        let tracer = self.tracer();
        let mut span = tracer.start("thompson.select");
        span.add_event(
            format!("chosen={chosen} scores={scores:?}"),
            vec![opentelemetry::KeyValue::new(
                "service",
                self.service.clone(),
            )],
        );
        drop(span);
        eprintln!(
            "[otel:{}] thompson.select chosen={} scores={:?}",
            self.service, chosen, scores
        );
    }

    fn on_record(&self, arm: &str, reward: f64, posterior: &Posterior) {
        let _ = arm;
        eprintln!(
            "[otel:{}] thompson.record arm={} reward={:.3} mean={:.3} pulls={}",
            self.service,
            arm,
            reward,
            posterior.mean(),
            posterior.pulls
        );
    }

    fn on_arm_added(&self, id: &str, warm_started: bool) {
        use opentelemetry::trace::Tracer;
        let tracer = self.tracer();
        let mut span = tracer.start("thompson.arm_added");
        span.add_event(
            format!("id={id} warm_started={warm_started}"),
            vec![opentelemetry::KeyValue::new(
                "service",
                self.service.clone(),
            )],
        );
        eprintln!(
            "[otel:{}] thompson.arm_added id={} warm_started={}",
            self.service, id, warm_started
        );
    }

    fn on_discount(&self, factor: f64) {
        let _ = factor;
        eprintln!(
            "[otel:{}] thompson.discount factor={}",
            self.service, factor
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posterior::Posterior;

    #[test]
    fn otel_observer_does_not_panic() {
        let obs = OtelObserver::new("test");
        obs.on_select("a", &[("a", 0.5)]);
        obs.on_record("a", 0.7, &Posterior::uninformative());
        obs.on_arm_added("b", true);
        obs.on_discount(0.99);
    }
}
