//! First-class observability for the bandit policy.
//!
//! Production routers need to answer three questions without forking the
//! library: how often each arm is chosen, how rewards map to posteriors, and
//! when operational events (warm-start, discounting, churn) happen. This module
//! makes that machinery explicit via [`PolicyObserver`], a trait you implement
//! once and attach with [`ThompsonSampling::set_observer`](crate::ThompsonSampling::set_observer).
//!
//! The trait is deliberately small and `Send + Sync` so the same observer can
//! be shared with the `Go` port's single-mutex policy. All callbacks are
//! synchronous and must be cheap — they run on the hot path.

use crate::posterior::Posterior;

/// Operational event emitted by the policy.
#[derive(Debug, Clone, PartialEq)]
pub enum Event<'a> {
    /// An arm was selected.
    Selected {
        /// Chosen arm id.
        chosen: &'a str,
        /// Scores that produced the choice (arm id, sampled value).
        candidates: Vec<(&'a str, f64)>,
    },
    /// A reward was recorded.
    Recorded {
        /// Arm that was updated.
        arm: &'a str,
        /// Raw reward in `[0, 1]`.
        reward: f64,
        /// Posterior after the update.
        posterior: Posterior,
    },
    /// A new arm was registered.
    ArmAdded {
        /// New arm id.
        id: &'a str,
        /// Whether an informed prior was applied.
        warm_started: bool,
    },
    /// Discount was applied to all posteriors.
    Discounted {
        /// Factor in `(0, 1]`.
        factor: f64,
    },
}

/// Hook for metrics, logging, and alerting.
///
/// Implement this trait and attach it to a policy. The default [`NoopObserver`]
/// does nothing and costs nothing (branch on `Option` is predicted not-taken).
pub trait PolicyObserver: Send + Sync + std::fmt::Debug {
    /// Called after [`ThompsonSampling::select`](crate::ThompsonSampling::select).
    fn on_select(&self, _chosen: &str, _scores: &[(&str, f64)]) {}

    /// Called after [`ThompsonSampling::record`](crate::ThompsonSampling::record).
    fn on_record(&self, _arm: &str, _reward: f64, _posterior: &Posterior) {}

    /// Called after [`ThompsonSampling::add_arm`](crate::ThompsonSampling::add_arm).
    fn on_arm_added(&self, _id: &str, _warm_started: bool) {}

    /// Called after discount is applied.
    fn on_discount(&self, _factor: f64) {}

    /// Generic event hook — convenience for forwarders that want one method.
    fn on_event(&self, event: Event<'_>) {
        match event {
            Event::Selected { chosen, candidates } => self.on_select(chosen, &candidates),
            Event::Recorded {
                arm,
                reward,
                posterior,
            } => self.on_record(arm, reward, &posterior),
            Event::ArmAdded { id, warm_started } => self.on_arm_added(id, warm_started),
            Event::Discounted { factor } => self.on_discount(factor),
        }
    }
}

/// No-op observer — the default when none is attached.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopObserver;

impl PolicyObserver for NoopObserver {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Default)]
    struct CountingObserver {
        selects: Mutex<usize>,
        records: Mutex<usize>,
        adds: Mutex<usize>,
    }

    impl PolicyObserver for CountingObserver {
        fn on_select(&self, _chosen: &str, _scores: &[(&str, f64)]) {
            *self.selects.lock().unwrap() += 1;
        }
        fn on_record(&self, _arm: &str, _reward: f64, _posterior: &Posterior) {
            *self.records.lock().unwrap() += 1;
        }
        fn on_arm_added(&self, _id: &str, _warm_started: bool) {
            *self.adds.lock().unwrap() += 1;
        }
    }

    #[test]
    fn observer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CountingObserver>();
        assert_send_sync::<NoopObserver>();
    }

    #[test]
    fn counting_observer_receives_events() {
        let obs = Arc::new(CountingObserver::default());
        obs.on_select("a", &[]);
        obs.on_record("a", 0.5, &Posterior::uninformative());
        obs.on_arm_added("b", true);
        assert_eq!(*obs.selects.lock().unwrap(), 1);
        assert_eq!(*obs.records.lock().unwrap(), 1);
        assert_eq!(*obs.adds.lock().unwrap(), 1);
    }
}
