//! First-class selection strategies.
//!
//! `policy::Selection` enumerates the built-in strategies. This module makes
//! the *machinery* behind them explicit: a [`SelectionStrategy`] trait that
//! can be implemented out-of-tree, a [`Registry`] that pairs a strategy with a
//! [`BetaSampler`](crate::sampler::BetaSampler), and the three built-in
//! strategies as standalone types. Plugging a custom strategy no longer
//! requires forking `policy.rs` — implement the trait and pass it to
//! [`ThompsonSampling::select_with`](crate::ThompsonSampling::select_with).

use crate::arm::Arm;
use crate::sampler::BetaSampler;
use rand::RngCore;
use std::collections::BTreeMap;

/// How an arm is chosen from the current posteriors.
pub trait SelectionStrategy: Send + Sync + std::fmt::Debug {
    /// Human-readable name for reports.
    fn name(&self) -> &'static str;

    /// Choose an arm. Must return an id present in `arms`.
    fn select(
        &self,
        rng: &mut dyn RngCore,
        arms: &BTreeMap<String, Arm>,
        sampler: &dyn BetaSampler,
        total_pulls: u64,
    ) -> String;
}

/// Plain Thompson Sampling: argmax of one Beta draw per arm.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThompsonStrategy;

impl SelectionStrategy for ThompsonStrategy {
    fn name(&self) -> &'static str {
        "thompson"
    }

    fn select(
        &self,
        rng: &mut dyn RngCore,
        arms: &BTreeMap<String, Arm>,
        sampler: &dyn BetaSampler,
        _total_pulls: u64,
    ) -> String {
        let mut best: Option<(&str, f64)> = None;
        for arm in arms.values() {
            let score = sampler.sample(rng, &arm.posterior);
            if best.map_or(true, |(_, b)| score > b) {
                best = Some((&arm.id, score));
            }
        }
        best.expect("arm set is non-empty").0.to_string()
    }
}

/// Thompson Sampling with a UCB-style bonus for under-explored arms.
#[derive(Debug, Clone, Copy)]
pub struct UcbRegularizedStrategy {
    /// Bonus coefficient.
    pub c: f64,
    /// Arms with at least this many pulls stop receiving the bonus.
    pub until_pulls: u64,
}

impl SelectionStrategy for UcbRegularizedStrategy {
    fn name(&self) -> &'static str {
        "ucb-regularized"
    }

    fn select(
        &self,
        rng: &mut dyn RngCore,
        arms: &BTreeMap<String, Arm>,
        sampler: &dyn BetaSampler,
        total_pulls: u64,
    ) -> String {
        let log_total = ((total_pulls + 1) as f64).ln();
        let mut best: Option<(&str, f64)> = None;
        for arm in arms.values() {
            let sample = sampler.sample(rng, &arm.posterior);
            let score = if arm.pulls() >= self.until_pulls {
                sample
            } else if arm.pulls() == 0 {
                f64::INFINITY
            } else {
                sample + self.c * (log_total / arm.pulls() as f64).sqrt()
            };
            if best.map_or(true, |(_, b)| score > b) {
                best = Some((&arm.id, score));
            }
        }
        best.expect("arm set is non-empty").0.to_string()
    }
}

/// Round-robin to a quota, then Thompson Sampling.
#[derive(Debug, Clone, Copy)]
pub struct PhasedStrategy {
    /// Pulls each arm receives before exploitation.
    pub bootstrap: u64,
    /// Pulls required before an arm may be selected by sampling.
    pub min_pulls_for_exploit: u64,
}

impl PhasedStrategy {
    fn quota(&self) -> u64 {
        self.bootstrap.max(self.min_pulls_for_exploit)
    }
}

impl SelectionStrategy for PhasedStrategy {
    fn name(&self) -> &'static str {
        "phased"
    }

    fn select(
        &self,
        rng: &mut dyn RngCore,
        arms: &BTreeMap<String, Arm>,
        sampler: &dyn BetaSampler,
        _total_pulls: u64,
    ) -> String {
        let quota = self.quota();
        if let Some(least) = arms
            .values()
            .filter(|arm| arm.pulls() < quota)
            .min_by_key(|arm| arm.pulls())
        {
            return least.id.clone();
        }
        ThompsonStrategy.select(rng, arms, sampler, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posterior::Posterior;
    use crate::sampler::Exact;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    fn arms(ids: &[&str]) -> BTreeMap<String, Arm> {
        ids.iter()
            .map(|id| {
                (
                    id.to_string(),
                    Arm::new(id.to_string(), Posterior::uninformative()),
                )
            })
            .collect()
    }

    #[test]
    fn thompson_strategy_selects_an_arm() {
        let arms = arms(&["a", "b", "c"]);
        let sampler = Exact;
        let mut rng = SmallRng::seed_from_u64(1);
        let chosen = ThompsonStrategy.select(&mut rng, &arms, &sampler, 0);
        assert!(["a", "b", "c"].contains(&chosen.as_str()));
    }

    #[test]
    fn phased_strategy_bootstraps_evenly() {
        let mut map = arms(&["a", "b"]);
        // give "a" one pull so "b" is least-pulled below quota 5
        map.get_mut("a").unwrap().posterior.pulls = 1;
        let sampler = Exact;
        let mut rng = SmallRng::seed_from_u64(2);
        let strat = PhasedStrategy {
            bootstrap: 5,
            min_pulls_for_exploit: 5,
        };
        assert_eq!(strat.select(&mut rng, &map, &sampler, 0), "b");
    }

    #[test]
    fn ucb_strategy_gives_infinity_to_unpulled() {
        let arms = arms(&["a", "b"]);
        let sampler = Exact;
        let mut rng = SmallRng::seed_from_u64(3);
        let strat = UcbRegularizedStrategy {
            c: 2.0,
            until_pulls: 30,
        };
        // both have 0 pulls -> both INFINITY, first in BTree order wins
        let chosen = strat.select(&mut rng, &arms, &sampler, 0);
        assert_eq!(chosen, "a");
    }
}
