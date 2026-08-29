//! Contextual bandit scaffold for thin-waist evolution.
//!
//! The current `ThompsonSampling` is non-contextual: one posterior per arm
//! (`policy.rs:79`). Real routing is contextual — `code` vs `summarize` prompts
//! have different best arms. This module partitions by context without forking
//! the core, so a gateway can start `NonContextual` and graduate to partitioned
//! or linear without changing the `select`/`record` waist.
//!
//! Phase 1 scaffold: `Context` trait + `PartitionedPolicy` (N independent
//! bandits). Future: linear contextual (`docs/FINDINGS.md:244`).

use crate::error::Result;
use crate::policy::{Config, Snapshot, ThompsonSampling};
use crate::sampler::BetaSampler;
use rand::RngCore;
use std::collections::BTreeMap;

/// Context that selects a bandit partition.
pub trait Context: Send + Sync + std::fmt::Debug + Clone + Eq + std::hash::Hash {
    /// Hashable key for partitioning. Override for custom bucketing.
    fn partition_key(&self) -> String;
}

/// Simple string context — e.g. task type, tenant, or prompt hash bucket.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SimpleContext(pub String);

impl Context for SimpleContext {
    fn partition_key(&self) -> String {
        self.0.clone()
    }
}

/// No context — single partition, preserves non-contextual behavior.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NoContext;

impl Context for NoContext {
    fn partition_key(&self) -> String {
        "default".to_string()
    }
}

/// Partitioned contextual policy: one `ThompsonSampling` per context key.
/// Thin-waist preserves `select`/`record` with added `ctx` param.
pub struct PartitionedPolicy<C: Context> {
    partitions: BTreeMap<String, ThompsonSampling>,
    config: Config,
    sampler_factory: Box<dyn Fn() -> Box<dyn BetaSampler> + Send + Sync>,
    _marker: std::marker::PhantomData<C>,
}

impl<C: Context> std::fmt::Debug for PartitionedPolicy<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PartitionedPolicy")
            .field("partitions", &self.partitions.keys().collect::<Vec<_>>())
            .field("config", &self.config)
            .finish()
    }
}

impl<C: Context> PartitionedPolicy<C> {
    /// Create with config and sampler factory (one per partition).
    pub fn new<F>(config: Config, sampler_factory: F) -> Self
    where
        F: Fn() -> Box<dyn BetaSampler> + Send + Sync + 'static,
    {
        Self {
            partitions: BTreeMap::new(),
            config,
            sampler_factory: Box::new(sampler_factory),
            _marker: std::marker::PhantomData,
        }
    }

    fn ensure_partition(&mut self, ctx: &C) -> &mut ThompsonSampling {
        let key = ctx.partition_key();
        if !self.partitions.contains_key(&key) {
            let sampler = (self.sampler_factory)();
            self.partitions
                .insert(key.clone(), ThompsonSampling::new(self.config, sampler));
        }
        self.partitions.get_mut(&key).unwrap()
    }

    /// Register arm in all partitions (or lazily per context).
    pub fn add_arm(&mut self, id: String) {
        for p in self.partitions.values_mut() {
            p.add_arm(id.clone());
        }
        // Also ensure default partition has it for future contexts
        // We do not eagerly create all partitions; they inherit on first use.
        // To guarantee arm exists, caller should `add_arm_to_all` or ensure ctx exists first.
    }

    /// Register arm in a specific context partition.
    pub fn add_arm_in(&mut self, ctx: &C, id: String) {
        self.ensure_partition(ctx).add_arm(id);
    }

    /// Select in context.
    pub fn select(&mut self, ctx: &C, rng: &mut dyn RngCore) -> Result<String> {
        self.ensure_partition(ctx).select(rng)
    }

    /// Record in context.
    pub fn record(&mut self, ctx: &C, rng: &mut dyn RngCore, id: &str, reward: f64) -> Result<()> {
        self.ensure_partition(ctx).record(rng, id, reward)
    }

    /// Number of partitions.
    pub fn len_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Snapshot all partitions.
    pub fn snapshots(&self) -> BTreeMap<String, Snapshot> {
        self.partitions
            .iter()
            .map(|(k, v)| (k.clone(), v.snapshot()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampler::Exact;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn partitioned_context_isolates_learning() {
        let mut policy = PartitionedPolicy::<SimpleContext>::new(
            Config::default(),
            || Box::new(Exact),
        );
        let ctx_code = SimpleContext("code".to_string());
        let ctx_chat = SimpleContext("chat".to_string());

        policy.add_arm_in(&ctx_code, "a".to_string());
        policy.add_arm_in(&ctx_code, "b".to_string());
        policy.add_arm_in(&ctx_chat, "a".to_string());
        policy.add_arm_in(&ctx_chat, "b".to_string());

        let mut rng = SmallRng::seed_from_u64(1);
        // Train code partition to prefer a=good, chat to prefer b=good
        for _ in 0..50 {
            policy.record(&ctx_code, &mut rng, "a", 0.9).unwrap();
            policy.record(&ctx_chat, &mut rng, "b", 0.9).unwrap();
        }

        assert_eq!(policy.len_partitions(), 2);
        assert_eq!(policy.partitions[&ctx_code.partition_key()].best_arm(1), Some("a"));
        assert_eq!(policy.partitions[&ctx_chat.partition_key()].best_arm(1), Some("b"));
    }

    #[test]
    fn no_context_is_single_partition() {
        let mut policy = PartitionedPolicy::<NoContext>::new(Config::default(), || Box::new(Exact));
        let ctx = NoContext;
        policy.add_arm_in(&ctx, "a".to_string());
        policy.add_arm_in(&ctx, "b".to_string());
        assert_eq!(policy.len_partitions(), 1);
        let mut rng = SmallRng::seed_from_u64(2);
        let chosen = policy.select(&ctx, &mut rng).unwrap();
        assert!(["a", "b"].contains(&chosen.as_str()));
    }
}
