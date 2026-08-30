//! Linear contextual scaffold — future evolution of `PartitionedPolicy`.
//!
//! `PartitionedPolicy` (context.rs:33) buckets by `Context::partition_key()`.
//! Linear contextual extends this to share strength across contexts via
//! `Posterior` weighting. Stubbed here so thin waist stays `select`/`record`
//! without breaking `thompson-sampling` 0.1.x semver.

use crate::posterior::Posterior;
use serde::{Deserialize, Serialize};

/// Weight vector for linear contextual bandit — placeholder.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearWeights {
    /// Per-dimension weights.
    pub weights: Vec<f64>,
}

impl LinearWeights {
    /// Create zero-initialized weights.
    pub fn new(dim: usize) -> Self {
        Self {
            weights: vec![0.0; dim],
        }
    }

    /// Score for context features — dot product stub.
    pub fn score(&self, features: &[f64]) -> f64 {
        self.weights
            .iter()
            .zip(features)
            .map(|(w, f)| w * f)
            .sum::<f64>()
            .clamp(0.0, 1.0)
    }
}

/// Config for linear blending.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LinearConfig {
    /// Weight for posterior mean (0..1). Context weight is `1 - posterior_weight`.
    pub posterior_weight: f64,
    /// Learning rate for SGD.
    pub learning_rate: f64,
}

impl Default for LinearConfig {
    fn default() -> Self {
        Self {
            posterior_weight: 0.7,
            learning_rate: 0.05,
        }
    }
}

/// Linear contextual policy stub — wraps `Posterior` per arm with shared weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearPolicy {
    /// Feature dimension.
    pub dim: usize,
    /// Shared weights.
    pub weights: LinearWeights,
    /// Blending config.
    pub config: LinearConfig,
}

impl LinearPolicy {
    /// Create with dimension.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            weights: LinearWeights::new(dim),
            config: LinearConfig::default(),
        }
    }

    /// Create with config.
    pub fn with_config(dim: usize, config: LinearConfig) -> Self {
        Self {
            dim,
            weights: LinearWeights::new(dim),
            config,
        }
    }

    /// Posterior mean adjusted by linear score — future hook for `select`.
    pub fn adjusted_mean(&self, posterior: &Posterior, features: &[f64]) -> f64 {
        let base = posterior.mean();
        let ctx = self.weights.score(features);
        let w = self.config.posterior_weight.clamp(0.0, 1.0);
        (base * w + ctx * (1.0 - w)).clamp(0.0, 1.0)
    }

    /// Update weights via SGD with configured learning rate.
    pub fn update_with_config(&mut self, features: &[f64], reward: f64) {
        let lr = self.config.learning_rate;
        self.update(features, reward, lr);
    }

    /// Update weights via simple SGD on reward error — shares strength across contexts.
    pub fn update(&mut self, features: &[f64], reward: f64, lr: f64) {
        let pred = self.weights.score(features);
        let err = reward - pred;
        for (w, f) in self.weights.weights.iter_mut().zip(features) {
            *w += lr * err * f;
            *w = w.clamp(-1.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posterior::Posterior;

    #[test]
    fn adjusted_mean_within_bounds() {
        let policy = LinearPolicy::new(2);
        let post = Posterior::new(8.0, 2.0).unwrap();
        let m = policy.adjusted_mean(&post, &[0.5, 0.5]);
        assert!((0.0..=1.0).contains(&m));
    }

    #[test]
    fn serde_round_trips() {
        let policy = LinearPolicy::with_config(
            3,
            LinearConfig {
                posterior_weight: 0.6,
                learning_rate: 0.1,
            },
        );
        let json = serde_json::to_string(&policy).unwrap();
        let restored: LinearPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.dim, 3);
        assert!((restored.config.posterior_weight - 0.6).abs() < 1e-9);
    }

    #[test]
    fn dim_mismatch_truncates_gracefully() {
        let policy = LinearPolicy::new(2);
        let post = Posterior::new(5.0, 5.0).unwrap();
        // features longer than dim -> truncated via zip
        let m = policy.adjusted_mean(&post, &[1.0, 1.0, 1.0, 1.0]);
        assert!((0.0..=1.0).contains(&m));
        // features shorter than dim -> partial dot product
        let m2 = policy.adjusted_mean(&post, &[0.5]);
        assert!((0.0..=1.0).contains(&m2));
    }

    #[test]
    fn update_clamps_weights() {
        let mut policy = LinearPolicy::new(1);
        // Large reward error with high lr should clamp to [-1,1]
        policy.update(&[10.0], 1.0, 10.0);
        assert!(policy.weights.weights[0] <= 1.0 && policy.weights.weights[0] >= -1.0);
    }

    #[test]
    fn config_posterior_weight_clamped() {
        let mut policy = LinearPolicy::with_config(
            2,
            LinearConfig {
                posterior_weight: 2.0,
                learning_rate: 0.05,
            },
        );
        let post = Posterior::new(1.0, 1.0).unwrap();
        let m = policy.adjusted_mean(&post, &[0.5, 0.5]);
        assert!((0.0..=1.0).contains(&m));
        policy.config.posterior_weight = -1.0;
        let m2 = policy.adjusted_mean(&post, &[0.5, 0.5]);
        assert!((0.0..=1.0).contains(&m2));
    }
}
