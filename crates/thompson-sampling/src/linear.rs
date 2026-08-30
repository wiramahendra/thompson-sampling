//! Linear contextual scaffold — future evolution of `PartitionedPolicy`.
//!
//! `PartitionedPolicy` (context.rs:33) buckets by `Context::partition_key()`.
//! Linear contextual extends this to share strength across contexts via
//! `Posterior` weighting. Stubbed here so thin waist stays `select`/`record`
//! without breaking `thompson-sampling` 0.1.x semver.

use crate::posterior::Posterior;

/// Weight vector for linear contextual bandit — placeholder.
#[derive(Debug, Clone)]
pub struct LinearWeights {
    pub weights: Vec<f64>,
}

impl LinearWeights {
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

/// Linear contextual policy stub — wraps `Posterior` per arm with shared weights.
#[derive(Debug)]
pub struct LinearPolicy {
    pub dim: usize,
    pub weights: LinearWeights,
}

impl LinearPolicy {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            weights: LinearWeights::new(dim),
        }
    }

    /// Posterior mean adjusted by linear score — future hook for `select`.
    pub fn adjusted_mean(&self, posterior: &Posterior, features: &[f64]) -> f64 {
        let base = posterior.mean();
        let ctx = self.weights.score(features);
        (base * 0.7 + ctx * 0.3).clamp(0.0, 1.0)
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
}
