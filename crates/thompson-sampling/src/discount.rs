//! First-class discounting for non-stationary environments.
//!
//! A bare `Option<f64>` buries the operational choice: how fast old evidence
//! should age, what that means for effective memory, and when discounting
//! should be disabled. This module promotes discounting to a trait so policies
//! can plug in fixed, adaptive, or schedule-based decay without changing the
//! core `record` path.
//!
//! The default [`FixedDiscount`] reproduces the existing `Option<f64>` behaviour
//! exactly, so existing snapshots and configs round-trip unchanged.

use crate::posterior::Posterior;

/// How posteriors are aged after each observation.
///
/// Implementations must be `Send + Sync` so a policy can be shared across
/// threads (and so the Go port can mirror the interface). `discount` is called
/// once per distinct arm after every `record`, which is the same point where
/// `PolicyObserver::on_discount` fires.
pub trait DiscountPolicy: Send + Sync + std::fmt::Debug {
    /// Factor in `(0, 1]` applied this round, or `None` if stationary.
    fn factor(&self) -> Option<f64>;

    /// Apply the factor to a posterior. The default is the standard
    /// `1 + (x - 1) * factor` pull toward `Beta(1,1)`.
    fn apply(&self, posterior: &mut Posterior) {
        if let Some(f) = self.factor() {
            posterior.discount(f);
        }
    }

    /// Effective memory in observations: `1 / (1 - factor)`. `INFINITY` when
    /// stationary.
    fn effective_memory(&self) -> f64 {
        match self.factor() {
            None => f64::INFINITY,
            Some(f) if f >= 1.0 => f64::INFINITY,
            Some(f) => 1.0 / (1.0 - f),
        }
    }

    /// Human-readable label for reports.
    fn label(&self) -> String {
        match self.factor() {
            None => "none".to_string(),
            Some(f) => format!("{f}"),
        }
    }
}

/// Fixed per-round discount — the production default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedDiscount(pub Option<f64>);

impl FixedDiscount {
    /// Stationary: evidence never expires.
    pub const NONE: Self = FixedDiscount(None);

    /// Create a fixed discount, clamping to `(0, 1]`. `None` or `1.0` means
    /// stationary; values `<= 0` are treated as stationary so a misconfigured
    /// rate cannot invert the posterior.
    pub fn new(factor: Option<f64>) -> Self {
        match factor {
            Some(f) if f > 0.0 && f < 1.0 => FixedDiscount(Some(f)),
            Some(f) if f >= 1.0 => FixedDiscount(None),
            _ => FixedDiscount(None),
        }
    }
}

impl Default for FixedDiscount {
    fn default() -> Self {
        Self::NONE
    }
}

impl DiscountPolicy for FixedDiscount {
    fn factor(&self) -> Option<f64> {
        self.0
    }
}

impl From<Option<f64>> for FixedDiscount {
    fn from(v: Option<f64>) -> Self {
        Self::new(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_discount_effective_memory() {
        assert!(FixedDiscount::NONE.effective_memory().is_infinite());
        assert!((FixedDiscount::new(Some(0.999)).effective_memory() - 1000.0).abs() < 1e-9);
        assert!((FixedDiscount::new(Some(0.99)).effective_memory() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn clamping_prevents_inversion() {
        assert_eq!(FixedDiscount::new(Some(0.0)).factor(), None);
        assert_eq!(FixedDiscount::new(Some(-0.5)).factor(), None);
        assert_eq!(FixedDiscount::new(Some(1.0)).factor(), None);
        assert_eq!(FixedDiscount::new(Some(2.0)).factor(), None);
    }

    #[test]
    fn apply_pulls_toward_uniform() {
        let mut p = Posterior::new(101.0, 11.0).unwrap();
        FixedDiscount::new(Some(0.5)).apply(&mut p);
        assert!((p.alpha - 51.0).abs() < 1e-12);
    }
}
