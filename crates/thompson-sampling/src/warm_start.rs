//! Priors for arms that have never been pulled.
//!
//! In provider routing the arm set is not fixed. Vendors ship a new model every
//! few weeks and retire the old one, so a router spends much of its life with a
//! freshly-added arm sitting at `Beta(1, 1)` while a well-understood arm beside
//! it has thousands of observations. Under a uniform prior the new arm has to
//! be rediscovered from scratch every time, and cold-start regret dominates the
//! total.
//!
//! The observation this module exploits: a new arm is usually not new. It is a
//! successor to something already measured. `openai/gpt-4.5-turbo` arriving
//! next to a well-characterised `openai/gpt-4` is far more informative than a
//! uniform prior admits.

use crate::arm::Arm;
use crate::posterior::Posterior;
use serde::{Deserialize, Serialize};

/// A prior expressed as Beta parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct InformedPrior {
    /// Prior successes.
    pub alpha: f64,
    /// Prior failures.
    pub beta: f64,
}

impl Default for InformedPrior {
    fn default() -> Self {
        // Mildly optimistic and weakly held: mean 0.8 with the weight of about
        // five observations. Strong enough to get a new arm tried, weak enough
        // that a handful of real failures overrides it.
        InformedPrior {
            alpha: 4.0,
            beta: 1.0,
        }
    }
}

impl InformedPrior {
    /// Build a prior, clamping both parameters to at least 1.0.
    pub fn new(alpha: f64, beta: f64) -> Self {
        InformedPrior {
            alpha: alpha.max(1.0),
            beta: beta.max(1.0),
        }
    }

    /// Derive a prior from an observed arm, scaled by `discount` in `(0, 1]`.
    ///
    /// The discount controls how much of the neighbour's evidence transfers.
    /// At 1.0 the new arm inherits the neighbour's full confidence, which is
    /// wrong — it is a different model. Values near 0.2 transfer the location
    /// of the neighbour's estimate while keeping the posterior loose enough
    /// that a few real observations dominate.
    pub fn from_neighbour(arm: &Arm, discount: f64) -> Self {
        let d = discount.clamp(0.0, 1.0);
        InformedPrior::new(arm.posterior.alpha * d, arm.posterior.beta * d)
    }

    /// The posterior this prior implies.
    pub fn to_posterior(self) -> Posterior {
        Posterior {
            alpha: self.alpha,
            beta: self.beta,
            pulls: 0,
        }
    }
}

/// Strategy for initialising a newly-added arm.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "strategy", rename_all = "snake_case")]
pub enum WarmStart {
    /// Always start at `Beta(1, 1)`.
    Cold,
    /// Always start from the same fixed prior.
    Fixed(InformedPrior),
    /// Inherit from the best-performing arm in the same model family, falling
    /// back to the same provider, then to `fallback`.
    FamilySimilarity {
        /// Fraction of the neighbour's evidence to carry over.
        discount: f64,
        /// Used when no related arm exists.
        fallback: InformedPrior,
    },
}

impl Default for WarmStart {
    fn default() -> Self {
        WarmStart::FamilySimilarity {
            discount: 0.2,
            fallback: InformedPrior::default(),
        }
    }
}

/// A parsed arm identifier of the form `provider/model`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelId {
    /// The portion before the first `/`, or the whole string if absent.
    pub provider: String,
    /// The portion after the first `/`, empty if absent.
    pub model: String,
    /// The model family: the model name truncated at its major version.
    pub family: String,
}

impl ModelId {
    /// Parse `provider/model`, deriving the family.
    pub fn parse(id: &str) -> Self {
        let (provider, model) = match id.split_once('/') {
            Some((p, m)) => (p.to_string(), m.to_string()),
            None => (id.to_string(), String::new()),
        };
        let family = model_family(&model);
        ModelId {
            provider,
            model,
            family,
        }
    }
}

/// Reduce a model name to its family by truncating at the major version.
///
/// The leading run of non-numeric tokens is the name; the first token starting
/// with a digit contributes its integer part; anything after is a variant.
///
/// ```
/// use thompson_sampling::warm_start::model_family;
/// assert_eq!(model_family("gpt-4.5-turbo"), "gpt-4");
/// assert_eq!(model_family("claude-3-5-sonnet"), "claude-3");
/// assert_eq!(model_family("claude-sonnet-4-5"), "claude-sonnet-4");
/// assert_eq!(model_family("gpt-4"), "gpt-4");
/// assert_eq!(model_family("mistral-large"), "mistral-large");
/// ```
pub fn model_family(model: &str) -> String {
    let mut name: Vec<&str> = Vec::new();

    for token in model.split('-') {
        if token.starts_with(|c: char| c.is_ascii_digit()) {
            let major = token.split('.').next().unwrap_or(token);
            if name.is_empty() {
                return major.to_string();
            }
            return format!("{}-{}", name.join("-"), major);
        }
        name.push(token);
    }

    name.join("-")
}

/// Choose a prior for `new_id` given the arms already present.
pub fn prior_for<'a, I>(strategy: &WarmStart, new_id: &str, existing: I) -> InformedPrior
where
    I: IntoIterator<Item = &'a Arm>,
{
    match strategy {
        WarmStart::Cold => InformedPrior::new(1.0, 1.0),
        WarmStart::Fixed(p) => *p,
        WarmStart::FamilySimilarity { discount, fallback } => {
            let target = ModelId::parse(new_id);

            let mut best_in_family: Option<&Arm> = None;
            let mut best_in_provider: Option<&Arm> = None;

            for arm in existing {
                // An arm with no observations carries no information to lend,
                // and chaining warm starts off other warm starts would let one
                // guess propagate through the whole arm set.
                if arm.posterior.pulls == 0 || arm.id == new_id {
                    continue;
                }
                let candidate = ModelId::parse(&arm.id);
                if candidate.provider != target.provider {
                    continue;
                }

                let better = |cur: &Option<&Arm>, next: &Arm| match cur {
                    None => true,
                    Some(c) => next.posterior.mean() > c.posterior.mean(),
                };

                if candidate.family == target.family && better(&best_in_family, arm) {
                    best_in_family = Some(arm);
                }
                if better(&best_in_provider, arm) {
                    best_in_provider = Some(arm);
                }
            }

            match best_in_family.or(best_in_provider) {
                Some(neighbour) => InformedPrior::from_neighbour(neighbour, *discount),
                None => *fallback,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm_with(id: &str, alpha: f64, beta: f64, pulls: u64) -> Arm {
        let mut arm = Arm::new(id.to_string(), Posterior::new(alpha, beta).unwrap());
        arm.posterior.pulls = pulls;
        arm
    }

    #[test]
    fn family_truncates_at_major_version() {
        assert_eq!(model_family("gpt-4.5-turbo"), "gpt-4");
        // `4o` is a distinct generation from `4`, and must not collapse into it.
        assert_eq!(model_family("gpt-4o-mini"), "gpt-4o");
        assert_ne!(model_family("gpt-4o-mini"), model_family("gpt-4-turbo"));
        assert_eq!(model_family("claude-3-5-sonnet"), "claude-3");
        assert_eq!(model_family("claude-3-opus"), "claude-3");
        assert_eq!(model_family("claude-sonnet-4-5"), "claude-sonnet-4");
        assert_eq!(model_family("gpt-4"), "gpt-4");
        assert_eq!(model_family("mistral-large"), "mistral-large");
        assert_eq!(model_family(""), "");
    }

    #[test]
    fn siblings_share_a_family() {
        assert_eq!(
            model_family("gpt-4.5-turbo"),
            model_family("gpt-4-turbo-preview")
        );
        assert_eq!(
            model_family("claude-3-5-sonnet"),
            model_family("claude-3-opus")
        );
        assert_ne!(model_family("gpt-4"), model_family("gpt-5"));
    }

    #[test]
    fn parses_provider_and_model() {
        let id = ModelId::parse("openai/gpt-4.5-turbo");
        assert_eq!(id.provider, "openai");
        assert_eq!(id.model, "gpt-4.5-turbo");
        assert_eq!(id.family, "gpt-4");

        let bare = ModelId::parse("local-llama");
        assert_eq!(bare.provider, "local-llama");
        assert_eq!(bare.model, "");
    }

    #[test]
    fn inherits_from_same_family() {
        let existing = vec![
            arm_with("openai/gpt-4", 90.0, 10.0, 100),
            arm_with("anthropic/claude-3-opus", 200.0, 1.0, 201),
        ];
        let strategy = WarmStart::default();
        let prior = prior_for(&strategy, "openai/gpt-4.5-turbo", &existing);

        // 20% of Beta(90, 10), not the stronger Anthropic arm from another
        // provider and not the fallback.
        assert!((prior.alpha - 18.0).abs() < 1e-9, "alpha {}", prior.alpha);
        assert!((prior.beta - 2.0).abs() < 1e-9, "beta {}", prior.beta);
    }

    #[test]
    fn falls_back_to_provider_then_to_default() {
        let existing = vec![arm_with("openai/gpt-3.5-turbo", 50.0, 50.0, 100)];
        let strategy = WarmStart::default();

        // Different family, same provider: still informative.
        let prior = prior_for(&strategy, "openai/gpt-5", &existing);
        assert!((prior.alpha - 10.0).abs() < 1e-9);

        // Unknown provider: fallback.
        let unknown = prior_for(&strategy, "cohere/command-r", &existing);
        assert_eq!(unknown, InformedPrior::default());
    }

    #[test]
    fn does_not_inherit_from_an_unpulled_arm() {
        // A warm-started arm has a concentrated posterior and zero pulls.
        // Letting it seed the next arm would launder a guess into evidence.
        let existing = vec![arm_with("openai/gpt-4", 18.0, 2.0, 0)];
        let prior = prior_for(&WarmStart::default(), "openai/gpt-4.5", &existing);
        assert_eq!(prior, InformedPrior::default());
    }

    #[test]
    fn cold_strategy_ignores_neighbours() {
        let existing = vec![arm_with("openai/gpt-4", 900.0, 10.0, 910)];
        let prior = prior_for(&WarmStart::Cold, "openai/gpt-4.5", &existing);
        assert_eq!(prior, InformedPrior::new(1.0, 1.0));
    }

    #[test]
    fn prior_parameters_never_drop_below_one() {
        let existing = vec![arm_with("openai/gpt-4", 1.2, 1.2, 2)];
        let strategy = WarmStart::FamilySimilarity {
            discount: 0.01,
            fallback: InformedPrior::default(),
        };
        let prior = prior_for(&strategy, "openai/gpt-4.5", &existing);
        assert!(prior.alpha >= 1.0 && prior.beta >= 1.0);
    }

    #[test]
    fn picks_the_best_performer_within_a_family() {
        let existing = vec![
            arm_with("openai/gpt-4-turbo", 30.0, 70.0, 100),
            arm_with("openai/gpt-4-preview", 80.0, 20.0, 100),
        ];
        let prior = prior_for(&WarmStart::default(), "openai/gpt-4.5", &existing);
        assert!((prior.alpha - 16.0).abs() < 1e-9, "alpha {}", prior.alpha);
    }
}
