//! Synthetic environments with known ground truth.
//!
//! Rewards are Bernoulli draws from a known per-arm success probability. That
//! keeps regret exactly computable — the whole point of a harness — and matches
//! the Beta-Bernoulli model the policy assumes, so any regret measured is
//! attributable to the policy rather than to model mismatch.

use rand::Rng;

/// How an arm's true success probability evolves over a run.
#[derive(Debug, Clone, Copy)]
pub enum Schedule {
    /// Fixed for the whole run.
    Constant(f64),
    /// One value up to `at`, another from `at` onward.
    Switch {
        /// Probability before the changepoint.
        before: f64,
        /// Probability from the changepoint onward.
        after: f64,
        /// Round at which the change happens.
        at: usize,
    },
}

impl Schedule {
    /// The true success probability at round `t`.
    pub fn at(&self, t: usize) -> f64 {
        match *self {
            Schedule::Constant(p) => p,
            Schedule::Switch { before, after, at } => {
                if t < at {
                    before
                } else {
                    after
                }
            }
        }
    }
}

/// One arm of a synthetic environment.
#[derive(Debug, Clone)]
pub struct ArmSpec {
    /// Arm identifier, conventionally `provider/model`.
    pub id: String,
    /// First round at which this arm exists. Arms with a non-zero value model
    /// a provider shipping a new model mid-run.
    pub available_from: usize,
    /// True success probability over time.
    pub schedule: Schedule,
}

impl ArmSpec {
    /// An arm present from the start with a fixed success probability.
    pub fn fixed(id: &str, p: f64) -> Self {
        ArmSpec {
            id: id.to_string(),
            available_from: 0,
            schedule: Schedule::Constant(p),
        }
    }

    /// An arm that appears at round `t`.
    pub fn arriving(id: &str, p: f64, t: usize) -> Self {
        ArmSpec {
            id: id.to_string(),
            available_from: t,
            schedule: Schedule::Constant(p),
        }
    }

    /// An arm whose quality changes at a changepoint.
    pub fn switching(id: &str, before: f64, after: f64, at: usize) -> Self {
        ArmSpec {
            id: id.to_string(),
            available_from: 0,
            schedule: Schedule::Switch { before, after, at },
        }
    }
}

/// The shape of the reward an arm pays out.
#[derive(Debug, Clone, Copy)]
pub enum RewardKind {
    /// Reward is 0 or 1, drawn with probability equal to the arm's mean.
    ///
    /// This matches the Beta-Bernoulli model exactly, so all three update rules
    /// coincide and any measured regret is down to selection alone.
    Bernoulli,

    /// Reward is continuous: the arm's mean plus uniform noise of half-width
    /// `spread`, clamped to `[0, 1]`.
    ///
    /// This is the realistic case for a router — a composite of latency, cost
    /// and quality is never binary — and it is where the update rule starts to
    /// matter, because thresholding maps a band of distinct rewards onto the
    /// same observation.
    Graded {
        /// Half-width of the uniform noise around the mean.
        spread: f64,
    },
}

/// A named scenario: a set of arms and a horizon.
#[derive(Debug, Clone)]
pub struct Scenario {
    /// Short identifier used in reports.
    pub name: &'static str,
    /// One-line description of what the scenario probes.
    pub description: &'static str,
    /// The arms.
    pub arms: Vec<ArmSpec>,
    /// Number of rounds per run.
    pub horizon: usize,
    /// How rewards are generated.
    pub reward_kind: RewardKind,
}

impl Scenario {
    /// Arms available at round `t`.
    pub fn available(&self, t: usize) -> impl Iterator<Item = &ArmSpec> {
        self.arms.iter().filter(move |a| a.available_from <= t)
    }

    /// Arms that first become available exactly at round `t`.
    pub fn arrivals_at(&self, t: usize) -> impl Iterator<Item = &ArmSpec> {
        self.arms.iter().filter(move |a| a.available_from == t)
    }

    /// True success probability of `id` at round `t`.
    pub fn mean(&self, id: &str, t: usize) -> f64 {
        self.arms
            .iter()
            .find(|a| a.id == id)
            .map(|a| a.schedule.at(t))
            .unwrap_or(0.0)
    }

    /// Best achievable success probability at round `t`.
    pub fn best_mean(&self, t: usize) -> f64 {
        self.available(t)
            .map(|a| a.schedule.at(t))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Whether `id` is optimal at round `t`.
    pub fn is_optimal(&self, id: &str, t: usize) -> bool {
        (self.mean(id, t) - self.best_mean(t)).abs() < 1e-12
    }

    /// Draw a reward for pulling `id` at round `t`.
    pub fn draw<R: Rng + ?Sized>(&self, rng: &mut R, id: &str, t: usize) -> f64 {
        let mean = self.mean(id, t);
        match self.reward_kind {
            RewardKind::Bernoulli => {
                if rng.gen::<f64>() < mean {
                    1.0
                } else {
                    0.0
                }
            }
            RewardKind::Graded { spread } => {
                let noise = (rng.gen::<f64>() - 0.5) * 2.0 * spread;
                (mean + noise).clamp(0.0, 1.0)
            }
        }
    }
}

/// The built-in scenario set.
pub fn scenarios() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "easy",
            description: "Three well-separated arms. Any working bandit should solve this.",
            arms: vec![
                ArmSpec::fixed("openai/gpt-4", 0.90),
                ArmSpec::fixed("anthropic/claude-3-opus", 0.55),
                ArmSpec::fixed("meta/llama-3", 0.20),
            ],
            horizon: 5_000,
            reward_kind: RewardKind::Bernoulli,
        },
        Scenario {
            name: "hard",
            description: "Five near-identical arms. Separating them needs real exploration.",
            arms: vec![
                ArmSpec::fixed("openai/gpt-4", 0.50),
                ArmSpec::fixed("openai/gpt-4-turbo", 0.48),
                ArmSpec::fixed("anthropic/claude-3-opus", 0.47),
                ArmSpec::fixed("anthropic/claude-3-haiku", 0.46),
                ArmSpec::fixed("meta/llama-3", 0.45),
            ],
            horizon: 20_000,
            reward_kind: RewardKind::Bernoulli,
        },
        Scenario {
            name: "drift",
            description: "Best and worst arms swap at the midpoint. Stale evidence is a trap.",
            arms: vec![
                ArmSpec::switching("openai/gpt-4", 0.80, 0.30, 7_500),
                ArmSpec::fixed("anthropic/claude-3-opus", 0.50),
                ArmSpec::switching("meta/llama-3", 0.30, 0.80, 7_500),
            ],
            horizon: 15_000,
            reward_kind: RewardKind::Bernoulli,
        },
        Scenario {
            name: "churn",
            description: "A better model ships mid-run. Cold-start cost is the whole story.",
            arms: vec![
                ArmSpec::fixed("openai/gpt-4", 0.60),
                ArmSpec::fixed("anthropic/claude-3-opus", 0.55),
                ArmSpec::fixed("meta/llama-3", 0.30),
                ArmSpec::arriving("openai/gpt-4.5-turbo", 0.85, 3_000),
            ],
            horizon: 10_000,
            reward_kind: RewardKind::Bernoulli,
        },
        Scenario {
            name: "treadmill",
            description:
                "Vendors keep shipping successors that are no better. Most churn is noise.",
            arms: vec![
                // The incumbent nobody should ever leave.
                ArmSpec::fixed("openai/gpt-4", 0.75),
                // Three families that keep releasing, none of which is
                // competitive. Under a uniform prior each arrival is a blank
                // slate that has to be re-measured from nothing; under family
                // inheritance each arrives already carrying its predecessor's
                // unimpressive record.
                ArmSpec::fixed("meta/llama-3", 0.25),
                ArmSpec::arriving("meta/llama-3.1-8b", 0.25, 500),
                ArmSpec::arriving("meta/llama-3.2-8b", 0.26, 1_000),
                ArmSpec::arriving("meta/llama-3.3-8b", 0.24, 1_500),
                ArmSpec::fixed("mistral/mixtral-8x7b", 0.30),
                ArmSpec::arriving("mistral/mixtral-8x22b", 0.31, 2_000),
                ArmSpec::fixed("cohere/command-r", 0.28),
                ArmSpec::arriving("cohere/command-r-plus", 0.29, 2_500),
            ],
            horizon: 5_000,
            reward_kind: RewardKind::Bernoulli,
        },
        Scenario {
            name: "graded",
            description: "Continuous rewards that a success threshold flattens into a tie.",
            arms: vec![
                // Both arms clear a 0.6 success threshold on essentially every
                // request, so a policy that binarises at 0.6 sees two perfect
                // arms and cannot prefer the genuinely better one.
                ArmSpec::fixed("openai/gpt-4", 0.95),
                ArmSpec::fixed("anthropic/claude-3-opus", 0.70),
                ArmSpec::fixed("meta/llama-3", 0.35),
            ],
            horizon: 10_000,
            reward_kind: RewardKind::Graded { spread: 0.05 },
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn switch_schedule_changes_at_the_changepoint() {
        let s = Schedule::Switch {
            before: 0.8,
            after: 0.3,
            at: 100,
        };
        assert_eq!(s.at(99), 0.8);
        assert_eq!(s.at(100), 0.3);
    }

    #[test]
    fn best_mean_accounts_for_arm_availability() {
        let scenarios = scenarios();
        let churn = scenarios.iter().find(|s| s.name == "churn").unwrap();

        // Before the new model ships, the incumbent is optimal.
        assert!((churn.best_mean(0) - 0.60).abs() < 1e-12);
        assert!(churn.is_optimal("openai/gpt-4", 0));

        // After it ships, it is.
        assert!((churn.best_mean(3_000) - 0.85).abs() < 1e-12);
        assert!(churn.is_optimal("openai/gpt-4.5-turbo", 3_000));
        assert!(!churn.is_optimal("openai/gpt-4", 3_000));
    }

    #[test]
    fn drift_scenario_swaps_the_optimal_arm() {
        let scenarios = scenarios();
        let drift = scenarios.iter().find(|s| s.name == "drift").unwrap();
        assert!(drift.is_optimal("openai/gpt-4", 0));
        assert!(drift.is_optimal("meta/llama-3", 7_500));
    }

    #[test]
    fn draws_match_the_configured_rate() {
        let scenarios = scenarios();
        let easy = scenarios.iter().find(|s| s.name == "easy").unwrap();
        let mut rng = SmallRng::seed_from_u64(4);
        let n = 100_000;
        let hits: f64 = (0..n).map(|_| easy.draw(&mut rng, "openai/gpt-4", 0)).sum();
        assert!((hits / n as f64 - 0.90).abs() < 0.01);
    }

    #[test]
    fn graded_rewards_are_indistinguishable_under_a_success_threshold() {
        let scenarios = scenarios();
        let graded = scenarios.iter().find(|s| s.name == "graded").unwrap();
        let mut rng = SmallRng::seed_from_u64(9);

        // The scenario only makes its point if the two good arms genuinely
        // straddle no threshold at 0.6 — assert the setup, not just the result.
        for _ in 0..20_000 {
            let a = graded.draw(&mut rng, "openai/gpt-4", 0);
            let b = graded.draw(&mut rng, "anthropic/claude-3-opus", 0);
            let c = graded.draw(&mut rng, "meta/llama-3", 0);
            assert!(a > 0.6 && b > 0.6, "a={a} b={b}");
            assert!(c < 0.6, "c={c}");
            assert!(a > b, "the better arm must stay better on raw reward");
        }
    }

    #[test]
    fn arrivals_fire_exactly_once() {
        let scenarios = scenarios();
        let churn = scenarios.iter().find(|s| s.name == "churn").unwrap();
        assert_eq!(churn.arrivals_at(0).count(), 3);
        assert_eq!(churn.arrivals_at(2_999).count(), 0);
        assert_eq!(churn.arrivals_at(3_000).count(), 1);
        assert_eq!(churn.arrivals_at(3_001).count(), 0);
    }
}
