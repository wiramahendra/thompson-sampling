//! Running a policy against a scenario and aggregating across seeds.

use crate::env::Scenario;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::time::Instant;
use thompson_sampling::ThompsonSampling;

/// A named policy configuration under test.
pub struct Treatment {
    /// Label used in reports.
    pub label: String,
    /// Which experiment group this belongs to.
    pub group: &'static str,
    /// Builds a fresh policy. Called once per run so no state leaks between
    /// seeds.
    pub build: Box<dyn Fn() -> ThompsonSampling>,
}

/// Outcome of a single run.
#[derive(Debug, Clone, Copy)]
pub struct RunResult {
    /// Cumulative regret against the per-round optimal arm.
    pub regret: f64,
    /// Fraction of rounds on which an optimal arm was chosen.
    pub optimal_share: f64,
    /// Mean wall-clock time per selection, in nanoseconds.
    pub nanos_per_decision: f64,
}

/// Run one policy against one scenario for one seed.
///
/// Regret is measured against the best arm *available at that round*, so an
/// arm arriving late does not retroactively penalise earlier decisions.
pub fn run(scenario: &Scenario, treatment: &Treatment, seed: u64) -> RunResult {
    let mut policy = (treatment.build)();
    let mut rng = SmallRng::seed_from_u64(seed);

    let mut regret = 0.0;
    let mut optimal_rounds = 0usize;
    let mut select_time = std::time::Duration::ZERO;

    for t in 0..scenario.horizon {
        // Register any arms that become available this round.
        for spec in scenario.arrivals_at(t) {
            policy.add_arm(spec.id.clone());
        }

        let started = Instant::now();
        let chosen = policy
            .select(&mut rng)
            .expect("scenario always has an available arm");
        select_time += started.elapsed();

        let reward = scenario.draw(&mut rng, &chosen, t);
        policy
            .record(&mut rng, &chosen, reward)
            .expect("selected arm is registered");

        regret += scenario.best_mean(t) - scenario.mean(&chosen, t);
        if scenario.is_optimal(&chosen, t) {
            optimal_rounds += 1;
        }
    }

    RunResult {
        regret,
        optimal_share: optimal_rounds as f64 / scenario.horizon as f64,
        nanos_per_decision: select_time.as_nanos() as f64 / scenario.horizon as f64,
    }
}

/// Aggregate of several seeds of the same treatment and scenario.
#[derive(Debug, Clone)]
pub struct Summary {
    /// Scenario name.
    pub scenario: &'static str,
    /// Treatment label.
    pub treatment: String,
    /// Experiment group.
    pub group: &'static str,
    /// Number of seeds.
    pub seeds: usize,
    /// Mean cumulative regret.
    pub mean_regret: f64,
    /// Standard error of the mean regret.
    pub stderr_regret: f64,
    /// Mean fraction of optimal choices.
    pub mean_optimal_share: f64,
    /// Mean nanoseconds per selection.
    pub nanos_per_decision: f64,
}

impl Summary {
    /// Half-width of an approximate 95% confidence interval on mean regret.
    pub fn regret_ci95(&self) -> f64 {
        1.96 * self.stderr_regret
    }
}

/// Run `seeds` independent runs and summarise.
pub fn evaluate(scenario: &Scenario, treatment: &Treatment, seeds: usize) -> Summary {
    assert!(seeds > 0, "need at least one seed");

    let results: Vec<RunResult> = (0..seeds)
        .map(|s| run(scenario, treatment, 0x5EED_0000 + s as u64))
        .collect();

    let n = seeds as f64;
    let mean_regret = results.iter().map(|r| r.regret).sum::<f64>() / n;

    // Sample standard deviation; undefined for a single seed.
    let stderr_regret = if seeds > 1 {
        let var = results
            .iter()
            .map(|r| (r.regret - mean_regret).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        (var / n).sqrt()
    } else {
        f64::NAN
    };

    Summary {
        scenario: scenario.name,
        treatment: treatment.label.clone(),
        group: treatment.group,
        seeds,
        mean_regret,
        stderr_regret,
        mean_optimal_share: results.iter().map(|r| r.optimal_share).sum::<f64>() / n,
        nanos_per_decision: results.iter().map(|r| r.nanos_per_decision).sum::<f64>() / n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::scenarios;
    use crate::treatments;

    fn easy() -> Scenario {
        scenarios().into_iter().find(|s| s.name == "easy").unwrap()
    }

    fn exact_treatment() -> Treatment {
        treatments::sampler_group()
            .into_iter()
            .find(|t| t.label == "exact")
            .unwrap()
    }

    #[test]
    fn a_run_is_deterministic_in_its_seed() {
        let scenario = easy();
        let t = exact_treatment();
        let a = run(&scenario, &t, 7);
        let b = run(&scenario, &t, 7);
        assert_eq!(a.regret, b.regret);
        assert_eq!(a.optimal_share, b.optimal_share);
    }

    #[test]
    fn different_seeds_give_different_runs() {
        let scenario = easy();
        let t = exact_treatment();
        assert_ne!(run(&scenario, &t, 1).regret, run(&scenario, &t, 2).regret);
    }

    #[test]
    fn regret_is_non_negative_and_bounded_by_the_horizon() {
        let scenario = easy();
        let summary = evaluate(&scenario, &exact_treatment(), 3);
        assert!(summary.mean_regret >= 0.0);
        // The worst possible arm loses 0.7 per round in this scenario.
        assert!(summary.mean_regret <= 0.7 * scenario.horizon as f64);
        assert!((0.0..=1.0).contains(&summary.mean_optimal_share));
    }

    #[test]
    fn exact_sampling_beats_no_exploration_on_the_hard_scenario() {
        // The headline claim of the harness. If this ever fails, either the
        // scenario stopped being hard or the policy stopped exploring.
        let scenario = scenarios().into_iter().find(|s| s.name == "hard").unwrap();
        let group = treatments::sampler_group();

        let exact = group.iter().find(|t| t.label == "exact").unwrap();
        let broken = group.iter().find(|t| t.label == "deterministic").unwrap();

        let a = evaluate(&scenario, exact, 5);
        let b = evaluate(&scenario, broken, 5);
        assert!(
            a.mean_regret < b.mean_regret,
            "exact regret {} should beat deterministic {}",
            a.mean_regret,
            b.mean_regret
        );
    }

    #[test]
    fn single_seed_reports_undefined_stderr_rather_than_zero() {
        let summary = evaluate(&easy(), &exact_treatment(), 1);
        assert!(summary.stderr_regret.is_nan());
    }
}
