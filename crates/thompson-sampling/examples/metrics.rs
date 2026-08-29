//! Observer -> metrics example. Emits Prometheus-style counters.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use thompson_sampling::observer::PolicyObserver;
use thompson_sampling::posterior::Posterior;

#[derive(Debug, Default, Clone)]
struct MetricsObserver {
    selects: Arc<Mutex<HashMap<String, u64>>>,
    rewards: Arc<Mutex<HashMap<String, (f64, u64)>>>, // sum, count
    discounts: Arc<Mutex<u64>>,
}

impl PolicyObserver for MetricsObserver {
    fn on_select(&self, chosen: &str, _scores: &[(&str, f64)]) {
        *self.selects.lock().unwrap().entry(chosen.to_string()).or_insert(0) += 1;
    }
    fn on_record(&self, arm: &str, reward: f64, _posterior: &Posterior) {
        let mut m = self.rewards.lock().unwrap();
        let e = m.entry(arm.to_string()).or_insert((0.0, 0));
        e.0 += reward;
        e.1 += 1;
    }
    fn on_discount(&self, _factor: f64) {
        *self.discounts.lock().unwrap() += 1;
    }
}

impl MetricsObserver {
    fn print_prometheus(&self) {
        println!("# HELP thompson_selects_total selects per arm");
        for (arm, cnt) in self.selects.lock().unwrap().iter() {
            println!("thompson_selects_total{{arm=\"{arm}\"}} {cnt}");
        }
        println!("# HELP thompson_reward_mean mean reward per arm");
        for (arm, (sum, cnt)) in self.rewards.lock().unwrap().iter() {
            println!("thompson_reward_mean{{arm=\"{arm}\"}} {:.3}", sum / *cnt as f64);
        }
        println!("# HELP thompson_discounts_total discounts applied");
        println!("thompson_discounts_total {}", *self.discounts.lock().unwrap());
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use rand::SeedableRng;
    use thompson_sampling::ThompsonSampling;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(1);
    let obs = MetricsObserver::default();
    let mut policy = ThompsonSampling::with_defaults(["a", "b"]).with_observer(Box::new(obs.clone()));
    for _ in 0..20 {
        let id = policy.select(&mut rng)?;
        let reward = if id == "a" { 0.9 } else { 0.1 };
        policy.record(&mut rng, &id, reward)?;
    }
    obs.print_prometheus();
    for s in policy.stats() {
        println!("arm {} mean {:.3} pulls {} width {:.3}", s.id, s.posterior_mean, s.pulls, s.credible_width);
    }
    Ok(())
}
