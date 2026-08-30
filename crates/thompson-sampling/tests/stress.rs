//! Stress harness: high-throughput, drift, churn, concurrency.

use rand::rngs::SmallRng;
use rand::SeedableRng;
use thompson_sampling::{Config, ThompsonSampling};

fn rng() -> SmallRng {
    SmallRng::seed_from_u64(0xBEEF)
}

#[test]
fn high_throughput_100_arms() {
    let mut policy = ThompsonSampling::new(Config::default(), Box::new(thompson_sampling::Exact));
    for i in 0..100 {
        policy.add_arm(format!("arm-{i}"));
    }
    let mut r = rng();
    for _ in 0..5000 {
        let id = policy.select(&mut r).unwrap();
        let reward = if id == "arm-0" { 0.9 } else { 0.3 };
        policy.record(&mut r, &id, reward).unwrap();
    }
    assert_eq!(policy.best_arm(1), Some("arm-0"));
}

#[test]
fn rapid_churn_every_500() {
    let mut policy = ThompsonSampling::with_defaults(["a"]);
    let mut r = rng();
    for t in 0..5000 {
        if t % 500 == 0 && t > 0 {
            policy.add_arm(format!("arm-{t}"));
        }
        let id = policy.select(&mut r).unwrap();
        policy.record(&mut r, &id, 0.5).unwrap();
    }
    assert!(policy.len() >= 10);
}

#[test]
fn drift_with_discount_follow() {
    let mut policy = ThompsonSampling::new(
        Config {
            discount: Some(0.99),
            ..Config::default()
        },
        Box::new(thompson_sampling::Exact),
    );
    for id in ["a", "b"] {
        policy.add_arm(id.into());
    }
    let mut r = rng();
    for _ in 0..500 {
        let id = policy.select(&mut r).unwrap();
        let reward = if id == "a" { 0.9 } else { 0.1 };
        policy.record(&mut r, &id, reward).unwrap();
    }
    for _ in 0..1500 {
        let id = policy.select(&mut r).unwrap();
        let reward = if id == "b" { 0.9 } else { 0.1 };
        policy.record(&mut r, &id, reward).unwrap();
    }
    assert_eq!(policy.best_arm(1), Some("b"));
}
