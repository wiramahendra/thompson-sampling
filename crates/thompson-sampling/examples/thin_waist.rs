//! Thin-waist integration example: two calls to add to any gateway.
//!
//! ```rust
//! use thompson_sampling::{ThompsonSampling, Outcome};
//! use rand::SeedableRng;
//! ```

use rand::rngs::SmallRng;
use rand::SeedableRng;
use thompson_sampling::observer::PolicyObserver;
use thompson_sampling::posterior::Posterior;
use thompson_sampling::{Outcome, SnapshotStore, ThompsonSampling};

#[derive(Debug, Default)]
struct LoggingObserver;

impl PolicyObserver for LoggingObserver {
    fn on_select(&self, chosen: &str, scores: &[(&str, f64)]) {
        // In production, emit to OTEL/Prometheus instead of println.
        eprintln!("select -> {chosen} scores={scores:?}");
    }
    fn on_record(&self, arm: &str, reward: f64, posterior: &Posterior) {
        eprintln!(
            "record {arm} reward={reward:.3} posterior mean={:.3}",
            posterior.mean()
        );
    }
    fn on_arm_added(&self, id: &str, warm_started: bool) {
        eprintln!("arm added {id} warm_started={warm_started}");
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut policy = ThompsonSampling::with_defaults(["openai/gpt-4", "anthropic/claude-3-opus"])
        .with_observer(Box::new(LoggingObserver));

    for _ in 0..5 {
        // Thin waist: 1. Select
        let provider = policy.select(&mut rng)?;

        // ... forward request to `provider` ...

        // 2. Record outcome collapsed via RewardPolicy
        let outcome = Outcome::new(320.0, true, 0.0012).with_quality(0.87);
        policy.record_outcome(&mut rng, &provider, &outcome)?;
        println!("stats: {:?}", policy.stats());
    }

    // New model ships — inherits from family via warm_start
    policy.add_arm("openai/gpt-4.5-turbo".to_string());
    assert!(policy.arm("openai/gpt-4.5-turbo").unwrap().warm_started);

    // Persistence is first-class: FileStore or MemoryStore
    let store = thompson_sampling::MemoryStore::new();
    policy.save_to_store(&store)?;
    let snapshot = store.load()?.unwrap();
    println!(
        "snapshot version {} with {} arms",
        snapshot.version,
        snapshot.arms.len()
    );

    // Discount introspection
    println!("effective_memory: {}", policy.effective_memory());

    Ok(())
}
