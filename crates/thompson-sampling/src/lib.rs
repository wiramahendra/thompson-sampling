//! Thompson Sampling for provider selection.
//!
//! A Beta-Bernoulli multi-armed bandit built for the case where the arms are
//! service providers or models: the arm set changes underneath you, rewards are
//! multi-objective rather than binary, and the cost of a bad draw is a real
//! request paid for in latency and money.
//!
//! Three things here are not in a textbook implementation:
//!
//! - [`sampler`] treats the Beta draw as a pluggable strategy, with an exact
//!   reference implementation and faithful reproductions of the approximations
//!   found in deployed routers, so the cost of approximating can be measured
//!   instead of assumed.
//! - [`warm_start`] gives a newly-added arm a prior derived from a related arm
//!   already in the set, on the theory that `gpt-4.5` arriving next to a
//!   well-measured `gpt-4` is not a blank slate.
//! - [`reward`] makes the collapse from a multi-objective outcome to the single
//!   scalar a Beta posterior can consume explicit and configurable.
//!
//! # Example
//!
//! ```
//! use rand::rngs::SmallRng;
//! use rand::SeedableRng;
//! use thompson_sampling::{Outcome, ThompsonSampling};
//!
//! let mut rng = SmallRng::seed_from_u64(42);
//! let mut policy = ThompsonSampling::with_defaults([
//!     "openai/gpt-4",
//!     "anthropic/claude-3-5-sonnet",
//! ]);
//!
//! for _ in 0..100 {
//!     let provider = policy.select(&mut rng).unwrap();
//!
//!     // Dispatch the request, then report what happened.
//!     let outcome = Outcome::new(320.0, true, 0.0012).with_quality(0.87);
//!     policy.record_outcome(&mut rng, &provider, &outcome).unwrap();
//! }
//!
//! // A new model ships. It inherits a prior from its closest relative rather
//! // than restarting from scratch.
//! policy.add_arm("openai/gpt-4.5-turbo".to_string());
//! assert!(policy.arm("openai/gpt-4.5-turbo").unwrap().warm_started);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod arm;
pub mod context;
pub mod discount;
pub mod error;
pub mod health;
pub mod linear;
pub mod observer;
pub mod otel;
pub mod persistence;
pub mod policy;
pub mod posterior;
pub mod reward;
pub mod sampler;
pub mod selection;
pub mod warm_start;

pub use arm::{Arm, ArmStats};
pub use discount::{DiscountPolicy, FixedDiscount};
pub use error::{Error, Result};
pub use linear::{LinearConfig, LinearPolicy, LinearWeights};
pub use observer::{Event, NoopObserver, PolicyObserver};
pub use persistence::{FileStore, MemoryStore, SnapshotStore};
pub use policy::{Config, Selection, Snapshot, ThompsonSampling};
pub use posterior::{Posterior, UpdateRule};
pub use reward::{Breakdown, Outcome, RewardPolicy, Weights};
pub use sampler::{BetaSampler, Exact};
pub use selection::{PhasedStrategy, SelectionStrategy, ThompsonStrategy, UcbRegularizedStrategy};
pub use warm_start::{InformedPrior, WarmStart};
