//! Regret and throughput harness as a library.
//!
//! The binary in `src/main.rs` is a thin CLI over this library. Importing
//! the harness directly lets you evaluate custom policies, samplers, or
//! scenarios without forking the harness.
//!
//! ```rust
//! use thompson_sim::{env, experiment, treatments};
//! use thompson_sim::experiment::{evaluate, run};
//! ```

pub mod env;
pub mod experiment;
pub mod trace;
pub mod treatments;
