# Thompson Sampling

A Beta-Bernoulli multi-armed bandit for **provider selection** — choosing which
model or service handles each request — in Rust and Go, with a simulation
harness that measures what the usual production shortcuts actually cost.

Thompson Sampling itself is from 1933. What is not standard, and what this
repository is actually about, is everything that goes wrong when you point it at
a fleet of LLM providers: the arm set changes underneath you, rewards are
multi-objective rather than binary, and the Beta draw at the centre of the
algorithm is expensive enough that people quietly replace it with something
cheaper.

```
crates/thompson-sampling/   Rust library
crates/thompson-sim/        regret + throughput harness
go/thompson/                Go port, dependency-free
docs/FINDINGS.md            what the harness found
docs/results.csv            full results, 50 seeds per cell
```

## The question

Sampling from `Beta(α, β)` needs two Gamma variates. That is a few hundred
nanoseconds, and on a hot routing path somebody eventually replaces it with the
posterior mean plus a bit of noise. The substitution is easy to justify, hard to
review, and the resulting policy still *looks* like it is working: it prefers
good arms, its numbers move in the right direction, and nothing crashes.

So this library treats the Beta draw as a pluggable strategy, ships an exact
reference implementation alongside faithful reproductions of the approximations
found in deployed routers, and measures them against each other.

```rust
use thompson_sampling::{Outcome, ThompsonSampling};

let mut policy = ThompsonSampling::with_defaults([
    "openai/gpt-4",
    "anthropic/claude-3-5-sonnet",
]);

let provider = policy.select(&mut rng)?;
let outcome = Outcome::new(320.0, true, 0.0012).with_quality(0.87);
policy.record_outcome(&mut rng, &provider, &outcome)?;

// A new model ships. It inherits a prior from its closest relative
// rather than restarting from a blank slate.
policy.add_arm("openai/gpt-4.5-turbo".to_string());
```

```go
policy := thompson.NewDefault("openai/gpt-4", "anthropic/claude-3-5-sonnet")

provider, err := policy.Select(rng)
outcome := thompson.NewOutcome(320, true, 0.0012).WithQuality(0.87)
err = policy.RecordOutcome(rng, provider, outcome)
```

## What the harness found

50 seeds per cell, cumulative regret, lower is better. Full tables in
[`docs/FINDINGS.md`](docs/FINDINGS.md).

**Approximations win on the benchmark you would naturally write, and lose on
the ones that resemble production.**

| sampler | easy | hard | drift | churn | ns/select |
|---|---|---|---|---|---|
| exact | 7.9 | **159.7** | 1330 | **28.2** | 490 |
| mean + gaussian | 10.4 | 171.4 | 1715 | 30.1 | 227 |
| mean + uniform | **2.5** | 383.4 | 3055 | 397.9 | 114 |
| concentration-switched | **2.5** | 250.8 | 3415 | 539.2 | 172 |
| deterministic | 179.0 | 493.2 | 3811 | 1462.7 | 115 |

On three well-separated arms the cheap samplers beat exact sampling, because
exploration is pure waste when the answer is obvious. Move to five near-identical
arms and they lose by 2.4×; add a mid-run arrival and they lose by 14×. A
benchmark built from obviously-different arms will certify an approximation that
fails on the workload you actually have.

**The reward-to-posterior step can dominate everything else.** Collapsing a
continuous reward with a success threshold — the most common choice — cost 187×
more regret than either principled alternative:

| update rule | regret | optimal |
|---|---|---|
| binarize @ 0.6 | 1289.3 | 48.4% |
| bernoulli (Agrawal & Goyal) | 8.1 | 99.8% |
| fractional | **6.9** | 99.8% |

Two arms that both clear the threshold become the same observation, and the
bandit stops being able to tell them apart. This is a one-line change with a
larger effect than the sampler.

**Warm-start priors mostly compensate for a broken sampler.** Inheriting a prior
from a related model is appealing — `gpt-4.5` arriving next to a
well-characterised `gpt-4` is not a blank slate — and under an approximate
sampler it is worth 9×. Under an exact sampler the benefit disappears:

| warm start | exact sampler | approximate sampler |
|---|---|---|
| cold `Beta(1,1)` | **28.2** | 397.9 |
| fixed optimistic | 27.0 | **44.0** |
| family similarity | 35.3 | 126.2 |

Exact Thompson Sampling already explores a fresh arm aggressively, because
`Beta(1,1)` is uniform and half its draws land above 0.5. There is little
cold-start cost left to eliminate. The machinery is real, and it is treating a
symptom.

**Discounting is worth more than any of it, if your arms drift.** On a scenario
where the best and worst arms swap places, a per-round discount of 0.999 cut
regret by 5.9×. Nothing else in this study came close on that scenario.

## Design notes

- **Arms live in an ordered map.** With a hash map, iteration order supplies
  incidental randomness that can mask a sampler doing no exploration of its own.
  Runs here are reproducible from a seed.
- **`select` does not mutate.** Nothing is learned until an outcome comes back.
  Selecting without recording is legitimate — requests get cancelled — and
  simply teaches nothing.
- **The Go port uses a single mutex.** A policy lock plus per-arm locks invites a
  lock-order inversion that only deadlocks under production concurrency;
  `TestConcurrentUseIsSafe` runs under `-race`.
- **Snapshots are not bit-exact.** JSON float round trips can shift a value by
  one ULP. Compare restored policies with a tolerance.

## Running it

```sh
cargo test --workspace
cargo run --release -p thompson-sim -- --seeds 50
cargo run --release -p thompson-sim -- --group sampler --scenario hard
cargo run --release -p thompson-sim -- --list

cd go && go test -race ./...
```

## Scope and limits

The environments are synthetic. Rewards are drawn from known distributions,
which is what makes regret exactly computable, but it also means every result
here is a statement about the algorithm rather than about any particular
provider fleet. Replaying real routing traces is the obvious next step and is
not done yet.

The reward model treats components as linearly separable and independent. Real
latency and quality correlate, often strongly, and no result here accounts for
that.

## References

- Thompson, W. R. (1933). On the likelihood that one unknown probability
  exceeds another in view of the evidence of two samples. *Biometrika*.
- Agrawal, S. & Goyal, N. (2012). Analysis of Thompson Sampling for the
  multi-armed bandit problem. *COLT*.
- Marsaglia, G. & Tsang, W. W. (2000). A simple method for generating gamma
  variables. *ACM TOMS*.
- Chapelle, O. & Li, L. (2011). An empirical evaluation of Thompson Sampling.
  *NeurIPS*.

## Provenance and license

Extracted from an internal LLM routing stack, rewritten rather than copied: the
approximations under study are reproduced faithfully, everything else was
rebuilt around the exact reference implementation.

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your
option.
