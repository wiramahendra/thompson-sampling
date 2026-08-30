# Thompson Wire Protocol — v1 Freeze Prep

Status: draft — thin waist not yet frozen. Validates via `thompson-sim` + `go/harness` before spec lock.

## Thin Waist (2 calls)

```rust
let provider = policy.select(&mut rng)?; // non-mutating `policy.rs:227`
policy.record_outcome(&mut rng, &provider, &Outcome::new(320.0,true,0.0012).with_quality(0.87))?; // `policy.rs:405`
```
```go
provider, _ := policy.Select(rng) // `go/thompson/policy.go:251`
policy.RecordOutcome(rng, provider, thompson.NewOutcome(320,true,0.0012).WithQuality(0.87))
```

## Wire Types

- `Outcome{latency_ms,success,cache_hit,cost_usd,quality:Option<f64>}` `reward.rs:12` / `go/thompson/reward.go:4` — `quality` `clamp01` `Nan→0` `reward.rs:136`, `RampDown` `INF→0` `reward.rs:150`.
- `Config{update_rule,reward_policy,warm_start,selection,discount}` `policy.rs:54` — `Selection::{Thompson,UcbRegularized,Phased}` `policy.rs:16`, `WarmStart::FamilySimilarity{discount:0.2}` `warm_start.rs:81`, `DiscountPolicy` `discount.rs:17`/`go/thompson/discount.go:1`.
- `Snapshot{version:1,config,arms:Vec<Arm>,total_pulls}` `policy.rs:498` `Version=1` `policy.rs:511` JSON float not bit-exact `policy.rs:490` — compare with `1e-9` tolerance.
- `Arm{ id, posterior:Beta(α,β), pulls, warm_started }` `arm.rs:51`, `Posterior{alpha,beta,pulls}` `posterior.rs:39`.

## Sampling Contract

`BetaSampler::sample` `sampler.rs:17` exact via `Exact::gamma` Marsaglia-Tsang `sampler.rs:69` `c=1/√(9d)` `squeeze 0.0331`; legacy `MeanPlusGaussian` etc. `sampler.rs:123` are conformance baselines. `SnapshotStore` `persistence.rs:13`/`go/thompson/persistence.go:1` persists snapshots atomically.

## Conformance

```
cargo test --workspace
cargo run --release -p thompson-sim -- --seeds 50 --csv docs/results.csv  # `FINDINGS.md:39` sampler table
cargo run --release -p thompson-sim -- --trace traces/*.jsonl           # `trace.rs:1`
go test -race ./...  # `thompson_test.go:579 TestConcurrentUseIsSafe`
```

Freeze gate: real trace replay reproduces `hard`/`churn` 2.4–14× `FINDINGS.md:43` and `graded` `binarize 187×` `FINDINGS.md:114` vs synthetic, plus `drift` `0.999` 5.9× `FINDINGS.md:206` before v1 lock. Contextual `PartitionedPolicy` `context.rs:33` stays partitioned until linear `linear.rs:1` validated.
