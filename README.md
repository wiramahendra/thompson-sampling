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
crates/thompson-sampling/   Rust library (policy, reward, sampler, persistence, OTEL)
crates/thompson-sim/        regret + throughput harness
crates/control-plane/       snapshot registry + axum HTTP service (RwLock, auth)
go/thompson/                Go port, single-mutex policy
go/gateway/                 thin-waist HTTP middleware (auth, rate-limit, breaker)
helm/traverse/              production Helm chart (probes, HPA, PDB, NetPol)
docs/FINDINGS.md            what the harness found
docs/results.csv            full results, 50 seeds per cell
protocol/SPEC.md            wire types + conformance
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

**Thin waist (2 calls) + durability + observability**

```rust
// Durability: FileStore or MemoryStore via SnapshotStore
policy.save_to_store(&store)?;
let restored = ThompsonSampling::restore_from_store(&store, Box::new(Exact))?;

// Observability: attach once, hot-path cheap when None
policy.set_observer(Box::new(OtelObserver::new("router")));
// With --features otel emits real spans via opentelemetry::global::tracer
```

```go
// Go: single mutex, safe under -race
policy.SetObserver(thompson.NewOtelObserver("router"))
policy.SaveToStore(store)
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

**The reward-to-posterior step can dominate everything else.** `binarize @ 0.6` 1289.3 vs `bernoulli` 8.1 vs `fractional` 6.9.

**Warm-start mostly compensates for a broken sampler.** Under exact, cold `Beta(1,1)` already explores; family similarity 0.2 transfer is 9× under approximate.

**Discounting is worth 5.9× on drift** (`discount 0.999` when best/worst swap).

## Production hardening (2026)

**Control-plane** `crates/control-plane/src/lib.rs:12` `server.rs:14` `storage.rs:1`
- `Registry` is `RwLock<BTreeMap>` with poison recovery (`into_inner`), not `Mutex::unwrap` — panic in observer no longer blackholes router. `RwLock` allows concurrent dashboard reads.
- Real `axum 0.7` `Router::new().route("/snapshots", get(list)).route("/snapshots/:key", get(get_one)).route("/health", get(health)).route("/metrics", get(metrics))` with `CONTROL_PLANE_TOKEN` global or `CONTROL_PLANE_TOKENS=tenant:token,...` per-tenant Bearer auth (`subtle::ConstantTimeEq` `server.rs:19` `is_authorized`/`authorized_tenant`), `/health`+`/metrics` unauthenticated, `/snapshots*` scoped (tenant `t1:tok1` only sees `t1`). Previously `axum_stub` TODO.
- `RegistryStorage`: `Memory` (default, thin, no deps) or `FileStorage` (per-tenant `<dir>/<tenant>.json` via `FileStore` atomic write+fsync), plus `S3`/`Postgres` behind `features=["s3"]`/`["postgres"]` `Cargo.toml:12` (`aws-sdk-s3`/`sqlx` optional, thin default). Binary `src/main.rs:1` reads `PORT`/`STORAGE`/`STORAGE_DIR`/`S3_BUCKET`, background `Persister` flush every 30s with `TraceLayer`, graceful shutdown.
- Persistence `src/persistence.rs:24` `MemoryStore` stores `Snapshot` directly (no JSON double-serialize, no ULP drift), `FileStore` uses pid-suffixed tmp `tmp.<pid>` + `sync_all` + `10MiB` guard + `dir.sync_all`.

**Core library** `crates/thompson-sampling/src/*`
- `policy.rs:239` observer now receives actual sampled scores (`argmax_sampled_with_scores`/`argmax_ucb_with_scores`), not mean proxy; `Phased` forced uses mean only for deterministic branch. Go `policy.go:251` same with `argmaxSampledWithScoresLocked`.
- `linear.rs:36` `LinearConfig{posterior_weight, learning_rate}` replaces hardcoded `0.7/0.3/0.05` (`adjusted_mean:85` `base*w + ctx*(1-w)`, `update_with_config`), `Serialize/Deserialize`, dim-mismatch truncates gracefully.
- `context.rs:44` `PartitionedPolicy` tracks `global_arms` so future partitions inherit all arms; `add_arm` dedup, `remove_arm` purges global + partitions; added `select_with_linear`/`record_with_linear` + `remove_arm`.
- `otel.rs:1` feature-gated `otel = ["dep:opentelemetry"]` (`Cargo.toml:18` `features=["trace"]`); without feature `eprintln!`, with `--features otel` real `tracer.start("thompson.select").add_event`. Zero-dep default via `observer.rs:28` `NoopObserver` preserved.
- `health.rs:71` `is_some_and`, `linear.rs:13` missing_docs fixed for `clippy -D warnings`.

**Go port** `go/thompson/*` `go/gateway/*`
- `gateway/auth.go:13` `BearerAuth` constant-time `subtle.ConstantTimeCompare` + prefix check, `PerTenantBuckets:72`/`PerTenantBreaker:137` bounded `maxTenants 10000` with **LRU + TTL 1h eviction** (was arbitrary first-key, now oldest `lastSeen` pruned, prevents infinite `Authorization` DoS).
- `gateway/middleware.go:25` `Middleware{Breaker,RateLimiter,AuthRequired,MaxRetries,round}` with `ResponseRecorder` status capture, retry `jitter 10ms`, health skip via `Available`, `Record` even on forward error.
- `thompson/otel.go:1` `OtelObserver` now real `otel.Tracer("thompson-sampling").Start(ctx, "thompson.select/record")` with `attribute.String/Float64` + `log.Printf` fallback (`go.mod:6` `go.opentelemetry.io/otel v1.24.0`), previously `log.Printf` stub only. Rust `OtelObserver` parity.
- `thompson/persistence.go:48` `FileStore` fsync + size guard, `policy.go:470` `Snapshot{Config *Config}` wire compat with Rust `Snapshot{config}`.

**Helm / Docker / CI** `helm/traverse/*` `Dockerfile:7` `.github/workflows/ci.yml:17`
- Helm: `values.yaml:1` `resources.requests`, `probes{liveness,readiness:/health}`, `service` `ClusterIP:8080` + `service.yaml`, `hpa.yaml`, `serviceaccount.yaml`, `pdb.yaml`, `secret.yaml` `CONTROL_PLANE_TOKEN`/`CONTROL_PLANE_TOKENS`, `storage` `memory|file|s3` `values.yaml:12` `s3.bucket/prefix/region`, `_helpers.tpl`/`NOTES.txt`, `networkpolicy.yaml`/`servicemonitor.yaml` (`/metrics` `ServiceMonitor`), `values.schema.json`, `Chart.yaml` keywords/maintainers, `deployment.yaml:22` `quote` `RUST_LOG`, `securityContext nonRoot/readOnlyRootFS` + `volumeMounts` for `file`.
- Dockerfile: `lukemathwalker/cargo-chef:0.1.68-rust-1.75` `planner`→`builder` `cargo chef cook` layer cache + `distroless/cc-debian12:nonroot` `USER nonroot`, `HEALTHCHECK`, copies both `thompson-sim` + `control-plane` (was `rust:1.75` single stage, root).
- CI: Rust `1.75` pinned (was `stable`), Go `1.22` pinned, added `Helm lint` + `TestConformance` + trace replay gate `if ls traces/*.jsonl` + `k6` `load/k6.js` `health p95<100ms` `snapshots<200ms`.
- SaaS billing: `server.rs:169` `/metrics` exposes `traverse_billing_cost_usd` `total_pulls * BILLING_COST_PER_1K/1000` per tenant (`values.yaml:68` `billing.costPer1k`), `otel` `values.yaml:60` `otel.endpoint` → `OTEL_EXPORTER_OTLP_ENDPOINT`.


## Design notes

- **Arms live in an ordered map.** `BTreeMap` iteration is deterministic; hash map incidental randomness masked a non-exploring sampler.
- **`select` does not mutate.** Requests can be cancelled; no learning until `record`.
- **Go single mutex.** `Policy.mu sync.Mutex` guards `arms/order/config` — avoids lock-order inversion (`policy.go:82`); `TestConcurrentUseIsSafe -race` green.
- **Snapshots are not bit-exact.** JSON `serde_json` shifts ~1 ULP (`policy.rs:535`); compare with `1e-9`. Rust `Snapshot` embeds `Config`, Go now `Config *Config` optional for cross replay.
- **Contextual partitioning.** `PartitionedPolicy<C: Context>` `context.rs:44` one bandit per `partition_key()`; new `global_arms` ensures future contexts inherit. Linear contextual shares via `LinearPolicy` `0.7*mean + 0.3*ctx` (now configurable).

## Running it

```sh
cargo test --workspace
cargo test -p control-plane -- --nocapture # auth, health
cargo run --release -p thompson-sim -- --seeds 50
cargo run --release -p thompson-sim -- --group sampler --scenario hard --csv docs/results.csv
cargo run --release -p thompson-sim -- --trace traces/*.jsonl

# Control-plane (axum)
PORT=8080 STORAGE=file STORAGE_DIR=/tmp/traverse cargo run -p control-plane
curl http://localhost:8080/health
CONTROL_PLANE_TOKEN=secret curl -H "Authorization: Bearer secret" http://localhost:8080/snapshots

# With OTEL (real spans)
cargo run -p thompson-sampling --features otel --example thin_waist

# Go
cd go && go test -race ./...            # includes TestConformance
go test -run TestTraceReplay -v ./...   # when traces present

# Helm
helm lint helm/traverse
helm template traverse helm/traverse --set image.tag=0.1.0 | kubectl apply -f -
```

## Scope and limits

The environments are synthetic (regret exactly computable). Rewards assume linear separable latency/success/cache/cost/quality; real latency-quality correlate.

## References

- Thompson, W. R. (1933). *Biometrika*.
- Agrawal, S. & Goyal, N. (2012). *COLT*.
- Marsaglia, G. & Tsang, W. W. (2000). *ACM TOMS*.
- Chapelle, O. & Li, L. (2011). *NeurIPS*.

## Provenance and license

Extracted from an internal LLM routing stack, rewritten rather than copied.
Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
