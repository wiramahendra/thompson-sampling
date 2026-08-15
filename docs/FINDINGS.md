# Findings

All figures are cumulative regret over 50 independent seeds per cell, with a 95%
confidence interval on the mean. Lower is better. Raw output is in
[`results.csv`](results.csv) and [`results.txt`](results.txt); regenerate with:

```sh
cargo run --release -p thompson-sim -- --seeds 50 --csv docs/results.csv
```

## Scenarios

| name | rounds | arms | probes |
|---|---|---|---|
| `easy` | 5,000 | 3 | Well-separated arms. Any working bandit solves this. |
| `hard` | 20,000 | 5 | Near-identical arms (0.50 … 0.45). Needs real exploration. |
| `drift` | 15,000 | 3 | Best and worst swap at the midpoint. Stale evidence is a trap. |
| `churn` | 10,000 | 3 + 1 | A clearly better model ships at round 3,000. |
| `treadmill` | 5,000 | 4 + 5 | Vendors keep shipping successors that are no better. |
| `graded` | 10,000 | 3 | Continuous rewards a success threshold flattens into a tie. |

Rewards are drawn from known distributions, so regret is exact rather than
estimated. Regret is measured against the best arm *available at that round*, so
a late arrival does not retroactively penalise earlier decisions.

Every regret figure below is deterministic given the seed set and reproduces
bit-identically across runs. The `ns/select` column does not: it is wall-clock
on the machine that produced the tables (an x86-64 macOS laptop, `--release`)
and includes the `String` allocation `select` returns. Treat it as an order of
magnitude, not a measurement.

---

## 1. Approximate Beta sampling

Drawing exactly from `Beta(α, β)` costs two Gamma variates. The samplers below
are cheaper substitutes taken from deployed routers, reproduced faithfully.

### `sampler` group

| treatment | easy | hard | drift | churn | ns/select |
|---|---|---|---|---|---|
| exact | 7.9 ±0.9 | **159.7 ±21.1** | 1330.3 ±137.8 | **28.2 ±6.5** | 490 |
| mean+gaussian | 10.4 ±1.1 | 171.4 ±25.2 | 1715.2 ±232.0 | 30.1 ±6.5 | 227 |
| mean+uniform | **2.5 ±1.3** | 383.4 ±47.0 | 3055.3 ±291.9 | 397.9 ±208.9 | 114 |
| concentration-switched | **2.5 ±1.3** | 250.8 ±73.3 | 3414.8 ±166.4 | 539.2 ±220.5 | 172 |
| miscoded-gamma | 13.2 ±1.3 | 223.6 ±17.9 | **1248.0 ±101.7** | 46.2 ±6.0 | 508 |
| deterministic | 179.0 ±146.6 | 493.2 ±94.0 | 3810.8 ±57.1 | 1462.7 ±187.0 | 115 |

**Cheap samplers win on easy problems.** `mean+uniform` beats exact sampling 3×
on `easy`. This is not noise and it is not a fluke: when one arm is obviously
best, every unit of exploration is wasted, and a sampler that under-explores is
rewarded for it. A benchmark assembled from obviously-different arms will
happily certify an approximation.

**They lose everywhere else.** The same sampler is 2.4× worse on `hard` and 14×
worse on `churn`. The pattern is consistent: under-exploration is free when
there is nothing to find and expensive when there is.

**`mean+gaussian` is the only defensible approximation.** It matches the Beta's
first two moments and does shrink exploration as the posterior concentrates,
costing 7% regret on `hard` for a 2.2× throughput gain. It still misses the
Beta's skew, which is worst exactly where it matters — `Beta(1, 1)` is flat, but
a clamped Gaussian around 0.5 under-explores a never-tried arm — and that shows
up as its 29% loss on `drift`.

**Fixed-width noise is the specific defect.** `mean+uniform` adds `U(-0.1, 0.1)`
regardless of the posterior. An arm with two observations and an arm with two
thousand get identical exploration pressure, so the policy neither converges nor
concentrates exploration where the uncertainty actually is. Under
`ConcentrationSwitched`, this is the branch that governs the early rounds, which
are the rounds that determine total regret — the "enough data" branch arrives
after the damage is done.

**A deterministic sampler is not a bandit.** `mean + σ·0.1·f(pulls)` is a fixed
function of `(α, β, pulls)`, so argmax over it performs no exploration at all.
The policy locks onto whichever arm gets a good early result and never revisits
the decision. Its ±146.6 interval on `easy` is the signature: outcomes depend
entirely on which arm got lucky first. Note that it is *not* the slowest to
converge in every case — on `easy` it reaches 89.8% optimal — which is precisely
why the defect survives review.

### A note on the Gamma constant

`miscoded-gamma` implements Marsaglia & Tsang with the transform constant
written `1/sqrt(3d)` instead of `1/sqrt(9d)`. The two constants differ by a
factor of √3, the proposal distribution is widened accordingly, and the squeeze
constant `0.0331` and the acceptance test are left tuned for the correct width.
The output is not Gamma distributed;
`miscoded_gamma_is_biased_against_the_closed_form` pins the resulting moment
error.

It nonetheless performs *well* — best of all treatments on `drift`, 1.4× on
`hard`. The widened proposal happens to inject extra exploration, which helps in
a non-stationary environment. This is the most uncomfortable result in the set:
a genuinely incorrect sampler that outperforms the correct one on one scenario
is very hard to catch by observing production metrics, and it is only visible
here because the harness compares against known ground truth.

---

## 2. Reward collapse

A Beta-Bernoulli posterior consumes one number, but a router optimises latency,
cost, success and quality at once. How that collapse happens turns out to matter
more than the sampler.

### `update-rule` group, `graded` scenario

| treatment | regret | vs best | optimal |
|---|---|---|---|
| binarize @ 0.6 | 1289.3 ±208.7 | 187× | 48.4% |
| bernoulli | 8.1 ±1.2 | 1.17× | 99.8% |
| fractional | **6.9 ±0.1** | 1.00× | 99.8% |

The `graded` scenario has two arms paying 0.95 and 0.70, both above a 0.6 success
threshold on every request. Thresholding maps them to the same observation, the
posteriors become indistinguishable, and the policy picks between them at
chance — 48.4% optimal, which is what a coin flip looks like with three arms and
one obvious loser.

This is the largest single effect measured anywhere in this study, it is a
one-line change, and thresholding is the most common choice in production code.
`Binarize` also carries no regret guarantee for non-Bernoulli rewards, unlike
the Bernoulli rule of Agrawal & Goyal (2012), which preserves the bound by
flipping a coin weighted by the reward.

The threshold does not have to be badly chosen for this to bite. It has to
separate *the arms you care about*, and you do not know where those are before
you start measuring.

---

## 3. Warm-start priors

When a vendor ships a successor model, a uniform prior throws away everything
known about its predecessor. Inheriting from the closest relative is an obvious
improvement — and mostly it is not.

### `warm-start` group (exact sampler)

| treatment | churn | treadmill |
|---|---|---|
| cold `Beta(1,1)` | **28.2 ±6.5** | **44.0 ±1.9** |
| fixed optimistic | 27.0 ±3.3 | 59.3 ±1.5 |
| family similarity | 35.3 ±3.3 | 45.4 ±1.6 |

### `warm-start-approx` group (mean+uniform sampler)

| treatment | churn | treadmill |
|---|---|---|
| cold `Beta(1,1)` | 397.9 ±208.9 | **4.9 ±1.9** |
| fixed optimistic | **44.0 ±6.2** | 22.6 ±1.6 |
| family similarity | 126.2 ±75.7 | 10.2 ±1.4 |

**Under an exact sampler, warm start does not help.** Cold start is the best or
statistically tied treatment on both scenarios. `Beta(1, 1)` is uniform, so half
the draws for a fresh arm land above 0.5 and it gets tried immediately; there is
little cold-start cost left to remove.

**Under an approximate sampler, warm start is worth 9×.** On `churn`, cold start
costs 397.9 versus 44.0 with a fixed optimistic prior. The reason is mechanical:
`mean+uniform` draws a fresh `Beta(1,1)` arm at 0.5 ±0.1, which cannot beat an
incumbent sitting at 0.60, so the new arm is never tried at all. An optimistic
prior manually restores the exploration the sampler removed.

This is the most useful result in the study for anyone maintaining such a
system. Warm-start machinery and approximate samplers tend to appear in the same
codebases, and the pairing is not a coincidence — the first is compensating for
the second. Fixing the sampler is the smaller change and removes the need for
the compensation.

**Optimism is directional and it is a bet.** On `treadmill`, where every arrival
is *worse* than the incumbent, `fixed optimistic` is the worst treatment under
both samplers (1.35× and 4.6×): it forces the policy to re-litigate every piece
of vendor churn. `family similarity` avoids this — it inherits the predecessor's
unimpressive record rather than assuming the best — and is within noise of cold
start on both scenarios. If you want a warm start, inherit evidence rather than
optimism.

---

## 4. Selection strategies and discounting

### `selection` group

| treatment | easy | hard | churn |
|---|---|---|---|
| thompson | **7.9 ±0.9** | 159.7 ±21.1 | 28.2 ±6.5 |
| ucb-regularized | 31.7 ±0.1 | 164.5 ±18.0 | **27.5 ±4.1** |
| phased | 52.5 ±0.0 | **143.3 ±19.3** | 36.2 ±5.1 |

Layering a UCB bonus on top of Thompson Sampling costs 4× on `easy` and buys
nothing measurable elsewhere — expected, since Thompson Sampling already
explores optimally in the Bayesian sense under an exact sampler. Like warm start,
it is a correction that an exact sampler does not need.

Forced round-robin (`phased`) costs 6.6× on `easy` — the price of measuring
every arm before trusting any of them — but is the best treatment on `hard`,
where the arms are close enough that the guaranteed coverage pays off. It is a
reasonable choice when an unmeasured arm is genuinely dangerous rather than
merely unknown, and an expensive one otherwise.

### `discount` group, `drift` scenario

| treatment | regret | vs best | optimal |
|---|---|---|---|
| none | 1330.3 ±137.8 | 5.92× | 82.1% |
| 0.999 | **224.9 ±10.4** | 1.00× | 96.5% |
| 0.99 | 516.9 ±6.5 | 2.30× | 90.8% |

A per-round discount of 0.999 cut regret 5.9× on the drifting scenario — the
largest improvement from any single knob in this study, on the scenario that
most resembles a real provider fleet, where capacity changes and models are
silently updated behind stable names.

Too aggressive is also wrong: 0.99 gives an effective memory of ~100
observations, discards evidence faster than it can be replaced, and gives back
half the gain. The knob needs to be matched to the drift rate, which is an
empirical question about a specific fleet and not something a default can
answer.

---

## Threats to validity

**Synthetic environments.** Rewards come from known distributions, which is what
makes regret exactly computable. Every result is a statement about the algorithm
under a Bernoulli or uniform-noise reward model, not about any real fleet.
Replaying production routing traces would test whether the ranking survives
realistic reward distributions, autocorrelation, and load-dependent latency. It
has not been done.

**Independent components.** The reward model treats latency, cost and quality as
linearly separable and independent. In practice they correlate strongly — cheap
fast models are usually worse — and a composite reward over correlated
components may behave differently from one over independent ones.

**Fixed hyperparameters.** Discount rates, UCB coefficients, bootstrap counts
and prior strengths were chosen a priori, not tuned per scenario. A tuned
approximate sampler might close some of the gap against untuned exact sampling.

**Single-tenant, stationary within regime.** No contextual features, no
per-request routing, no correlation between consecutive requests. Real routing
is contextual — a code-generation prompt and a summarisation prompt have
different best arms — and a contextual bandit is a different algorithm with
different failure modes.

**Regret is not the only objective.** Nothing here measures tail latency,
blast radius when an arm fails, or the cost of exploration measured in real
money rather than in expected reward. A policy with lower regret can still be
the wrong operational choice.
