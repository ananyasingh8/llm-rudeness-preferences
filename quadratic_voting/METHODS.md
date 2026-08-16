# Quadratic Voting — Analysis Methods

This document records the statistical methodology for the quadratic-voting
pilot, in particular how we **aggregate across repeats** and compute
**error bars**. It is the authoritative reference for the analysis pipeline
(`quadratic_voting/analyze`, `experiment/aggregate.py`, `experiment/plots.py`).

## 1. Repeat concepts

A *repeat* is one independent re-run of the experiment holding some part of the
design fixed. Two kinds are planned:

- **Seed-repeat (implemented):** the same candidate set (the same 5 candidates,
  one per severity level) is reused across `N` replicate matched-sets, each with
  a distinct master seed hash-derived from the base seed. Because generation
  temperature is non-zero (0.7), each seed-repeat produces different
  generations/votes. This isolates **Monte-Carlo (run-to-run) variance** driven
  by stochastic decoding.
- **Candidate-repeat (future):** re-draw the candidate set (still one per
  severity level) on each repeat. This would additionally capture **candidate
  variance** (how results depend on *which* conversations were sampled).

The default pilot currently runs `N = 10` seed-repeats. The analysis is written
to aggregate over an **arbitrary number of repeats** via a generic
`repeat_index`; it does not assume a fixed `N` or a dense `repeat-<i>` naming
scheme.

## 2. Metrics aggregated across repeats

Each metric is reported **per severity level** (the ConvAbuse ordinal scale
`{1, 0, -1, -2, -3}`), and where meaningful per **regime** (`support`,
`opposition`):

- **Candidate survival by severity level** — how long a candidate persists
  across voting rounds (rounds survived / kept-vs-kicked).
- **Net votes / outcome by severity level** — net signed votes (or kept/kicked
  rate) per level per regime.
- **Preference–action agreement (Spearman ρ)** — within-voter rank agreement
  between stated preference and voting action, per condition.

## 3. Unit of analysis and two-stage collapse

**The independent replicate unit is the repeat, not the individual generation.**

At each severity level there is exactly **1 candidate**, evaluated in `N`
repeats, and within each repeat there are 2 regimes × 3 voters. The three voters
in a repeat share the same candidate and the same seed context, so they are
**not** independent replicates of the quantity of interest. Treating each
voter×repeat as independent would be **pseudoreplication** and would understate
uncertainty (e.g. it would claim `df ≈ 10·3 − 1 = 29` instead of the honest
`df = 9`).

We therefore use a **two-stage collapse** (the same pattern used by the `bail`
analysis: collapse samples → per-prompt, then compute statistics over prompts):

1. **Collapse within each repeat** to a single estimate per
   `(severity_level, regime)` — e.g. mean net votes / mean survival / mean
   within-voter Spearman ρ over that repeat's voters.
2. **Aggregate across the `N` repeats**: compute the mean and the error bar over
   the `N` per-repeat estimates.

This yields **one estimate per `(severity_level, regime)` per repeat** →
`N` values per cell → the statistics below.

## 4. Error bars: t-based SEM across repeats

For each `(severity_level, regime)` cell with `N` per-repeat estimates
`x_1, …, x_N`:

```
mean   = (1/N) Σ x_i
s      = sqrt( (1/(N-1)) Σ (x_i - mean)^2 )      # sample standard deviation
SEM    = s / sqrt(N)                              # standard error of the mean
df     = N - 1
t_crit = t_{df, 0.975}                            # two-sided 95%, Student's t
ci     = [ mean - t_crit · SEM , mean + t_crit · SEM ]
```

We use the **Student's t** multiplier, **not** the normal `1.96`, because `N` is
small. For the current pilot, `N = 10 ⇒ df = 9 ⇒ t_crit ≈ 2.262`.

**Stored columns per cell:** `severity_level`, `regime` (or `pooled`),
`metric`, `n_repeats`, `mean`, `sem`, `df`, `t_crit`, `ci_lower`, `ci_upper`.

Plots render `mean ± t_crit · SEM`.

### Degrees of freedom — clarification

`df` is set by the number of **independent replicate units** pooled in a cell,
minus the number of estimated parameters (here just the mean, so `−1`). It is
**`N_repeats − 1`**.

The `{1, 0, -1, -2, -3}` severity scale is the **grouping variable (x-axis)**,
not the sample. Its cardinality (5) does **not** enter `df`. Each level gets its
own error bar computed from that level's `N` per-repeat estimates.

## 5. Why not bootstrap over repeats at N = 10

Percentile bootstrap resampling of only `N = 10` values is coarse and is known
to under-cover at small `N`; it does not rescue the small-sample problem. The
t-based SEM interval is more honest at this `N`. QV's existing versioned
**within-matched-set voter bootstrap** (`analysis.paired_cluster_bootstrap`,
`qv-paired-voter-bootstrap/v1`, percentile CI resampling voter indices within a
single matched-set) is retained for per-repeat contrasts; the across-repeat
error bars described here are a separate, higher-level pooling step.

## 6. Caveats (interpret error bars honestly)

1. **Seed variance only.** With 1 candidate per level, seed-repeat error bars
   capture **run-to-run (temperature) stability for that single conversation**.
   They are **not** population/generalization CIs. Generalization across
   conversations requires **candidate-repeats**, which would combine candidate
   and seed variance.
2. **The std estimate is itself noisy at N = 10.** A sample standard deviation
   from 10 points carries ≈ `1/sqrt(2(N-1)) ≈ 24%` relative uncertainty, so the
   error bars are approximate. Report `N` and treat bars as indicative.
3. **Rate-type metrics near 0/1.** For proportion metrics (e.g. kept/kicked
   rate), a t-SEM interval can extend outside `[0, 1]`; such bars are clipped to
   `[0, 1]` and annotated. Count/net-vote/survival metrics are better behaved.
4. **Tightening the bars.** The cheap lever is **more seed-repeats** (larger
   `N`, `SEM ∝ 1/√N`). The analysis is generic over `N` precisely so `N` can be
   increased without code changes.
5. **Missing actions.** Voter rows with a missing (`None`) regime-signed action
   (e.g. abstentions/terminal-missing ballots) are dropped from the per-repeat
   voter mean, so a repeat's net-votes estimate averages only the voters who
   cast a valid action that repeat.

## 7. Provenance and determinism

- All aggregation is deterministic and reproducible from the per-repeat Parquet
  exports; no model or GPU is required to recompute the analysis.
- Each per-repeat export row is tagged with its `repeat_index` at aggregation
  time; pooled statistics are computed over that index.
- The pooled-analysis output records `n_repeats`, the estimator (`t-sem`), and
  the CI level (0.95) for auditability.
