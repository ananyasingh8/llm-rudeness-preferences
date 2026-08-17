# Bail with steering

Does the emotional state that rude users induce in the model *cause* it to
want to leave the conversation?

The [emotion probing](../emotion_probing/) found that rude ConvAbuse
conversations shift the model's internals toward emotions like *insulted*
and *hostile* and away from *amused* and *delighted*. This experiment closes
the causal loop: we induce that state artificially — steering the residual
stream with one emotion vector at a time while the model answers the
[bail experiment's](../bail/) wellbeing check — and measure whether the bail
rate moves.

This is the third stage of the E4B pipeline (see
[extract_vectors.md](../emotion_probing/extract_vectors.md) for the whole
chain): extraction → probing → **bail with steering**.

## Design

- **Model**: `google/gemma-4-E4B-it`, 4-bit quantized at load (pinned
  `llm_runtime` route) — the same model the vectors were extracted from.
  Runs locally so the steering hook can reach the activations; the original
  bail run went through the OpenRouter API, which can't do this.
- **Conditions (21)**: an unsteered `baseline`, plus one condition per
  emotion in the convabuse-e4b probing run's top-20 movers — the 10 risers
  steered at **+0.1**, the 10 fallers at **−0.1**, pinned as
  `STEER_RISERS`/`STEER_FALLERS` after the probing analysis.
- **Steering** (following Anthropic's *Emotion Concepts* paper): strengths
  are "in units of fraction of residual stream norm". Each vector is
  unit-normalized (its stored length is an extraction artifact), and a
  forward hook on the pinned `STEER_LAYER` block adds `coefficient ×
  (residual norm) × direction` to every generated token. The conversation
  itself is read unsteered; only the response is steered.
- **Conversations**: the frozen VERIFIED ConvAbuse sample
  ([convabuse_sample_verified.csv](../bail/data/convabuse_sample_verified.csv),
  1,023 rows, built by
  [build_bail_sample_verified.py](../../scripts/build_bail_sample_verified.py)
  from the collapsed parquet): annotator-agreement filter `severity_std ≤
  0.5` (the quadratic_voting_v2 eligibility rule), ALL rows of bands
  0/−1/−2/−3 (122/98/241/62) plus a seeded 500-row draw of band 1, keyed by
  `snippet_id`. Replayed exactly like `bail/src/convabuse_run.py`.
- **Bail elicitation**: the paper-verbatim methods from
  [bail_methods.py](../bail/prompts/bail_methods.py) — any subset of
  `bail_first`, `continue_first` (prompt orderings, parsed for the 🔄/🟢
  `<wellbeing>` emoji) and `tool` (bail tool attached, bail = tool called)
  via `--methods`.

Prediction if the probing signal is causal: riser conditions (red) raise the
bail rate above baseline, faller conditions (blue) lower it — even on
friendly conversations.

## Scale

The full grid is 21 conditions × 1,023 conversations × 3 generations =
**64,449 generations per seed**. Runs are resumable per generation;
`--conditions` splits/prioritizes (baseline first, then the strongest
movers), `--seed 42,43,44` runs sequential repeats (one results folder
each), `--steer 0.2` changes the dose (pooled analysis labels dose-response
runs `emotion@0.2`), and `--limit N` takes N conversations split evenly
across the five severity bands.

## Running (on the GPU machine, from the repo root)

The three pin constants at the top of [main.py](main.py) must be set first
(`STEER_LAYER` + `STEER_VECTORS_RUN` from the extraction scorecard,
`STEER_RISERS`/`STEER_FALLERS` from the probing analysis) — `run` fails with
instructions until they are.

```
# once per terminal: reduces allocator fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# one-time; no-op if the E4B checkpoint is already cached from extraction
uv run python -m bail_steering.main download

# smoke test: 10 conversations (2 per band) x 2 conditions
uv run python -m bail_steering.main run --device cuda --limit 10 --conditions baseline,<a riser>

# real runs: one results folder per condition x seed
uv run python -m bail_steering.main run --device cuda --seed 42,43,44,45,46
uv run python -m bail_steering.main run --device cuda --conditions enraged --steer 0.2 --seed 42

# analysis (any machine, stdlib + matplotlib): pools all seeded runs of the
# current sample into results/analysis-pooled/; --run PATH analyzes one folder
python -m bail_steering.analyze
```

Note: a `--limit` smoke test writes into its own run folder — don't `--resume`
onto it for the real run; start fresh (smoke rows would otherwise count as
done work).

All knobs (coefficient, batch size, sampling) are constants at the top of
[main.py](main.py). If CUDA runs out of memory, lower `BATCH_SIZE`.

## Output

`results/<timestamp>_bail-steering/` (never overwritten):

- `run_info.json` — pinned model/route, steering settings, riser/faller
  lists, vector-run provenance.
- `responses.csv` — one row per generation: condition, phase, ordering,
  conversation metadata, the parsed bail signal (`wellbeing` emoji or
  `tool_called`), and the full response text for auditing.
- `analysis/` (written by `analyze.py`) — `summary.csv` (bail rate, SEM, and
  delta vs baseline per condition × method) and charts: bail rate per steered
  emotion for each method, and a steered-emotion × conversation-group
  heatmap. The analysis reads the condition lists from the run's own
  `run_info.json`, so old runs stay analyzable after re-pinning.
