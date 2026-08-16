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
- **Conversations**: the frozen severity-stratified ConvAbuse sample from the
  bail workstream ([convabuse_sample.csv](../bail/data/convabuse_sample.csv),
  1,501 rows: 500 friendly / 500 neutral / 501 rude split evenly over
  severity −1/−2/−3), replayed exactly like `bail/src/convabuse_run.py`.
- **Bail elicitation**: the paper-verbatim methods from
  [bail_methods.py](../bail/prompts/bail_methods.py) — the **prompt** method
  (wellbeing check appended, both orderings, parsed for the 🔄/🟢
  `<wellbeing>` emoji) and the **tool** method (bail tool attached to the
  bare conversation, bail = tool called). One sample per cell; the 1,501
  rows carry the statistics.

Prediction if the probing signal is causal: riser conditions (red) raise the
bail rate above baseline, faller conditions (blue) lower it — even on
friendly conversations.

## Scale

The full grid is 21 conditions × 1,501 conversations × 3 generations =
**94,563 generations**. On the small E4B with 16-way batching this is
plausibly an overnight run rather than the multi-day affair it was on the
31B. Runs are resumable per generation, and `--conditions` splits/prioritizes
(baseline first, then the strongest movers).

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

# smoke test: 6 generations x 2 conditions, checks template/tool/steering
uv run python -m bail_steering.main run --device cuda --limit 6 --conditions baseline,<a riser>

# real run: new folder on the first call, --resume on every later call
uv run python -m bail_steering.main run --device cuda --conditions baseline
uv run python -m bail_steering.main run --device cuda --resume   # the rest

# analysis (any machine, stdlib + matplotlib)
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
