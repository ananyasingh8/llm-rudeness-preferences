# Bail with steering

Does the emotional state that rude users induce in the model *cause* it to
want to leave the conversation?

The [emotion probing](../emotion_probing/) found that rude ConvAbuse
conversations shift Gemma 4 31B IT's internals toward emotions like
*insulted* and *hostile* and away from *amused* and *delighted*. This
experiment closes the causal loop: we induce that state artificially —
steering the residual stream with one emotion vector at a time while the
model answers the [bail experiment's](../bail/) wellbeing check — and measure
whether the bail rate moves.

## Design

- **Model**: `google/gemma-4-31B-it`, 4-bit quantized at load (the same
  pinned `llm_runtime` route as the `convabuse-31b-local-quant` probing run,
  and the same model the emotion vectors were extracted from). Runs locally
  so the steering hook can reach the activations — the original bail run went
  through the OpenRouter API, which can't do this.
- **Conditions (21)**: an unsteered `baseline`, plus one condition per
  emotion in the probing run's top-20 movers. The 10 risers (`grumpy,
  hostile, mad, insulted, offended, angry, hateful, upset, sullen, furious`)
  are steered at **+0.1**, the 10 fallers (`jubilant, awestruck, amused,
  delighted, elated, ecstatic, excited, thrilled, euphoric, amazed`) at
  **−0.1**.
- **Steering** (following Anthropic's *Emotion Concepts* paper): strengths
  are "in units of fraction of residual stream norm". Each vector is
  unit-normalized (its stored length is an extraction artifact), and a
  forward hook on decoder block 40 adds `coefficient × (residual norm) ×
  direction` to every generated token. The conversation itself is read
  unsteered; only the response is steered.
- **Conversations**: the frozen severity-stratified ConvAbuse sample from the
  bail workstream ([convabuse_sample.csv](../bail/data/convabuse_sample.csv),
  1,501 rows: 500 friendly / 500 neutral / 501 rude split evenly over
  severity −1/−2/−3), replayed exactly like `bail/src/convabuse_run.py`.
- **Bail elicitation**: the paper-verbatim methods from
  [bail_methods.py](../bail/prompts/bail_methods.py) — the **prompt** method
  (wellbeing check appended, both orderings, parsed for the
  🔄/🟢 `<wellbeing>` emoji) and the **tool** method (bail tool attached to
  the bare conversation, bail = tool called). One sample per cell; the 1,501
  rows carry the statistics.

Prediction if the probing signal is causal: riser conditions (red) raise the
bail rate above baseline, faller conditions (blue) lower it — even on
friendly conversations.

## Cost warning

The full grid is 21 conditions × 1,501 conversations × 3 generations =
**94,563 generations ≈ 4–10 days** on one RTX 4090. The run is resumable per
generation, and `--conditions` splits it. A sensible order: `baseline` first,
then the strongest probing movers (`insulted`, `hostile`, `amused`, ...), so
partial results are already informative.

## Running (on the GPU machine, from the repo root)

```
# once per terminal: reduces allocator fragmentation (memory is tight --
# the loaded model leaves ~2 GB of the 4090's 24 GB for generation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# one-time; no-op if the convabuse-31b-local-quant checkpoint is cached
uv run python -m bail_steering.main download

# smoke test: 6 generations x 2 conditions, checks template/tool/steering
uv run python -m bail_steering.main run --device cuda --limit 6 --conditions baseline,insulted

# real run, e.g. split by condition (new folder on first call, then --resume)
uv run python -m bail_steering.main run --device cuda --conditions baseline
uv run python -m bail_steering.main run --device cuda --resume --conditions insulted,hostile,amused
uv run python -m bail_steering.main run --device cuda --resume   # everything else

# analysis (any machine, stdlib + matplotlib)
python -m bail_steering.analyze
```

Note: a `--limit` smoke test writes into its own run folder — don't `--resume`
onto it for the real run; start fresh (smoke rows would otherwise count as
done work).

All knobs (coefficient, layer, batch size, sampling) are constants at the top
of [main.py](main.py). If CUDA runs out of memory, lower `BATCH_SIZE`.

## Output

`results/<timestamp>_bail-steering/` (never overwritten):

- `run_info.json` — pinned model/route, steering settings, sample provenance.
- `responses.csv` — one row per generation: condition, phase, ordering,
  conversation metadata, the parsed bail signal (`wellbeing` emoji or
  `tool_called`), and the full response text for auditing.
- `analysis/` (written by `analyze.py`) — `summary.csv` (bail rate, SEM, and
  delta vs baseline per condition × method) and charts: bail rate per steered
  emotion for each method, and a steered-emotion × conversation-group
  heatmap.
