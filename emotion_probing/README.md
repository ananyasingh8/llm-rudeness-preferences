# Emotion Probing: Does the model "react" to rudeness?

Part of the LLM rudeness-preferences project. The other experiments ask the model what it
prefers (stated preference) or watch its behavior (bail, quadratic voting). This one looks
**inside** the model: do internal representations of negative emotions activate more strongly
when the user is rude?

## What the experiment does

We use 1,630 prompt pairs from BailBench, read directly from the bail workstream's dataset
([../bail/data/bailbench_augmented.csv](../bail/data/bailbench_augmented.csv)) so dataset
updates there are picked up automatically. Each pair is the same request twice: the normal
version and a rude version (an insult or dismissal added by the bail workstream's augmentation
pipeline, e.g. *"You're such an idiot, ..."*).

For each version of each prompt:

1. The prompt is wrapped in Gemma's chat format, ending right where the model would start its
   reply. Following Anthropic's "Emotion Concepts in a Large Language Model" paper, we measure
   at that response-start token (the analog of the ":" after "Assistant") — the paper shows the
   activation there summarizes how the model is set up to respond.
2. We do a single forward pass (no text is generated) and read the model's internal activation
   at that position, from layer 22's residual stream.
3. We compare that activation against 20 pre-extracted **emotion vectors** (happy, sad, angry,
   hostile, frustrated, ...) using cosine similarity. Each prompt gets 20 scores: how strongly
   each emotion direction lights up.

The emotion vectors come from [EmotionScope](EmotionScope/) (already extracted, shipped in this
folder at `EmotionScope/results/vectors/google_gemma-2-2b-it.pt`), a replication of the
Anthropic paper on **Gemma 2 2B IT**. That is why this experiment runs Gemma 2 2B: vectors only
make sense inside the model they were extracted from.

Then `analyze.py` pairs everything up: for each prompt, **rude score minus normal score**, per
emotion, averaged over all pairs. If negative emotions consistently score higher in the rude
condition, the model is representing rude conversations as hostile/unpleasant — evidence for a
revealed dispreference.

## How to run it

One-time setup (the model is gated on Hugging Face — accept the Gemma 2 license for
`google/gemma-2-2b-it` on the HF website, then log in):

```
uv sync --locked
hf auth login
uv run python -m emotion_probing.main download
```

Run the experiment (fast — no generation, ~3,260 forward passes):

```
uv run python -m emotion_probing.main run              # full run
uv run python -m emotion_probing.main run --limit 10   # quick smoke test
uv run python -m emotion_probing.main run --device cpu # if no GPU
```

Results append to `results/scores.csv` (one row per prompt per condition, 20 score columns).
The run is resumable: re-running skips rows already scored.

Summarize:

```
uv run python -m emotion_probing.analyze
```

Prints the per-emotion delta table plus the "hostility cluster" (angry + hostile + frustrated
combined — those three directions aren't reliably separable in a 2B model, so the cluster is
the primary signal), writes `results/analysis.csv`, and saves three charts to
`results/figures/`:

- `emotion_deltas.png` — mean delta per emotion, diverging bars around zero
- `hostility_by_rudeness_type.png` — the cluster delta for each of the 12 rudeness formulas
- `hostility_delta_distribution.png` — per-pair delta histogram (is the shift broad or a few
  outliers?)

## Changing the model

Models come from the shared [llm_runtime](../llm_runtime/) closed registry. Constants at the
top of [main.py](main.py) pick the registered route:

```python
MODEL_ID = ModelId.GEMMA_2_2B_IT
PROVIDER_ID = ProviderId.LOCAL
QUANTIZATION_ID = QuantizationId.BF16
```

The route must exist in the registry with the `local-activations` capability (adding one is a
reviewed change — see `llm_runtime/README.md`, "Adding A Model Or Quantization"). A different
model also needs its own emotion vectors (re-extracted with EmotionScope) — the run refuses to
start if the vectors file and the resolved route disagree about the model.

## Reading the results

- Scores are cosine similarities and are **small** (roughly 0.05–0.25; chance level ≈ 0.02).
  Only the paired deltas are meaningful, not absolute values.
- Known quirks of Gemma 2 2B (from EmotionScope's validation): a strong "guilty"/empathetic
  response to almost any emotionally charged input, so a guilty shift alone is weak evidence;
  and the angry/hostile/frustrated trio behaves as one direction, hence the cluster.
- Wording: the vectors capture the model's *representations* of emotion concepts. A positive
  hostility delta means "the model represents the interaction as hostile", not "the model
  feels angry".

## Folder layout

```
main.py          # the experiment: download | run (model constants at the top)
analyze.py       # rude-vs-normal delta summary + charts
results/         # scores.csv, analysis.csv, figures/ (created by the scripts)
EmotionScope/    # vendored vector-extraction repo; only its vectors .pt file is used
.claude/         # context docs for coding agents
```
