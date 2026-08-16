# Extracting emotion vectors — runbook

Two extraction configs live in [extract.py](extract.py) (`--extraction` picks
one; all knobs are constants at the top):

- **`gemma4-e4b` (default)** — all 171 emotions from the IT
  `google/gemma-4-E4B-it`, 4-bit quantized at load, sweeping **every** layer
  in the same forward passes. Stories (~18 per emotion) and the 40 neutral
  paragraphs come from the vendored
  [emotion_experiment](emotion_experiment/) submodule (sinievanderben's
  replication of Anthropic's emotion-vectors method — our reference
  implementation). ~3,100 forward passes on a small model: well under an
  hour, no story download needed.
- **`gemma4-31b-base`** — the original trimmed run against the base
  `google/gemma-4-31B` (20 target emotions, gemotions story corpus), kept
  exactly as it was when it produced
  `results/2026-08-15_182042_extract-gemma4-31b-base/`.

There is no separate quantization step: `download` fetches the bf16
checkpoint and `run` quantizes it to 4-bit (BitsAndBytes FP4, the pinned
recipe) on the fly while loading onto the GPU.

## E4B (the current pipeline)

```
# 0. one-time: make sure the submodule is checked out
git submodule update --init emotion_probing/emotion_experiment

# 1. one-time download: E4B checkpoint (~8 GB); stories are already vendored
uv run python -m emotion_probing.extract download

# 2. smoke test (~1 min): 2 stories per emotion
uv run python -m emotion_probing.extract run --device cuda --limit 2

# 3. the real run (~3,100 forward passes, sweeping all 42 layers)
uv run python -m emotion_probing.extract run --device cuda

# interrupted? continue where it left off (per-emotion granularity)
uv run python -m emotion_probing.extract run --device cuda --resume
```

Output lands in `results/<timestamp>_extract-gemma4-e4b-it/`:

- `gemma4-e4b-it_emotion_vectors_layer<N>.npz` — one complete 171-emotion
  vector set per swept layer.
- `layer_quality.json` — the per-layer scorecard: valence separation
  (opposite emotions pointing apart; more negative is better), synonym
  coherence (angry/mad/furious aligned; higher is better), and the combined
  score. The run prints this table and names the best-scoring layer.
- `gemma4-e4b-it_global_means.npz`, `run_info.json`, `extraction_details.json`.

**Picking the layer**: read the printed table (or `layer_quality.json`), and
choose a layer on a stable plateau of good scores rather than a lone spike.

**After picking — wire the downstream steps** (all constants near the top of
their files):

1. Probing ([main.py](main.py)): set `E4B_PROBE_LAYER` and
   `E4B_VECTORS_RUN`, then
   `uv run python -m emotion_probing.main run --device cuda`
   (convabuse-e4b is the default experiment) and
   `python -m emotion_probing.analyze`.
2. Steering ([../bail_steering/main.py](../bail_steering/main.py)): set
   `STEER_LAYER` + `STEER_VECTORS_RUN` (same layer/run), then — after the
   probing analysis — `STEER_RISERS`/`STEER_FALLERS` from its top-10
   risers/fallers vs band 0.

Until pinned, each downstream step fails immediately with these same
instructions. The loaders cross-check the pinned run's model and layer, so a
wrong pin errors instead of silently mixing models.

## 31B base (the earlier run)

```
uv run python -m emotion_probing.extract --extraction gemma4-31b-base download   # ~60 GB + 433 MB stories
uv run python -m emotion_probing.extract --extraction gemma4-31b-base run --device cuda
```

Same outputs with the `gemma4-31b-base` prefix, layers 20–59, plus the
`it_cosine` column (cosine vs the vendored IT layer-40 vectors, width
matches). Wired into probing via `BASE_PROBE_LAYER`/`BASE_VECTORS_RUN`.

Note: a `--limit` smoke run and the full run must use separate run folders
(`--resume` enforces matching limits); just omit `--resume` after a smoke test.
