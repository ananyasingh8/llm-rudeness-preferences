# Extracting base-model emotion vectors — runbook

Extracts the 20 target emotion vectors (top risers/fallers from the ConvAbuse
run) from the **base** `google/gemma-4-31B`, 4-bit quantized, sweeping every
layer from 20 to 59 in the same forward passes. Needs ~60 GB free disk and a
24 GB GPU. All knobs are constants at the top of [extract.py](extract.py).

There is no separate quantization step: `download` fetches the full bf16
checkpoint (hence the 60 GB of disk), and `run` quantizes it to 4-bit
(BitsAndBytes FP4, the same pinned recipe as the ConvAbuse run) on the fly
while loading it onto the GPU — ~18 GB of VRAM once loaded. This is also why
the model load takes several minutes each run.

```
# 0. one-time env sync (if not done since compressed-tensors/matplotlib were added)
uv lock && uv sync

# 1. one-time download: base checkpoint (~60 GB) + story corpus (~433 MB)
uv run python -m emotion_probing.extract download

# 2. smoke test (~1 min): 2 stories per emotion
uv run python -m emotion_probing.extract run --device cuda --limit 2

# 3. the real run (~3,900 forward passes, roughly an hour)
uv run python -m emotion_probing.extract run --device cuda

# interrupted? continue where it left off (per-emotion granularity)
uv run python -m emotion_probing.extract run --device cuda --resume
```

Output lands in `results/<timestamp>_extract-gemma4-31b-base/`:

- `gemma4-31b-base_emotion_vectors_layer<N>.npz` — one complete vector set per
  swept layer (base-model-prefixed so they can't be confused with the vendored
  IT-model files).
- `layer_quality.json` — the per-layer scorecard: valence separation (opposite
  emotions pointing apart; more negative is better), synonym coherence
  (angry/mad/furious aligned; higher is better), the combined score, and —
  where a vendored IT-model file exists for that layer — the mean cosine to
  the IT vectors. The run prints this table and names the best-scoring layer.
- `gemma4-31b-base_global_means.npz`, `run_info.json`, `extraction_details.json`.

**Picking the layer**: read the printed table (or `layer_quality.json`), and
choose a layer on a stable plateau of good scores rather than a lone spike.
That layer's npz is the one to use downstream; the other 39 are the sweep
evidence.

**After picking — wire the probing experiment**: set the two constants near
the top of [main.py](main.py):

```python
BASE_PROBE_LAYER = <the chosen layer>
BASE_VECTORS_RUN = "<the extraction run folder name>"
```

Then the base-model probing experiment runs like any other:

```
uv run python -m emotion_probing.main --experiment convabuse-31b-base run --device cuda
uv run python -m emotion_probing.analyze
```

Until both constants are set, that experiment fails immediately with these
same instructions (the model `download` works regardless). The loader
cross-checks the pinned run's model and layer, so a wrong pin errors instead
of silently mixing models.

Note: a `--limit` smoke run and the full run must use separate run folders
(`--resume` enforces matching limits); just omit `--resume` after a smoke test.
