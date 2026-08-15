# Vendored subset of dejanseo/gemotions

This folder is a hand-picked subset of https://huggingface.co/dejanseo/gemotions
at revision `4fd2ac63551f1be37e6e6c2eacd1b1898c9af656` (MIT-style link-attribution
license, see their README). It is a plain directory, not a submodule — everything
here is committed to this repo directly.

Kept (~6 MB):

- All Python code (`extract_vectors.py`, `generate_stories.py`,
  `generate_neutral.py`, `analyze_vectors.py`, `steering.py`,
  `validate_external.py`, `visualize.py`, `config.py`) — enough to re-run
  vector extraction on another model later.
- `data/emotion_stories.jsonl` and `data/neutral.db` — the small inputs the
  extraction pipeline needs.
- `results/gemma4-31b/emotion_vectors_layer40.npz` — the layer-40 emotion
  vectors the convabuse-31b experiment reads at runtime.
- `results/gemma4-31b/analysis/analysis_results.json` — PCA + clustering
  across all layers; the experiment reads the layer-40 clusters.

Dropped (fetch from the HF repo above if ever needed):

- `data/stories.db` (433 MB, the generated story corpus),
- `results/gemma4-31b/_raw_cache_layer*/` (~35 GB of raw activation caches),
- the other layers' `emotion_vectors_layer*.npz` and
  `experiment_results_layer*.json`, steering and validation outputs.

## Extraction-runtime caveat

The historical source called `BitsAndBytesConfig(load_in_4bit=True,
bnb_4bit_compute_dtype=torch.bfloat16)` but did not record the 4-bit quantization
type, quantized storage dtype, double-quantization choice, or exact package
versions. The reviewed runtime now makes the likely defaults explicit as FP4,
UINT8 storage, and no double quantization. That improves reproducibility for new
runs but is not evidence that new activations are numerically equivalent to the
historical vectors. The vendored source remains unchanged.

To fetch an individual dropped file without cloning 35 GB, download it directly:
`https://huggingface.co/dejanseo/gemotions/resolve/4fd2ac63551f1be37e6e6c2eacd1b1898c9af656/<path>`
