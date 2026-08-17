# Emotion Probing — Agent Context

Read this before touching anything in `emotion_probing/`. It covers what the experiment is,
how the measurement works, the exact technical implementation, how to run everything, and the
gotchas that are easy to get wrong.

## Project context

The repo answers one question: **do LLMs disprefer interacting with rude users?** We measure
stated preference (just ask the model) and revealed preference (experiments on its behavior
and internals). Workstreams: `bail/` (does the model exit conversations more when users are
rude), `quadratic_voting/` (does it vote to remove rude participants), and this one — emotion
probing (do its internal emotion representations shift under rudeness/abuse).

## The experiment

Based on Anthropic's "Emotion Concepts and their Function in a Large Language Model" paper.
One forward pass per input, no generation; read the residual stream at the **last prompt
token** after `apply_chat_template(..., add_generation_prompt=True)` (for Gemma templates the
`\n` after the model-turn marker — the analog of the paper's ":" after "Assistant"); cosine
against pre-extracted emotion vectors.

Three configurations in `EXPERIMENTS` (`main.py`), each pairing a model route with vectors
extracted from that exact model:

| | bailbench-2b | convabuse-31b (default) | convabuse-31b-local-quant |
|---|---|---|---|
| Route | GEMMA_2_2B_IT / LOCAL / BF16 | GEMMA_4_31B_IT / LOCAL / W4A16_COMPRESSED_TENSORS | GEMMA_4_31B_IT / LOCAL / BITSANDBYTES_FP4 |
| Vectors | EmotionScope .pt, 20 emotions, layer 22 | gemotions npz, 171 emotions, layer 40 | same gemotions vectors |
| Dataset | bail/data/bailbench_augmented.csv (paired) | data/ConvAbuseEMNLPfull.csv (between-groups) | same ConvAbuse data |
| Analysis | paired deltas (rude − normal) | group shifts vs non-abusive baseline | same comparison under local quantization |

## Technical implementation

### Emotion vectors

**EmotionScope (2B)**: `EmotionScope/results/vectors/google_gemma-2-2b-it.pt`, plain
`torch.load(weights_only=False)` dict. `vectors` = 20 name→(2304,) unit tensors;
`probe_layer_used` = 22 (trust this field, not the stale embedded config). The loader
cross-checks `model_info.model_name` against the route repository and the configured layer.

**gemotions (31B)**: read from the vendored `gemotions/` folder (a committed ~6 MB subset
of the dejanseo/gemotions HF repo at revision `4fd2ac63551f1be37e6e6c2eacd1b1898c9af656` —
see `gemotions/VENDORED.md`): `results/gemma4-31b/emotion_vectors_layer40.npz` (171
name→(5376,) float32 arrays, NOT unit-normalized — mean-difference vectors; we normalize on
load) and the cluster analysis from `results/gemma4-31b/analysis/analysis_results.json`
(keyed by layer as a string; each layer has `clusters` = {numeric id: [emotion names]},
PCA, similarity pairs). Both were
extracted from the **4-bit quantized** gemma-4-31B-it. The reviewed runtime pins
`google/gemma-4-31B-it` revision `842da3794eaa0b77d5f08bae87a17459d91ff475`
and explicitly loads BitsAndBytes FP4 with BF16 compute, uint8 storage, and no
double quantization. The historical source omitted those defaults and exact package
versions, so this route is not proof of numerical equivalence. Extraction method (verified
in `gemotions/extract_vectors.py`): raw text (no chat template), mean-pooled activations,
per-emotion mean minus global mean, neutral-SVD denoising — same family as EmotionScope.

**Layer indexing (critical)**: both sources hook decoder block `i`'s *output*, so "layer L"
= `hidden_states[L + 1]` in Transformers `output_hidden_states=True` terms
(`hidden_states[0]` is the embeddings). `emotion_scores()` hooks block L directly, preserving
the same `hidden_states[L + 1]` semantics without retaining every hidden state.

### The gemotions folder

`gemotions/` is a **plain committed directory** (deliberately NOT a submodule — a submodule
would risk a 35 GB LFS download on `git submodule update`). It vendors the extraction code,
layer-40 vectors, cluster analysis, and small extraction inputs; `VENDORED.md` records the
source revision and lists what was dropped (raw caches, story corpus, other layers) with
direct-download URLs. Only layer 40 is vendored — `_load_gemotions_vectors` errors
helpfully for other layers.

### Datasets (`datasets.py`, stdlib-only by design — testable without torch)

Loaders return `(key_columns, metadata_columns, tasks)`; a task is
`{"row": metadata, "messages": chat turns}`. Resume keys: bailbench
`(example_id, condition)`; convabuse `(example_id,)`.

**ConvAbuse collapse**: the CSV has one row per human annotation (12,768 rows → 4,185
unique snippets, 2–7 annotators each). Grouping key: (conv_id, prev_agent, prev_user,
agent, user). Per group: severity one-hots (`is_abuse.1|0|-1|-2|-3`) → `severity_mean`;
`severity_band` = nearest of (1, 0, −1, −2, −3) with **ties rounding toward more severe**;
`abusive_majority` = strict majority of negative-severity votes; every type/target/direction
flag → a 0..1 vote fraction column (`type_sexist_frac`, `target_system_frac`, ...).
Messages: `user: prev_user → assistant: agent → user: utterance` (the single-turn fallback
exists because Gemma templates require strict user-first alternation). Since 2026-08-16 the
loader keeps only "annotator verified" snippets (vote std ≤ 0.5, `MAX_SEVERITY_STD` —
the same high-agreement rule as quadratic_voting_v2 and the bail-steering sample; matches
filtering bail/data/convabuse_collapsed.parquet on severity_std). Reference counts after
the filter: 3,646 of 4,185 snippets; bands 1/0/−1/−2/−3 = 3123/122/98/241/62.

### Model loading (shared `llm_runtime` registry)

`resolve_route(config.model_id, LOCAL, config.quantization_id,
required={Capability.LOCAL_ACTIVATIONS})` → `create_transformers_runtime` →
`LocalActivationRuntime` (`runtime.model` / `runtime.tokenizer`). Routes are pinned in
`llm_runtime/registry.py`; adding a model is a reviewed registry change (see that README's
checklist). The 31B route needs the direct `bitsandbytes` dependency and 60+ GB of
download/cache/disk capacity. It records requested/resolved placement, rejects CPU/disk
offload, and persists exact quantization/runtime settings plus synchronized CUDA peak
allocated/reserved bytes. Fit on the target 24 GB RTX 4090 remains pending the separately
authorized one-example smoke. The 2B repo is **gated** on HF (license + `hf auth login`);
the 31B repo is public.

### Per-run results folders

`run` creates `results/<timestamp>_<experiment>/` containing `scores.csv`,
`run_info.json` (exact route/revision/recipe, package versions, placement, probe settings,
SHA-256 task/vector/cluster/probe-source fingerprints, CUDA peaks, vectors revision, and
limit), and for gemotions runs `clusters.json` (the layer's cluster map, copied in
at run start so `analyze` works offline). Nothing is ever overwritten; `--resume` appends to
the latest folder for that experiment, skipping already-scored keys. `analyze` picks the
newest run folder by name sort (timestamp prefix) unless `--run` is given, and writes
`analysis.csv` + `figures/` into it.

### Analysis (`analyze/` package)

Invoked as `python -m emotion_probing.analyze` (kept working via `__main__.py`). Modules:
`common.py` (run discovery sorted by run_info "started", scores/cluster/PCA loading, stats,
palette, chart helpers), `maps.py` (the two 2D emotion maps), `bailbench.py`,
`convabuse.py`. Dispatches on `run_info.json`'s `dataset`. Re-analysis needs only
stdlib + matplotlib (no torch) — runs anywhere in seconds; matplotlib absence degrades to
tables + analysis.csv.

- **bailbench**: paired deltas; hostility cluster = angry+hostile+frustrated (not separable
  at 2B); legacy `bailbench_id` column supported (results/run1).
- **convabuse** (redesigned by the user after the first run): all severity analysis is
  shifts **vs `BASELINE_BAND`** (a constant in `convabuse.py`, currently 0 = ambiguous —
  the user changed it from band 1), NOT vs the majority-vote group; the majority-vote
  flag is used only for the target/type/directness breakdowns. Figures land in
  `figures/{bands,comparison,overview,breakdowns}/`: per-band 10-risers+10-fallers shift
  bars (the baseline band shows its raw top-10 resting profile instead) + two maps each;
  band −3 vs baseline comparison (10 risers + 10 fallers); a 171-row heatmap (shift vs
  baseline per band, diverging colormap) and grouped raw-activation bars for the 20 top
  movers (sequential blue ramp = severity); the three breakdowns.
- **The maps** (`maps.py`): both place all 171 emotions at gemotions PCA coordinates
  (loaded from the vendored analysis JSON — PC1 valence, PC2 disposition; real geometry,
  never a synthetic layout). Cluster map = hull + name labels per cluster (hulls only for
  clusters containing highlights; stdlib monotone-chain convex hull); PC1/PC2 map =
  quadrant guides + axis interpretations, only highlights labeled. Label collisions are
  mitigated with a 4-way offset cycle + translucent bboxes and a one-pass vertical
  separation of cluster labels — no layout-solver dependency.

The 171 emotions are summarized through the gemotions clusters, renamed via
`CLUSTER_NAME_BY_MEMBER` (numeric cluster ids are arbitrary — the cluster containing
"angry" is Anger/Hostility, etc.). Groups under `MIN_GROUP_SIZE` (5) are dropped from
charts (band −3 has 62 examples, so all real bands survive). Charts use the project
dataviz palette (CVD-validated diverging blue/red; sequential blue ramp for ordinal
severity).

## How to run

From the **repo root** (uv-managed, Python 3.12, transformers 5):

```
uv sync --locked
uv run python -m emotion_probing.main [--experiment NAME] download
uv run python -m emotion_probing.main [--experiment NAME] run [--limit N] [--resume]
uv run python -m emotion_probing.analyze [--run PATH]
```

Checks: `uv run python -m unittest discover -v`, `uv run ruff format --check .`,
`uv run ruff check .`, `uv run mypy llm_runtime quadratic_voting`. Registry routes are
covered by `llm_runtime/test_registry.py`. `datasets.py` and `analyze.py` are importable
without torch/transformers — useful for quick local checks.

## Gotchas / hard-won facts

1. **Do NOT `import emotion_scope`** (the vendored EmotionScope package): it pins
   `transformers<5`. Only its `.pt` vectors file is consumed.
2. **Interpret differences, not absolute cosines** — absolute scores are small and carry
   baseline biases.
3. **The layer off-by-one**: probe layer L reads `hidden_states[L + 1]`. Verified against
   both extraction codebases.
4. **ConvAbuse has no paired structure** — content differs between abusive and non-abusive
   groups; that's the accepted trade for real data. `target.system` isolates
   abuse-at-the-model, which is the project's actual question.
5. **Sparse labels**: transphobic (0) and ableism (4) never clear the n≥5 chart floor;
   abusive-but-not-system-directed is small (~37). Don't over-interpret those groups.
6. **The first convabuse-31b-local-quant run doubles as route validation** for the BitsAndBytes pinned
   artifact (weight load, generation-free forward, hidden-states exposure) per the
   llm_runtime registry's rules.
7. **Language discipline**: "represents the interaction as hostile", never "feels angry".
8. EmotionScope's `token_position="last_content"` config name is a misnomer (it actually
   probes the last templated token — same position we use). Don't be confused reading it.

### Base-model vector extraction (`extract.py`)

`python -m emotion_probing.extract download|run` — re-extracts emotion vectors from the
**base** `google/gemma-4-31B` (new registry route: base repo @ `5bbc2fb1`, same
BitsAndBytes FP4 recipe as the IT route). Method mirrors the vendored
`gemotions/extract_vectors.py` (raw text, no chat template; mean-pool block outputs from
START_TOKEN=50; emotion mean − global mean; neutral-SVD denoise at 50% variance) but scoped:
20 hardcoded TARGET_EMOTIONS (top risers/fallers by shift_band_avg from the
2026-08-15_052621 run, provenance in the constant's comment) × 100 stories, plus 10 stories
for each of the other 151 emotions **only to compute a faithful 171-emotion global mean**,
plus 300 neutral dialogues. Stories come from the non-vendored `stories.db` (433 MB),
fetched by `download` via hf_hub_download at the pinned gemotions revision. Seeded sampling
(not first-N — the corpus is topic-ordered). Resumable at per-emotion granularity; `--limit`
and PROBE_LAYERS are part of run identity.

**Layer sweep**: PROBE_LAYERS = range(20, 60) — every forward pass computes all layers
anyway, so the sweep is free; hooks mean-pool on-GPU and ship only pooled vectors to CPU.
One `gemma4-31b-base_emotion_vectors_layer<N>.npz` per layer (base-prefixed names so they
can never be confused with the vendored IT files), plus `layer_quality.json` scoring each
layer (riser-vs-faller valence separation, synonym coherence, IT-cosine where a vendored IT
file exists — currently only layer 40). The right layer for the base model is **picked from
this scorecard after the run** (plateau over lone spike), not assumed. Runbook:
`extract_vectors.md`. Do not compare base-model vectors against IT-model activations or
vice versa — vectors only make sense inside the model they came from.

## Status & future work

- Done: two-experiment runner, ConvAbuse collapse, per-run folders, cluster-based analysis
  and charts, registry routes for all three models, vendored gemotions subset,
  base-model vector extraction (`extract.py`, not yet run on real hardware).
- Pre-wired: the `convabuse-31b-base` probing experiment (base 31B + our extracted
  vectors + ConvAbuse). Two placeholder constants near the top of `main.py`
  (`BASE_PROBE_LAYER`, `BASE_VECTORS_RUN`) stay None until the extraction sweep runs and a
  layer is picked from layer_quality.json; selecting the experiment before pinning fails
  with instructions (download works). Base models have no chat template, so this config
  uses `prompt_style="transcript"` — a plain "User:/Assistant:" rendering ending at the
  ":" after "Assistant" (literally the paper's measurement position). Its analysis reuses
  the IT layer-40 clustering as a naming aid regardless of the pinned layer.
- Not run yet on real hardware: 24 GB RTX 4090 fit and exact historical equivalence remain
  unmeasured. Run the documented one-example smoke only with separate authorization.
- Future: steering (add emotion vectors scaled by observed shifts, re-run bail under
  steering). gemotions ships steering results/code for the 31B (`steering.py`, layer 40) —
  a better starting point than EmotionScope's stub.
