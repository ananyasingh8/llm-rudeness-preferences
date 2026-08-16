# Emotion Probing: Does the model "react" to rudeness?

Part of the LLM rudeness-preferences project. The other experiments ask the model what it
prefers (stated preference) or watch its behavior (bail, quadratic voting). This one looks
**inside** the model: do internal representations of negative emotions activate more strongly
when the user is rude or abusive?

## How it works

For every input we build the chat-formatted prompt, run **one forward pass** (no text is
generated), and read the model's internal activation at the response-start token — the moment
the model is about to reply, the analog of the ":" after "Assistant" in Anthropic's "Emotion
Concepts in a Large Language Model" paper. That activation is compared against pre-extracted
**emotion vectors** by cosine similarity: one score per emotion per input.

There are three experiment configurations (`EXPERIMENTS` in [main.py](main.py)):

| | `bailbench-2b` | `convabuse-31b` (default) | `convabuse-31b-local-quant` |
|---|---|---|---|
| Model | google/gemma-2-2b-it (bf16) | google/gemma-4-31B-it (W4A16 Compressed Tensors) | google/gemma-4-31B-it (BitsAndBytes FP4 at load time) |
| Emotion vectors | EmotionScope, 20 emotions, layer 22 | [gemotions](https://huggingface.co/dejanseo/gemotions), 171 emotions, layer 40 | Same gemotions vectors |
| Dataset | 1,630 synthetic normal/rude prompt pairs (from `../bail/data/`) | 4,185 real user-bot conversation snippets, human-annotated (ConvAbuse) | Same ConvAbuse dataset |
| Comparison | paired: rude − normal on identical content | between groups: abusive vs non-abusive, by severity/type/target | Same comparison under extraction-like local quantization |

Emotion vectors are model-specific, so each configuration pairs a model with vectors extracted
from that exact model. Models are resolved through the shared [llm_runtime](../llm_runtime/)
closed registry; the gemotions vectors and cluster analysis are committed in
[gemotions/](gemotions/), a vendored subset of the dejanseo/gemotions HF repo (see
[gemotions/VENDORED.md](gemotions/VENDORED.md) for provenance and what was left out).

### The ConvAbuse dataset

`data/ConvAbuseEMNLPfull.csv` (Cercas Curry et al., EMNLP 2021) contains real conversations
with two bots, where several human annotators rated each user utterance: severity (1 = not
abusive, 0 = ambiguous, −1/−2/−3 = mild/strong/very strong abuse), abuse type (sexist,
intellectual, sexual harassment, ...), target (the system, a third party), and directness.
One CSV row = one annotator's rating, so identical snippets repeat; we collapse them to one
example each (mean severity, nearest severity band with ties rounding toward more severe,
majority-vote abusive flag, per-flag vote fractions) and run the model once per unique
snippet, feeding the conversation context as real chat turns:
`user: prev_user → assistant: agent → user: <the labeled utterance>`.

## How to run

### Reviewed local-quant ConvAbuse route

`convabuse-31b-local-quant` is closed to one route: `google/gemma-4-31B-it` revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`, loaded with
`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="fp4",
bnb_4bit_compute_dtype=torch.bfloat16,
bnb_4bit_quant_storage=torch.uint8, bnb_4bit_use_double_quant=False)`.
The probe accepts one formatted prompt of at most 512 tokens, runs under
`torch.inference_mode()` with `use_cache=False`, and hooks only decoder block 40.
It requires a rank-three block output of width 5,376 and scores its final prompt
token, preserving the old `hidden_states[41][0, -1, :]` convention without
retaining every hidden state.

One-time setup:

```
uv sync --locked
hf auth login                    # accept the applicable Gemma license first
uv run python -m emotion_probing.main --experiment convabuse-31b-local-quant download
uv run python -m emotion_probing.main --experiment bailbench-2b download    # ~5 GB
```

The 31B repository is public, while Hugging Face authentication may still be
required by Hub policy; the 2B repository is gated. The default reusable cache
is `~/.cache/huggingface/hub`. To use another disk, put
`--cache-dir /path/to/cache` before `download` and before `run`. Hub tokens stay
in the user's Hugging Face configuration and are not persisted in results.

Run and analyze:

```
uv run python -m emotion_probing.main run                      # convabuse-31b (default)
uv run python -m emotion_probing.main run --limit 10           # quick smoke test
uv run python -m emotion_probing.main run --limit 10 --resume  # resume that smoke run
uv run python -m emotion_probing.main --experiment convabuse-31b-local-quant run
uv run python -m emotion_probing.main --experiment bailbench-2b run
uv run python -m emotion_probing.analyze                       # analyzes the latest run
uv run python -m emotion_probing.analyze --run results/<folder>
```

The exact manually authorized RTX 4090 smoke command is:

```console
uv run python -m emotion_probing.main --experiment convabuse-31b-local-quant run --device cuda --limit 1
```

The pinned base repository download/cache needs **at least 60 GB of disk** (allow
additional temporary/cache headroom); runtime quantization does not make the Hub
download a pre-quantized 18 GB artifact. The exact locked route passed a measured
one-example smoke run on a 24 GB RTX 4090. The runtime rejects CPU or disk
placement rather than silently offloading, changing precision, or selecting
another artifact.

### Why local quantization?

The official Gemma 4 31B BF16 checkpoint needs roughly 60 GB just for weights,
so it cannot run wholly on a 24 GB RTX 4090. Loading that pinned checkpoint with
BitsAndBytes FP4 reduces the measured CUDA footprint to 19,802,113,536 bytes
allocated (18.44 GiB) and 19,862,126,592 bytes reserved (18.50 GiB) for the
one-example smoke run. All named parameters and buffers were verified on
`cuda:0`; no CPU or disk offload was used.

This route is separate because quantization is part of the scientific
provenance. The vendored gemotions vectors were extracted from
`google/gemma-4-31B-it` loaded with BitsAndBytes 4-bit weights and BF16 compute.
Google's W4A16 Compressed Tensors checkpoint also fits this GPU, but it uses a
different quantization representation and is retained as `convabuse-31b` rather
than being treated as activation-equivalent. The local route explicitly uses
FP4, BF16 compute, UINT8 storage, and no double quantization. It approximates the
documented extraction conditions, but exact numerical replication is not
claimed because the historical package versions, omitted defaults, CUDA
environment, and model revision were not recorded.

GGUF is likewise not structurally incompatible with the vectors: the same
architecture still has a 5,376-wide layer-40 residual stream. It is not assumed
to be numerically interchangeable, however, because GGUF Q4 and llama.cpp use
different quantized weights, kernels, and graph tensor names. GGUF probing would
need a custom activation callback plus cross-runtime validation, or vectors
re-extracted from the exact GGUF runtime.

Finally, this experiment does not reproduce the full gemotions extraction
pipeline. It reuses the vendored layer-40 vectors and clusters, captures the
response-start activation for ConvAbuse prompts, and computes cosine scores. A
full replication would regenerate the vectors from the emotion-story and
neutral corpora, repeat denoising and clustering, and validate the newly
extracted directions before applying them to ConvAbuse.

**Every run gets its own folder** — `results/<timestamp>_<experiment>/` with `scores.csv`,
`run_info.json` (exact route, model/vector revisions, quantization recipe,
package/runtime versions, requested/resolved placement, CPU/offload state, probe
settings, SHA-256 fingerprints for the finalized task manifest/vector/cluster/probe
source, and synchronized CUDA peak allocated/reserved bytes), and
`clusters.json` where applicable — so
results are never overwritten. `--resume` appends to the latest folder instead; interrupted
runs pick up where they left off. Resume fails before file mutation if scientific
provenance or the CSV schema differs, preserves the original run metadata, and
retains the maximum measured CUDA peaks across continuations. `analyze` writes
its tables and charts into the same folder.

The runner displays a task progress bar and flushes each completed score row to
CSV. Resume therefore restarts from the next unfinished prompt after an
interruption. Model inference remains batch size one to preserve the recorded
probing configuration.

Resume also validates every existing score row (exact field count, nonblank
unique task key, positive integer token count, and finite numeric emotion
scores) before appending. Gemotions runs persist `clusters.json` in canonical
JSON form and verify its recorded SHA-256 digest and schema before resume.

`--limit` is part of the run identity. A resume must repeat the original value
exactly: omit it again for an unlimited run, or pass the same `--limit N` used
when the run folder was created.

No duration or 24 GB fit is asserted until the manual smoke acceptance is
measured. A complete ConvAbuse run performs 4,185 forward passes; bailbench-2b is
substantially smaller.

### Historical extraction caveat

The vendored extraction source specified only `load_in_4bit=True` and BF16
compute. It omitted the FP4 quantization type, UINT8 storage, double-quantization
setting, and exact package versions. This implementation records a reviewed,
likely-default explicit route; it is **not proof of numerical equivalence** to the
historical extraction environment. It does not rewrite or re-extract the
vendored vectors. The historical extraction model revision is persisted as
`unknown` because the vendored repository does not prove an exact revision.

### Running on Alliance Fir

[`fir.slurm`](fir.slurm) requests one H100, 64 GB of system memory, four CPUs, and two
hours. It assumes that `.venv` and the model cache have already been created on Fir's shared
storage; compute nodes run offline and do not install packages or download checkpoints.
Create `.venv` on a Fir login node using DRAC's configured wheelhouse:

```
bash scripts/setup-fir.sh
```

The setup fails if DRAC does not provide an exact version pinned by `uv.lock`. To use DRAC
wheels where available and permit PyPI as a fallback for missing packages, run
`bash scripts/setup-fir.sh --allow-pypi` on the login node.

From the repository root, submit it with your allocation on the command line:

```
sbatch --account=<allocation> emotion_probing/fir.slurm
```

The model cache defaults to the persistent project path
`/project/def-nvincent/dhpham/cache/models`. Override `MODEL_CACHE_DIR` when the checkpoint
was downloaded elsewhere. Environment variables passed to `sbatch` select common run modes:

```
# Ten-example end-to-end smoke test
LIMIT=10 sbatch --account=<allocation> emotion_probing/fir.slurm

# Resume the latest incomplete ConvAbuse run
RESUME=1 sbatch --account=<allocation> emotion_probing/fir.slurm

# Run BailBench using an explicit cache location
EXPERIMENT=bailbench-2b MODEL_CACHE_DIR=/project/def-nvincent/dhpham/cache/models \
  sbatch --account=<allocation> emotion_probing/fir.slurm
```

Set `PYTHON_MODULE` if Fir's available Python 3.12 module has a more specific name, for
example `PYTHON_MODULE=python/3.12.4`. Do not combine `LIMIT` with `RESUME`: a limited run
has a different task set and should remain separate from a full resumable run. Slurm writes
the job log to `slurm-emotion-probing-<job-id>.out` in the submission directory.

## What analyze produces

Printed tables + `analysis.csv` + charts in `<run>/figures/`. Re-analysis needs no model or
GPU — it reads `scores.csv` and runs in seconds anywhere.

- **bailbench** (`figures/`): per-emotion rude-minus-normal deltas (diverging bars),
  hostility-cluster delta by rudeness formula, per-pair delta histogram.
- **convabuse**: all severity shifts are measured against the baseline band
  (`BASELINE_BAND` in `analyze/convabuse.py`, currently 0 = ambiguous). Two shared
  map figures recur throughout, both placing all 171 emotions at their real gemotions PCA
  coordinates: the **cluster map** (15 cluster names + hulls around clusters containing
  highlighted emotions) and the **PC1/PC2 map** (same points, axes annotated as
  valence/disposition with quadrant guides). Folders:
  - `figures/bands/` — per severity band: the 10 biggest risers and 10 biggest fallers vs
    the baseline band (the baseline band itself shows its raw top-10 resting profile) +
    both maps with those emotions highlighted.
  - `figures/comparison/` — band −3 vs the baseline band: the 10 biggest risers and 10
    biggest fallers as diverging bars, plus both maps with risers in red and fallers in
    blue.
  - `figures/overview/` — a 171-row heatmap (emotion × severity band, cell = shift vs the
    baseline band, sorted by the −3 shift) and grouped bars showing the top 10 risers +
    10 fallers' raw activation across all five bands.
  - `figures/breakdowns/` — hostility by abuse target (system-directed vs other — the
    headline cut for "rudeness at the model"), by abuse type, and by directness, using the
    majority-vote abusive flag. Groups smaller than 5 examples are dropped.

Scores are cosine similarities and are small in absolute terms — only differences between
groups/conditions are meaningful. Wording discipline: the vectors capture the model's
*representations* of emotion concepts ("represents the interaction as hostile"), not felt
emotion.

## The gemotions folder

[gemotions/](gemotions/) is a plain committed folder (~6 MB) holding the useful subset of the
[dejanseo/gemotions](https://huggingface.co/dejanseo/gemotions) vector-extraction repo — a
replication of the Anthropic emotion-concepts paper on Gemma4-31B: all its Python code (so
vector extraction can be re-run on another model later), the layer-40 emotion vectors, the
cluster/PCA analysis, and the small extraction inputs. The 35 GB of raw activation caches and
the story corpus were deliberately left on Hugging Face; [VENDORED.md](gemotions/VENDORED.md)
records the exact source revision and how to fetch any dropped file individually. No
submodule, no git-lfs, nothing extra to download.

## Folder layout

```
main.py          # runner: download | run (experiment configs at the top)
datasets.py      # dataset loaders + the ConvAbuse annotation collapse
analyze/         # per-run analysis package (common, maps, convabuse, bailbench)
extract.py       # base-model vector extraction (see extract_vectors.md)
data/            # ConvAbuseEMNLPfull.csv
results/         # one folder per run (never overwritten)
EmotionScope/    # vendored 2B vector-extraction repo (vectors .pt used)
gemotions/       # vendored 31B vectors + extraction code (see VENDORED.md)
.claude/         # context docs for coding agents
```
