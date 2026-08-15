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

There are two experiment configurations (`EXPERIMENTS` in [main.py](main.py)):

| | `bailbench-2b` | `convabuse-31b` (default) |
|---|---|---|
| Model | google/gemma-2-2b-it (bf16) | google/gemma-4-31B-it (BitsAndBytes FP4 at load time) |
| Emotion vectors | EmotionScope, 20 emotions, layer 22 | [gemotions](https://huggingface.co/dejanseo/gemotions), 171 emotions, layer 40 |
| Dataset | 1,630 synthetic normal/rude prompt pairs (from `../bail/data/`) | 4,185 real user-bot conversation snippets, human-annotated (ConvAbuse) |
| Comparison | paired: rude − normal on identical content | between groups: abusive vs non-abusive, by severity/type/target |

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

### Reviewed ConvAbuse route

`convabuse-31b` is closed to one route: `google/gemma-4-31B-it` revision
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
uv run python -m emotion_probing.main --experiment convabuse-31b download
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
uv run python -m emotion_probing.main --experiment bailbench-2b run
uv run python -m emotion_probing.analyze                       # analyzes the latest run
uv run python -m emotion_probing.analyze --run results/<folder>
```

The exact manually authorized RTX 4090 smoke command is:

```console
uv run python -m emotion_probing.main --experiment convabuse-31b run --device cuda --limit 1
```

The pinned base repository download/cache needs **at least 60 GB of disk** (allow
additional temporary/cache headroom); runtime quantization does not make the Hub
download a pre-quantized 18 GB artifact. The target is a 24 GB RTX 4090 with the
entire reviewed route on CUDA. Whether this exact lock fits is **pending measured
acceptance** from the separately authorized smoke run. The runtime rejects CPU or
disk placement rather than silently offloading, changing precision, or selecting
another artifact.

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

## What analyze produces

Printed tables + `analysis.csv` + charts in `<run>/figures/`:

- **bailbench**: per-emotion rude-minus-normal deltas (diverging bars), hostility-cluster
  delta by rudeness formula, per-pair delta histogram.
- **convabuse**: severity trend for the Anger/Hostility and Positive/Joy clusters,
  per-cluster shift under abuse (using the gemotions unsupervised clustering of all 171
  emotions into 15 groups), top-10 rising/falling emotions labeled by cluster, hostility by
  abuse target (system-directed vs other — the headline cut for "rudeness at the model"),
  by abuse type, and by directness (explicit/implicit). Groups smaller than 5 examples are
  dropped from charts.

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
analyze.py       # per-run analysis: tables, analysis.csv, charts
data/            # ConvAbuseEMNLPfull.csv
results/         # one folder per run (never overwritten)
EmotionScope/    # vendored 2B vector-extraction repo (vectors .pt used)
gemotions/       # vendored 31B vectors + extraction code (see VENDORED.md)
.claude/         # context docs for coding agents
```
