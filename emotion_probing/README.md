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
| Model | google/gemma-2-2b-it (bf16) | google/gemma-4-31B-it (W4A16 4-bit) |
| Emotion vectors | EmotionScope, 20 emotions, layer 22 | [gemotions](https://huggingface.co/dejanseo/gemotions), 171 emotions, layer 40 |
| Dataset | 1,630 synthetic normal/rude prompt pairs (from `../bail/data/`) | 4,185 real user-bot conversation snippets, human-annotated (ConvAbuse) |
| Comparison | paired: rude − normal on identical content | between groups: abusive vs non-abusive, by severity/type/target |

Emotion vectors are model-specific, so each configuration pairs a model with vectors extracted
from that exact model. Models are resolved through the shared [llm_runtime](../llm_runtime/)
closed registry; the gemotions vectors (~4 MB) are fetched from the Hugging Face Hub at a
pinned revision — the local `gemotions/` submodule clone is reference material only.

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

One-time setup:

```
uv sync --locked
hf auth login                    # only needed for the gated 2B model
uv run python -m emotion_probing.main --experiment convabuse-31b download   # ~18 GB
uv run python -m emotion_probing.main --experiment bailbench-2b download    # ~5 GB
```

Run and analyze:

```
uv run python -m emotion_probing.main run                      # convabuse-31b (default)
uv run python -m emotion_probing.main run --limit 10           # quick smoke test
uv run python -m emotion_probing.main run --resume             # continue the latest run
uv run python -m emotion_probing.main --experiment bailbench-2b run
uv run python -m emotion_probing.analyze                       # analyzes the latest run
uv run python -m emotion_probing.analyze --run results/<folder>
```

**Every run gets its own folder** — `results/<timestamp>_<experiment>/` with `scores.csv`,
`run_info.json` (exact model/vectors revisions), and `clusters.json` where applicable — so
results are never overwritten. `--resume` appends to the latest folder instead; interrupted
runs pick up where they left off. `analyze` writes its tables and charts into the same folder.

Expect the convabuse-31b run to take ~30–90 minutes on a 24 GB GPU (4,185 forward passes
through a quantized 31B model); bailbench-2b takes a few minutes.

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

## The gemotions submodule

[gemotions/](gemotions/) is a skip-smudge clone of the vector-extraction repo (a replication
of the Anthropic paper on Gemma4-31B): all code and analysis files are real, but the ~35 GB
of raw activation caches are 133-byte LFS pointers. To clone it yourself:

```
$env:GIT_LFS_SKIP_SMUDGE = "1"; git submodule update --init emotion_probing/gemotions
git -C emotion_probing/gemotions lfs pull --include "results/gemma4-31b/emotion_vectors_layer*.npz"
```

Never run a bare `git submodule update` / `git lfs pull` inside it without the include filter
unless you actually want 35 GB. The experiment itself never reads this clone — it downloads
its two files from the Hub at a pinned revision.

## Folder layout

```
main.py          # runner: download | run (experiment configs at the top)
datasets.py      # dataset loaders + the ConvAbuse annotation collapse
analyze.py       # per-run analysis: tables, analysis.csv, charts
data/            # ConvAbuseEMNLPfull.csv
results/         # one folder per run (never overwritten)
EmotionScope/    # vendored 2B vector-extraction repo (vectors .pt used)
gemotions/       # submodule: 31B vector-extraction repo (reference only)
.claude/         # context docs for coding agents
```
