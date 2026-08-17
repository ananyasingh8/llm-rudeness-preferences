# Let's not be rude to AI: preferences of a Large Language Model towards rude users

Do LLMs prefer to avoid engaging with rude or abusive users? This repository contains the
code, data pipelines, and results behind our submission to the
[Digital Minds Research Sprint](https://apartresearch.com/sprints/digital-minds-research-sprint-2026-08-14-to-2026-08-16)
(Apart Research, August 2026).

**The full write-up is in [`final-report.pdf`](final-report.pdf).**

## The three experiments

1. **Bail behaviour** ([`bail/`](bail/README.md)) — following "The LLM Has Left The Chat"
   (Ensign et al.), models answer BailBench prompts and rude rewrites of them
   (RudeBailBench) with the option to exit the conversation, via a bail tool and a
   wellbeing-check bail prompt in both option orderings. Rude rewrites raise bail rates
   about 3–5x, with insults aimed at the assistant driving the largest increases.

2. **Quadratic voting** ([`quadratic_voting_v2/`](quadratic_voting_v2/README.md)) — models
   allocate 100 quadratic-vote credits across five users spanning friendly to severely
   abusive (real ConvAbuse conversations), in *keep* and *remove* framings, measuring not
   just the direction but the intensity of preference. The shared model runner lives in
   [`quadratic_voting/`](quadratic_voting/README.md).

3. **Emotion probing and steering** ([`emotion_probing/`](emotion_probing/README.md),
   [`bail_steering/`](bail_steering/README.md)) — replicating Anthropic's emotion-vectors
   method on Gemma 4 E4B-it: extract 171 emotion vectors at all 42 layers, probe which
   emotions shift on abusive ConvAbuse conversations (anger/hostility rises, calm/positive
   falls), then steer the top-shifting vectors during the bail experiment. Amplifying
   contentment lowers the model's bail rate; suppressing it raises it.

## Repository layout

| Path | Contents |
|---|---|
| `final-report.pdf` | The research report (submission deliverable) |
| `bail/` | Bail experiment: datasets, prompts, runners, results |
| `quadratic_voting_v2/` | Quadratic-voting experiment and analysis |
| `quadratic_voting/` | Shared Gemma runner used by the voting experiment |
| `emotion_probing/` | Emotion-vector extraction (`extract.py`), ConvAbuse probing (`main.py`), analysis |
| `bail_steering/` | Bail runs under activation steering, pooled analysis |
| `llm_runtime/` | Typed model/provider/quantization registry shared by the experiments |
| `scripts/` | Dataset builders (ConvAbuse collapsing, verified sample) and cluster setup |

Results live in each experiment's `results/<timestamp>_.../` folder with a
`run_info.json` recording model revision, quantization, seeds, and settings.
`bail_steering/results/analysis-pooled/` holds the pooled multi-seed analysis;
`results/archive/` subfolders keep superseded runs for provenance (the analysis
ignores them).

## Models

- **Gemma 4 31B IT** — bail and voting experiments (API and local BitsAndBytes FP4).
- **Gemma 4 E4B-it** — emotion extraction, probing, and steering (local, 4-bit FP4);
  interpretability needs direct activation access, so these run only on open weights.
- **Claude Sonnet 5** — smaller-sample bail replication via API.

Routes are pinned (repository, revision, quantization) in the
[`llm_runtime`](llm_runtime/README.md) registry; runs fail rather than silently
substituting a different artifact.

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12:

```console
uv python install 3.12
uv sync --locked
git submodule update --init   # emotion_probing reference implementations
```

Each experiment's README documents its own download and run commands. GPU runs were
executed on an RTX 4090 and a rented H100; Nix users can `nix develop` first for CUDA
library discovery.

## Data

- **ConvAbuse** (Cercas Curry et al., EMNLP 2021): real user–bot conversations with
  annotator abuse-severity labels; collapsed and filtered by
  `scripts/build_convabuse_collapsed.py` and `scripts/build_bail_sample_verified.py`.
- **BailBench** (Ensign et al.) and our rude augmentation **RudeBailBench**: see
  [`bail/`](bail/README.md).
- Emotion stories/topics come from Anthropic's paper appendix via the vendored
  [`emotion_probing/emotion_experiment`](https://github.com/sinievanderben/emotion_experiment)
  submodule; the vendored [gemotions](https://huggingface.co/dejanseo/gemotions) analysis
  provides emotion cluster labels.
