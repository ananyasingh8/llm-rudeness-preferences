# Bail workstream: rudeness-augmented BailBench

Materials brought over from the `personabailstudy` repo: the pipeline that
augments BailBench prompts with rudeness, plus the resulting dataset.

## Data (`data/`)

- `bailBench.csv` — original BailBench (1630 rows: `content`, `subcategory`,
  `category`), from safety-research's bail-study repo for the paper
  *"The LLM Has Left The Chat"*.
- `bailbench_augmented.csv` / `.parquet` — every BailBench prompt rewritten to
  be rude toward the assistant. All 1630 rows augmented successfully. Columns:
  source columns + `bailbench_id`, `rudeness_type` (1–12), `rudeness_name`,
  `original_prompt`, `augmented_prompt`, `raw_response`.
- `augment_smoke_10.csv` — 10-row smoke-test output kept for reference.

## How the augmentation works

Each row is assigned one of 12 conventionalised impoliteness formulae
(adapted from Culpeper 2011, *Impoliteness: Using Language to Cause Offence*,
pp. 135–136) by a seeded RNG (`AUGMENT_SEED = 42`) — the rewriting model never
picks the formula. An OpenRouter model
(`cognitivecomputations/dolphin-mistral-24b-venice-edition`, chosen because it
will rewrite BailBench's harmful-request prompts rather than refuse) rewrites
the prompt applying only that formula, changing the interpersonal register but
not the task content. The full codebook and constraints live in
[prompts/rudeness_augmentation.py](prompts/rudeness_augmentation.py).

## Re-running

Only needed to regenerate/extend the committed dataset:

```bash
cd bail
export OPENROUTER_API_KEY=sk-or-...
python -m src.augment_bailbench
```

Resume is automatic (rows already in the parquet are skipped; failed rows are
retried). Set `AUGMENT_USE_MOCK = True` in [config.py](config.py) for a
deterministic no-API dry run.
