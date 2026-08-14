# Bail workstream: rudeness-augmented BailBench

The augmentation pipeline assigns one of 12 seeded Culpeper impoliteness
formulae to each BailBench row, asks an injected `TextGenerator` to rewrite the
prompt, parses `<augmented>` tags, and preserves bounded retries, concurrency,
Parquet resume, and committed outputs.

## Run

From the repository root:

```console
uv sync --locked
export OPENROUTER_API_KEY=sk-or-...
uv run python -m bail.src.augment_bailbench
```

Only `OPENROUTER_API_KEY` is read. There is no local secrets-file fallback. Set
`AUGMENT_USE_MOCK = True` in `bail/config.py` for a deterministic no-network
run.

Resume filtering happens before credentials or HTTP clients are constructed.
Only transport failures and HTTP 408/429/selected 5xx responses are retried;
authentication, validation, malformed success payloads, and programmer errors
become `API_ERROR` rows immediately. Internally created HTTP clients are closed
after each batch, while injected clients remain caller-owned.

The real route is the registered OpenRouter model
`dolphin-mistral-24b-venice` with quantization `none`, because OpenRouter does
not disclose an enforceable upstream quantization. Remote generation has no
activation access. Adding or changing a model requires a reviewed entry in
`llm_runtime.registry` and adapter sanity test; do not add raw URLs, model slugs,
or provider SDK construction to the bail pipeline.

## Data

- `data/bailBench.csv`: 1,630 source prompts.
- `data/bailbench_augmented.csv` and `.parquet`: committed augmented dataset.
- `data/augment_smoke_10.csv`: small reference output.
