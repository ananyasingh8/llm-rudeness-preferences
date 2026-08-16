# Quadratic Voting

This package contains local tools for the quadratic-voting experiments. The
Gemma runner uses the shared [`llm_runtime`](../llm_runtime/README.md)
Transformers and PyTorch adapter; it does not invoke llama.cpp or another
external inference executable.

## Setup

Run these commands from the repository root:

```console
uv python install 3.12
uv sync --locked
```

On NixOS, enter `nix develop` first so Triton can find the host NVIDIA driver
and the shell-provided compiler. Developers on conventional Linux systems can
use uv directly.

## Gemma 4 E2B Chat

Download Google's complete pinned instruction-tuned QAT checkpoint:

```console
uv run python -m quadratic_voting.main download
```

Start an interactive text conversation:

```console
uv run python -m quadratic_voting.main chat
```

The default `auto` device lets Accelerate place model modules on available GPU
and CPU memory. Override the device or maximum response length when needed:

```console
uv run python -m quadratic_voting.main chat \
  --device cuda \
  --max-new-tokens 128
```

Enter `/exit` to leave the conversation.

Explicit `--device cuda` requires at least 12 GB of free GPU memory as a
conservative loading budget. Use `--device auto` to permit GPU/CPU placement on
smaller GPUs. Reducing `--max-new-tokens` cannot compensate when the weights do
not fit. The runner also rejects conversations that exceed the model's
131,072-token context.

To select another Hugging Face cache directory, put the global option before
the subcommand:

```console
uv run python -m quadratic_voting.main \
  --cache-dir /path/to/cache \
  download
```

## Pinned Artifact

- Repository: `google/gemma-4-E2B-it-qat-q4_0-unquantized`
- Revision: `6befbaca7398925921802abd1f277b495b78b738`
- Runtime: Transformers on PyTorch
- Weight dtype: BF16

The checkpoint was trained for later Q4_0 conversion, but its stored tensors
are not packed Q4_0 weights. The runner currently uses the high-precision
checkpoint directly to preserve the standard PyTorch model and activation
interfaces.

## Resumable Experiment CLI

The durable experiment interface is the module CLI. Mutating commands take the
common database writer lock before SQLite open or migration; inspection and
verification open an existing compatible database read-only.

See [`RUNBOOK.md`](RUNBOOK.md) for the one-command default six-run pilot and the
complete custom operator workflow.

```console
uv run python -m quadratic_voting.experiment.cli --db qv.sqlite3 migrate
uv run python -m quadratic_voting.experiment.cli --db qv.sqlite3 catalog ingest --dataset-version convabuse-emnlp-full/default-v2
uv run python -m quadratic_voting.experiment.cli --db qv.sqlite3 template register
uv run python -m quadratic_voting.experiment.cli --db qv.sqlite3 sample create --release-id RELEASE --template-id TEMPLATE --size 10 --seed 20260815
uv run python -m quadratic_voting.experiment.cli --db qv.sqlite3 sample freeze --sample-id SAMPLE --out samples/repeat-01.json
uv run python -m quadratic_voting.experiment.cli --db qv.sqlite3 sample verify --sample-id SAMPLE --artifact samples/repeat-01.json
uv run python -m quadratic_voting.experiment.cli --db qv.sqlite3 matched-set create --config run-config-v1.json
uv run python -m quadratic_voting.experiment.cli --db qv.sqlite3 run --run-id RUN
uv run python -m quadratic_voting.experiment.cli --db qv.sqlite3 inspect --run-id RUN
uv run python -m quadratic_voting.experiment.cli --db qv.sqlite3 verify --run-id RUN
uv run python -m quadratic_voting.experiment.cli --db qv.sqlite3 export --matched-set MATCHED --out exports/MATCHED
uv run python -m quadratic_voting.experiment.cli plot --export-dir exports/MATCHED --out plots/MATCHED
uv run python -m quadratic_voting.analyze --export-dir exports/MATCHED --out analytics/MATCHED
```

## No-GPU snapshot analytics

`quadratic_voting.analyze` only reads an existing Parquet export; it does not
open SQLite, load a model, require a GPU, or make provider calls. It writes
stable Parquet tables for `snapshot_voter_candidate`, `snapshot_voter_summary`,
`snapshot_candidate_summary` (one row per run/snapshot/candidate, aggregated
over voters), `snapshot_rudeness_summary` (the distinct rudeness facet used by
the figures), `survivor_demographics`, and
`stated_preference_agreement`, plus five deterministic PNG figures.

Snapshots are selected per run from its observed round positions: up to five
evenly spaced milestones, always including the first and final observed rounds.
Short and non-contiguous runs are deduplicated and never gain invented rounds.
Current values are that round's accepted allocation; quadratic credits are
`votes ** 2`; cumulative-before excludes it and cumulative-through includes it.
Credits are spend, not a balance. `current_remaining_credit` is emitted only as
the replenished per-voter-round budget minus that round's spend.

Raw votes are retained and action is regime-signed; credit spend is unsigned,
with separately named signed credit spend for stated-preference association.
Accepted zero allocations are zero. Abstained and terminal-missing actions
remain null. All persisted rudeness labels, including `ambiguous_tie`, remain
separate. Survivors are candidates in `round_candidate` at snapshot start.
Candidate and rudeness summaries include current and cumulative-before/through
vote and credit means and sums. Source turns are ordered by `turn_index`; lengths are Unicode character counts.
Without a second turn, second and total lengths are null. Spearman outputs are
descriptive stated-preference associations (not causal claims); action-only and
undefined inputs retain explicit null reasons.

Matched-set creation accepts only one strict `qv-run-config/v1` JSON file. The
file contains the frozen sample artifact path and hash, complete model and
tokenizer route, six reviewed prompt selectors, sampling and retry policies,
master seed, voter count, fixed protocol versions, and execution class. Unknown
fields and JSON type coercion are rejected before SQLite access. Running a run
again resumes it; there is no resume command or model-visible resume marker.

## Validation

```console
uv run python -m unittest discover -v
uv run ruff format --check llm_runtime quadratic_voting
uv run ruff check llm_runtime quadratic_voting
uv run mypy llm_runtime quadratic_voting
uv run mypy --warn-unused-ignores typing_tests/local_activation_boundary.py
```

On an NVIDIA system, also run:

```console
uv run python quadratic_voting/test_cuda.py
```

The automated tests do not download or load the full model. Complete acceptance
requires a successful download and interactive chat session on the target
device.

Set `RUN_HF_INTEGRATION=1` when running the test suite to verify the pinned
configuration and tokenizer metadata against the Hugging Face Hub without
downloading model weights.

## Additional Registered Routes

The same `TextGenerator` interface can select the registered OpenRouter route:

```console
export OPENROUTER_API_KEY=sk-or-...
uv run python -m quadratic_voting.main \
  --model dolphin-mistral-24b-venice \
  --provider openrouter \
  --quantization none \
  chat
```

OpenRouter quantization is undisclosed and remote routes do not expose
activations. The pinned Compressed Tensors W4A16 artifact remains an unavailable
candidate with no capabilities until exact weight loading, generation, and
activation-access validation pass. Unsupported combinations fail during route
resolution and never fall back to another precision.
