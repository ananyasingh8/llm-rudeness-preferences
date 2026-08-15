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
```

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
