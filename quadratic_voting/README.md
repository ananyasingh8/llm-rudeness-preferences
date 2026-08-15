# Quadratic Voting

This package contains local tools for the quadratic-voting experiments. The
Gemma runner uses the shared `llm_runtime` Transformers and PyTorch adapter; it
does not invoke llama.cpp or another external inference executable.

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
