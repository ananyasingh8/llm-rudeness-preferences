# llm-rudeness-preferences

## Project Context

This repo is our submission for the [Digital Minds Research Sprint](https://apartresearch.com/sprints/digital-minds-research-sprint-2026-08-14-to-2026-08-16) (Apart Research, Aug 14-16, 2026), a hackathon focused on building empirical foundations for AI welfare research: probing the preferences, welfare signals, introspective abilities, and identity of frontier AI models.

## What we're doing

We're running a small set of experiments on frontier LLMs, loosely spanning the sprint's tracks on welfare/valence signals and preference elicitation. Three workstreams:

### 1. Bail behavior

We're studying "bail" - cases where a model chooses to exit or end a conversation when given the option - as a behavioral welfare signal. The rough idea is to measure when and why models opt out of interactions, and how that relates to the content/conditions of the conversation.

### 2. Emotion probes

Experiment design still TBD. Broadly: interpretability-style probes related to emotion/valence in model internals. Details will be added as they're settled.

### 3. Quadratic voting (QV)

Experiment design still TBD. Broadly: using QV-style mechanisms as a preference-elicitation method for models. Details will be added as they're settled.

## Practical notes for agents

- Deliverable is a short research report (PDF), optionally with code and a demo. Deadline: Sunday, Aug 16, 11:59 PM AoE.
- This is a weekend sprint - prefer simple, working, well-scoped code over polish or generality.
- Don't invent experimental details for the TBD workstreams; ask or leave placeholders.

## Gemma 4 E2B Runner

The current runner uses Transformers and PyTorch to download and run Google's
official instruction-tuned Gemma 4 E2B QAT checkpoint. Developers do not need
to install llama.cpp or another C++ inference program.

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Network access for the first dependency and model downloads
- Enough disk and memory for the Python environment and model checkpoint

An NVIDIA GPU is optional. CUDA execution requires a working NVIDIA driver.
PyTorch and the other application dependencies are installed from locked Python
packages by uv.

### Setup With uv

Install Python 3.12 through uv:

```console
uv python install 3.12
```

From the repository root, create `.venv` and install the locked application and
development dependencies:

```console
uv sync --locked
```

Run Python through uv so commands use the project environment instead of the
system Python installation:

```console
uv run python --version
```

### Download The Model

Download the complete pinned Transformers checkpoint into the Hugging Face Hub
cache:

```console
uv run python -m quadratic_voting.main download
```

The default cache is normally `~/.cache/huggingface/hub`. Select another cache
directory by placing the global option before the subcommand:

```console
uv run python -m quadratic_voting.main \
  --cache-dir /path/to/cache \
  download
```

Pinned artifact:

- Repository: `google/gemma-4-E2B-it-qat-q4_0-unquantized`
- Revision: `6befbaca7398925921802abd1f277b495b78b738`
- Runtime: Transformers on PyTorch
- Weight dtype: BF16

`qat-q4_0` identifies the quantization target used during quantization-aware
training. `unquantized` means this checkpoint stores the resulting weights in a
high-precision Transformers format. The current runner does not pack the
weights into Q4_0 before inference.

### Run Interactive Chat

After the download completes, start the Python conversation loop:

```console
uv run python -m quadratic_voting.main chat
```

The default `auto` device lets Accelerate place model modules on available GPU
and CPU memory. Device selection and response length can be set explicitly:

```console
uv run python -m quadratic_voting.main chat \
  --device cuda \
  --max-new-tokens 128
```

Use `--device cpu` for CPU inference. Enter `/exit` to leave the interactive
conversation. Run `uv run python -m quadratic_voting.main --help` for all
commands and options.

The BF16 weights require approximately 10.2 GB before CUDA context, KV cache,
temporary buffers, and other runtime allocations. Explicit `--device cuda`
requires at least 12 GB of free GPU memory as a conservative loading budget and
can still require more for long conversations. Use `--device auto` to permit
GPU/CPU placement on smaller GPUs. Reducing `--max-new-tokens` helps with
generation memory only after the model weights fit.

Conversation history is limited by the model's 131,072-token context. The
runner returns an actionable error instead of generating past that boundary.

See [`quadratic_voting/README.md`](quadratic_voting/README.md) for the
runner-specific reference.

### Validate The Environment

Run the test suite:

```console
uv run python -m unittest discover -v
```

Run the CUDA matrix-multiplication smoke test on an NVIDIA system:

```console
uv run python quadratic_voting/test_cuda.py
```

This smoke test verifies native BF16 CUDA execution, which is required by the
current checkpoint path.

Run static checks:

```console
uv run ruff format --check llm_runtime quadratic_voting
uv run ruff check llm_runtime quadratic_voting
uv run mypy llm_runtime quadratic_voting
uv run mypy --warn-unused-ignores typing_tests/local_activation_boundary.py
```

The automated tests do not download the multi-gigabyte checkpoint. A complete
local acceptance test requires the model download followed by an interactive
chat session.

To verify the pinned model configuration and chat-template metadata against the
Hugging Face Hub without downloading model weights, run:

```console
RUN_HF_INTEGRATION=1 uv run python -m unittest \
  llm_runtime.test_transformers.TransformersRuntimeTests.test_pinned_bf16_metadata_and_text_chat_template
```

### Optional Nix Environment

Nix users can enter the included development shell before running the same uv
commands:

```console
nix develop
uv sync --locked
```

The shell supplies Triton's compiler and NixOS NVIDIA driver-library discovery;
run local CUDA inference from inside this shell. Nix remains optional for
developers on conventional Linux systems.

## Typed Runtime Registry

[`llm_runtime`](llm_runtime/README.md) separates model identity, provider,
quantization, pinned artifact metadata, credentials, and per-run generation
settings. Public identifiers use strongly typed `StrEnum` values, and dynamic
CLI values are validated once against a closed route registry.

| Model | Provider | Quantization | Status |
|---|---|---|---|
| `gemma-4-e2b-it` | `local` | `bf16` | enabled |
| `gemma-4-e2b-it` | `local` | `w4a16-compressed-tensors` | unavailable candidate |
| `dolphin-mistral-24b-venice` | `openrouter` | `none` | enabled |

The W4A16 candidate remains unavailable and advertises no capabilities until
its exact pinned revision passes real weight loading, text generation, and
model/tokenizer activation-access validation. OpenRouter requires
`OPENROUTER_API_KEY`, does not expose activations, and has no enforceable
quantization identity. The active Bail experiment remains unchanged from
`origin/main`; this registry does not alter or re-run its completed augmentation
pipeline.
