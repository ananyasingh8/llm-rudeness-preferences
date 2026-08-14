# llm-rudeness-preferences

Research-sprint experiments on bail behavior, emotion/valence probes, and
quadratic-voting preference elicitation. The repository favors small,
reproducible experiment paths over a general inference framework.

## Setup

```console
uv python install 3.12
uv sync --locked
```

## Reviewed model routes

`llm_runtime` keeps model identity, provider, quantization, pinned artifacts,
credentials, and per-run generation settings separate. Its closed registry
contains:

| Model | Provider | Quantization | Status | Artifact/runtime |
|---|---|---|---|---|
| `gemma-4-e2b-it` | `local` | `bf16` | enabled | `google/gemma-4-E2B-it-qat-q4_0-unquantized` at `6befbaca7398925921802abd1f277b495b78b738` |
| `gemma-4-e2b-it` | `local` | `w4a16-compressed-tensors` | unavailable candidate | `google/gemma-4-E2B-it-qat-w4a16-ct` at `971342c08f607aa7779983f6b5289778b5d271a7` |
| `dolphin-mistral-24b-venice` | `openrouter` | `none` | enabled | `cognitivecomputations/dolphin-mistral-24b-venice-edition` |

The W4A16 route uses `compressed-tensors`; the BF16 route preserves the
previous local behavior. OpenRouter does not disclose an enforceable upstream
quantization, so its route is deliberately `none`. Only local routes expose the
real model and tokenizer through `LocalActivationRuntime`; remote activation
access is not supported.

W4A16 metadata parsing succeeded, but metadata cannot prove model loading or
generation. The registry therefore rejects this candidate before download or
runtime construction and advertises no capabilities for it. Enabling it
or granting any capability requires reviewed validation of exact pinned-revision
weight loading, text generation, and real model/tokenizer exposure for activation
access.

## Usage

The QV CLI defaults to local Gemma BF16:

```console
uv run python -m quadratic_voting.main download
uv run python -m quadratic_voting.main chat
```

OpenRouter calls require only the process environment variable:

```console
export OPENROUTER_API_KEY=sk-or-...
uv run python -m bail.src.augment_bailbench
```

See [`quadratic_voting/README.md`](quadratic_voting/README.md) and
[`bail/README.md`](bail/README.md) for experiment-specific commands.

## Adding a route

Add a new enum identity and a pinned entry in `llm_runtime/registry.py`, add its
quantization-specific loader behavior in the provider adapter, validate the
exact artifact load, generation, and required activation path, and review focused
boundary tests. Experiments must continue to depend only on `TextGenerator` (or
`LocalActivationRuntime` when local activations are required); do not accept
arbitrary model slugs.

## Checks

```console
uv run python -m unittest discover -v
uv run ruff format --check llm_runtime bail quadratic_voting
uv run ruff check llm_runtime bail quadratic_voting
uv run mypy llm_runtime bail quadratic_voting
uv run mypy --no-incremental --warn-unused-ignores \
  typing_tests/local_activation_boundary.py
```

Tests do not download model weights or call OpenRouter live. Set
`RUN_HF_INTEGRATION=1` only to check the small pinned BF16 Hub metadata and text
chat template.
