# Quadratic Voting

The interactive runner is a thin composition root over `llm_runtime`. Its
conversation function accepts any injected `TextGenerator`; experiment logic
does not import Transformers or OpenRouter.

## Local Gemma

The default is the pinned BF16 route:

```console
uv run python -m quadratic_voting.main download
uv run python -m quadratic_voting.main chat --device auto --max-new-tokens 128
```

The official Compressed Tensors W4A16 artifact remains pinned as an unavailable
candidate. Selecting `--quantization w4a16-compressed-tensors` fails at route
resolution because it has not passed exact pinned-revision weight loading, text
generation, and real model/tokenizer activation-access validation. It advertises
no capabilities and cannot be enabled until all three checks pass; there is no
BF16 fallback.

Explicit CUDA retains the 12 GB conservative device check; `auto` permits
Accelerate placement. The runtime rejects generations beyond Gemma's
131,072-token context and uses the text-only tokenizer path. Enabled local
runtimes expose the actual model and tokenizer for activations.

## OpenRouter chat

```console
export OPENROUTER_API_KEY=sk-or-...
uv run python -m quadratic_voting.main \
  --model dolphin-mistral-24b-venice \
  --provider openrouter \
  --quantization none \
  chat
```

OpenRouter quantization is undisclosed and remote routes do not expose
activations. To add a model or quantization, add and review a pinned closed
registry route and its provider loader; do not place provider branches or raw
artifact slugs in this package.
