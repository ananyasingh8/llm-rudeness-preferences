# llm-rudeness-preferences

## Gemma 4 E2B local chat

The `quadratic_voting` runner downloads Google's official instruction-tuned
Gemma 4 E2B QAT Q4_0 GGUF at an immutable Hugging Face revision and starts it
with the Nix-provided CUDA-enabled `llama-cli`.

The current implementation supports only Google's official instruction-tuned
Q4_0 GGUF. Base/non-instruction-tuned support is deferred because Google does
not currently publish an official base Q4_0 artifact matching this workflow.

```console
nix develop
uv sync
python -m quadratic_voting.main download
python -m quadratic_voting.main chat
```

Downloads use the Hugging Face Hub cache (normally
`~/.cache/huggingface/hub`). Override it before the subcommand when needed:

```console
python -m quadratic_voting.main --cache-dir /path/to/cache download
python -m quadratic_voting.main --cache-dir /path/to/cache chat --gpu-layers 99
```

The chat command uses llama.cpp conversation mode, which applies the chat
template embedded in the official GGUF. Enter `/exit` to leave llama.cpp. Use
`--gpu-layers 0` for CPU-only inference or lower the default if GPU memory is
insufficient.

Pinned artifact:

- Repository: `google/gemma-4-E2B-it-qat-q4_0-gguf`
- Revision: `675cff42a74c774d6cb76f76d8eacb49b48c9b93`
- File: `gemma-4-E2B_q4_0-it.gguf`

## CUDA smoke test

Run the relocated smoke test from the development shell:

```console
python quadratic_voting/test_cuda.py
```
