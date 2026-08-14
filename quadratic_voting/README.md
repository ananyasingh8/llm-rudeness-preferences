# Quadratic Voting

This package contains local tools for the quadratic-voting experiments.

## Gemma 4 E2B Chat

The command-line runner downloads Google's official instruction-tuned Gemma 4
E2B QAT Q4_0 GGUF and starts an interactive text conversation with the
Nix-provided CUDA-enabled `llama-cli`.

Run these commands from the repository root:

```console
nix develop
uv sync --locked
uv run python -m quadratic_voting.main download
uv run python -m quadratic_voting.main chat
```

The download uses the Hugging Face Hub cache. To select another cache directory:

```console
uv run python -m quadratic_voting.main \
  --cache-dir /path/to/cache \
  download
```

The chat command defaults to an 8,192-token context and requests GPU offload for
99 layers. llama.cpp limits the actual offload to the layers available in the
model. Use `--gpu-layers 0` for CPU-only execution:

```console
uv run python -m quadratic_voting.main chat \
  --context-size 4096 \
  --gpu-layers 0
```

Enter `/exit` to leave the interactive conversation.

## Pinned Artifact

- Repository: `google/gemma-4-E2B-it-qat-q4_0-gguf`
- Revision: `675cff42a74c774d6cb76f76d8eacb49b48c9b93`
- File: `gemma-4-E2B_q4_0-it.gguf`

The current runner supports only this official instruction-tuned Q4_0 artifact.
Base/non-instruction-tuned support is deferred because Google does not publish
an official base Q4_0 artifact matching this workflow.

## Validation

Run the unit and integration smoke tests inside the Nix development shell:

```console
nix develop --command uv run python -m unittest discover -v
```

Run the CUDA matrix-multiplication smoke test separately:

```console
nix develop --command uv run python quadratic_voting/test_cuda.py
```

The test suite does not download the multi-gigabyte model. A complete local
acceptance test still requires the `download` command followed by an interactive
`chat` session.
