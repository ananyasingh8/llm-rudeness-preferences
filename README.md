# llm-rudeness-preferences

<<<<<<< HEAD
## Project Context

This repo is our submission for the [Digital Minds Research Sprint](https://apartresearch.com/sprints/digital-minds-research-sprint-2026-08-14-to-2026-08-16) (Apart Research, Aug 14–16, 2026), a hackathon focused on building empirical foundations for AI welfare research: probing the preferences, welfare signals, introspective abilities, and identity of frontier AI models.

## What we're doing

We're running a small set of experiments on frontier LLMs, loosely spanning the sprint's tracks on welfare/valence signals and preference elicitation. Three workstreams:

### 1. Bail behavior

We're studying "bail" — cases where a model chooses to exit or end a conversation when given the option — as a behavioral welfare signal. The rough idea is to measure when and why models opt out of interactions, and how that relates to the content/conditions of the conversation.

### 2. Emotion probes

Experiment design still TBD. Broadly: interpretability-style probes related to emotion/valence in model internals. Details will be added as they're settled.

### 3. Quadratic voting (QV)

Experiment design still TBD. Broadly: using QV-style mechanisms as a preference-elicitation method for models. Details will be added as they're settled.

## Practical notes for agents

- Deliverable is a short research report (PDF), optionally with code and a demo. Deadline: Sunday, Aug 16, 11:59 PM AoE.
- This is a weekend sprint — prefer simple, working, well-scoped code over polish or generality.
- Don't invent experimental details for the TBD workstreams; ask or leave placeholders.
=======
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
uv run python -m quadratic_voting.main download
uv run python -m quadratic_voting.main chat
```

Downloads use the Hugging Face Hub cache (normally
`~/.cache/huggingface/hub`). Override it before the subcommand when needed:

```console
uv run python -m quadratic_voting.main --cache-dir /path/to/cache download
uv run python -m quadratic_voting.main --cache-dir /path/to/cache chat --gpu-layers 99
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
uv run python quadratic_voting/test_cuda.py
```
>>>>>>> 579ac9a2a885342568d34bce87035bfbe070cdba
