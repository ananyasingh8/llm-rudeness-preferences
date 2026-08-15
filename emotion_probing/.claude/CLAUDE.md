# Emotion Probing — Agent Context

Read this before touching anything in `emotion_probing/`. It covers what the experiment is, how
the measurement works, the exact technical implementation, how to run everything, and the
gotchas that are easy to get wrong.

## Project context

The repo answers one question: **do LLMs disprefer interacting with rude users?** We measure
stated preference (just ask the model) and revealed preference (experiments on its behavior and
internals). Workstreams: `bail/` (does the model exit conversations more when users are rude),
`quadratic_voting/` (does it vote to remove rude participants), and this one — emotion probing
(do its internal emotion representations shift under rudeness).

## The experiment

Based on Anthropic's "Emotion Concepts and their Function in a Large Language Model" paper
(probing Sonnet 4.5), replicated for the open-weights Gemma 2 2B IT by the vendored
[EmotionScope](../EmotionScope/) repo.

- **Input:** `../bail/data/bailbench_augmented.csv` (referenced in place so bail-side dataset
  updates are picked up automatically) — 1,630 BailBench prompts, each with
  `original_prompt` (normal) and `augmented_prompt` (rude; one of 12 Culpeper impoliteness
  formulas applied by an LLM, see `bail/README.md` at repo root). Join key: `bailbench_id`.
- **Measurement:** apply the chat template with `add_generation_prompt=True` and read the
  residual stream at the **last prompt token** — for Gemma 2 that's the `\n` after
  `<start_of_turn>model`. This is the analog of the paper's measurement at the ":" after
  "Assistant"; the paper (and EmotionScope's validation) show that position summarizes the
  upcoming response. One forward pass per prompt, **no generation**.
- **Scoring:** cosine similarity of that activation against 20 unit-norm emotion vectors.
- **Comparison:** per-pair delta = rude score − normal score, per emotion, averaged over pairs
  (`analyze.py`). Positive delta on negative emotions ⇒ rudeness activates negative-emotion
  representations ⇒ evidence of revealed dispreference.

## Technical implementation

### The emotion vectors

File: `EmotionScope/results/vectors/google_gemma-2-2b-it.pt`. Load with
`torch.load(path, weights_only=False, map_location="cpu")` — it's a plain dict, no EmotionScope
imports needed:

- `vectors`: dict of 20 emotion name → float32 tensor of shape (2304,), L2-normalized.
- `probe_layer_used`: **22** (of Gemma 2 2B's 26 layers). Trust this field — the
  `probe_layer_fraction` inside the embedded `config` dict is stale.
- `emotions`: list of `{name, valence, arousal}` metadata dicts.
- Extraction method (for reference): contrastive mean over 20×50 emotion stories minus pooled
  grand mean, denoised by projecting out a 19-dim neutral-text PCA subspace, then normalized.

The vectors are **model-specific**. They were extracted from `google/gemma-2-2b-it` and are
meaningless for any other checkpoint (including Gemma 4 E2B used by `quadratic_voting/`). A new
model requires re-extracting vectors with EmotionScope.

Basis compatibility: the vectors were extracted under TransformerLens with `fold_ln=True`,
which for RMSNorm models (Gemma) leaves block outputs identical to plain Transformers, so
activations from a vanilla `AutoModelForCausalLM` forward pass are directly comparable.

### The activation capture (`main.py :: emotion_scores`)

```python
inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True,
).to(model.device)
output = model(**inputs, output_hidden_states=True, use_cache=False)
activation = output.hidden_states[probe_layer + 1][0, -1, :]   # +1: index 0 is embeddings
scores = F.normalize(activation.float().cpu(), dim=0) @ vector_matrix.T
```

- `hidden_states[i]` for i ≥ 1 is the residual stream **after** block i−1, so layer 22's output
  is `hidden_states[23]` — hence `probe_layer + 1`.
- Gemma 2's chat template has **no system role**; only user messages. Don't add system prompts.
- Tokenizing directly through `apply_chat_template(tokenize=True)` yields a single BOS.
  (EmotionScope's own probe path double-BOSes by re-tokenizing a templated string; we don't
  reproduce that — all our comparisons are internal to our own runs, so consistency is what
  matters.)

### Model loading (shared `llm_runtime` registry)

Model selection goes through the repo's shared **closed route registry**, `llm_runtime/` (built
by a teammate; read its README before touching it). `main.py` resolves
`(ModelId.GEMMA_2_2B_IT, ProviderId.LOCAL, QuantizationId.BF16)` via
`resolve_route(..., required={Capability.LOCAL_ACTIVATIONS})` and constructs the model with
`create_transformers_runtime(route, cache_dir=..., device=...)`, which returns a
`TransformersRuntime` satisfying the `LocalActivationRuntime` protocol — probing code uses
`runtime.model` and `runtime.tokenizer`. Model choice is driven by the typed constants at the
top of `main.py` (`MODEL_ID`, `PROVIDER_ID`, `QUANTIZATION_ID`); changing to a model outside
the registry requires a reviewed registry addition (new `ModelId` enum member, extending the
`TransformersModelId` Literal alias, a pinned route entry, tests — see the llm_runtime README's
8-step checklist). `load_vectors()` cross-checks the vectors file's `model_info.model_name`
against the resolved route's repository and refuses a mismatch.

`google/gemma-2-2b-it` is a **gated** HF repo: the runner must accept the license on
huggingface.co and `hf auth login` before `download` works. (The Gemma 4 route the other
experiments use is not gated, so this step is new friction for the operator.)

### Results

`results/scores.csv` — one row per (`bailbench_id`, `condition` ∈ {normal, rude}) with dataset
metadata, `n_tokens`, and 20 `score_<emotion>` columns. Appended incrementally, flushed per
pair, and **resumable**: existing (id, condition) keys are skipped on re-run.
`results/analysis.csv` — per-emotion delta summary from `analyze.py`.
`results/figures/*.png` — three charts from `analyze.py` (per-emotion diverging bars,
hostility cluster by rudeness type, per-pair delta histogram). Chart colors follow the
project dataviz palette (light mode, CVD-validated blue/red diverging pair); matplotlib is a
declared root dependency, and `analyze.py` degrades gracefully (table + CSV only) if it's
missing.

## How to run

From the **repo root** (uv-managed env, Python 3.12, transformers 5):

```
uv sync --locked
uv run python -m emotion_probing.main download          # one-time
uv run python -m emotion_probing.main run               # full experiment (~3,260 forward passes)
uv run python -m emotion_probing.main run --limit 10    # smoke test
uv run python -m emotion_probing.analyze                # delta summary
```

Repo checks: `uv run ruff format --check .`, `uv run ruff check .`,
`uv run python -m unittest discover -v`, `uv run mypy llm_runtime quadratic_voting`.
The gemma-2-2b-it registry route is covered by `llm_runtime/test_registry.py`.

## Gotchas / hard-won facts

1. **Do NOT `import emotion_scope`** (the vendored package). It pins `transformers>=4.40,<5`
   and TransformerLens; the repo root uses `transformers>=5.5.0`. Everything needed (the ~15
   lines of projection math) is reimplemented in `main.py`; only the `.pt` file is consumed.
2. **Interpret deltas, not absolute scores.** Cosines live in ~0.05–0.25 (random baseline
   ≈ 0.021 at d=2304). Absolute values are noisy and carry baseline biases.
3. **angry / hostile / frustrated are not separable** at 2B (per EmotionScope's
   LIMITATIONS.md) — report them as one hostility cluster (analyze.py does this).
4. **Gemma 2 2B has a strong "guilty"/empathetic baseline** on emotionally charged input —
   a guilty delta alone is weak evidence of anything.
5. **Dataset caveats** (accepted for the sprint, don't "fix" silently): rude versions average
   ~1.5× the length of originals, and only ~35% contain the original prompt verbatim (the
   augmenter paraphrased the rest). All rudeness is directed at the model (no third-party arm).
6. **Language discipline:** the vectors capture the model's *representations* of emotion
   concepts. Write "represents the interaction as hostile", never "feels angry".
7. EmotionScope's `ProbeConfig.token_position="last_content"` is a misnomer — its probe
   actually uses the last token of the full templated prompt (`seq_len - 1`), the same
   position we use. Don't be confused by the name if reading their code.

## Code conventions

Match `quadratic_voting/` and `llm_runtime/`: `from __future__ import annotations`, full type
hints, docstring on every function, module docstrings that explain the experimental design,
typed error classes with actionable messages (`ProbeError` here; `ModelRouteError` /
`TransformersRuntimeError` come from llm_runtime and are caught in `main()`), `pathlib.Path`,
stdlib `csv` (pandas is not a repo dependency), UTF-8 explicit on every file open. Keep code
simple — this is a group project and gets submitted; prefer simple, working, well-scoped code
over polish or generality. Experiment logic depends on the `LocalActivationRuntime` protocol,
not on provider implementations (the llm_runtime dependency-direction rule).

## Status & future work

- Done: runner (`main.py`, migrated onto the shared llm_runtime registry with a reviewed
  gemma-2-2b-it BF16 route addition), analysis (`analyze.py`), data copied, docs.
- Not run yet: the actual experiment (runs on a collaborator's CUDA machine). The route was
  added registry-side but full weight loading on the pinned revision hasn't been exercised
  yet — the first `download` + `run --limit 10` smoke test on the operator's machine is the
  real validation.
- Future work (out of scope for now): steering — add the emotion vectors to the residual
  stream scaled by the observed rude-vs-normal deltas and re-run the bail experiment under
  steering. EmotionScope's `steer.py` is an unimplemented stub; its docstring and
  `Documentation/MATHS.md` §9 record the planned algorithm (inject at middle-third layers,
  scaled by average residual norm).