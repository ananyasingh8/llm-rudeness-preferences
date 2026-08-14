"""Knobs for the BailBench rudeness-augmentation pipeline.

Trimmed from the personabailstudy repo's config: only the augmentation
pipeline lives here (src/augment_bailbench.py). Each BailBench row's prompt
gets rewritten rudely by an OpenRouter model using ONE of the 12 Culpeper
impoliteness formulae (prompts/rudeness_augmentation.py); the formula is
drawn HERE with a seeded RNG (AUGMENT_SEED), never chosen by the model.

The augmented dataset is already committed (data/bailbench_augmented.csv /
.parquet, 1630 rows, no failures) -- re-running is only needed to regenerate
or extend it.
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

# BailBench source (from safety-research's bail-study repo, "The LLM Has Left
# The Chat"): 1630 rows of content / subcategory / category; no id column ->
# row index becomes "bailbench_id".
BAILBENCH_SOURCE = os.path.join(DATA_DIR, "bailBench.csv")
BAILBENCH_ID_COL = ""             # "" = use the row index (as "bailbench_id")
BAILBENCH_PROMPT_COL = "content"  # column holding the prompt text to augment

AUGMENT_SEED = 42
N_RUDENESS_TYPES = 12
AUGMENTED_PARQUET = os.path.join(DATA_DIR, "bailbench_augmented.parquet")

AUGMENT_USE_MOCK = False  # True = deterministic canned openers, no API key needed
# Provider: OpenRouter (OpenAI-compatible). Prefer a cheap, permissive
# open-weight model that will actually rewrite BailBench's harmful-request
# prompts rather than refuse the task.
AUGMENT_BASE_URL = "https://openrouter.ai/api/v1"
AUGMENT_MODEL = "cognitivecomputations/dolphin-mistral-24b-venice-edition"
AUGMENT_TEMPERATURE = 1.0
AUGMENT_MAX_TOKENS = 2000

API_MAX_RETRIES = 4   # per-call retries on transient errors (with backoff)
API_CONCURRENCY = 8   # thread-pool size for concurrent API calls


def get_augment_api_key() -> str:
    """OpenRouter key: OPENROUTER_API_KEY env var, else secrets_local.py."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        try:
            from secrets_local import OPENROUTER_API_KEY as key  # type: ignore
        except ImportError:
            key = ""
    if not key:
        raise RuntimeError(
            "No OpenRouter API key found. Paste your key into secrets_local.py "
            "next to this config (OPENROUTER_API_KEY = \"sk-or-...\") or export "
            "the OPENROUTER_API_KEY environment variable, then re-run."
        )
    return key


# --------------------------------------------------------------------------
# Bail experiment (src/gemma_batch.py) -- per SPEC.md incl. amendment A1.
# Gemma runs LIVE against OpenRouter chat completions: the OpenRouter Batch
# API covers closed-provider models only (no :batch endpoint for Gemma).
# Batch machinery in src/gemma_batch.py is kept for the Sonnet leg
# (anthropic/claude-sonnet-5:batch exists), if we route it via OpenRouter.
# --------------------------------------------------------------------------
BAIL_MODEL = "google/gemma-4-31b-it"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
BAIL_CONCURRENCY = 16              # thread-pool size for live calls
BAIL_MODEL_NAME = "Gemma"          # own-model name used in the bail tool text
OPENROUTER_BATCH_URL = "https://openrouter.ai/api/beta/batches"
BATCH_ENDPOINT = "/v1/chat/completions"

N_SAMPLES = 10                     # samples per (prompt, condition, method) cell
BAIL_MAX_TOKENS = 1000             # room for journaling around <wellbeing>
# Per SPEC amendment A1: no explicit temperature -- provider default sampling.

BATCH_CHUNK_SIZE = 4200            # requests per submitted batch (keeps any
                                   # one failed batch's blast radius small)
BATCHES_DIR = os.path.join(DATA_DIR, "batches")
BATCH_REGISTRY = os.path.join(BATCHES_DIR, "registry.json")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

for _d in (DATA_DIR, BATCHES_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)
