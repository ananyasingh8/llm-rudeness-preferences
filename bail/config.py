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

from llm_runtime import GenerationSettings

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

# BailBench source (from safety-research's bail-study repo, "The LLM Has Left
# The Chat"): 1630 rows of content / subcategory / category; no id column ->
# row index becomes "bailbench_id".
BAILBENCH_SOURCE = os.path.join(DATA_DIR, "bailBench.csv")
BAILBENCH_ID_COL = ""  # "" = use the row index (as "bailbench_id")
BAILBENCH_PROMPT_COL = "content"  # column holding the prompt text to augment

AUGMENT_SEED = 42
N_RUDENESS_TYPES = 12
AUGMENTED_PARQUET = os.path.join(DATA_DIR, "bailbench_augmented.parquet")

AUGMENT_USE_MOCK = False  # True = deterministic canned openers, no API key needed
AUGMENT_GENERATION = GenerationSettings(max_new_tokens=2000, temperature=1.0)

API_MAX_RETRIES = 4  # per-call retries on transient errors (with backoff)
API_CONCURRENCY = 8  # thread-pool size for concurrent API calls
os.makedirs(DATA_DIR, exist_ok=True)
