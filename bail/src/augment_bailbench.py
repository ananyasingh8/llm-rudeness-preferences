"""Augment BailBench prompts with rudeness via an OpenRouter model.

Each source row is assigned ONE of the 12 Culpeper impoliteness formulae by a
seeded RNG (config.AUGMENT_SEED) -- the model never picks the number. The model
(OpenRouter, OpenAI-compatible API) then rewrites the row's prompt applying only that
formula; wordings, constraints, and the <augmented> output parser live in
prompts/rudeness_augmentation.py.

Two paths, toggled by config.AUGMENT_USE_MOCK:
- Mock: deterministic synthetic response (canned per-type opener + the source
  prompt) so the pipeline runs end to end with no key.
- Real: one OpenRouter chat call per row, thread pool (config.API_CONCURRENCY)
  with retries; a call that still fails records raw_response "API_ERROR: ..."
  and augmented_prompt None. If config.AUGMENTED_PARQUET exists, rows already
  present are skipped and new results appended (resume).

Output: all source columns plus [id, rudeness_type, rudeness_name,
original_prompt, augmented_prompt, raw_response] -> config.AUGMENTED_PARQUET.

Run: python -m src.augment_bailbench
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import config
from prompts.rudeness_augmentation import (
    MOCK_OPENERS,
    RUDENESS_TYPE_NAMES,
    build_augmentation_messages,
    extract_augmented_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("augment_bailbench")

DEFAULT_ID_COL = "bailbench_id"  # used when config.BAILBENCH_ID_COL is ""

LOG_EVERY = 25  # progress log interval (real path)


def load_bailbench() -> tuple[pd.DataFrame, str]:
    """Load the BailBench source file (csv or parquet by extension) and return
    (df, id_col). If config.BAILBENCH_ID_COL is "", a bailbench_id column is
    created from the row index (stable as long as the source file's row order
    is stable -- resume relies on this)."""
    path = config.BAILBENCH_SOURCE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BailBench source not found at {path}. Set config.BAILBENCH_SOURCE "
            "to the dataset file (csv or parquet)."
        )
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if config.BAILBENCH_PROMPT_COL not in df.columns:
        raise KeyError(
            f"Prompt column {config.BAILBENCH_PROMPT_COL!r} not in BailBench "
            f"columns {list(df.columns)}. Fix config.BAILBENCH_PROMPT_COL."
        )

    id_col = config.BAILBENCH_ID_COL or DEFAULT_ID_COL
    if not config.BAILBENCH_ID_COL:
        df = df.copy()
        df[id_col] = range(len(df))
    elif id_col not in df.columns:
        raise KeyError(
            f"Id column {id_col!r} not in BailBench columns {list(df.columns)}. "
            "Fix config.BAILBENCH_ID_COL (or set it to \"\" to use the row index)."
        )
    if df[id_col].duplicated().any():
        raise ValueError(f"Id column {id_col!r} has duplicate values.")
    return df, id_col


def assign_rudeness_types(df: pd.DataFrame) -> pd.DataFrame:
    """Seeded draw of one rudeness type (1..N_RUDENESS_TYPES) per row.
    Deterministic given AUGMENT_SEED and the source row order; drawn for ALL
    rows before any resume-filtering so previously-done rows don't shift the
    stream for the rest."""
    rng = np.random.default_rng(config.AUGMENT_SEED)
    out = df.copy()
    out["rudeness_type"] = rng.integers(1, config.N_RUDENESS_TYPES + 1, size=len(out))
    out["rudeness_name"] = out["rudeness_type"].map(RUDENESS_TYPE_NAMES)
    out["original_prompt"] = out[config.BAILBENCH_PROMPT_COL].astype(str)
    return out


# --------------------------------------------------------------------------
# Mock path
# --------------------------------------------------------------------------

def _mock_raw_response(rudeness_type: int, source_prompt: str) -> str:
    """Deterministic stand-in for a real response: canned opener for the
    assigned type prepended to the untouched source prompt, in tags."""
    return f"<augmented>{MOCK_OPENERS[rudeness_type]} {source_prompt}</augmented>"


def run_mock_batch(cells: pd.DataFrame) -> pd.DataFrame:
    out = cells.copy()
    out["raw_response"] = [
        _mock_raw_response(t, p)
        for t, p in zip(out["rudeness_type"], out["original_prompt"])
    ]
    return out


# --------------------------------------------------------------------------
# Real path (OpenRouter via the OpenAI-compatible endpoint)
# --------------------------------------------------------------------------

def _call_one(client, rudeness_type: int, source_prompt: str) -> str:
    """One augmentation call with retry/backoff; returns the raw response text
    or an 'API_ERROR: ...' string after config.API_MAX_RETRIES failed retries."""
    messages = build_augmentation_messages(rudeness_type, source_prompt)
    last_err = None
    for attempt in range(config.API_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=config.AUGMENT_MODEL,
                messages=messages,
                temperature=config.AUGMENT_TEMPERATURE,
                max_tokens=config.AUGMENT_MAX_TOKENS,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < config.API_MAX_RETRIES:
                time.sleep(min(2 ** attempt, 30))
    return f"API_ERROR: {type(last_err).__name__}: {last_err}"


def run_real_batch(cells: pd.DataFrame) -> pd.DataFrame:
    from openai import OpenAI

    client = OpenAI(
        api_key=config.get_augment_api_key(),
        base_url=config.AUGMENT_BASE_URL,
        max_retries=0,  # we do our own retries
    )
    pairs = list(zip(cells["rudeness_type"], cells["original_prompt"]))
    responses = []
    with ThreadPoolExecutor(max_workers=config.API_CONCURRENCY) as pool:
        for raw in pool.map(lambda tp: _call_one(client, *tp), pairs):
            responses.append(raw)
            if len(responses) % LOG_EVERY == 0 or len(responses) == len(pairs):
                log.info("progress: %d/%d calls", len(responses), len(pairs))
    out = cells.copy()
    out["raw_response"] = responses
    n_fail = sum(r.startswith("API_ERROR:") for r in responses)
    log.info("real run summary: %d calls made, %d failed after retries", len(pairs), n_fail)
    return out


# --------------------------------------------------------------------------

def main() -> None:
    df, id_col = load_bailbench()
    log.info("BailBench source: %d rows from %s", len(df), config.BAILBENCH_SOURCE)

    cells = assign_rudeness_types(df)
    log.info("rudeness type counts (seed=%d):\n%s", config.AUGMENT_SEED,
             cells["rudeness_type"].value_counts().sort_index().to_string())

    out_cols = list(df.columns) + [
        "rudeness_type", "rudeness_name", "original_prompt",
        "augmented_prompt", "raw_response",
    ]

    existing = None
    if config.AUGMENT_USE_MOCK:
        results = run_mock_batch(cells)
    else:
        config.get_augment_api_key()  # fail fast, before touching the API
        if os.path.exists(config.AUGMENTED_PARQUET):
            existing = pd.read_parquet(config.AUGMENTED_PARQUET)
            # "Done" = present AND successfully augmented. Rows that failed last
            # time (augmented_prompt is null, e.g. an API_ERROR) are retried.
            done_ids = set(existing.loc[existing["augmented_prompt"].notna(), id_col])
            existing = existing[existing[id_col].isin(done_ids)]  # drop failed rows; they get re-run
            todo = ~cells[id_col].isin(done_ids)
            log.info("resume: %d of %d rows already done in %s, %d to run",
                     (~todo).sum(), len(cells), config.AUGMENTED_PARQUET, todo.sum())
            cells = cells[todo]
            if len(cells) == 0:
                log.info("nothing to do")
                return
        results = run_real_batch(cells)

    results["augmented_prompt"] = results["raw_response"].map(extract_augmented_prompt)

    out = results[out_cols]
    if existing is not None:
        out = pd.concat([existing[out_cols], out], ignore_index=True).sort_values(
            id_col, kind="stable").reset_index(drop=True)
    out.to_parquet(config.AUGMENTED_PARQUET, index=False)

    n_missing = out["augmented_prompt"].isna().sum()
    log.info("wrote %d rows to %s (%d with no parseable <augmented> output)",
             len(out), config.AUGMENTED_PARQUET, n_missing)


if __name__ == "__main__":
    main()
