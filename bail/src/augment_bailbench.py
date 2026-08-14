"""Augment BailBench prompts through an injected typed text generator.

Run from the repository root with:
`uv run python -m bail.src.augment_bailbench`.
"""

from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from bail import config
from bail.prompts.rudeness_augmentation import (
    MOCK_OPENERS,
    RUDENESS_TYPE_NAMES,
    build_augmentation_messages,
    extract_augmented_prompt,
)
from llm_runtime import (
    GenerationError,
    MAX_RETRY_DELAY_SECONDS,
    ModelId,
    OpenRouterRoute,
    ProviderId,
    TextGenerator,
    resolve_route,
)
from llm_runtime.openrouter import (
    OpenRouterGenerator,
    create_openrouter_generator,
    openrouter_credentials_from_env,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("augment_bailbench")

DEFAULT_ID_COL = "bailbench_id"
LOG_EVERY = 25


def load_bailbench() -> tuple[pd.DataFrame, str]:
    path = config.BAILBENCH_SOURCE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"BailBench loading failed because the source does not exist at {path}. "
            "The failure occurred in bail.src.augment_bailbench.load_bailbench "
            "before augmentation, so no output can be produced. Set "
            "bail.config.BAILBENCH_SOURCE to an existing CSV or Parquet file and "
            "retry."
        )
    dataframe = (
        pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    )
    if config.BAILBENCH_PROMPT_COL not in dataframe.columns:
        raise KeyError(
            "BailBench loading failed because prompt column "
            f"{config.BAILBENCH_PROMPT_COL!r} is absent from {list(dataframe.columns)}. "
            "Validation failed in bail.src.augment_bailbench.load_bailbench before "
            "augmentation, so no rows can run. Fix BAILBENCH_PROMPT_COL and retry."
        )

    id_col = config.BAILBENCH_ID_COL or DEFAULT_ID_COL
    if not config.BAILBENCH_ID_COL:
        dataframe = dataframe.copy()
        dataframe[id_col] = range(len(dataframe))
    elif id_col not in dataframe.columns:
        raise KeyError(
            f"BailBench loading failed because id column {id_col!r} is absent from "
            f"{list(dataframe.columns)}. Validation failed in load_bailbench before "
            "resume processing, so row identity is unsafe. Fix BAILBENCH_ID_COL or "
            "set it to an empty string to use stable row indices."
        )
    if dataframe[id_col].duplicated().any():
        raise ValueError(
            f"BailBench loading failed because id column {id_col!r} has duplicates. "
            "Validation failed in load_bailbench before resume processing, so rows "
            "cannot be safely skipped. Deduplicate the source IDs and retry."
        )
    return dataframe, id_col


def assign_rudeness_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(config.AUGMENT_SEED)
    output = dataframe.copy()
    output["rudeness_type"] = rng.integers(
        1, config.N_RUDENESS_TYPES + 1, size=len(output)
    )
    output["rudeness_name"] = output["rudeness_type"].map(RUDENESS_TYPE_NAMES)
    output["original_prompt"] = output[config.BAILBENCH_PROMPT_COL].astype(str)
    return output


def _mock_raw_response(rudeness_type: int, source_prompt: str) -> str:
    return f"<augmented>{MOCK_OPENERS[rudeness_type]} {source_prompt}</augmented>"


def run_mock_batch(cells: pd.DataFrame) -> pd.DataFrame:
    output = cells.copy()
    output["raw_response"] = [
        _mock_raw_response(int(rudeness_type), str(prompt))
        for rudeness_type, prompt in zip(
            output["rudeness_type"], output["original_prompt"], strict=True
        )
    ]
    return output


def _call_one(
    generator: TextGenerator,
    rudeness_type: int,
    source_prompt: str,
) -> str:
    """Generate one tagged augmentation, preserving bounded retry behavior."""
    messages = build_augmentation_messages(rudeness_type, source_prompt)
    for attempt in range(config.API_MAX_RETRIES + 1):
        try:
            return generator.generate(messages, config.AUGMENT_GENERATION)
        except GenerationError as error:
            if not error.retryable or attempt == config.API_MAX_RETRIES:
                return _api_error(error)
            time.sleep(_bounded_retry_delay(error, attempt))
        except Exception as error:
            return _api_error(error)
    raise AssertionError("bounded Bail retry loop exited without a result")


def _api_error(error: Exception) -> str:
    return f"API_ERROR: {type(error).__name__}: {error}"


def _bounded_retry_delay(error: GenerationError, attempt: int) -> float:
    suggested = error.retry_after_seconds
    if suggested is None or not math.isfinite(suggested) or suggested < 0:
        suggested = float(2**attempt)
    return min(suggested, MAX_RETRY_DELAY_SECONDS)


def run_real_batch(
    cells: pd.DataFrame,
    generator: TextGenerator,
) -> pd.DataFrame:
    pairs = [
        (int(rudeness_type), str(prompt))
        for rudeness_type, prompt in zip(
            cells["rudeness_type"], cells["original_prompt"], strict=True
        )
    ]

    def generate(pair: tuple[int, str]) -> str:
        return _call_one(generator, pair[0], pair[1])

    responses: list[str] = []
    with ThreadPoolExecutor(max_workers=config.API_CONCURRENCY) as pool:
        for raw_response in pool.map(generate, pairs):
            responses.append(raw_response)
            if len(responses) % LOG_EVERY == 0 or len(responses) == len(pairs):
                log.info("progress: %d/%d calls", len(responses), len(pairs))
    output = cells.copy()
    output["raw_response"] = responses
    failures = sum(response.startswith("API_ERROR:") for response in responses)
    log.info(
        "real run summary: %d calls made, %d failed after retries",
        len(pairs),
        failures,
    )
    return output


def create_augmentation_generator() -> OpenRouterGenerator:
    route = resolve_route(
        ModelId.DOLPHIN_MISTRAL_24B_VENICE,
        ProviderId.OPENROUTER,
        None,
    )
    if not isinstance(route, OpenRouterRoute):
        raise AssertionError(f"unexpected closed route type: {type(route).__name__}")
    return create_openrouter_generator(
        route,
        credentials=openrouter_credentials_from_env(),
    )


def main(generator: TextGenerator | None = None) -> None:
    dataframe, id_col = load_bailbench()
    log.info(
        "BailBench source: %d rows from %s", len(dataframe), config.BAILBENCH_SOURCE
    )
    cells = assign_rudeness_types(dataframe)
    log.info(
        "rudeness type counts (seed=%d):\n%s",
        config.AUGMENT_SEED,
        cells["rudeness_type"].value_counts().sort_index().to_string(),
    )
    output_columns = list(dataframe.columns) + [
        "rudeness_type",
        "rudeness_name",
        "original_prompt",
        "augmented_prompt",
        "raw_response",
    ]

    existing: pd.DataFrame | None = None
    if config.AUGMENT_USE_MOCK:
        results = run_mock_batch(cells)
    else:
        if os.path.exists(config.AUGMENTED_PARQUET):
            existing = pd.read_parquet(config.AUGMENTED_PARQUET)
            done_ids = set(existing.loc[existing["augmented_prompt"].notna(), id_col])
            existing = existing[existing[id_col].isin(done_ids)]
            todo = ~cells[id_col].isin(done_ids)
            log.info(
                "resume: %d of %d rows already done in %s, %d to run",
                (~todo).sum(),
                len(cells),
                config.AUGMENTED_PARQUET,
                todo.sum(),
            )
            cells = cells[todo]
            if len(cells) == 0:
                log.info("nothing to do")
                return
        owned_generator: OpenRouterGenerator | None = None
        active_generator = generator
        if active_generator is None:
            owned_generator = create_augmentation_generator()
            active_generator = owned_generator
        try:
            results = run_real_batch(cells, active_generator)
        finally:
            if owned_generator is not None:
                owned_generator.close()

    results["augmented_prompt"] = results["raw_response"].map(extract_augmented_prompt)
    output = results[output_columns]
    if existing is not None:
        output = (
            pd.concat([existing[output_columns], output], ignore_index=True)
            .sort_values(id_col, kind="stable")
            .reset_index(drop=True)
        )
    output.to_parquet(config.AUGMENTED_PARQUET, index=False)
    missing = output["augmented_prompt"].isna().sum()
    log.info(
        "wrote %d rows to %s (%d with no parseable <augmented> output)",
        len(output),
        config.AUGMENTED_PARQUET,
        missing,
    )


if __name__ == "__main__":
    main()
