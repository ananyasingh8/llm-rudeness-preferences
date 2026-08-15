"""Apply human cleanup verdicts to the augmented dataset.

Reads the reviewed CSVs (outputs/augment_flags_clean.csv falls back to
outputs/augment_flags.csv, plus outputs/augment_spotcheck.csv), then:

  fix         -> the reviewer's edited augmented_prompt is written into
                 data/bailbench_augmented.parquet (raw_response kept, and
                 the row is marked human_edited=True)
  regenerate  -> the row is re-rolled through the SAME dolphin-mistral
                 augmentation pipeline (same system prompt, same seeded
                 rudeness formula) with REJECTION SAMPLING: up to
                 MAX_ATTEMPTS draws, first one passing the automated audit
                 checks (src.audit_augmented.flags_for) is kept. If none
                 pass, the least-flagged draw is kept and reported.
  ok / empty  -> untouched

Afterwards re-exports data/bailbench_augmented.csv.
Run: python -m src.regen_augmented
"""
from __future__ import annotations

import io
import logging
import os

import pandas as pd

import config
from prompts.rudeness_augmentation import (
    build_augmentation_messages,
    extract_augmented_prompt,
)
from src.audit_augmented import flags_for

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("regen_augmented")

MAX_ATTEMPTS = 4
OUTPUTS = os.path.join(config.ROOT, "outputs")


def decode_mixed(b: bytes) -> str:
    """UTF-8 with per-byte mac_roman fallback (spreadsheet apps can save
    edits in a different encoding than the original file)."""
    out, i = [], 0
    while i < len(b):
        try:
            out.append(b[i:].decode("utf-8"))
            break
        except UnicodeDecodeError as e:
            out.append(b[i:i + e.start].decode("utf-8"))
            out.append(b[i + e.start:i + e.start + 1].decode("mac_roman"))
            i += e.start + 1
    return "".join(out)


def read_review(path: str) -> pd.DataFrame:
    with open(path, "rb") as fh:
        df = pd.read_csv(io.StringIO(decode_mixed(fh.read())))
    df["verdict"] = df["verdict"].astype(str).str.strip().str.lower()
    return df


def regen_one(client, row) -> tuple[str | None, str, int]:
    """Rejection-sample one row. Returns (augmented, raw_response, n_flags)."""
    best = (None, "", 99)
    for attempt in range(MAX_ATTEMPTS):
        messages = build_augmentation_messages(int(row["rudeness_type"]),
                                               str(row["original_prompt"]))
        resp = client.chat.completions.create(
            model=config.AUGMENT_MODEL, messages=messages,
            temperature=config.AUGMENT_TEMPERATURE,
            max_tokens=config.AUGMENT_MAX_TOKENS)
        raw = resp.choices[0].message.content or ""
        aug = extract_augmented_prompt(raw)
        probe = {"original_prompt": row["original_prompt"], "augmented_prompt": aug}
        n = len(flags_for(probe)) if aug else 99
        if n == 0:
            return aug, raw, 0
        if n < best[2]:
            best = (aug, raw, n)
    return best


def main() -> None:
    flags_path = os.path.join(OUTPUTS, "augment_flags_clean.csv")
    if not os.path.exists(flags_path):
        flags_path = os.path.join(OUTPUTS, "augment_flags.csv")
    reviews = pd.concat([
        read_review(flags_path),
        read_review(os.path.join(OUTPUTS, "augment_spotcheck.csv")),
    ], ignore_index=True).drop_duplicates(subset="bailbench_id", keep="first")

    df = pd.read_parquet(config.AUGMENTED_PARQUET)
    if "human_edited" not in df.columns:
        df["human_edited"] = False
    if "regenerated" not in df.columns:
        df["regenerated"] = False
    df = df.set_index("bailbench_id")

    fixes = reviews[reviews["verdict"] == "fix"]
    for _, r in fixes.iterrows():
        df.loc[r["bailbench_id"], "augmented_prompt"] = str(r["augmented_prompt"]).strip()
        df.loc[r["bailbench_id"], "human_edited"] = True
    log.info("applied %d human fixes", len(fixes))

    regens = reviews[reviews["verdict"] == "regenerate"]
    if len(regens):
        from openai import OpenAI
        client = OpenAI(api_key=config.get_augment_api_key(),
                        base_url=config.AUGMENT_BASE_URL, max_retries=2)
        n_clean = 0
        for i, (_, r) in enumerate(regens.iterrows(), 1):
            src_row = df.loc[r["bailbench_id"]]
            aug, raw, n_flags = regen_one(client, src_row)
            df.loc[r["bailbench_id"], "augmented_prompt"] = aug
            df.loc[r["bailbench_id"], "raw_response"] = raw
            df.loc[r["bailbench_id"], "regenerated"] = True
            n_clean += (n_flags == 0)
            log.info("regen %d/%d id=%s: %s", i, len(regens), r["bailbench_id"],
                     "clean" if n_flags == 0 else f"kept best draw ({n_flags} flags)")
        log.info("regenerated %d rows (%d passed all checks)", len(regens), n_clean)

    df = df.reset_index()
    df.to_parquet(config.AUGMENTED_PARQUET, index=False)
    df.to_csv(os.path.join(config.DATA_DIR, "bailbench_augmented.csv"), index=False)
    log.info("wrote %d rows to parquet + csv", len(df))


if __name__ == "__main__":
    main()
