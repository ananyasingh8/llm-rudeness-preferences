#!/usr/bin/env python3
"""Build the verified bail-steering sample from the collapsed ConvAbuse.

Source: bail/data/convabuse_collapsed.parquet (one row per unique snippet,
built by scripts/build_convabuse_collapsed.py). Rows are "annotator
verified" using the same eligibility rule as quadratic_voting_v2:
severity_std <= 0.5 (high annotator agreement). Bands come from the
parquet's severity_band column (mean of votes -> nearest band, ties toward
the more severe band).

Sample design (the rare bands are taken whole — balancing down to the
smallest band would only discard precision elsewhere):

- bands 0 / -1 / -2 / -3: ALL eligible rows
- band 1: 500 rows, drawn with numpy's seeded generator from the pool
  sorted by snippet_id (same determinism recipe as quadratic_voting_v2, so
  regenerating this file always yields identical rows)

Run from the repo root:  python3 scripts/build_bail_sample_verified.py
Writes bail/data/convabuse_sample_verified.csv. Does not modify any
existing file (the legacy convabuse_sample.csv stays untouched).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
PARQUET = REPO_ROOT / "bail" / "data" / "convabuse_collapsed.parquet"
OUT_CSV = REPO_ROOT / "bail" / "data" / "convabuse_sample_verified.csv"

MAX_SEVERITY_STD = 0.5  # quadratic_voting_v2's "high annotator agreement"
BANDS = (1, 0, -1, -2, -3)
BAND_1_SAMPLE = 500
SEED = 42
COLUMNS = (
    "snippet_id",
    "severity_band",
    "group",
    "prev_agent",
    "prev_user",
    "agent",
    "user",
)


def band_group(band: int) -> str:
    return "friendly" if band == 1 else ("neutral" if band == 0 else "rude")


def main() -> None:
    df = pd.read_parquet(PARQUET)
    eligible = df[df["severity_std"].fillna(0) <= MAX_SEVERITY_STD]
    parts = []
    for band in BANDS:
        pool = eligible[eligible["severity_band"] == band]
        pool = pool.sort_values("snippet_id").reset_index(drop=True)
        if band == 1:
            rng = np.random.default_rng(SEED)
            pool = pool.iloc[np.sort(rng.permutation(len(pool))[:BAND_1_SAMPLE])]
        parts.append(pool)
    sample = pd.concat(parts, ignore_index=True)
    sample["group"] = sample["severity_band"].map(band_group)
    sample = sample[list(COLUMNS)]
    sample.to_csv(OUT_CSV, index=False)
    print(f"eligible rows (std <= {MAX_SEVERITY_STD}): {len(eligible)} of {len(df)}")
    print("sample per band:")
    print(
        sample["severity_band"].value_counts().sort_index(ascending=False).to_string()
    )
    print(f"wrote {OUT_CSV} ({len(sample)} rows)")


if __name__ == "__main__":
    main()
