#!/usr/bin/env python3
"""Build a collapsed (one row per unique conversation snippet) ConvAbuse dataset.

ConvAbuse ships one row per human annotation, so identical conversation
snippets repeat (2-7 annotators each). Sampling per-annotation rows (as
bail/data/convabuse_filtered.parquet was) causes label noise and duplicate
snippets across severity bins. This script collapses the FULL csv to one row
per unique snippet, mirroring the conventions of
emotion_probing/datasets.py::load_convabuse exactly:

- grouping key: (conv_id, prev_agent, prev_user, agent, user)
- rows with an empty/whitespace `user` utterance are dropped
- groups with no parseable severity annotation are dropped
- severity_band: nearest of (1, 0, -1, -2, -3); exact ties round toward the
  more severe (smaller) band
- abusive_majority: strict majority of negative-severity votes
- one 0..1 vote-fraction column per type/target/direction flag

It additionally emits:
- snippet_id: deterministic sha256[:16] of conv_id plus the four turn texts
  joined with a unit separator, so any experiment can join on it. (conv_id is
  part of the payload because 133 snippet rows share identical turn texts
  under different conv_ids — the emotion_probing grouping keeps them
  distinct, so a texts-only hash would collide.)
- ratings / severity_min / severity_max / severity_std / annotators_disagree
- target_system_majority (fraction > 0.5) so bail-style target filtering is a
  column filter
- bail_example_nos: the example_no values this snippet had in the legacy
  bail/data/convabuse_filtered.parquet (empty list if filtered out there)

Run from the repo root:  python3 scripts/build_convabuse_collapsed.py
Writes bail/data/convabuse_collapsed.parquet.
Does not modify any existing file.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_CSV = REPO_ROOT / "emotion_probing" / "data" / "ConvAbuseEMNLPfull.csv"
BAIL_FILTERED = REPO_ROOT / "bail" / "data" / "convabuse_filtered.parquet"
OUT_PARQUET = REPO_ROOT / "bail" / "data" / "convabuse_collapsed.parquet"

SEVERITY_BANDS = (1, 0, -1, -2, -3)
# Same flag list as emotion_probing.datasets.CONVABUSE_FLAGS.
CONVABUSE_FLAGS = (
    "type.ableism",
    "type.homophobic",
    "type.intellectual",
    "type.racist",
    "type.sexist",
    "type.sex_harassment",
    "type.transphobic",
    "target.generalised",
    "target.individual",
    "target.system",
    "direction.explicit",
    "direction.implicit",
)
TURN_COLUMNS = ("prev_agent", "prev_user", "agent", "user")
GROUP_KEY = ("conv_id",) + TURN_COLUMNS


def snippet_id(
    conv_id: str, prev_agent: str, prev_user: str, agent: str, user: str
) -> str:
    """Deterministic stable id: sha256 hex prefix of conv_id + turn texts."""
    payload = "\x1f".join((conv_id, prev_agent, prev_user, agent, user))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def annotation_severity(row: pd.Series) -> float:
    # One-hot columns is_abuse.1 ... is_abuse.-3; first (most mild-first order,
    # matching datasets.py) band marked "1" wins. NaN if none marked.
    for band in SEVERITY_BANDS:
        if str(row[f"is_abuse.{band}"]) == "1":
            return float(band)
    return math.nan


def severity_band(mean: float) -> int:
    # Nearest band; on an exact tie the more severe (smaller) band wins.
    return min(SEVERITY_BANDS, key=lambda band: (abs(mean - band), band))


def main() -> None:
    # Read everything as strings so grouping matches the csv-module semantics
    # of emotion_probing/datasets.py exactly (no float coercion of conv_id).
    df = pd.read_csv(SOURCE_CSV, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    total_rows_in = len(df)

    # Mirror datasets.py: drop annotations whose labeled utterance is blank.
    df = df[df["user"].str.strip() != ""].copy()
    df["severity"] = df.apply(annotation_severity, axis=1)

    # Legacy bail sample: map snippet -> the example_no values it had there.
    bail = pd.read_parquet(BAIL_FILTERED)
    bail_example_set = set(int(x) for x in bail["example_no"])

    records = []
    for key, group in df.groupby(list(GROUP_KEY), sort=False):
        conv_id, prev_agent, prev_user, agent, user = key
        severities = group["severity"].dropna().tolist()
        if not severities:
            continue
        mean = sum(severities) / len(severities)
        n = len(severities)
        abusive_votes = sum(1 for s in severities if s < 0)
        rec: dict[str, object] = {
            "snippet_id": snippet_id(conv_id, prev_agent, prev_user, agent, user),
            "conv_id": conv_id,
            "prev_agent": prev_agent,
            "prev_user": prev_user,
            "agent": agent,
            "user": user,
            "bot": group["bot"].iloc[0],
            "example_id": group["example_no"].astype(int).min(),
            "n_annotations": n,
            "ratings": [int(s) for s in severities],
            "severity_mean": round(mean, 4),
            "severity_min": int(min(severities)),
            "severity_max": int(max(severities)),
            "severity_band": severity_band(mean),
            "severity_std": round(
                math.sqrt(sum((s - mean) ** 2 for s in severities) / n), 4
            ),
            "annotators_disagree": len(set(severities)) > 1,
            "abusive_majority": abusive_votes > n / 2,
        }
        for flag in CONVABUSE_FLAGS:
            votes = group[flag].astype(int)
            rec[flag.replace(".", "_") + "_frac"] = round(votes.mean(), 4)
        rec["target_system_majority"] = rec["target_system_frac"] > 0.5
        rec["bail_example_nos"] = sorted(
            e for e in group["example_no"].astype(int) if e in bail_example_set
        )
        records.append(rec)

    out = pd.DataFrame.from_records(records).sort_values("example_id").reset_index(drop=True)

    # snippet_id must be a valid join key — fail loudly on any collision.
    dupes = int(out["snippet_id"].duplicated().sum())
    if dupes:
        raise SystemExit(f"snippet_id collision across {dupes} rows")

    out.to_parquet(OUT_PARQUET, index=False)

    matched = int((out["bail_example_nos"].str.len() > 0).sum())
    n_bail_rows_mapped = int(out["bail_example_nos"].str.len().sum())
    print(f"source csv rows in:            {total_rows_in}")
    print(f"annotation rows kept:          {len(df)} (blank-user rows dropped)")
    print(f"unique snippets out:           {len(out)}")
    print(f"target_system_majority == True: {int(out['target_system_majority'].sum())}")
    print(f"target_system_frac > 0 (any vote): {int((out['target_system_frac'] > 0).sum())}")
    print(f"abusive_majority == True:      {int(out['abusive_majority'].sum())}")
    print(f"annotators_disagree == True:   {int(out['annotators_disagree'].sum())}")
    print("severity_band value counts:")
    print(out["severity_band"].value_counts().sort_index(ascending=False).to_string())
    print(
        f"snippets matched to legacy bail parquet: {matched} "
        f"({n_bail_rows_mapped} of {len(bail)} bail rows mapped)"
    )
    print(f"wrote {OUT_PARQUET} ({len(out)} rows, {len(out.columns)} columns)")


if __name__ == "__main__":
    main()
