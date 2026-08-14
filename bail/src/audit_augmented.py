"""Audit the rudeness-augmented dataset before running the rude condition.

Automated flags (heuristics -- flag for human review, not auto-drop):
  missing        augmented_prompt empty/NaN
  residual_tags  leftover <augmented>/<source> tags or meta-commentary
  refusal        rewriter refusal markers ("I can't", "As an AI", ...)
  identical      augmented == original (no rudeness applied)
  too_short/long length outside 40%-250% of the original
  no_change_hint augmented contains the original verbatim with <15 chars added

Outputs:
  outputs/augment_flags.csv       every flagged row with flag reasons
  outputs/augment_spotcheck.csv   stratified manual-review sample
                                  (5 per rudeness type, seed 42), with an
                                  empty `verdict` column to fill in
                                  (ok / fix / regenerate)

Run: python -m src.audit_augmented
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

import config

OUTPUTS = os.path.join(config.ROOT, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

REFUSAL_MARKERS = [
    "i can't", "i cannot", "i won't", "i'm sorry", "i am sorry",
    "as an ai", "i'm not able", "i am not able", "cannot assist",
    "can't assist", "unable to comply",
]
TAG_MARKERS = ["<augmented>", "</augmented>", "<source>", "</source>",
               "assigned rudeness type", "rewritten prompt"]


def flags_for(row) -> list[str]:
    orig = str(row["original_prompt"])
    aug = row["augmented_prompt"]
    out = []
    if not isinstance(aug, str) or not aug.strip():
        return ["missing"]
    low = aug.lower()
    if any(m in low for m in TAG_MARKERS):
        out.append("residual_tags")
    if any(low.startswith(m) or f". {m}" in low[:120] for m in REFUSAL_MARKERS):
        out.append("refusal")
    if aug.strip() == orig.strip():
        out.append("identical")
    ratio = len(aug) / max(len(orig), 1)
    if ratio < 0.4:
        out.append("too_short")
    elif ratio > 2.5:
        out.append("too_long")
    if orig.strip() and orig.strip() in aug and len(aug) - len(orig) < 15:
        out.append("no_change_hint")
    return out


def main() -> None:
    df = pd.read_csv(os.path.join(config.DATA_DIR, "bailbench_augmented.csv"))
    df["flags"] = df.apply(lambda r: ",".join(flags_for(r)), axis=1)

    flagged = df[df["flags"] != ""][
        ["bailbench_id", "rudeness_type", "rudeness_name", "flags",
         "original_prompt", "augmented_prompt"]]
    flag_path = os.path.join(OUTPUTS, "augment_flags.csv")
    flagged.to_csv(flag_path, index=False)
    print(f"{len(flagged)}/{len(df)} rows flagged -> {flag_path}")
    if len(flagged):
        print(flagged["flags"].str.split(",").explode().value_counts().to_string())

    rng = np.random.default_rng(42)
    spot = (df[df["flags"] == ""]
            .groupby("rudeness_type", group_keys=False)
            .apply(lambda g: g.sample(min(5, len(g)), random_state=rng.integers(1 << 31))))
    spot = spot[["bailbench_id", "rudeness_type", "rudeness_name",
                 "original_prompt", "augmented_prompt"]].copy()
    spot["verdict"] = ""
    spot_path = os.path.join(OUTPUTS, "augment_spotcheck.csv")
    spot.to_csv(spot_path, index=False)
    print(f"{len(spot)} unflagged rows sampled for manual review -> {spot_path}")


if __name__ == "__main__":
    main()
