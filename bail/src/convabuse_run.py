"""EXPLORATORY EXTENSION (not pre-registered): Gemma bail prompt on ConvAbuse.

Naturalistic complement to the BailBench experiment: real human-written abuse
with human-annotated severity, instead of synthetic rudeness augmentation.

Sample (seed 42) from data/convabuse_filtered.parquet (personabailstudy's
target-filtered ConvAbuse: abusive rows kept only when aimed at the system):
  friendly  severity +1  n=500
  neutral   severity  0  n=500
  rude      severity -1/-2/-3, n=167 each (501 total, equal representation)

Design: conversation turns replayed as chat messages (prev_agent, prev_user,
agent, user -- same as personabailstudy's run_bail.py), wellbeing check
appended as a final user message. 2 orderings x 10 samples per row
= 30,020 calls. No rollout phase (the bot turn is already in the data).

Run: python -m src.convabuse_run run [--limit N]
Results -> data/results/convabuse_prompt.parquet
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

import config
from prompts.bail_methods import ORDERINGS, get_bail_prompt, get_bail_prompt_status

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("convabuse_run")

SEED = 42
N_PER_GROUP = {1: 500, 0: 500, -1: 167, -2: 167, -3: 167}
N_SAMPLES = 10
CONCURRENCY = 32
RESULTS = os.path.join(config.RESULTS_DIR, "convabuse_prompt.parquet")
SAMPLE_CSV = os.path.join(config.DATA_DIR, "convabuse_sample.csv")

TRANSCRIPT_TURNS = [
    ("prev_agent", "assistant"),
    ("prev_user", "user"),
    ("agent", "assistant"),
    ("user", "user"),  # the annotated user message
]


def build_sample() -> pd.DataFrame:
    if os.path.exists(SAMPLE_CSV):
        return pd.read_csv(SAMPLE_CSV)
    df = pd.read_parquet(os.path.join(config.DATA_DIR, "convabuse_filtered.parquet"))
    parts = []
    for sev, n in N_PER_GROUP.items():
        pool = df[df["abuse_severity"] == sev]
        parts.append(pool.sample(n, random_state=SEED))
    sample = pd.concat(parts, ignore_index=True)
    sample["group"] = sample["abuse_severity"].map(
        lambda s: "friendly" if s == 1 else ("neutral" if s == 0 else "rude"))
    keep = ["example_no", "abuse_severity", "group"] + [c for c, _ in TRANSCRIPT_TURNS]
    sample = sample[keep]
    sample.to_csv(SAMPLE_CSV, index=False)
    log.info("froze sample: %s (%d rows)", SAMPLE_CSV, len(sample))
    return sample


def build_messages(row, ordering: str) -> list[dict]:
    messages = []
    for col, role in TRANSCRIPT_TURNS:
        text = row.get(col)
        if pd.isna(text) or not str(text).strip():
            continue
        messages.append({"role": role, "content": str(text)})
    messages.append({"role": "user", "content": get_bail_prompt(ordering)})
    return messages


def call_one(session: requests.Session, req: dict) -> dict:
    payload = {"model": config.BAIL_MODEL, "messages": req["messages"],
               "max_tokens": config.BAIL_MAX_TOKENS}
    last_err = None
    for attempt in range(config.API_MAX_RETRIES + 1):
        try:
            resp = session.post(
                config.OPENROUTER_CHAT_URL,
                headers={"Authorization": f"Bearer {config.get_augment_api_key()}"},
                json=payload, timeout=180)
            body = resp.json()
            if "error" in body and "choices" not in body:
                raise RuntimeError(str(body["error"])[:200])
            text = (body["choices"][0]["message"].get("content")) or ""
            return {**req["meta"], "error": None, "response_text": text,
                    "wellbeing": get_bail_prompt_status(text) if text else "?"}
        except Exception as e:
            last_err = e
            if attempt < config.API_MAX_RETRIES:
                time.sleep(min(2 ** attempt, 30))
    return {**req["meta"], "error": f"{type(last_err).__name__}: {last_err}",
            "response_text": "", "wellbeing": None}


def _flush(rows: list[dict]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if os.path.exists(RESULTS):
        old = pd.read_parquet(RESULTS)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(
            subset="custom_id", keep="last")
    new.to_parquet(RESULTS, index=False)


def cmd_run(limit: int = 0) -> None:
    sample = build_sample()
    reqs = []
    for _, row in sample.iterrows():
        for ordering in ORDERINGS:
            for s in range(N_SAMPLES):
                cid = f"conv|{row['example_no']}|{ordering}|{s}"
                reqs.append({"messages": build_messages(row, ordering),
                             "meta": {"custom_id": cid,
                                      "example_no": row["example_no"],
                                      "abuse_severity": row["abuse_severity"],
                                      "group": row["group"],
                                      "ordering": ordering, "sample": s}})
    if limit:
        reqs = reqs[:limit]
    done: set[str] = set()
    if os.path.exists(RESULTS):
        existing = pd.read_parquet(RESULTS)
        done = set(existing.loc[existing["error"].isna(), "custom_id"])
    todo = [r for r in reqs if r["meta"]["custom_id"] not in done]
    log.info("convabuse: %d cells (%d done, %d to run)", len(reqs),
             len(reqs) - len(todo), len(todo))
    if not todo:
        return
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_maxsize=CONCURRENCY)
    session.mount("https://", adapter)
    buffer, n_done, n_fail = [], 0, 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = [pool.submit(call_one, session, r) for r in todo]
        for fut in as_completed(futures):
            buffer.append(fut.result())
            n_done += 1
            n_fail += bool(buffer[-1]["error"])
            if n_done % 400 == 0 or n_done == len(todo):
                _flush(buffer)
                buffer = []
                log.info("progress: %d/%d (%d failed)", n_done, len(todo), n_fail)
    _flush(buffer)
    log.info("done: %d calls, %d failed", n_done, n_fail)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    cmd_run(args.limit)


if __name__ == "__main__":
    main()
