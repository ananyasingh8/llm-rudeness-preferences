"""EXPLORATORY: Sonnet 5 bail prompt on ConvAbuse, via OpenRouter Batch API.

Subsample of the frozen Gemma ConvAbuse sample (strict subset, seed 42):
  friendly 200, neutral 200, rude 201 (67 each of severity -1/-2/-3)
x 2 orderings x 5 samples = 6,010 calls, batched at 50% off.

Usage (from bail/):
  python -m src.convabuse_sonnet submit
  python -m src.convabuse_sonnet collect     # poll + write results
Results -> data/results/convabuse_sonnet_prompt.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import pandas as pd

import config
from prompts.bail_methods import ORDERINGS, get_bail_prompt, get_bail_prompt_status
from src.convabuse_run import build_messages

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("convabuse_sonnet")

SEED = 42
N_PER_SEV = {1: 200, 0: 200, -1: 67, -2: 67, -3: 67}
N_SAMPLES = 5
SAMPLE_CSV = os.path.join(config.DATA_DIR, "convabuse_sonnet_sample.csv")
RESULTS = os.path.join(config.RESULTS_DIR, "convabuse_sonnet_prompt.parquet")
REGISTRY = os.path.join(config.BATCHES_DIR, "convabuse_sonnet_batches.json")


def build_sample() -> pd.DataFrame:
    if os.path.exists(SAMPLE_CSV):
        return pd.read_csv(SAMPLE_CSV)
    base = pd.read_csv(os.path.join(config.DATA_DIR, "convabuse_sample.csv"))
    parts = [base[base["abuse_severity"] == sev].sample(n, random_state=SEED)
             for sev, n in N_PER_SEV.items()]
    sample = pd.concat(parts, ignore_index=True)
    sample.to_csv(SAMPLE_CSV, index=False)
    log.info("froze Sonnet subsample: %d rows (subset of Gemma sample)", len(sample))
    return sample


OR_BATCH_MODEL = "anthropic/claude-sonnet-5:batch"


def cmd_submit() -> None:
    from src.gemma_batch import submit_one  # OpenRouter batch POST (uses OPENROUTER key)
    import src.gemma_batch as gb
    sample = build_sample()
    reqs = []
    for _, row in sample.iterrows():
        for ordering in ORDERINGS:
            for s in range(N_SAMPLES):
                cid = f"conv--{row['example_no']}--{ordering}--{s}"
                reqs.append({
                    "custom_id": cid,
                    "body": {"messages": build_messages(row, ordering),
                             "max_tokens": config.BAIL_MAX_TOKENS,
                             "reasoning": {"enabled": False}},
                })
    # submit chunked via OpenRouter batch endpoint with the Sonnet batch slug
    CHUNK = 1000
    chunks = [reqs[i:i + CHUNK] for i in range(0, len(reqs), CHUNK)]
    entries = []
    if os.path.exists(REGISTRY):
        with open(REGISTRY) as f:
            entries = json.load(f).get("batches", [])
    done_chunks = {e["chunk"] for e in entries}
    orig_model = config.BAIL_MODEL
    config.BAIL_MODEL = OR_BATCH_MODEL
    try:
        for i, chunk in enumerate(chunks):
            if i in done_chunks:
                log.info("chunk %d already submitted, skipping", i)
                continue
            batch = submit_one(chunk)
            log.info("submitted chunk %d/%d: %s (%d requests, %s)", i + 1,
                     len(chunks), batch.get("id"), len(chunk), batch.get("status"))
            entries.append({"id": batch["id"], "chunk": i,
                            "n_requests": len(chunk),
                            "status": batch.get("status", "validating")})
            with open(REGISTRY, "w") as f:
                json.dump({"batches": entries}, f, indent=2)
    finally:
        config.BAIL_MODEL = orig_model


def cmd_collect() -> None:
    from src.gemma_batch import fetch_batch
    with open(REGISTRY) as f:
        reg = json.load(f)
    sample = build_sample().set_index("example_no")
    rows, pending = [], 0
    all_results = []
    for b in reg["batches"]:
        remote = fetch_batch(b["id"])
        log.info("%s chunk %d: %s -- %s", b["id"], b["chunk"],
                 remote.get("status"), remote.get("request_counts"))
        if remote.get("status") == "failed":
            continue  # moderation-failed originals; replaced by retry batches
        if remote.get("status") != "completed":
            pending += 1
            continue
        all_results.extend(remote.get("results") or [])
    if pending:
        log.info("%d chunk(s) still processing; run collect again later", pending)
        return
    for result in all_results:
        _, eno, ordering, s = result["custom_id"].split("--")
        row = {"custom_id": result["custom_id"].replace("--", "|"),
               "example_no": int(eno) if eno.isdigit() else eno,
               "ordering": ordering, "sample": int(s)}
        meta = sample.loc[row["example_no"]] if row["example_no"] in sample.index else None
        row["abuse_severity"] = meta["abuse_severity"] if meta is not None else None
        row["group"] = meta["group"] if meta is not None else None
        err = result.get("error")
        body = (result.get("response") or {}).get("body") or {}
        msg = ((body.get("choices") or [{}])[0].get("message")) or {}
        text = msg.get("content") or ""
        row |= {"error": json.dumps(err) if err else None,
                "stop_reason": (body.get("choices") or [{}])[0].get("finish_reason"),
                "response_text": text,
                "wellbeing": get_bail_prompt_status(text) if text else ("?" if not err else None)}
        rows.append(row)
    pd.DataFrame(rows).to_parquet(RESULTS, index=False)
    n_err = sum(1 for r in rows if r["error"])
    log.info("wrote %d rows (%d errored) -> %s", len(rows), n_err, RESULTS)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("submit")
    sub.add_parser("collect")
    args = p.parse_args()
    (cmd_submit if args.cmd == "submit" else cmd_collect)()


if __name__ == "__main__":
    main()
