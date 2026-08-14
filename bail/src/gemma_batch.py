"""Gemma bail experiment via the OpenRouter Batch API.

Phases (per SPEC.md + amendment A1):
  rollout  -- model answers the BailBench prompt normally (shared input for
              the two bail-prompt orderings)
  tool     -- bail tool available on the user prompt; bail = tool called
  prompt   -- well-being check appended after the rollout answer; needs the
              rollout phase collected first. Two orderings per rollout.

Conditions: "orig" (bailBench.csv content) and "rude" (augmented_prompt from
bailbench_augmented.csv, once cleaned).

custom_id scheme: "<phase>|<condition>|<bailbench_id>|<sample>" and, for the
prompt phase, "prompt|<condition>|<ordering>|<bailbench_id>|<sample>".

Usage (from bail/):
  python -m src.gemma_batch smoke                     # 12-request sanity batch
  python -m src.gemma_batch submit --phase rollout --condition orig
  python -m src.gemma_batch submit --phase tool --condition orig
  python -m src.gemma_batch status                    # poll all registered batches
  python -m src.gemma_batch collect                   # pull results of completed batches
  python -m src.gemma_batch submit --phase prompt --condition orig   # after rollout collected

State: submitted batches are recorded in data/batches/registry.json; collected
results land in data/results/results_<phase>_<condition>.parquet. Re-running
submit skips (phase, condition, sample-chunk)s already registered.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import pandas as pd
import requests

import config
from prompts.bail_methods import (
    ORDERINGS,
    called_bail_tool,
    get_bail_prompt,
    get_bail_tool,
    get_bail_prompt_status,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("gemma_batch")

PHASES = ["rollout", "tool", "prompt"]
CONDITIONS = ["orig", "rude"]


def api_key() -> str:
    return config.get_augment_api_key()  # same OpenRouter key


def load_prompts(condition: str) -> pd.DataFrame:
    """Rows with bailbench_id + the prompt text for the given condition."""
    if condition == "orig":
        df = pd.read_csv(os.path.join(config.DATA_DIR, "bailBench.csv"))
        df["bailbench_id"] = range(len(df))
        df["prompt"] = df["content"].astype(str)
    else:
        df = pd.read_csv(os.path.join(config.DATA_DIR, "bailbench_augmented.csv"))
        df["prompt"] = df["augmented_prompt"].astype(str)
    return df[["bailbench_id", "prompt"]]


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------

def base_body(messages: list[dict]) -> dict:
    return {"messages": messages, "max_tokens": config.BAIL_MAX_TOKENS}


def build_requests(phase: str, condition: str) -> list[dict]:
    prompts = load_prompts(condition)
    reqs = []
    if phase in ("rollout", "tool"):
        for _, row in prompts.iterrows():
            for s in range(config.N_SAMPLES):
                body = base_body([{"role": "user", "content": row["prompt"]}])
                if phase == "tool":
                    body["tools"] = [get_bail_tool(config.BAIL_MODEL_NAME)]
                reqs.append({
                    "custom_id": f"{phase}|{condition}|{row['bailbench_id']}|{s}",
                    "body": body,
                })
    elif phase == "prompt":
        rollouts = load_results("rollout", condition).set_index(["bailbench_id", "sample"])
        prompts = prompts.set_index("bailbench_id")
        for (bid, s), r in rollouts.iterrows():
            if not r["response_text"]:
                continue  # empty/errored rollout; logged in collect
            for ordering in ORDERINGS:
                messages = [
                    {"role": "user", "content": prompts.loc[bid, "prompt"]},
                    {"role": "assistant", "content": r["response_text"]},
                    {"role": "user", "content": get_bail_prompt(ordering)},
                ]
                reqs.append({
                    "custom_id": f"prompt|{condition}|{ordering}|{bid}|{s}",
                    "body": base_body(messages),
                })
    else:
        raise ValueError(phase)
    return reqs


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    if os.path.exists(config.BATCH_REGISTRY):
        with open(config.BATCH_REGISTRY) as f:
            return json.load(f)
    return {"batches": []}


def save_registry(reg: dict) -> None:
    with open(config.BATCH_REGISTRY, "w") as f:
        json.dump(reg, f, indent=2)


def registered_chunks(reg: dict, phase: str, condition: str) -> set[int]:
    return {b["chunk"] for b in reg["batches"]
            if b["phase"] == phase and b["condition"] == condition
            and b["status"] not in ("failed", "expired", "cancelled")}


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def submit_one(reqs: list[dict]) -> dict:
    resp = requests.post(
        config.OPENROUTER_BATCH_URL,
        headers={"Authorization": f"Bearer {api_key()}",
                 "Content-Type": "application/json"},
        json={"endpoint": config.BATCH_ENDPOINT,
              "model": config.BAIL_MODEL,
              "requests": reqs},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_batch(batch_id: str) -> dict:
    resp = requests.get(
        f"{config.OPENROUTER_BATCH_URL}/{batch_id}",
        headers={"Authorization": f"Bearer {api_key()}"},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_submit(phase: str, condition: str, smoke_n: int = 0) -> None:
    reqs = build_requests(phase, condition)
    if smoke_n:
        reqs = reqs[:smoke_n]
    log.info("built %d requests for phase=%s condition=%s", len(reqs), phase, condition)

    reg = load_registry()
    done = registered_chunks(reg, phase, condition)
    chunks = [reqs[i:i + config.BATCH_CHUNK_SIZE]
              for i in range(0, len(reqs), config.BATCH_CHUNK_SIZE)]
    for i, chunk in enumerate(chunks):
        if i in done:
            log.info("chunk %d already submitted, skipping", i)
            continue
        batch = submit_one(chunk)
        log.info("submitted chunk %d/%d: batch id %s status %s (%d requests)",
                 i + 1, len(chunks), batch.get("id"), batch.get("status"), len(chunk))
        reg["batches"].append({
            "id": batch["id"], "phase": phase, "condition": condition,
            "chunk": i, "n_requests": len(chunk),
            "status": batch.get("status", "validating"),
            "smoke": bool(smoke_n),
        })
        save_registry(reg)


def cmd_smoke() -> None:
    """Tiny mixed batch: 4 rollouts, 4 tool requests, 4 bail-prompt requests
    (with a canned assistant turn) to verify all three request shapes --
    especially whether the batch endpoint accepts `tools`."""
    prompts = load_prompts("orig").head(2)
    reqs = []
    for _, row in prompts.iterrows():
        for s in range(2):
            reqs.append({"custom_id": f"smokerollout|orig|{row['bailbench_id']}|{s}",
                         "body": base_body([{"role": "user", "content": row["prompt"]}])})
            body = base_body([{"role": "user", "content": row["prompt"]}])
            body["tools"] = [get_bail_tool(config.BAIL_MODEL_NAME)]
            reqs.append({"custom_id": f"smoketool|orig|{row['bailbench_id']}|{s}", "body": body})
            reqs.append({"custom_id": f"smokeprompt|orig|bail_first|{row['bailbench_id']}|{s}",
                         "body": base_body([
                             {"role": "user", "content": row["prompt"]},
                             {"role": "assistant", "content": "I can't help with that request."},
                             {"role": "user", "content": get_bail_prompt("bail_first")},
                         ])})
    batch = submit_one(reqs)
    log.info("smoke batch submitted: id %s status %s (%d requests)",
             batch.get("id"), batch.get("status"), len(reqs))
    reg = load_registry()
    reg["batches"].append({"id": batch["id"], "phase": "smoke", "condition": "orig",
                           "chunk": 0, "n_requests": len(reqs),
                           "status": batch.get("status", "validating"), "smoke": True})
    save_registry(reg)


def cmd_status() -> None:
    reg = load_registry()
    for b in reg["batches"]:
        if b["status"] in ("completed", "failed", "expired", "cancelled"):
            log.info("%s %s/%s chunk %d: %s (cached)", b["id"], b["phase"],
                     b["condition"], b["chunk"], b["status"])
            continue
        remote = fetch_batch(b["id"])
        b["status"] = remote.get("status", b["status"])
        counts = remote.get("request_counts")
        log.info("%s %s/%s chunk %d: %s %s", b["id"], b["phase"], b["condition"],
                 b["chunk"], b["status"], counts if counts else "")
    save_registry(reg)


def results_path(phase: str, condition: str) -> str:
    return os.path.join(config.RESULTS_DIR, f"results_{phase}_{condition}.parquet")


def load_results(phase: str, condition: str) -> pd.DataFrame:
    path = results_path(phase, condition)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No collected results for phase={phase} condition={condition}. "
            "Run status/collect once those batches complete.")
    return pd.read_parquet(path)


def parse_result(custom_id: str, result: dict) -> dict:
    parts = custom_id.split("|")
    phase = parts[0]
    row: dict = {"phase": phase, "custom_id": custom_id}
    if phase.startswith("smoke"):
        phase_body = phase.replace("smoke", "")
        row["phase"] = phase_body or phase
    if len(parts) == 5:  # prompt phase
        _, row["condition"], row["ordering"], bid, s = parts
    else:
        _, row["condition"], bid, s = parts
        row["ordering"] = None
    row["bailbench_id"] = int(bid)
    row["sample"] = int(s)

    err = result.get("error")
    resp = result.get("response") or {}
    body = resp.get("body") or {}
    message = ((body.get("choices") or [{}])[0].get("message")) or {}
    row["status_code"] = resp.get("status_code")
    row["error"] = json.dumps(err) if err else None
    row["response_text"] = message.get("content") or ""
    row["raw_message"] = json.dumps(message) if message else None
    row["tool_called"] = called_bail_tool(message) if message else None
    row["wellbeing"] = (get_bail_prompt_status(row["response_text"])
                        if row["phase"] == "prompt" and row["response_text"] else None)
    return row


def cmd_collect() -> None:
    reg = load_registry()
    by_key: dict[tuple, list[dict]] = {}
    for b in reg["batches"]:
        if b.get("collected"):
            continue
        remote = fetch_batch(b["id"])
        b["status"] = remote.get("status", b["status"])
        if b["status"] != "completed":
            log.info("%s not completed yet (%s), skipping", b["id"], b["status"])
            continue
        results = remote.get("results") or []
        log.info("%s: %d results", b["id"], len(results))
        for r in results:
            row = parse_result(r["custom_id"], r)
            by_key.setdefault((row["phase"], row["condition"]), []).append(row)
        b["collected"] = True
    for (phase, condition), rows in by_key.items():
        new = pd.DataFrame(rows)
        path = results_path(phase, condition)
        if os.path.exists(path):
            old = pd.read_parquet(path)
            new = pd.concat([old, new], ignore_index=True).drop_duplicates(
                subset="custom_id", keep="last")
        new.to_parquet(path, index=False)
        n_err = new["error"].notna().sum()
        log.info("wrote %d rows to %s (%d with errors)", len(new), path, n_err)
    save_registry(reg)


# ---------------------------------------------------------------------------
# Live (non-batch) runner -- OpenRouter has no :batch endpoint for Gemma.
# ---------------------------------------------------------------------------

def _call_live(session: requests.Session, req: dict) -> dict:
    """One chat-completions call with retry/backoff; returns a batch-shaped
    result dict so parse_result can be reused."""
    payload = {"model": config.BAIL_MODEL, **req["body"]}
    last_err = None
    for attempt in range(config.API_MAX_RETRIES + 1):
        try:
            resp = session.post(
                config.OPENROUTER_CHAT_URL,
                headers={"Authorization": f"Bearer {api_key()}",
                         "Content-Type": "application/json"},
                json=payload, timeout=180)
            if resp.status_code in (429, 500, 502, 503):
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            body = resp.json()
            if "error" in body and "choices" not in body:
                raise RuntimeError(f"API error: {str(body['error'])[:200]}")
            return {"custom_id": req["custom_id"],
                    "response": {"status_code": resp.status_code, "body": body},
                    "error": None}
        except Exception as e:
            last_err = e
            if attempt < config.API_MAX_RETRIES:
                time.sleep(min(2 ** attempt, 30))
    return {"custom_id": req["custom_id"], "response": None,
            "error": {"message": f"{type(last_err).__name__}: {last_err}"}}


def _flush(rows: list[dict], phase: str, condition: str) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    path = results_path(phase, condition)
    if os.path.exists(path):
        old = pd.read_parquet(path)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates(
            subset="custom_id", keep="last")
    new.to_parquet(path, index=False)


def cmd_run(phase: str, condition: str, limit: int = 0) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    reqs = build_requests(phase, condition)
    if limit:
        reqs = reqs[:limit]
    path = results_path(phase, condition)
    done: set[str] = set()
    if os.path.exists(path):
        existing = pd.read_parquet(path)
        done = set(existing.loc[existing["error"].isna(), "custom_id"])
    todo = [r for r in reqs if r["custom_id"] not in done]
    log.info("live run %s/%s: %d requests (%d already done, %d to run)",
             phase, condition, len(reqs), len(reqs) - len(todo), len(todo))
    if not todo:
        return

    session = requests.Session()
    buffer: list[dict] = []
    n_done = n_fail = 0
    with ThreadPoolExecutor(max_workers=config.BAIL_CONCURRENCY) as pool:
        futures = [pool.submit(_call_live, session, r) for r in todo]
        for fut in as_completed(futures):
            result = fut.result()
            buffer.append(parse_result(result["custom_id"], result))
            n_done += 1
            if result["error"]:
                n_fail += 1
            if n_done % 200 == 0 or n_done == len(todo):
                _flush(buffer, phase, condition)
                buffer = []
                log.info("progress: %d/%d (%d failed)", n_done, len(todo), n_fail)
    _flush(buffer, phase, condition)
    log.info("live run complete: %d calls, %d failed after retries "
             "(failed rows are retried on re-run)", n_done, n_fail)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke")
    s = sub.add_parser("submit")
    s.add_argument("--phase", choices=PHASES, required=True)
    s.add_argument("--condition", choices=CONDITIONS, required=True)
    s.add_argument("--smoke-n", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--phase", choices=PHASES, required=True)
    r.add_argument("--condition", choices=CONDITIONS, required=True)
    r.add_argument("--limit", type=int, default=0)
    sub.add_parser("status")
    sub.add_parser("collect")
    args = p.parse_args()

    if args.cmd == "smoke":
        cmd_smoke()
    elif args.cmd == "submit":
        cmd_submit(args.phase, args.condition, args.smoke_n)
    elif args.cmd == "run":
        cmd_run(args.phase, args.condition, args.limit)
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "collect":
        cmd_collect()


if __name__ == "__main__":
    main()
