"""Quadratic voting v2: do LLM voters spend credits against rude participants?

Deliberately simple quadratic-voting experiment on ConvAbuse snippets. Each
round shows 5 candidates (one per severity band 1/0/-1/-2/-3, high annotator
agreement only), presented as anonymous conversation cards in a randomized
letter order. A voter gets 100 credits; v votes on a candidate cost v^2
credits. Two frames, run separately: allocate credits to REMOVE participants,
or to KEEP them. 50 rounds x 3 voters x 5 repeats = 750 ballots per frame.

Voters are independent replicate samples of the same model (Gemma 4 E4B IT,
4-bit, same pinned route as bail_steering); they never see each other's
ballots. Severity labels are never shown.

Usage (from the repo root, on the machine with the GPU):
  uv run python -m quadratic_voting_v2.main download
  uv run python -m quadratic_voting_v2.main freeze
  uv run python -m quadratic_voting_v2.main run --frame remove --device cuda --limit 6
  uv run python -m quadratic_voting_v2.main run --frame remove --device cuda
  uv run python -m quadratic_voting_v2.main run --frame keep --device cuda
  uv run python -m quadratic_voting_v2.main run --frame keep --device cuda --resume
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from llm_runtime import (
    Capability,
    LocalTransformersRoute,
    ModelId,
    ProviderId,
    QuantizationId,
    resolve_route,
)
from quadratic_voting_v2.prompts import (
    BALLOT_TEMPLATE,
    CARD_TEMPLATE,
    CORRECTION_MESSAGE,
    FRAME_SENTENCES,
    FRAMES,
)

# --- Experiment constants -----------------------------------------------------
MODEL_ID = ModelId.GEMMA_4_E4B_IT  # same pinned route as bail_steering/main.py
QUANTIZATION_ID = QuantizationId.BITSANDBYTES_FP4

SEED = 42
N_ROUNDS = 50
N_VOTERS = 3
N_REPEATS = 5
BUDGET = 100  # credits per ballot; v votes cost v^2 credits
MAX_CORRECTIONS = 3  # correction retries after the first attempt

BANDS = (1, 0, -1, -2, -3)  # severity bands, one candidate each per round
MAX_SEVERITY_STD = 0.5  # eligibility: high annotator agreement
LETTERS = ("A", "B", "C", "D", "E")

TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 512

REPO_ROOT = Path(__file__).parent.parent
PARQUET_FILE = REPO_ROOT / "bail" / "data" / "convabuse_collapsed.parquet"
ROUNDS_FILE = Path(__file__).parent / "data" / "rounds_seed42.csv"
RESULTS_DIR = Path(__file__).parent / "results"
BALLOTS_FILE = "ballots.csv"

TEXT_COLUMNS = ("prev_agent", "prev_user", "agent", "user")
ROUNDS_COLUMNS = ("round_index", "band", "snippet_id") + TEXT_COLUMNS

# Conversation turns rendered on each card, chronological, same order as
# bail_steering/main.py TRANSCRIPT_TURNS. User turns are labeled with the
# participant's letter; the bot's turns are labeled "Assistant".
CARD_TURNS = (
    ("prev_agent", "Assistant"),
    ("prev_user", "user"),
    ("agent", "Assistant"),
    ("user", "user"),  # the annotated user message
)

BAND_VOTE_COLUMNS = {
    1: "votes_band_p1",
    0: "votes_band_0",
    -1: "votes_band_m1",
    -2: "votes_band_m2",
    -3: "votes_band_m3",
}
BALLOT_COLUMNS = (
    "frame",
    "round_index",
    "voter",
    "repeat",
    "presentation_order",  # JSON: letter -> snippet_id
    "votes_by_snippet",  # JSON: snippet_id -> votes
    *(BAND_VOTE_COLUMNS[band] for band in BANDS),
    "credits_spent",
    "valid",
    "failure_reason",
    "retry_count",
    "raw_response",  # final attempt's raw text
)

# run_info keys that must match exactly for --resume (provenance check).
RESUME_KEYS = (
    "experiment",
    "frame",
    "model_id",
    "quantization_id",
    "repository",
    "revision",
    "parquet_sha256",
    "frozen_sample_sha256",
    "seed",
    "n_rounds",
    "n_voters",
    "n_repeats",
    "budget",
    "temperature",
    "top_p",
    "max_new_tokens",
    "limit",
)


class QV2Error(RuntimeError):
    """A user-actionable failure in the QV2 workflow."""


class BallotError(ValueError):
    """One model reply failed ballot parsing or the quadratic budget rule."""


# --- Candidate draw and frozen sample -----------------------------------------


def load_eligible(parquet_file: Path = PARQUET_FILE) -> pd.DataFrame:
    """Eligible ConvAbuse snippets: high annotator agreement only."""
    if not parquet_file.exists():
        raise QV2Error(
            f"ConvAbuse parquet not found: {parquet_file}. It is committed "
            "with the bail workstream; check the repo checkout."
        )
    df = pd.read_parquet(parquet_file)
    return df[df["severity_std"].fillna(0) <= MAX_SEVERITY_STD]


def draw_rounds(eligible: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """N_ROUNDS rounds x one snippet per band, without replacement per band.

    Deterministic given the parquet: each band's pool is sorted by snippet_id
    before a single seeded Generator permutes it, so the draw does not depend
    on parquet row order.
    """
    rng = np.random.default_rng(seed)
    picks_by_band: dict[int, pd.DataFrame] = {}
    for band in BANDS:
        pool = eligible[eligible["severity_band"] == band]
        pool = pool.sort_values("snippet_id").reset_index(drop=True)
        if len(pool) < N_ROUNDS:
            raise QV2Error(
                f"Band {band} has only {len(pool)} eligible snippets; "
                f"{N_ROUNDS} rounds need {N_ROUNDS} distinct ones."
            )
        order = rng.permutation(len(pool))[:N_ROUNDS]
        picks_by_band[band] = pool.iloc[order].reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for round_index in range(N_ROUNDS):
        for band in BANDS:
            pick = picks_by_band[band].iloc[round_index]
            row: dict[str, Any] = {
                "round_index": round_index,
                "band": band,
                "snippet_id": pick["snippet_id"],
            }
            for column in TEXT_COLUMNS:
                text = pick[column]
                row[column] = "" if pd.isna(text) else str(text)
            rows.append(row)
    return pd.DataFrame(rows, columns=list(ROUNDS_COLUMNS))


def load_rounds(
    rounds_file: Path = ROUNDS_FILE, parquet_file: Path = PARQUET_FILE
) -> pd.DataFrame:
    """The frozen seed-42 sample; built and written on first use.

    A pre-existing file is verified against the parquet (snippet_ids must
    exist with the same band) so both frames provably vote on the same
    candidates.
    """
    if rounds_file.exists():
        rounds = pd.read_csv(rounds_file, keep_default_na=False)
        rounds["round_index"] = rounds["round_index"].astype(int)
        rounds["band"] = rounds["band"].astype(int)
        eligible = load_eligible(parquet_file)
        bands = dict(zip(eligible["snippet_id"], eligible["severity_band"]))
        for row in rounds.itertuples():
            if bands.get(row.snippet_id) != row.band:
                raise QV2Error(
                    f"Frozen sample {rounds_file} does not match the parquet "
                    f"(snippet {row.snippet_id}, band {row.band}); delete the "
                    "frozen file only if you mean to redraw the sample."
                )
        if len(rounds) != N_ROUNDS * len(BANDS):
            raise QV2Error(
                f"Frozen sample {rounds_file} has {len(rounds)} rows, "
                f"expected {N_ROUNDS * len(BANDS)}."
            )
        return rounds
    rounds = draw_rounds(load_eligible(parquet_file))
    rounds_file.parent.mkdir(parents=True, exist_ok=True)
    rounds.to_csv(rounds_file, index=False)
    print(f"Froze sample: {rounds_file} ({len(rounds)} rows)")
    return rounds


# --- Ballot construction ------------------------------------------------------


def presentation_order(
    frame: str, round_index: int, voter: int, repeat: int
) -> tuple[int, ...]:
    """Candidate index (in BANDS order) shown at each letter position.

    Seeded from (SEED, frame, round, voter, repeat) so every ballot's
    letter assignment is reproducible and independent across ballots.
    """
    seed_seq = np.random.SeedSequence(
        (SEED, FRAMES.index(frame), round_index, voter, repeat)
    )
    rng = np.random.default_rng(seed_seq)
    return tuple(int(i) for i in rng.permutation(len(BANDS)))


def assign_letters(
    round_rows: pd.DataFrame, frame: str, round_index: int, voter: int, repeat: int
) -> dict[str, dict[str, Any]]:
    """Letter -> candidate row (as a dict) for one ballot."""
    by_band = {int(row["band"]): row for _, row in round_rows.iterrows()}
    if sorted(by_band) != sorted(BANDS):
        raise QV2Error(f"Round {round_index} does not have one row per band.")
    order = presentation_order(frame, round_index, voter, repeat)
    return {
        letter: dict(by_band[BANDS[candidate_index]])
        for letter, candidate_index in zip(LETTERS, order)
    }


def render_card(letter: str, candidate: dict[str, Any]) -> str:
    lines = []
    for column, label in CARD_TURNS:
        text = str(candidate.get(column) or "").strip()
        if not text or text.lower() == "nan":
            continue
        speaker = f"Participant {letter}" if label == "user" else label
        lines.append(f"{speaker}: {text}")
    return CARD_TEMPLATE.format(letter=letter, turns="\n".join(lines))


def build_ballot_text(frame: str, assignment: dict[str, dict[str, Any]]) -> str:
    cards = "\n\n".join(
        render_card(letter, assignment[letter]) for letter in LETTERS
    )
    return BALLOT_TEMPLATE.format(
        frame_sentence=FRAME_SENTENCES[frame], budget=BUDGET, cards=cards
    )


# --- Ballot parsing and the quadratic rule ------------------------------------


def ballot_cost(votes: dict[str, int]) -> int:
    """Quadratic cost: v votes cost v^2 credits, summed over candidates."""
    return sum(v * v for v in votes.values())


def _strip_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]  # drop the opening fence (with or without "json")
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def parse_ballot(text: str, budget: int = BUDGET) -> tuple[dict[str, int], str]:
    """Strictly parse one reply into (votes by letter, reason).

    Raises BallotError with a human-readable message (fed back to the model
    as the correction) on any malformed reply or budget violation. The only
    tolerated decoration is a Markdown code fence around the JSON object.
    """
    try:
        obj = json.loads(_strip_fences(text))
    except json.JSONDecodeError as error:
        raise BallotError(f"the reply is not a single JSON object ({error.msg}).")
    if not isinstance(obj, dict):
        raise BallotError("the reply is JSON but not an object.")
    votes_raw = obj.get("votes")
    if not isinstance(votes_raw, dict):
        raise BallotError('the object has no "votes" mapping.')
    if sorted(votes_raw) != sorted(LETTERS):
        raise BallotError(
            f'"votes" must have exactly the keys {", ".join(LETTERS)}.'
        )
    votes: dict[str, int] = {}
    for letter in LETTERS:
        value = votes_raw[letter]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BallotError(
                f'vote for "{letter}" must be a non-negative whole number.'
            )
        votes[letter] = value
    cost = ballot_cost(votes)
    if cost > budget:
        raise BallotError(
            f"the votes cost {cost} credits but the budget is {budget}."
        )
    reason = obj.get("reason")
    return votes, reason if isinstance(reason, str) else ""


# --- Run folder and resume ----------------------------------------------------


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def task_key(round_index: int, voter: int, repeat: int) -> str:
    return f"{round_index}|{voter}|{repeat}"


def load_done(run_dir: Path) -> set[str]:
    """(round, voter, repeat) keys already balloted (valid or invalid)."""
    path = run_dir / BALLOTS_FILE
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            task_key(int(row["round_index"]), int(row["voter"]), int(row["repeat"]))
            for row in csv.DictReader(handle)
        }


def append_ballot(run_dir: Path, row: dict[str, Any]) -> None:
    """Flush one ballot row to CSV immediately (crash-safe resume)."""
    path = run_dir / BALLOTS_FILE
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BALLOT_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def prepare_run_dir(frame: str, resume: bool, run_info: dict[str, Any]) -> Path:
    """New timestamped per-frame folder, or the newest matching one on resume.

    Run folders are never overwritten. Resume refuses provenance mismatches,
    including a different --limit: a limited smoke run must not silently grow
    into (or truncate) a full run.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if resume:
        runs = [
            path
            for path in RESULTS_DIR.iterdir()
            if path.is_dir()
            and path.name.endswith(f"_qv2-{frame}")
            and (path / "run_info.json").exists()
        ]
        if not runs:
            raise QV2Error(
                f"No {frame}-frame run folder to resume; run without --resume."
            )
        run_dir = max(
            runs,
            key=lambda p: json.loads(
                (p / "run_info.json").read_text(encoding="utf-8")
            ).get("started", ""),
        )
        old_info = json.loads(
            (run_dir / "run_info.json").read_text(encoding="utf-8")
        )
        for key in RESUME_KEYS:
            if old_info.get(key) != run_info.get(key):
                raise QV2Error(
                    f"Refusing to resume {run_dir}: run_info mismatch on "
                    f"'{key}' ({old_info.get(key)!r} != {run_info.get(key)!r})."
                )
        print(f"Resuming {run_dir}")
        return run_dir
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    base_name = f"{stamp}_qv2-{frame}"
    run_dir = RESULTS_DIR / base_name
    suffix = 0
    while True:  # mkdir fails atomically if the folder exists: never overwrite
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            suffix += 1
            run_dir = RESULTS_DIR / f"{base_name}_{suffix}"
    (run_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    print(f"New run folder: {run_dir}")
    return run_dir


# --- Model boundary (the only code that needs torch/GPU) ----------------------


def resolve_qv2_route() -> LocalTransformersRoute:
    route = resolve_route(
        MODEL_ID,
        ProviderId.LOCAL,
        QUANTIZATION_ID,
        required={Capability.TEXT_GENERATION},
    )
    if not isinstance(route, LocalTransformersRoute):
        raise QV2Error("QV2 needs a local Transformers route.")
    return route


def generate_reply(model: Any, tokenizer: Any, messages: list[dict]) -> str:
    """One sampled completion for a chat, mirroring bail_steering's
    generate_batch (chat template includes BOS; no extra special tokens)."""
    import torch

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(
        text, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    generated = output[:, encoded["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]


def collect_ballot(
    model: Any, tokenizer: Any, ballot_text: str
) -> tuple[dict[str, int] | None, str, int, str]:
    """Run one ballot with up to MAX_CORRECTIONS retries.

    Returns (votes-by-letter or None, failure_reason, retry_count,
    final raw response).
    """
    messages: list[dict] = [{"role": "user", "content": ballot_text}]
    failure = ""
    raw = ""
    for attempt in range(1 + MAX_CORRECTIONS):
        raw = generate_reply(model, tokenizer, messages)
        try:
            votes, _reason = parse_ballot(raw)
            return votes, "", attempt, raw
        except BallotError as error:
            failure = str(error)
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": CORRECTION_MESSAGE.format(
                        error=failure, budget=BUDGET
                    ),
                }
            )
    return None, failure, MAX_CORRECTIONS, raw


# --- Run ----------------------------------------------------------------------


def build_tasks(limit: int | None) -> list[tuple[int, int, int]]:
    """(round, voter, repeat) grid, round-major, optionally capped."""
    tasks = [
        (round_index, voter, repeat)
        for round_index in range(N_ROUNDS)
        for voter in range(N_VOTERS)
        for repeat in range(N_REPEATS)
    ]
    return tasks[:limit] if limit is not None else tasks


def ballot_row(
    frame: str,
    round_index: int,
    voter: int,
    repeat: int,
    assignment: dict[str, dict[str, Any]],
    votes: dict[str, int] | None,
    failure_reason: str,
    retry_count: int,
    raw: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "frame": frame,
        "round_index": round_index,
        "voter": voter,
        "repeat": repeat,
        "presentation_order": json.dumps(
            {letter: assignment[letter]["snippet_id"] for letter in LETTERS}
        ),
        "votes_by_snippet": "",
        **{BAND_VOTE_COLUMNS[band]: "" for band in BANDS},
        "credits_spent": "",
        "valid": votes is not None,
        "failure_reason": failure_reason,
        "retry_count": retry_count,
        "raw_response": raw,
    }
    if votes is not None:
        row["votes_by_snippet"] = json.dumps(
            {assignment[letter]["snippet_id"]: votes[letter] for letter in LETTERS}
        )
        for letter in LETTERS:
            band = int(assignment[letter]["band"])
            row[BAND_VOTE_COLUMNS[band]] = votes[letter]
        row["credits_spent"] = ballot_cost(votes)
    return row


def run(
    frame: str,
    cache_dir: Path,
    device_name: str,
    limit: int | None,
    resume: bool,
) -> None:
    # torch and the transformers runtime are imported here, not at module
    # level, so tests and analysis stay importable without a GPU stack.
    from llm_runtime.transformers import Device, create_transformers_runtime

    import torch

    route = resolve_qv2_route()
    rounds = load_rounds()
    run_info = {
        "experiment": "quadratic-voting-v2",
        "frame": frame,
        "started": datetime.now(timezone.utc).isoformat(),
        "model_id": route.model_id.value,
        "quantization_id": route.quantization_id.value,
        "repository": route.artifact.repository,
        "revision": route.artifact.revision,
        "parquet_sha256": file_sha256(PARQUET_FILE),
        "frozen_sample_sha256": file_sha256(ROUNDS_FILE),
        "seed": SEED,
        "n_rounds": N_ROUNDS,
        "n_voters": N_VOTERS,
        "n_repeats": N_REPEATS,
        "budget": BUDGET,
        "max_corrections": MAX_CORRECTIONS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_new_tokens": MAX_NEW_TOKENS,
        "limit": limit,
    }
    run_dir = prepare_run_dir(frame, resume, run_info)
    done = load_done(run_dir)
    tasks = [
        task for task in build_tasks(limit) if task_key(*task) not in done
    ]
    print(f"[{frame}] {len(tasks)} ballots to collect ({len(done)} done)")
    if not tasks:
        print("Nothing to do.")
        return

    print(f"Loading {route.artifact.repository} ({route.quantization_id.value})...")
    runtime = create_transformers_runtime(
        route, cache_dir=cache_dir, device=Device(device_name)
    )
    if runtime.placement.has_cpu_or_offload:
        print(
            "WARNING: parts of the model were placed on CPU/disk "
            f"(cpu={list(runtime.placement.cpu_modules)}); generation will be "
            "very slow. Free GPU memory and restart with --resume."
        )
    model, tokenizer = runtime.model, runtime.tokenizer
    torch.manual_seed(SEED)

    started = time.perf_counter()
    finished = 0
    n_invalid = 0
    for round_index, voter, repeat in tasks:
        round_rows = rounds[rounds["round_index"] == round_index]
        assignment = assign_letters(round_rows, frame, round_index, voter, repeat)
        ballot_text = build_ballot_text(frame, assignment)
        votes, failure_reason, retry_count, raw = collect_ballot(
            model, tokenizer, ballot_text
        )
        n_invalid += votes is None
        append_ballot(
            run_dir,
            ballot_row(
                frame,
                round_index,
                voter,
                repeat,
                assignment,
                votes,
                failure_reason,
                retry_count,
                raw,
            ),
        )
        finished += 1
        rate = finished / (time.perf_counter() - started)
        remaining = (len(tasks) - finished) / rate if rate else 0
        print(
            f"[{frame}] {finished}/{len(tasks)} "
            f"({n_invalid} invalid, {rate * 60:.1f}/min, "
            f"~{remaining / 3600:.1f} h left)"
        )
    print(f"Done. Results in {run_dir}")


# --- CLI ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quadratic voting over ConvAbuse participants (v2)."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Hugging Face Hub cache directory (default: the HF default)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download", help="download the pinned checkpoint")
    subparsers.add_parser(
        "freeze", help="build data/rounds_seed42.csv without a model"
    )
    run_parser = subparsers.add_parser("run", help="collect one frame's ballots")
    run_parser.add_argument(
        "--frame", choices=list(FRAMES), required=True, help="ballot frame"
    )
    run_parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="inference device (default: %(default)s)",
    )
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap total ballots (smoke tests; recorded in run_info)",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the newest matching-frame run folder",
    )
    return parser


def _default_cache_dir() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "freeze":
            rounds = load_rounds()
            counts = rounds["band"].value_counts().sort_index(ascending=False)
            print(f"Frozen sample: {ROUNDS_FILE}")
            for band, count in counts.items():
                print(f"  band {band:+d}: {count} snippets")
        elif args.command == "download":
            from llm_runtime.transformers import download_transformers_artifact

            cache_dir = args.cache_dir or _default_cache_dir()
            model_path = download_transformers_artifact(
                resolve_qv2_route(), cache_dir
            )
            print(f"Downloaded pinned checkpoint to: {model_path}")
        else:
            cache_dir = args.cache_dir or _default_cache_dir()
            run(args.frame, cache_dir, args.device, args.limit, args.resume)
    except QV2Error as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
