"""Deterministic payload construction for the read-only timeline renderer."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import pyarrow.parquet as pq  # type: ignore[import-untyped]

Row: TypeAlias = dict[str, object]
_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
)


@dataclass(frozen=True)
class SourceSeveritySummary:
    """Selected one-hot source-abuse ratings, excluding malformed annotators."""

    selected_ratings: tuple[int, ...]
    mean: float | None
    malformed_annotator_count: int


def _int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, (str, float)):
        return int(value)
    raise TypeError(
        f"timeline expected integer-compatible exported value, got {value!r}"
    )


def _read(directory: Path, name: str) -> list[Row]:
    return [
        dict(row) for row in pq.read_table(directory / f"{name}.parquet").to_pylist()
    ]


def candidate_label_sort_key(label: object) -> tuple[int, int | str]:
    """Order C1, C2, … C10 numerically, with nonstandard labels last."""
    value = str(label)
    return (
        (0, int(value[1:]))
        if value.startswith("C") and value[1:].isdigit()
        else (1, value)
    )


def _regime(regime: object) -> tuple[str, str]:
    if regime == "support":
        return (
            "Most Votes Kept",
            "Highest aggregate support is protected; a different active candidate is removed by seeded uniform draw.",
        )
    return "Most Votes Kicked", "Highest aggregate opposition is removed."


def source_severity_summary(rows: list[Row]) -> SourceSeveritySummary:
    """Select one numeric nonzero abuse rating per valid annotator, never zeros."""
    selected: dict[str, list[int]] = defaultdict(list)
    observed: set[str] = set()
    for row in rows:
        label, value = (
            str(row.get("source_label", "")),
            str(row.get("source_value", "")).lower(),
        )
        if not label.startswith("is_abuse."):
            continue
        annotator = str(row.get("annotator_hash", ""))
        observed.add(annotator)
        if value not in {"1", "true", "yes"}:
            continue
        try:
            selected[annotator].append(int(label.removeprefix("is_abuse.")))
        except ValueError:
            continue
    ratings = tuple(
        selected[annotator][0]
        for annotator in sorted(observed)
        if len(selected[annotator]) == 1
    )
    malformed = sum(len(selected[annotator]) != 1 for annotator in observed)
    return SourceSeveritySummary(
        selected_ratings=ratings,
        mean=None if not ratings else sum(ratings) / len(ratings),
        malformed_annotator_count=malformed,
    )


def build_timeline_payload(export_dir: Path, labels: list[Row]) -> dict[str, object]:
    """Shape persisted relations without reconstructing omitted source turns."""
    label = {str(row["candidate_id"]): str(row["candidate_label"]) for row in labels}
    metadata = {
        str(row["candidate_id"]): row for row in _read(export_dir, "candidate_metadata")
    }
    annotations: dict[str, list[Row]] = defaultdict(list)
    sources: dict[str, list[Row]] = defaultdict(list)
    for row in _read(export_dir, "source_annotations"):
        annotations[str(row["candidate_id"])].append(row)
    for row in _read(export_dir, "candidate_source_turns"):
        sources[str(row["candidate_id"])].append(row)
    active: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row in _read(export_dir, "round_candidates"):
        active[(str(row["run_id"]), _int(row["round_index"]))].append(
            str(row["candidate_id"])
        )
    outcomes = {str(row["round_id"]): row for row in _read(export_dir, "outcomes")}
    rounds = _read(export_dir, "rounds")
    round_ids = {
        (str(row["run_id"]), _int(row["round_index"])): str(row["round_id"])
        for row in rounds
    }
    voter_indices = {
        str(row["voter_id"]): _int(row["voter_index"])
        for row in _read(export_dir, "voters")
    }
    turns = {str(row["turn_id"]): row for row in _read(export_dir, "turns")}
    rationale: dict[tuple[str, int], str | None] = {}
    for row in _read(export_dir, "ballots"):
        turn = turns.get(str(row["turn_id"]))
        if turn:
            rationale[(str(turn["round_id"]), voter_indices[str(turn["voter_id"])])] = (
                str(row["rationale"])
                if row.get("status") == "accepted" and row.get("rationale") is not None
                else None
            )
    analysis: dict[tuple[str, int, int], list[Row]] = defaultdict(list)
    for row in _read(export_dir, "candidate_analysis"):
        analysis[
            (str(row["run_id"]), _int(row["round_index"]), _int(row["voter_index"]))
        ].append(row)
    survival_rows = _read(export_dir, "candidate_survival")
    survival = {
        (str(row["run_id"]), str(row["candidate_id"])): bool(row["winner"])
        for row in survival_rows
    }
    aggregate_votes: dict[tuple[str, str, int], int] = {}
    for row in survival_rows:
        round_indices = row.get("round_indices")
        votes_by_round = row.get("votes_by_round")
        if not isinstance(round_indices, list) or not isinstance(votes_by_round, list):
            raise TypeError(
                "timeline expected candidate_survival round_indices and votes_by_round lists"
            )
        for round_index, votes in zip(round_indices, votes_by_round, strict=True):
            aggregate_votes[
                (str(row["run_id"]), str(row["candidate_id"]), _int(round_index))
            ] = _int(votes)
    runs: list[Row] = []
    for run in sorted(
        _read(export_dir, "runs"), key=lambda value: str(value["run_id"])
    ):
        run_id = str(run["run_id"])
        title, explanation = _regime(run["regime"])
        run_rounds = sorted(
            index for candidate_run, index in active if candidate_run == run_id
        )
        all_candidates = sorted(
            {
                candidate
                for candidate_run, _ in active
                if candidate_run == run_id
                for candidate in active[(candidate_run, _)]
            },
            key=lambda value: candidate_label_sort_key(label.get(value, value)),
        )
        removed_round = {
            str(outcome["removed_candidate_id"]): index
            for (candidate_run, index), round_id in round_ids.items()
            if candidate_run == run_id
            for outcome in [outcomes.get(round_id, {})]
            if outcome.get("removed_candidate_id") is not None
        }
        frames: list[Row] = []
        for index in run_rounds:
            outcome = outcomes.get(round_ids[(run_id, index)], {})
            candidates: list[Row] = []
            for candidate in all_candidates:
                is_active = candidate in active[(run_id, index)]
                source_turns = sorted(
                    sources[candidate], key=lambda value: _int(value["turn_index"])
                )
                severity = source_severity_summary(annotations[candidate])
                candidates.append(
                    {
                        "id": candidate,
                        "label": label.get(candidate, candidate),
                        "rudeness": metadata.get(candidate, {}).get("rudeness_label"),
                        "sourceSeverityMean": severity.mean,
                        "sourceSeverityRatings": severity.selected_ratings,
                        "sourceSeverityN": len(severity.selected_ratings),
                        "sourceSeverityMalformedAnnotatorCount": severity.malformed_annotator_count,
                        "sourceTurns": [
                            {"role": row.get("role"), "text": row.get("text")}
                            for row in source_turns
                        ],
                        "aggregateVotes": aggregate_votes.get(
                            (run_id, candidate, index)
                        )
                        if is_active
                        else None,
                        "protected": outcome.get("protected_candidate_id") == candidate,
                        "removed": outcome.get("removed_candidate_id") == candidate,
                        "winner": survival.get((run_id, candidate), False)
                        and index == run_rounds[-1],
                        "active": is_active,
                        "previouslyRemoved": removed_round.get(candidate, index)
                        < index,
                    }
                )
            voters: list[Row] = []
            for voter in sorted(
                key[2] for key in analysis if key[:2] == (run_id, index)
            ):
                rows = analysis[(run_id, index, voter)]
                valid = all(row.get("raw_votes") is not None for row in rows)
                spend = (
                    None
                    if not valid
                    else sum(_int(row["raw_votes"]) ** 2 for row in rows)
                )
                allocations = [
                    {
                        "label": label.get(
                            str(row["candidate_id"]), str(row["candidate_id"])
                        ),
                        "votes": row.get("raw_votes"),
                        "credits": None
                        if row.get("raw_votes") is None
                        else _int(row["raw_votes"]) ** 2,
                        "rating": row.get("rating_code"),
                        "statement": row.get("statement_text"),
                    }
                    for row in rows
                ]
                voters.append(
                    {
                        "voter": voter + 1,
                        "ballotStatus": rows[0].get("ballot_status"),
                        "statementStatus": rows[0].get("statement_status"),
                        "rationale": rationale.get((round_ids[(run_id, index)], voter)),
                        "spend": spend,
                        "budget": run.get("credit_budget"),
                        "allocations": sorted(
                            allocations,
                            key=lambda allocation: candidate_label_sort_key(
                                allocation["label"]
                            ),
                        ),
                    }
                )
            frames.append(
                {
                    "round": index,
                    "candidates": candidates,
                    "voters": voters,
                    "outcome": {
                        "protected": outcome.get("protected_candidate_id"),
                        "protectedLabel": label.get(
                            str(outcome.get("protected_candidate_id"))
                        )
                        if outcome.get("protected_candidate_id") is not None
                        else None,
                        "removed": outcome.get("removed_candidate_id"),
                        "removedLabel": label.get(
                            str(outcome.get("removed_candidate_id"))
                        )
                        if outcome.get("removed_candidate_id") is not None
                        else None,
                        "tie": outcome.get("tie_flag"),
                    },
                }
            )
        runs.append(
            {
                "id": run_id,
                "arm": run["arm"],
                "regime": title,
                "explanation": explanation,
                "frames": frames,
            }
        )
    return {
        "runs": runs,
        "colors": {f"C{index + 1}": color for index, color in enumerate(_COLORS)},
        "unspent": "#8A8A8A",
    }
