"""Strict parsing and quadratic-cost validation for model ballots."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from quadratic_voting.experiment.types import CandidateId, ValidationErrorCode


@dataclass(frozen=True, slots=True)
class ParsedBallot:
    """A validated ballot in active-snapshot order.

    Explicit zero-vote entries are equivalent to omission and are omitted here.
    """

    rationale: str
    allocations: tuple[tuple[CandidateId, int], ...]


@dataclass(frozen=True, slots=True)
class ValidationFailure:
    code: ValidationErrorCode
    ordinal: int
    message: str


@dataclass(frozen=True, slots=True)
class _Failure:
    code: ValidationErrorCode
    position: int
    message: str


_CODE_ORDER = {code: index for index, code in enumerate(ValidationErrorCode)}


def _finish(failures: list[_Failure]) -> tuple[ValidationFailure, ...]:
    ordered = sorted(failures, key=lambda item: (_CODE_ORDER[item.code], item.position))
    return tuple(
        ValidationFailure(code=item.code, ordinal=ordinal, message=item.message)
        for ordinal, item in enumerate(ordered)
    )


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    return "object"


def ballot_cost(allocations: Iterable[tuple[CandidateId, int]]) -> int:
    """Return the engine-computed quadratic cost of allocations."""
    return sum(votes**2 for _, votes in allocations)


def parse_and_validate_ballot(
    raw: str,
    active: tuple[CandidateId, ...],
    budget: int,
    known: tuple[CandidateId, ...] | None = None,
) -> ParsedBallot | tuple[ValidationFailure, ...]:
    """Parse a ballot without coercion and report every deterministic failure."""
    try:
        value: Any = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as error:
        detail = error.msg if isinstance(error, json.JSONDecodeError) else str(error)
        return (
            ValidationFailure(
                ValidationErrorCode.MALFORMED_JSON,
                0,
                f"Malformed JSON: {detail}.",
            ),
        )

    if not isinstance(value, dict):
        return (
            ValidationFailure(
                ValidationErrorCode.INVALID_TYPE,
                0,
                f"Top-level ballot must be a JSON object; received {_json_type(value)}.",
            ),
        )

    failures: list[_Failure] = []
    position = 0

    def add(code: ValidationErrorCode, message: str) -> None:
        nonlocal position
        failures.append(_Failure(code, position, message))
        position += 1

    required = ("reason", "allocations")
    for field in required:
        if field not in value:
            add(ValidationErrorCode.MISSING_FIELD, f"Missing required field: {field}.")
    for field in value:
        if field not in required:
            add(
                ValidationErrorCode.EXTRA_FIELD, f"Unexpected top-level field: {field}."
            )

    reason = value.get("reason")
    if "reason" in value and not isinstance(reason, str):
        add(
            ValidationErrorCode.INVALID_TYPE,
            f"Field reason must be a string; received {_json_type(reason)}.",
        )
    elif isinstance(reason, str) and not any(not char.isspace() for char in reason):
        add(
            ValidationErrorCode.EMPTY_RATIONALE,
            "Field reason must contain at least one non-whitespace Unicode character.",
        )

    raw_allocations = value.get("allocations")
    if "allocations" in value and not isinstance(raw_allocations, list):
        add(
            ValidationErrorCode.INVALID_TYPE,
            "Field allocations must be an array; received "
            f"{_json_type(raw_allocations)}.",
        )

    parsed: dict[CandidateId, int] = {}
    submitted_votes: list[int] = []
    encountered: set[CandidateId] = set()
    active_set = set(active)
    known_set = set(known) if known is not None else active_set
    if isinstance(raw_allocations, list):
        for index, item in enumerate(raw_allocations):
            if not isinstance(item, dict):
                add(
                    ValidationErrorCode.INVALID_TYPE,
                    f"Allocation at index {index} must be an object; received {_json_type(item)}.",
                )
                continue
            item_fields = ("candidate_id", "votes")
            for field in item_fields:
                if field not in item:
                    add(
                        ValidationErrorCode.MISSING_FIELD,
                        f"Allocation at index {index} is missing required field: {field}.",
                    )
            for field in item:
                if field not in item_fields:
                    add(
                        ValidationErrorCode.EXTRA_FIELD,
                        f"Allocation at index {index} has unexpected field: {field}.",
                    )

            candidate_raw = item.get("candidate_id")
            candidate: CandidateId | None = None
            if "candidate_id" in item:
                if not isinstance(candidate_raw, str):
                    add(
                        ValidationErrorCode.INVALID_TYPE,
                        "Allocation at index "
                        f"{index} field candidate_id must be a string; received "
                        f"{_json_type(candidate_raw)}.",
                    )
                else:
                    candidate = CandidateId(candidate_raw)
                    if candidate not in known_set:
                        add(
                            ValidationErrorCode.UNKNOWN_CANDIDATE,
                            f"Unknown candidate ID at allocation index {index}: {candidate}.",
                        )
                    elif candidate not in active_set:
                        add(
                            ValidationErrorCode.INACTIVE_CANDIDATE,
                            f"Inactive candidate ID at allocation index {index}: {candidate}.",
                        )
                    if candidate in encountered:
                        add(
                            ValidationErrorCode.DUPLICATE_CANDIDATE,
                            f"Duplicate candidate ID at allocation index {index}: {candidate}.",
                        )
                    encountered.add(candidate)

            votes_raw = item.get("votes")
            votes: int | None = None
            if "votes" in item:
                if type(votes_raw) is not int:
                    add(
                        ValidationErrorCode.NON_INTEGER_VOTES,
                        f"Votes for allocation at index {index} must be a JSON integer; "
                        f"received {_json_type(votes_raw)}.",
                    )
                else:
                    votes = votes_raw
                    submitted_votes.append(votes)
                    if votes < 0:
                        add(
                            ValidationErrorCode.NEGATIVE_VOTES,
                            f"Votes for allocation at index {index} must be non-negative; received {votes}.",
                        )
            if (
                candidate is not None
                and candidate in active_set
                and candidate not in parsed
                and votes is not None
                and votes >= 0
            ):
                parsed[candidate] = votes

    allocations = tuple(
        (candidate, parsed[candidate])
        for candidate in active
        if parsed.get(candidate, 0) > 0
    )
    # Cost is computed from every well-typed, non-negative submitted entry so a
    # ballot can report budget excess alongside duplicate/unknown-ID failures.
    cost = sum(votes**2 for votes in submitted_votes)
    if cost > budget:
        add(
            ValidationErrorCode.BUDGET_EXCEEDED,
            f"Ballot quadratic cost is {cost}, which exceeds the budget of {budget}.",
        )
    if failures:
        return _finish(failures)
    assert isinstance(reason, str)
    return ParsedBallot(rationale=reason, allocations=allocations)
