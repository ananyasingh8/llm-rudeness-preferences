"""Strict parsing for per-candidate stated preferences."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from quadratic_voting.experiment.ballots import ValidationFailure
from quadratic_voting.experiment.types import (
    CandidateId,
    LikertRating,
    ValidationErrorCode,
)


@dataclass(frozen=True, slots=True)
class ParsedStatement:
    items: tuple[tuple[CandidateId, LikertRating, str], ...]


@dataclass(frozen=True, slots=True)
class _Failure:
    code: ValidationErrorCode
    position: int
    message: str


_CODE_ORDER = {code: index for index, code in enumerate(ValidationErrorCode)}


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, list):
        return "array"
    return "object"


def parse_and_validate_statement(
    raw: str, active: tuple[CandidateId, ...]
) -> ParsedStatement | tuple[ValidationFailure, ...]:
    """Require exactly one valid statement for every active candidate."""
    try:
        value: Any = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as error:
        detail = error.msg if isinstance(error, json.JSONDecodeError) else str(error)
        return (
            ValidationFailure(
                ValidationErrorCode.MALFORMED_JSON, 0, f"Malformed JSON: {detail}."
            ),
        )
    if not isinstance(value, dict):
        return (
            ValidationFailure(
                ValidationErrorCode.INVALID_TYPE,
                0,
                f"Top-level statement response must be a JSON object; received {_json_type(value)}.",
            ),
        )

    failures: list[_Failure] = []
    position = 0

    def add(code: ValidationErrorCode, message: str) -> None:
        nonlocal position
        failures.append(_Failure(code, position, message))
        position += 1

    if "statements" not in value:
        add(ValidationErrorCode.MISSING_FIELD, "Missing required field: statements.")
    for field in value:
        if field != "statements":
            add(
                ValidationErrorCode.EXTRA_FIELD, f"Unexpected top-level field: {field}."
            )
    raw_items = value.get("statements")
    if "statements" in value and not isinstance(raw_items, list):
        add(
            ValidationErrorCode.INVALID_TYPE,
            f"Field statements must be an array; received {_json_type(raw_items)}.",
        )

    active_set = set(active)
    encountered: set[CandidateId] = set()
    parsed: dict[CandidateId, tuple[LikertRating, str]] = {}
    if isinstance(raw_items, list):
        for index, item in enumerate(raw_items):
            if not isinstance(item, dict):
                add(
                    ValidationErrorCode.INVALID_TYPE,
                    f"Statement at index {index} must be an object; received {_json_type(item)}.",
                )
                continue
            item_fields = ("candidate_id", "rating", "statement")
            for field in item_fields:
                if field not in item:
                    add(
                        ValidationErrorCode.MISSING_FIELD,
                        f"Statement at index {index} is missing required field: {field}.",
                    )
            for field in item:
                if field not in item_fields:
                    add(
                        ValidationErrorCode.EXTRA_FIELD,
                        f"Statement at index {index} has unexpected field: {field}.",
                    )

            candidate_raw = item.get("candidate_id")
            candidate: CandidateId | None = None
            if "candidate_id" in item:
                if not isinstance(candidate_raw, str):
                    add(
                        ValidationErrorCode.INVALID_TYPE,
                        f"Statement at index {index} field candidate_id must be a string; "
                        f"received {_json_type(candidate_raw)}.",
                    )
                else:
                    candidate = CandidateId(candidate_raw)
                    if candidate not in active_set:
                        add(
                            ValidationErrorCode.UNKNOWN_CANDIDATE,
                            f"Unknown candidate ID at statement index {index}: {candidate}.",
                        )
                    if candidate in encountered:
                        add(
                            ValidationErrorCode.DUPLICATE_CANDIDATE,
                            f"Duplicate candidate ID at statement index {index}: {candidate}.",
                        )
                    encountered.add(candidate)

            rating_raw = item.get("rating")
            rating: LikertRating | None = None
            if "rating" in item:
                if not isinstance(rating_raw, str):
                    add(
                        ValidationErrorCode.INVALID_TYPE,
                        f"Rating at statement index {index} must be a string; received "
                        f"{_json_type(rating_raw)}.",
                    )
                else:
                    try:
                        rating = LikertRating(rating_raw)
                    except ValueError:
                        add(
                            ValidationErrorCode.UNKNOWN_RATING,
                            f"Unknown rating at statement index {index}: {rating_raw!r}.",
                        )

            statement_raw = item.get("statement")
            statement: str | None = None
            if "statement" in item:
                if not isinstance(statement_raw, str):
                    add(
                        ValidationErrorCode.INVALID_TYPE,
                        f"Statement text at index {index} must be a string; received "
                        f"{_json_type(statement_raw)}.",
                    )
                else:
                    statement = statement_raw
                    if not any(not char.isspace() for char in statement):
                        add(
                            ValidationErrorCode.EMPTY_STATEMENT,
                            f"Statement text at index {index} must not be empty or whitespace-only.",
                        )
            if (
                candidate is not None
                and candidate in active_set
                and candidate not in parsed
                and rating is not None
                and statement is not None
                and any(not char.isspace() for char in statement)
            ):
                parsed[candidate] = (rating, statement)

    for candidate in active:
        if candidate not in encountered:
            add(
                ValidationErrorCode.MISSING_CANDIDATE,
                f"Missing statement for active candidate: {candidate}.",
            )
    if failures:
        ordered = sorted(
            failures, key=lambda item: (_CODE_ORDER[item.code], item.position)
        )
        return tuple(
            ValidationFailure(item.code, ordinal, item.message)
            for ordinal, item in enumerate(ordered)
        )
    return ParsedStatement(
        items=tuple((candidate, *parsed[candidate]) for candidate in active)
    )
