"""Tests for strict ballot parsing and quadratic budget validation."""

from __future__ import annotations

import json
import unittest

from quadratic_voting.experiment.ballots import (
    ParsedBallot,
    ValidationFailure,
    ballot_cost,
    parse_and_validate_ballot,
)
from quadratic_voting.experiment.types import CandidateId, ValidationErrorCode


class BallotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.active = (CandidateId("C1"), CandidateId("C2"))

    def failures(self, raw: str, **kwargs: object) -> tuple[ValidationFailure, ...]:
        result = parse_and_validate_ballot(raw, self.active, 100, **kwargs)  # type: ignore[arg-type]
        self.assertIsInstance(result, tuple)
        return result  # type: ignore[return-value]

    def test_valid_ballot_is_ordered_and_zero_is_canonicalized(self) -> None:
        result = parse_and_validate_ballot(
            '{"rationale":"why","allocations":['
            '{"candidate_id":"C2","votes":3},'
            '{"candidate_id":"C1","votes":0}]}',
            self.active,
            100,
        )
        self.assertEqual(result, ParsedBallot("why", ((CandidateId("C2"), 3),)))
        self.assertEqual(ballot_cost(result.allocations), 9)  # type: ignore[union-attr]
        self.assertEqual(
            parse_and_validate_ballot(
                '{"rationale":"abstain","allocations":[]}', self.active, 100
            ),
            ParsedBallot("abstain", ()),
        )

    def test_rationale_preserves_exact_text_but_rejects_unicode_whitespace_only(
        self,
    ) -> None:
        exact = "  reason\u2003"
        self.assertEqual(
            parse_and_validate_ballot(
                json.dumps({"rationale": exact, "allocations": []}), self.active, 100
            ),
            ParsedBallot(exact, ()),
        )
        failures = self.failures(
            json.dumps({"rationale": " \t\n\u2003", "allocations": []})
        )
        self.assertEqual(failures[0].code, ValidationErrorCode.EMPTY_RATIONALE)

    def test_malformed_missing_extra_and_invalid_types(self) -> None:
        self.assertEqual(
            self.failures("{")[0].message,
            "Malformed JSON: Expecting property name enclosed in double quotes.",
        )
        errors = self.failures('{"rationale":2,"extra":1}')
        self.assertEqual(
            [(error.code, error.ordinal, error.message) for error in errors],
            [
                (
                    ValidationErrorCode.MISSING_FIELD,
                    0,
                    "Missing required field: allocations.",
                ),
                (
                    ValidationErrorCode.EXTRA_FIELD,
                    1,
                    "Unexpected top-level field: extra.",
                ),
                (
                    ValidationErrorCode.INVALID_TYPE,
                    2,
                    "Field rationale must be a string; received integer.",
                ),
            ],
        )
        self.assertEqual(
            self.failures("[]")[0].message,
            "Top-level ballot must be a JSON object; received array.",
        )

    def test_candidate_errors_include_unknown_inactive_and_duplicate(self) -> None:
        raw = json.dumps(
            {
                "rationale": "x",
                "allocations": [
                    {"candidate_id": "C1", "votes": 0},
                    {"candidate_id": "C1", "votes": 2},
                    {"candidate_id": "C3", "votes": 1},
                    {"candidate_id": "C4", "votes": 1},
                ],
            }
        )
        errors = self.failures(
            raw, known=(CandidateId("C1"), CandidateId("C2"), CandidateId("C3"))
        )
        self.assertEqual(
            [error.code for error in errors],
            [
                ValidationErrorCode.UNKNOWN_CANDIDATE,
                ValidationErrorCode.INACTIVE_CANDIDATE,
                ValidationErrorCode.DUPLICATE_CANDIDATE,
            ],
        )
        self.assertEqual([error.ordinal for error in errors], [0, 1, 2])

    def test_non_integer_classes_negative_and_budget_are_exact(self) -> None:
        for value, received in [(True, "boolean"), (3.0, "number"), ("3", "string")]:
            with self.subTest(value=value):
                errors = self.failures(
                    json.dumps(
                        {
                            "rationale": "x",
                            "allocations": [{"candidate_id": "C1", "votes": value}],
                        }
                    )
                )
                self.assertEqual(errors[0].code, ValidationErrorCode.NON_INTEGER_VOTES)
                self.assertEqual(
                    errors[0].message,
                    "Votes for allocation at index 0 must be a JSON integer; "
                    f"received {received}.",
                )
        self.assertEqual(
            self.failures(
                '{"rationale":"x","allocations":[{"candidate_id":"C1","votes":-1}]}'
            )[0].code,
            ValidationErrorCode.NEGATIVE_VOTES,
        )
        error = self.failures(
            '{"rationale":"x","allocations":['
            '{"candidate_id":"C1","votes":8},'
            '{"candidate_id":"C2","votes":7}]}',
        )[0]
        self.assertEqual(error.code, ValidationErrorCode.BUDGET_EXCEEDED)
        self.assertEqual(
            error.message,
            "Ballot quadratic cost is 113, which exceeds the budget of 100.",
        )

    def test_nested_missing_extra_and_invalid_item_are_reported(self) -> None:
        errors = self.failures(
            '{"rationale":"x","allocations":[{"candidate_id":"C1","x":1},3]}'
        )
        self.assertEqual(
            [error.code for error in errors],
            [
                ValidationErrorCode.MISSING_FIELD,
                ValidationErrorCode.EXTRA_FIELD,
                ValidationErrorCode.INVALID_TYPE,
            ],
        )

    def test_budget_is_computed_even_when_candidate_validation_also_fails(self) -> None:
        errors = self.failures(
            '{"rationale":"x","allocations":[{"candidate_id":"C9","votes":11}]}'
        )
        self.assertEqual(
            [error.code for error in errors],
            [
                ValidationErrorCode.UNKNOWN_CANDIDATE,
                ValidationErrorCode.BUDGET_EXCEEDED,
            ],
        )
        self.assertEqual(
            errors[1].message,
            "Ballot quadratic cost is 121, which exceeds the budget of 100.",
        )


if __name__ == "__main__":
    unittest.main()
