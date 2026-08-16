"""Tests for strict statement parsing."""

from __future__ import annotations

import json
import unittest

from quadratic_voting.experiment.ballots import ValidationFailure
from quadratic_voting.experiment.statements import (
    ParsedStatement,
    parse_and_validate_statement,
)
from quadratic_voting.experiment.types import (
    CandidateId,
    LikertRating,
    ValidationErrorCode,
)


class StatementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.active = (CandidateId("C1"), CandidateId("C2"))

    def failures(self, raw: str) -> tuple[ValidationFailure, ...]:
        result = parse_and_validate_statement(raw, self.active)
        self.assertIsInstance(result, tuple)
        return result  # type: ignore[return-value]

    def test_all_exact_labels_are_accepted_and_normalized_variants_rejected(
        self,
    ) -> None:
        for rating in LikertRating:
            with self.subTest(rating=rating):
                raw = json.dumps(
                    {
                        "statements": [
                            {
                                "candidate_id": "C2",
                                "rating": rating.value,
                                "reason": "two",
                            },
                            {
                                "candidate_id": "C1",
                                "rating": rating.value,
                                "reason": "one",
                            },
                        ]
                    }
                )
                self.assertEqual(
                    parse_and_validate_statement(raw, self.active),
                    ParsedStatement(
                        (
                            (CandidateId("C1"), rating, "one"),
                            (CandidateId("C2"), rating, "two"),
                        )
                    ),
                )
                altered = raw.replace(rating.value, rating.value.upper(), 1)
                failures = self.failures(altered)
                self.assertIn(
                    ValidationErrorCode.UNKNOWN_RATING,
                    [failure.code for failure in failures],
                )

    def test_strict_candidate_set_reports_unknown_duplicate_and_missing(self) -> None:
        raw = json.dumps(
            {
                "statements": [
                    {"candidate_id": "C1", "rating": "neutral", "reason": "a"},
                    {"candidate_id": "C1", "rating": "neutral", "reason": "b"},
                    {"candidate_id": "C3", "rating": "neutral", "reason": "c"},
                ]
            }
        )
        errors = self.failures(raw)
        self.assertEqual(
            [error.code for error in errors],
            [
                ValidationErrorCode.UNKNOWN_CANDIDATE,
                ValidationErrorCode.DUPLICATE_CANDIDATE,
                ValidationErrorCode.MISSING_CANDIDATE,
            ],
        )
        self.assertEqual(
            errors[-1].message, "Missing statement for active candidate: C2."
        )

    def test_unknown_rating_empty_statement_and_types_are_rejected(self) -> None:
        raw = json.dumps(
            {
                "statements": [
                    {
                        "candidate_id": "C1",
                        "rating": "sort of neutral",
                        "reason": " ",
                    },
                    {"candidate_id": "C2", "rating": 3, "reason": 4},
                ]
            }
        )
        errors = self.failures(raw)
        self.assertEqual(
            [error.code for error in errors],
            [
                ValidationErrorCode.INVALID_TYPE,
                ValidationErrorCode.INVALID_TYPE,
                ValidationErrorCode.UNKNOWN_RATING,
                ValidationErrorCode.EMPTY_STATEMENT,
            ],
        )
        self.assertEqual([error.ordinal for error in errors], [0, 1, 2, 3])

    def test_structural_errors_are_strict(self) -> None:
        errors = self.failures('{"extra":1}')
        self.assertEqual(
            [error.code for error in errors],
            [
                ValidationErrorCode.MISSING_FIELD,
                ValidationErrorCode.EXTRA_FIELD,
                ValidationErrorCode.MISSING_CANDIDATE,
                ValidationErrorCode.MISSING_CANDIDATE,
            ],
        )
        self.assertEqual(self.failures("{")[0].code, ValidationErrorCode.MALFORMED_JSON)


if __name__ == "__main__":
    unittest.main()
