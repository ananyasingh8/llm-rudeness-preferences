"""Tests for deterministic pure round aggregation."""

from __future__ import annotations

import unittest

from quadratic_voting.experiment.engine import aggregate_round
from quadratic_voting.experiment.seeds import SeededDraw
from quadratic_voting.experiment.types import CandidateId, VotingRegime


class EngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.active = tuple(CandidateId(value) for value in ("C1", "C2", "C3"))

    def test_support_single_max_and_removal_excludes_protected(self) -> None:
        outcome = aggregate_round(
            VotingRegime.SUPPORT,
            self.active,
            ({CandidateId("C1"): 2, CandidateId("C2"): 1}, {CandidateId("C1"): 3}),
            SeededDraw("tie", 1),
            SeededDraw("removal", 2),
        )
        self.assertEqual(dict(outcome.result.totals), {"C1": 5, "C2": 1, "C3": 0})
        self.assertEqual(outcome.result.protected, "C1")
        self.assertIsNone(outcome.tie_draw)
        self.assertEqual(outcome.removal_draw.eligible, ("C2", "C3"))  # type: ignore[union-attr]
        self.assertNotEqual(outcome.result.removed, "C1")

    def test_tie_uses_ordered_maxima_and_same_seeds_are_deterministic(self) -> None:
        args = (
            VotingRegime.SUPPORT,
            self.active,
            ({CandidateId("C1"): 2, CandidateId("C3"): 2},),
            SeededDraw("tie", 9),
            SeededDraw("remove", 10),
        )
        first = aggregate_round(*args)
        second = aggregate_round(*args)
        self.assertEqual(first, second)
        self.assertEqual(first.tie_draw.eligible, ("C1", "C3"))  # type: ignore[union-attr]
        self.assertEqual(first.result.tie_among, frozenset(("C1", "C3")))

    def test_opposition_removes_max_and_never_uses_removal_draw(self) -> None:
        outcome = aggregate_round(
            VotingRegime.OPPOSITION,
            self.active,
            ({CandidateId("C2"): 4},),
            SeededDraw("tie", 1),
            SeededDraw("unused", 1),
        )
        self.assertEqual(outcome.result.removed, "C2")
        self.assertIsNone(outcome.result.protected)
        self.assertIsNone(outcome.removal_draw)

    def test_support_rejects_impossible_removal_population_actionably(self) -> None:
        with self.assertRaisesRegex(ValueError, "sole active.*terminal result"):
            aggregate_round(
                VotingRegime.SUPPORT,
                (CandidateId("C1"),),
                (),
                SeededDraw("tie", 1),
                SeededDraw("remove", 2),
            )


if __name__ == "__main__":
    unittest.main()
