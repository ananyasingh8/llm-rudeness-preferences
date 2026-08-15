"""Tests for versioned, matched experiment RNG utilities."""

from __future__ import annotations

import hashlib
import unittest

from quadratic_voting.experiment.seeds import (
    SEED_ALGORITHM_VERSION,
    SeededDraw,
    call_seed,
    derive_seed,
    support_removal_draw,
    tie_break_draw,
    voter_permutation_draw,
)
from quadratic_voting.experiment.types import (
    CandidateId,
    ElicitationArm,
    RngAlgorithm,
    SeedDomain,
    TurnKind,
    VotingRegime,
)


class SeedTests(unittest.TestCase):
    def test_seed_derivation_is_versioned_domain_separated_and_unambiguous(
        self,
    ) -> None:
        seed = derive_seed(7, SeedDomain.GENERATION, "run", 1, "ballot")
        self.assertEqual(
            seed, derive_seed(7, SeedDomain.GENERATION, "run", 1, "ballot")
        )
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 1 << 64)
        self.assertNotEqual(
            derive_seed(7, SeedDomain.GENERATION, "a", 1),
            derive_seed(7, SeedDomain.GENERATION, "a1"),
        )
        self.assertNotEqual(
            derive_seed(7, SeedDomain.GENERATION, "same", 1),
            derive_seed(7, SeedDomain.TIE_BREAK, "same", 1),
        )

    def test_seed_vector_pins_exact_qv_seed_v1_wire_format(self) -> None:
        def lp(value: bytes) -> bytes:
            return len(value).to_bytes(4, "big") + value

        payload = b"".join(
            (
                lp(b"qv-seed/v1"),
                lp(b"generation"),
                lp((42).to_bytes(8, "big")),
                lp(b"action-only"),
                lp((3).to_bytes(8, "big")),
            )
        )
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        self.assertEqual(
            derive_seed(42, SeedDomain.GENERATION, "action-only", 3), expected
        )
        self.assertEqual(SEED_ALGORITHM_VERSION, "qv-seed/v1")

    def test_call_seed_uses_all_stable_matched_coordinates(self) -> None:
        seed = call_seed(
            42,
            ElicitationArm.ACTION_ONLY,
            VotingRegime.SUPPORT,
            3,
            4,
            TurnKind.BALLOT,
            1,
        )
        self.assertEqual(
            seed,
            call_seed(
                42,
                ElicitationArm.ACTION_ONLY,
                VotingRegime.SUPPORT,
                3,
                4,
                TurnKind.BALLOT,
                1,
            ),
        )
        self.assertNotEqual(
            seed,
            call_seed(
                42,
                ElicitationArm.ACTION_THEN_STATEMENT,
                VotingRegime.SUPPORT,
                3,
                4,
                TurnKind.BALLOT,
                1,
            ),
        )

    def test_voter_permutation_is_invariant_across_arm_and_regime(self) -> None:
        expected = voter_permutation_draw(99, 2)
        self.assertEqual(expected.stream_name, "voter-permutation/99/2")
        for arm in ElicitationArm:
            for regime in VotingRegime:
                with self.subTest(arm=arm, regime=regime):
                    # The helper accepts neither condition coordinate by design.
                    self.assertEqual(voter_permutation_draw(99, 2), expected)

    def test_condition_specific_draw_helpers_are_domain_separated(self) -> None:
        tie = tie_break_draw(9, ElicitationArm.ACTION_ONLY, VotingRegime.SUPPORT, 1)
        removal = support_removal_draw(
            9, ElicitationArm.ACTION_ONLY, VotingRegime.SUPPORT, 1
        )
        self.assertEqual(tie.stream_name, "tie-break/9/action-only/support/1")
        self.assertEqual(removal.stream_name, "support-removal/9/action-only/support/1")
        self.assertNotEqual(tie.seed, removal.seed)

    def test_seeded_draw_preserves_explicit_population_order(self) -> None:
        candidates = [CandidateId("C3"), CandidateId("C1"), CandidateId("C2")]
        draw = SeededDraw(stream_name="test", seed=17)
        selection = draw.choose(candidates)
        permutation = draw.permutation(candidates)

        self.assertEqual(selection.eligible, ("C3", "C1", "C2"))
        self.assertGreaterEqual(selection.selected_index, 0)
        self.assertLess(selection.selected_index, len(selection.eligible))
        self.assertEqual(
            selection.selected, selection.eligible[selection.selected_index]
        )
        self.assertEqual(selection.stream_name, "test")
        self.assertEqual(selection.seed, 17)
        self.assertEqual(selection.algorithm, RngAlgorithm.PYRANDOM_RANDRANGE_V1)

        self.assertEqual(permutation.eligible, ("C3", "C1", "C2"))
        self.assertCountEqual(permutation.permutation, candidates)
        self.assertEqual(permutation.stream_name, "test")
        self.assertEqual(permutation.seed, 17)
        self.assertEqual(permutation.algorithm, RngAlgorithm.FISHER_YATES_PYRANDOM_V1)

    def test_draws_reject_empty_populations_actionably(self) -> None:
        draw = SeededDraw(stream_name="empty", seed=1)
        with self.assertRaisesRegex(ValueError, "SeededDraw.choose.*retry"):
            draw.choose(())
        with self.assertRaisesRegex(ValueError, "SeededDraw.permutation.*retry"):
            draw.permutation(())


if __name__ == "__main__":
    unittest.main()
