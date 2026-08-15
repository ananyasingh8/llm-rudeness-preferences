"""Hand-calculated tests for the versioned executable analysis contract."""

from __future__ import annotations

import unittest
from collections.abc import Sequence
from typing import cast

from quadratic_voting.experiment.analysis import (
    AnalysisInputs,
    _agreement_cells,
    analyze,
    midranks,
    paired_cluster_bootstrap,
    spearman_with_ties,
)


class RankAndBootstrapTests(unittest.TestCase):
    def test_tied_midranks_and_spearman(self) -> None:
        self.assertEqual(midranks((4, 4, 1)), (2.5, 2.5, 1.0))
        rho = spearman_with_ties((2, 2, -2), (3, 0, 1))
        self.assertIsNotNone(rho)
        self.assertAlmostEqual(cast(float, rho), 0.0)
        self.assertIsNone(spearman_with_ties((1, 1), (0, 2)))

    def test_paired_voter_bootstrap_is_fixed_and_reproducible(self) -> None:
        first, first_rows = paired_cluster_bootstrap(
            {0: -1.0, 1: 3.0}, contrast_id="fixture", replicates=100
        )
        second, second_rows = paired_cluster_bootstrap(
            {1: 3.0, 0: -1.0}, contrast_id="fixture", replicates=100
        )
        self.assertEqual(first, second)
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(first["voter_population"], [0, 1])
        self.assertEqual(first["estimate"], 1.0)
        self.assertEqual(first["bootstrap_replicates"], 100)
        self.assertTrue(
            all(
                set(cast(Sequence[int], row["sampled_voter_indices"])).issubset({0, 1})
                for row in first_rows
            )
        )


class AnalysisSemanticsTests(unittest.TestCase):
    def test_every_null_reason_including_constant_action_has_strict_precedence(
        self,
    ) -> None:
        rows: list[dict[str, object]] = []
        cases = (
            (0, "invalid-missing", "abstained", ((0, 0), (1, 1))),
            (1, "accepted", "abstained", ((0, 0), (1, 1))),
            (2, "accepted", "accepted", ((0, 0),)),
            (3, "accepted", "accepted", ((0, 0), (0, 1))),
            (4, "accepted", "accepted", ((0, 1), (1, 1))),
        )
        for voter, statement_status, ballot_status, values in cases:
            for position, (rating, action) in enumerate(values):
                rows.append(
                    {
                        "matched_set_id": "m",
                        "run_id": "r",
                        "arm": "statement-then-action",
                        "regime": "support",
                        "round_index": 1,
                        "voter_index": voter,
                        "candidate_id": f"c{position}",
                        "rating_code": None
                        if statement_status != "accepted"
                        else rating,
                        "signed_action": None
                        if ballot_status != "accepted"
                        else action,
                        "rudeness_label": "non_rude",
                        "statement_status": statement_status,
                        "ballot_status": ballot_status,
                        "active_pool_size": len(values),
                        "intersection_pool_size": len(values),
                        "label_policy_version": "v1",
                        "label_policy_id": "policy",
                        "label_policy_sha256": "p" * 64,
                    }
                )
        overall = sorted(
            (row for row in _agreement_cells(rows) if row["scope"] == "overall"),
            key=lambda row: int(cast(int, row["voter_index"])),
        )
        self.assertEqual(
            [row["null_reason"] for row in overall],
            [
                "MISSING_STATEMENT",
                "ABSTAINED_BALLOT",
                "N_LT_2",
                "CONSTANT_RATING",
                "CONSTANT_ACTION",
            ],
        )

    def test_null_reason_precedence_and_label_denominators(self) -> None:
        inputs = AnalysisInputs(
            runs=(
                {
                    "run_id": "run",
                    "matched_set_id": "matched",
                    "arm": "statement-then-action",
                    "regime": "opposition",
                    "status": "complete",
                },
            ),
            voters=(
                {"voter_id": "v0", "run_id": "run", "voter_index": 0},
                {"voter_id": "v1", "run_id": "run", "voter_index": 1},
            ),
            rounds=({"round_id": "round", "run_id": "run", "round_index": 1},),
            round_candidate_rows=tuple(
                {
                    "round_id": "round",
                    "run_id": "run",
                    "round_index": 1,
                    "candidate_id": candidate,
                    "sample_position": position,
                }
                for position, candidate in enumerate(("c1", "c2", "c3"))
            ),
            candidate_rows=tuple(
                {
                    "candidate_id": candidate,
                    "rudeness_label": "rude" if candidate == "c1" else "non_rude",
                    "label_policy_id": "policy",
                    "label_policy_name": "fixture-policy",
                    "label_policy_version": "v1",
                    "label_policy_sha256": "p" * 64,
                    "presentation_id": f"presentation-{candidate}",
                    "template_id": "template",
                    "presentation_template_name": "card",
                    "presentation_template_version": "v1",
                    "presentation_template_sha256": "t" * 64,
                    "presentation_sha256": candidate * 32,
                }
                for candidate in ("c1", "c2", "c3")
            ),
            source_annotation_rows=tuple(
                {
                    "candidate_id": candidate,
                    "annotation_index": 0,
                    "annotator_hash": "a" * 64,
                    "source_label": "rudeness",
                    "source_value": "rude" if candidate == "c1" else "non_rude",
                }
                for candidate in ("c1", "c2", "c3")
            ),
            run_definition_rows=(
                {
                    "run_id": "run",
                    "label_policy_id": "policy",
                    "presentation_template_id": "template",
                },
            ),
            turns=(
                {
                    "turn_id": "s0t",
                    "round_id": "round",
                    "voter_id": "v0",
                    "kind": "statement",
                },
                {
                    "turn_id": "b0t",
                    "round_id": "round",
                    "voter_id": "v0",
                    "kind": "ballot",
                },
                {
                    "turn_id": "s1t",
                    "round_id": "round",
                    "voter_id": "v1",
                    "kind": "statement",
                },
                {
                    "turn_id": "b1t",
                    "round_id": "round",
                    "voter_id": "v1",
                    "kind": "ballot",
                },
            ),
            calls=(),
            validation_failures=(),
            runtime_failures=(),
            ballots=(
                {"ballot_id": "b0", "turn_id": "b0t", "status": "abstained"},
                {"ballot_id": "b1", "turn_id": "b1t", "status": "accepted"},
            ),
            allocations=(
                {"ballot_id": "b1", "candidate_id": "c1", "votes": 1},
                {"ballot_id": "b1", "candidate_id": "c2", "votes": 2},
            ),
            statements=(
                {"statement_id": "s0", "turn_id": "s0t", "status": "invalid-missing"},
                {"statement_id": "s1", "turn_id": "s1t", "status": "accepted"},
            ),
            statement_items=tuple(
                {
                    "statement_id": "s1",
                    "candidate_id": candidate,
                    "rating": "neutral",
                    "text": candidate,
                }
                for candidate in ("c1", "c2", "c3")
            ),
            outcomes=(),
        )
        output = analyze(inputs, bootstrap_replicates=10)
        overall = sorted(
            (row for row in output.agreement_cells if row["scope"] == "overall"),
            key=lambda row: int(cast(int, row["voter_index"])),
        )
        self.assertEqual(
            [row["null_reason"] for row in overall],
            ["MISSING_STATEMENT", "CONSTANT_RATING"],
        )
        missing_candidates = [
            row for row in output.candidate_rows if row["voter_index"] == 0
        ]
        self.assertTrue(
            all(
                row["rating_code"] is None
                and row["raw_votes"] is None
                and row["signed_action"] is None
                for row in missing_candidates
            )
        )
        rude = [
            row
            for row in output.agreement_cells
            if row["scope"] == "rudeness-label"
            and row["rudeness_label"] == "rude"
            and row["voter_index"] == 1
        ]
        self.assertEqual(rude[0]["null_reason"], "N_LT_2")
        actions = {
            row["candidate_id"]: row["signed_action"]
            for row in output.candidate_rows
            if row["voter_index"] == 1
        }
        self.assertEqual(actions, {"c1": -1, "c2": -2, "c3": 0})

    def test_round_one_is_causal_and_later_intersection_is_descriptive(self) -> None:
        runs = (
            {
                "run_id": "a",
                "matched_set_id": "m",
                "arm": "action-only",
                "regime": "support",
                "status": "complete",
            },
            {
                "run_id": "s",
                "matched_set_id": "m",
                "arm": "statement-then-action",
                "regime": "support",
                "status": "complete",
            },
        )
        rounds = tuple(
            {"round_id": f"{run}{index}", "run_id": run, "round_index": index}
            for run in ("a", "s")
            for index in (1, 2)
        )
        pool_members = {
            "a1": ("c1", "c2", "c3"),
            "s1": ("c1", "c2", "c3"),
            "a2": ("c1", "c2"),
            "s2": ("c2", "c3"),
        }
        pools = tuple(
            {
                "round_id": round_id,
                "run_id": round_id[0],
                "round_index": int(round_id[1]),
                "candidate_id": candidate,
                "sample_position": position,
            }
            for round_id, candidates in pool_members.items()
            for position, candidate in enumerate(candidates)
        )
        turns = []
        ballots = []
        allocations = []
        statements = []
        items = []
        for run in ("a", "s"):
            for index in (1, 2):
                ballot_turn = f"b{run}{index}t"
                turns.append(
                    {
                        "turn_id": ballot_turn,
                        "round_id": f"{run}{index}",
                        "voter_id": f"v{run}",
                        "kind": "ballot",
                    }
                )
                ballots.append(
                    {
                        "ballot_id": f"b{run}{index}",
                        "turn_id": ballot_turn,
                        "status": "accepted",
                    }
                )
                for candidate in pool_members[f"{run}{index}"]:
                    allocations.append(
                        {
                            "ballot_id": f"b{run}{index}",
                            "candidate_id": candidate,
                            "votes": 1 if run == "a" else 2,
                        }
                    )
                if run == "s":
                    statement_turn = f"s{index}t"
                    turns.append(
                        {
                            "turn_id": statement_turn,
                            "round_id": f"s{index}",
                            "voter_id": "vs",
                            "kind": "statement",
                        }
                    )
                    statements.append(
                        {
                            "statement_id": f"s{index}",
                            "turn_id": statement_turn,
                            "status": "accepted",
                        }
                    )
                    for offset, candidate in enumerate(pool_members[f"s{index}"]):
                        rating = (
                            "strongly prefer not to continue",
                            "neutral",
                            "strongly prefer to continue",
                        )[offset]
                        items.append(
                            {
                                "statement_id": f"s{index}",
                                "candidate_id": candidate,
                                "rating": rating,
                                "text": candidate,
                            }
                        )
        output = analyze(
            AnalysisInputs(
                runs=runs,
                voters=(
                    {"voter_id": "va", "run_id": "a", "voter_index": 0},
                    {"voter_id": "vs", "run_id": "s", "voter_index": 0},
                ),
                rounds=rounds,
                round_candidate_rows=pools,
                candidate_rows=tuple(
                    {
                        "candidate_id": candidate,
                        "rudeness_label": "non_rude",
                        "label_policy_id": "policy",
                        "label_policy_name": "fixture-policy",
                        "label_policy_version": "v1",
                        "label_policy_sha256": "p" * 64,
                        "presentation_id": f"presentation-{candidate}",
                        "template_id": "template",
                        "presentation_template_name": "card",
                        "presentation_template_version": "v1",
                        "presentation_template_sha256": "t" * 64,
                        "presentation_sha256": candidate * 32,
                    }
                    for candidate in ("c1", "c2", "c3")
                ),
                source_annotation_rows=(),
                run_definition_rows=tuple(
                    {
                        "run_id": run,
                        "label_policy_id": "policy",
                        "presentation_template_id": "template",
                    }
                    for run in ("a", "s")
                ),
                turns=tuple(turns),
                calls=(),
                validation_failures=(),
                runtime_failures=(),
                ballots=tuple(ballots),
                allocations=tuple(allocations),
                statements=tuple(statements),
                statement_items=tuple(items),
                outcomes=(),
            ),
            bootstrap_replicates=20,
        )
        action = sorted(
            (row for row in output.contrasts if row["metric"] == "mean-signed-action"),
            key=lambda row: int(cast(int, row["round_index"])),
        )
        self.assertEqual(action[0]["claim_kind"], "round-1-causal-order")
        self.assertTrue(action[0]["causal"])
        self.assertEqual(action[0]["intersection_pool_size"], 3)
        self.assertEqual(action[1]["claim_kind"], "descriptive-post-treatment")
        self.assertTrue(action[1]["post_treatment"])
        self.assertEqual(action[1]["original_pool_size"], 2)
        self.assertEqual(action[1]["intersection_pool_size"], 1)


if __name__ == "__main__":
    unittest.main()
