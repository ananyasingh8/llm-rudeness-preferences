"""Focused production-path tests for no-GPU snapshot analytics."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from quadratic_voting.experiment.export import export_parquet
from quadratic_voting.experiment.snapshots import (
    build_snapshot_tables,
    select_snapshot_rounds,
)
from quadratic_voting.experiment.test_export import AnalysisFixture


class SnapshotTests(AnalysisFixture):
    def test_position_snapshots_include_endpoints_without_inventing_rounds(
        self,
    ) -> None:
        self.assertEqual(
            select_snapshot_rounds((2, 4, 9, 20, 33, 40)), (2, 4, 9, 33, 40)
        )
        self.assertEqual(select_snapshot_rounds((3, 8)), (3, 8))
        self.assertEqual(select_snapshot_rounds((7,), count=1), (7,))
        with self.assertRaisesRegex(ValueError, "at least 2"):
            select_snapshot_rounds((2, 4), count=1)

    def test_synthetic_rounds_have_per_candidate_and_rudeness_grains(self) -> None:
        """Hand-calculated two-voter values prove snapshot aggregate semantics."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export, out = root / "export", root / "out"
            export.mkdir()

            def write(name: str, rows: list[dict[str, object]]) -> None:
                pq.write_table(pa.Table.from_pylist(rows), export / f"{name}.parquet")

            rounds = (2, 4, 9)
            write(
                "runs",
                [
                    {
                        "run_id": "r",
                        "matched_set_id": "m",
                        "arm": "statement-then-action",
                        "regime": "support",
                        "credit_budget": 19,
                    }
                ],
            )
            write(
                "round_candidates",
                [
                    {
                        "run_id": "r",
                        "round_index": round_index,
                        "candidate_id": candidate,
                    }
                    for round_index in rounds
                    for candidate in ("c1", "c2")
                ],
            )
            write(
                "candidate_source_turns",
                [
                    {
                        "candidate_id": "c1",
                        "turn_index": 1,
                        "role": "user",
                        "text": "é",
                    },
                    {
                        "candidate_id": "c2",
                        "turn_index": 1,
                        "role": "user",
                        "text": "first",
                    },
                    {
                        "candidate_id": "c2",
                        "turn_index": 2,
                        "role": "agent",
                        "text": "two",
                    },
                ],
            )
            votes = {
                2: ((1, 0), (2, 1)),
                4: ((2, 1), (0, 2)),
                9: ((0, 2), (1, 0)),
            }
            write(
                "candidate_analysis",
                [
                    {
                        "matched_set_id": "m",
                        "run_id": "r",
                        "arm": "statement-then-action",
                        "regime": "support",
                        "round_index": round_index,
                        "voter_index": voter,
                        "candidate_id": candidate,
                        "rudeness_label": "rude"
                        if candidate == "c1"
                        else "ambiguous_tie",
                        "rating_code": 1 if candidate == "c1" else -1,
                        "statement_status": "accepted",
                        "ballot_status": "accepted",
                        "raw_votes": values[candidate_index],
                        "signed_action": values[candidate_index],
                    }
                    for round_index, per_voter in votes.items()
                    for voter, values in enumerate(per_voter)
                    for candidate_index, candidate in enumerate(("c1", "c2"))
                ],
            )
            build_snapshot_tables(export, out, snapshot_count=3)
            candidate = pq.read_table(
                out / "snapshot_candidate_summary.parquet"
            ).to_pylist()
            rudeness = pq.read_table(
                out / "snapshot_rudeness_summary.parquet"
            ).to_pylist()
            self.assertEqual(len(candidate), 6)  # one row for each round/candidate
            self.assertEqual(len(rudeness), 6)  # separate truthful rudeness facet
            row = next(
                row
                for row in candidate
                if row["snapshot_round"] == 4 and row["candidate_id"] == "c1"
            )
            self.assertEqual(row["matched_set_id"], "m")
            self.assertEqual(
                (row["sum_current_votes"], row["sum_current_credits"]), (2.0, 4.0)
            )
            self.assertEqual(
                (
                    row["sum_cumulative_before_votes"],
                    row["sum_cumulative_before_credits"],
                ),
                (3.0, 5.0),
            )
            self.assertEqual(
                (
                    row["sum_cumulative_through_votes"],
                    row["sum_cumulative_through_credits"],
                ),
                (5.0, 9.0),
            )
            voter = pq.read_table(out / "snapshot_voter_summary.parquet").to_pylist()
            voter_zero = next(
                row
                for row in voter
                if row["snapshot_round"] == 4 and row["voter_index"] == 0
            )
            self.assertEqual(
                (voter_zero["current_votes"], voter_zero["current_credits"]), (3, 5)
            )
            self.assertEqual(
                (
                    voter_zero["cumulative_before_votes"],
                    voter_zero["cumulative_before_credits"],
                ),
                (1, 1),
            )
            self.assertEqual(
                (
                    voter_zero["cumulative_through_votes"],
                    voter_zero["cumulative_through_credits"],
                ),
                (4, 6),
            )
            demographics = pq.read_table(
                out / "survivor_demographics.parquet"
            ).to_pylist()
            one_turn = next(row for row in demographics if row["candidate_id"] == "c1")
            self.assertEqual(
                (
                    one_turn["first_turn_length"],
                    one_turn["second_turn_length"],
                    one_turn["total_two_turn_length"],
                ),
                (1, None, None),
            )

    def test_cli_materializes_credit_and_source_turn_semantics(self) -> None:
        export_dir, out_dir = self.root / "exports", self.root / "analysis"
        export_parquet(self.export_store, export_dir)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "quadratic_voting.analyze",
                "--export-dir",
                str(export_dir),
                "--out",
                str(out_dir),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        detail = pq.read_table(out_dir / "snapshot_voter_candidate.parquet").to_pylist()
        self.assertTrue(detail)
        vote_two = next(row for row in detail if row["raw_votes"] == 2)
        self.assertEqual(vote_two["current_credits"], 4)
        self.assertEqual(vote_two["cumulative_before_credits"], 0)
        self.assertEqual(vote_two["cumulative_through_credits"], 4)
        demographics = pq.read_table(
            out_dir / "survivor_demographics.parquet"
        ).to_pylist()
        self.assertTrue(all(row["second_turn_length"] == 2 for row in demographics))
        self.assertTrue((out_dir / "stated_preference_agreement.png").exists())
        self.assertEqual(
            {path.name for path in out_dir.glob("*.parquet")},
            {
                "snapshot_voter_candidate.parquet",
                "snapshot_voter_summary.parquet",
                "snapshot_candidate_summary.parquet",
                "snapshot_rudeness_summary.parquet",
                "survivor_demographics.parquet",
                "stated_preference_agreement.parquet",
            },
        )
        self.assertEqual(len(list(out_dir.glob("*.png"))), 5)
        agreement = pq.read_table(
            out_dir / "stated_preference_agreement.parquet"
        ).to_pylist()
        self.assertEqual(
            {row["measure"] for row in agreement},
            {"signed_action", "signed_credit_spend"},
        )
        self.assertTrue(
            any(row["null_reason"] == "NOT_APPLICABLE_ACTION_ONLY" for row in agreement)
        )
        self.assertTrue(
            any(row["null_reason"] == "MISSING_STATEMENT" for row in agreement)
        )
        self.assertTrue(
            any(row["null_reason"] == "ABSTAINED_BALLOT" for row in agreement)
        )
