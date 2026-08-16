"""Focused production-path tests for no-GPU snapshot analytics."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgba

from quadratic_voting.analyze.__main__ import _confirm_replacement, _publish
from quadratic_voting.experiment.export import export_parquet
from quadratic_voting.experiment.snapshots import (
    aggregate_preference_agreement,
    aggregate_preference_agreement_by_rudeness,
    build_snapshot_figures,
    build_snapshot_tables,
    select_snapshot_rounds,
)
from quadratic_voting.experiment.timeline import (
    SourceSeveritySummary,
    build_timeline_payload,
    candidate_label_sort_key,
    source_severity_summary,
)
from quadratic_voting.experiment.timeline_flow import render_timeline_html
from quadratic_voting.experiment.test_export import AnalysisFixture


class SnapshotTests(AnalysisFixture):
    def test_analysis_replacement_confirmation_and_restore(self) -> None:
        out = self.root / "analysis"
        out.mkdir()
        marker = out / "marker.txt"
        marker.write_text("old", encoding="utf-8")
        with (
            mock.patch.object(sys.stdin, "isatty", return_value=True),
            mock.patch("builtins.input", return_value="yes") as prompt,
        ):
            _confirm_replacement(out, overwrite=False)
        prompt.assert_called_once_with(
            f"Analysis output already exists at {out}. Replace it? [y/N] "
        )
        with (
            mock.patch.object(sys.stdin, "isatty", return_value=True),
            mock.patch("builtins.input", return_value="no"),
            self.assertRaisesRegex(FileExistsError, "replacement cancelled"),
        ):
            _confirm_replacement(out, overwrite=False)
        self.assertEqual(marker.read_text(encoding="utf-8"), "old")

        staging = self.root / "staging"
        staging.mkdir()
        (staging / "marker.txt").write_text("new", encoding="utf-8")
        replace = os.replace
        replacement_count = 0

        def fail_publication(source: Path, destination: Path) -> None:
            nonlocal replacement_count
            replacement_count += 1
            if replacement_count == 2:
                raise OSError("fixture publication failure")
            replace(source, destination)

        with (
            mock.patch(
                "quadratic_voting.analyze.__main__.os.replace",
                side_effect=fail_publication,
            ),
            self.assertRaisesRegex(OSError, "fixture publication failure"),
        ):
            _publish(staging, out)
        self.assertEqual(marker.read_text(encoding="utf-8"), "old")

    def test_timeline_payload_keeps_persisted_source_turns_in_order(self) -> None:
        """The renderer must not reconstruct omitted context from another source."""
        export_dir = self.root / "exports"
        export_parquet(self.export_store, export_dir)
        source_path = export_dir / "candidate_source_turns.parquet"
        source_table = pq.read_table(source_path)
        source_rows = source_table.to_pylist()
        candidate_id = source_rows[0]["candidate_id"]
        two_turn_candidate_id = next(
            row["candidate_id"]
            for row in source_rows
            if row["candidate_id"] != candidate_id
        )
        source_rows = (
            [
                row
                for row in source_rows
                if row["candidate_id"] not in {candidate_id, two_turn_candidate_id}
            ]
            + [
                {
                    "candidate_id": candidate_id,
                    "turn_index": index,
                    "role": role,
                    "text": text,
                }
                for index, role, text in (
                    (1, "assistant", "model one"),
                    (2, "user", "user two"),
                    (3, "assistant", "</script><script>unsafe()</script>"),
                    (4, "user", "user four"),
                )
            ]
            + [
                {
                    "candidate_id": two_turn_candidate_id,
                    "turn_index": index,
                    "role": role,
                    "text": text,
                }
                for index, role, text in (
                    (1, "assistant", "model only"),
                    (2, "user", "user reply"),
                )
            ]
        )
        pq.write_table(
            pa.Table.from_pylist(source_rows, schema=source_table.schema), source_path
        )
        candidate_ids = sorted({row["candidate_id"] for row in source_rows})
        labels = [
            {
                "candidate_id": value,
                "candidate_label": f"C{10 if value == candidate_id else index + 1}",
            }
            for index, value in enumerate(candidate_ids)
        ]
        payload = build_timeline_payload(export_dir, labels)
        frame = payload["runs"][0]["frames"][0]  # type: ignore[index]
        candidate = next(
            row
            for row in frame["candidates"]
            if row["id"] == candidate_id  # type: ignore[index]
        )
        self.assertEqual(
            candidate["sourceTurns"],  # type: ignore[index]
            [
                {"role": "assistant", "text": "model one"},
                {"role": "user", "text": "user two"},
                {"role": "assistant", "text": "</script><script>unsafe()</script>"},
                {"role": "user", "text": "user four"},
            ],
        )
        two_turn_candidate = next(
            row
            for row in frame["candidates"]
            if row["id"] == two_turn_candidate_id  # type: ignore[index]
        )
        self.assertEqual(
            two_turn_candidate["sourceTurns"],  # type: ignore[index]
            [
                {"role": "assistant", "text": "model only"},
                {"role": "user", "text": "user reply"},
            ],
        )
        self.assertEqual(
            sorted(("C1", "C10", "C2"), key=candidate_label_sort_key),
            ["C1", "C2", "C10"],
        )
        expected_votes = {
            str(self.candidates[0]): 4,
            str(self.candidates[1]): 2,
            str(self.candidates[2]): 1,
        }
        runs = payload["runs"]
        self.assertIsInstance(runs, list)
        assert isinstance(runs, list)
        for run in runs:
            if run["arm"] != "statement-then-action":
                continue
            run_frame = run["frames"][0]
            actual_votes = {
                row["id"]: row["aggregateVotes"] for row in run_frame["candidates"]
            }
            self.assertEqual(actual_votes, expected_votes)
            if run["regime"] == "Most Votes Kept":
                self.assertEqual(
                    run_frame["outcome"]["protected"], str(self.candidates[0])
                )
            else:
                self.assertEqual(
                    run_frame["outcome"]["removed"], str(self.candidates[0])
                )
        out_dir = self.root / "analysis"
        build_snapshot_tables(export_dir, out_dir)
        rendered = render_timeline_html(export_dir, out_dir).read_text(encoding="utf-8")
        self.assertIn("candidate-sidebar", rendered)
        self.assertIn("Credits: ", rendered)
        self.assertIn("Votes: ", rendered)
        self.assertIn("aggregateCredits", rendered)
        self.assertIn("Voter statements and ballot evidence", rendered)
        self.assertIn("Optional raw source annotation provenance", rendered)
        self.assertNotIn(".slice(0,42)", rendered)
        self.assertNotIn("</script><script>unsafe()", rendered)
        self.assertIn(r"\u003c/script\u003e", rendered)

    def test_source_severity_requires_one_selected_numeric_category_per_annotator(
        self,
    ) -> None:
        self.assertEqual(
            source_severity_summary(
                [
                    {
                        "annotator_hash": "a",
                        "source_label": "is_abuse.-2",
                        "source_value": "1",
                    },
                    {
                        "annotator_hash": "b",
                        "source_label": "is_abuse.-3",
                        "source_value": "true",
                    },
                ]
            ),
            SourceSeveritySummary(
                selected_ratings=(-2, -3), mean=-2.5, malformed_annotator_count=0
            ),
        )
        self.assertEqual(
            source_severity_summary(
                [
                    {
                        "annotator_hash": "a",
                        "source_label": "is_abuse.-2",
                        "source_value": "1",
                    },
                    {
                        "annotator_hash": "a",
                        "source_label": "is_abuse.-3",
                        "source_value": "1",
                    },
                ]
            ),
            SourceSeveritySummary(
                selected_ratings=(), mean=None, malformed_annotator_count=1
            ),
        )

    def test_source_severity_omits_zero_rows_and_excludes_malformed_annotators(
        self,
    ) -> None:
        summary = source_severity_summary(
            [
                {
                    "annotator_hash": "c",
                    "source_label": "is_abuse.-2",
                    "source_value": "1",
                },
                {
                    "annotator_hash": "a",
                    "source_label": "is_abuse.-1",
                    "source_value": "1",
                },
                {
                    "annotator_hash": "b",
                    "source_label": "is_abuse.-1",
                    "source_value": "true",
                },
                {
                    "annotator_hash": "a",
                    "source_label": "is_abuse.0",
                    "source_value": "0",
                },
                {
                    "annotator_hash": "b",
                    "source_label": "is_abuse.1",
                    "source_value": "0",
                },
                {
                    "annotator_hash": "c",
                    "source_label": "is_abuse.0",
                    "source_value": "0",
                },
                {
                    "annotator_hash": "bad",
                    "source_label": "is_abuse.-1",
                    "source_value": "1",
                },
                {
                    "annotator_hash": "bad",
                    "source_label": "is_abuse.-2",
                    "source_value": "1",
                },
            ]
        )
        self.assertEqual(summary.selected_ratings, (-1, -1, -2))
        self.assertEqual(summary.mean, -4 / 3)
        self.assertEqual(summary.malformed_annotator_count, 1)

    def test_plot_artists_use_condition_facets_and_aggregate_preferences(self) -> None:
        """Pin readable grouping rather than fragile rendered pixels."""
        export_dir, out_dir = self.root / "exports", self.root / "analysis"
        export_parquet(self.export_store, export_dir)
        build_snapshot_tables(export_dir, out_dir)
        figures = dict(build_snapshot_figures(out_dir))
        try:
            current = figures["average_current_votes_credits.png"]
            self.assertEqual(len(current.axes), 6)
            self.assertTrue(all("01M" not in axis.get_title() for axis in current.axes))
            self.assertTrue(all(len(axis.lines) >= 2 for axis in current.axes))
            self.assertTrue(
                all(list(axis.get_xticks()) == [1] for axis in current.axes)
            )

            cumulative = figures["cumulative_votes_credits_before_through.png"]
            self.assertEqual(len(cumulative.axes), 12)
            self.assertTrue(
                all(
                    "Votes" in axis.get_title() or "Credits" in axis.get_title()
                    for axis in cumulative.axes
                )
            )

            voter = figures["per_voter_current_votes_by_rudeness.png"]
            self.assertEqual(len(voter.axes), 12)
            self.assertTrue(
                all(
                    "Non-rude" in axis.get_title() or "Rude" in axis.get_title()
                    for axis in voter.axes
                )
            )
            self.assertTrue(
                all(
                    "V1" in [tick.get_text() for tick in axis.get_yticklabels()]
                    for axis in voter.axes
                )
            )

            candidate = figures["per_candidate_current_credits_by_rudeness.png"]
            self.assertEqual(len(candidate.axes), 12)
            self.assertTrue(
                all("01M" not in axis.get_title() for axis in candidate.axes)
            )

            totals = figures["cumulative_vote_totals_before_through_by_rudeness.png"]
            self.assertIn("totals (sums)", totals.texts[0].get_text())
            self.assertTrue(
                all(axis.get_ylabel() == "Total (sum)" for axis in totals.axes)
            )

            demographics = figures["candidate_rudeness_demographics.png"]
            self.assertEqual(len(demographics.axes), 6)
            self.assertTrue(all(axis.patches for axis in demographics.axes))
            first_panel = demographics.axes[0]
            self.assertEqual(list(first_panel.get_xticks()), [1])
            first_round_centers = [
                patch.get_x() + patch.get_width() / 2
                for patch in first_panel.patches
                if isinstance(patch, Rectangle) and patch.get_x() < 2
            ]
            self.assertEqual(len(first_round_centers), 2)
            self.assertNotEqual(*first_round_centers)
            self.assertTrue(
                all(
                    patch.get_y() == 0
                    for patch in first_panel.patches
                    if isinstance(patch, Rectangle)
                )
            )
            self.assertTrue(
                all(
                    axis.get_ylabel() == "Surviving candidates"
                    for axis in demographics.axes
                )
            )
            self.assertEqual(
                [text.get_text() for text in demographics.legends[0].get_texts()],
                ["Non-rude", "Rude"],
            )

            lengths = figures["survivor_message_lengths.png"]
            self.assertEqual(len(lengths.axes), 6)
            self.assertTrue(all(len(axis.lines) == 3 for axis in lengths.axes))
            self.assertIn(
                "First + second",
                [text.get_text() for text in lengths.legends[0].get_texts()],
            )

            length_distributions = figures[
                "survivor_total_message_length_distribution_by_rudeness.png"
            ]
            self.assertEqual(len(length_distributions.axes), 12)
            self.assertIn(
                "survivor composition", length_distributions.texts[0].get_text()
            )

            preference = figures["stated_preference_agreement.png"]
            self.assertEqual(len(preference.axes), 4)
            self.assertTrue(
                all(axis.get_ylim() == (-1.0, 1.0) for axis in preference.axes)
            )
            self.assertTrue(
                all("Action Only" not in axis.get_title() for axis in preference.axes)
            )
            self.assertTrue(
                all(list(axis.get_xticks()) == [1] for axis in preference.axes)
            )

            preference_by_rudeness = figures[
                "stated_preference_agreement_by_rudeness.png"
            ]
            self.assertEqual(len(preference_by_rudeness.axes), 8)
            self.assertIn("not causal", preference_by_rudeness.texts[0].get_text())
            budget = figures["voter_credit_budget_distribution.png"]
            self.assertEqual(len(budget.axes), 6)
            self.assertTrue(
                any(
                    patch.get_facecolor() == to_rgba("#8A8A8A")
                    for axis in budget.axes
                    for patch in axis.patches
                )
            )
        finally:
            import matplotlib.pyplot as plt

            for figure in figures.values():
                plt.close(figure)

    def test_preference_plot_rows_are_aggregated_per_voter_measure(self) -> None:
        rows: list[dict[str, object]] = [
            {
                "run_id": "opaque-run-id",
                "arm": "statement-then-action",
                "regime": "support",
                "snapshot_round": 3,
                "measure": "signed_action",
                "spearman_rho": rho,
            }
            for rho in (-1.0, 0.5)
        ]
        rows.append(
            {
                "run_id": "excluded",
                "arm": "action-only",
                "regime": "support",
                "snapshot_round": 3,
                "measure": "signed_action",
                "spearman_rho": 1.0,
            }
        )
        self.assertEqual(
            aggregate_preference_agreement(rows),
            (
                {
                    "run_id": "opaque-run-id",
                    "arm": "statement-then-action",
                    "regime": "support",
                    "snapshot_round": 3,
                    "measure": "signed_action",
                    "mean_spearman_rho": -0.25,
                    "min_spearman_rho": -1.0,
                    "max_spearman_rho": 0.5,
                    "n_voters": 2,
                },
            ),
        )
        for row in rows:
            row["rudeness_label"] = "rude"
        self.assertEqual(
            aggregate_preference_agreement_by_rudeness(rows)[0]["mean_spearman_rho"],
            -0.25,
        )

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
            figures = dict(build_snapshot_figures(out))
            try:
                self.assertEqual(
                    list(
                        figures["average_current_votes_credits.png"]
                        .axes[0]
                        .get_xticks()
                    ),
                    [2, 4, 9],
                )
                self.assertEqual(
                    list(
                        figures["candidate_rudeness_demographics.png"]
                        .axes[0]
                        .get_xticks()
                    ),
                    [2, 4, 9],
                )
                budget_axis = figures["voter_credit_budget_distribution.png"].axes[0]
                separators = [
                    line for line in budget_axis.lines if line.get_zorder() == 0
                ]
                self.assertEqual(len(separators), 5)
                self.assertEqual(
                    sum(line.get_linestyle() == ":" for line in separators), 3
                )
                self.assertEqual(
                    sum(line.get_linestyle() == "-" for line in separators), 2
                )
                self.assertGreater(
                    next(
                        line.get_linewidth()
                        for line in separators
                        if line.get_linestyle() == "-"
                    ),
                    next(
                        line.get_linewidth()
                        for line in separators
                        if line.get_linestyle() == ":"
                    ),
                )
            finally:
                import matplotlib.pyplot as plt

                for figure in figures.values():
                    plt.close(figure)
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
            voter_rudeness = pq.read_table(
                out / "snapshot_voter_rudeness_summary.parquet"
            ).to_pylist()
            self.assertEqual(
                next(
                    row
                    for row in voter_rudeness
                    if row["snapshot_round"] == 4
                    and row["voter_index"] == 0
                    and row["rudeness_label"] == "rude"
                )["current_votes"],
                2,
            )
            agreement_by_rudeness = pq.read_table(
                out / "stated_preference_agreement_by_rudeness.parquet"
            ).to_pylist()
            self.assertTrue(
                all(row["null_reason"] == "N_LT_2" for row in agreement_by_rudeness)
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
                "--input-dir",
                str(export_dir),
                "--out",
                str(out_dir),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        marker = out_dir / "existing-marker.txt"
        marker.write_text("preserve", encoding="utf-8")
        refused = subprocess.run(
            [
                sys.executable,
                "-m",
                "quadratic_voting.analyze",
                "--input-dir",
                str(export_dir),
                "--out",
                str(out_dir),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(refused.returncode, 1)
        self.assertIn(
            "non-interactive replacement requires --overwrite", refused.stderr
        )
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
        overwritten = subprocess.run(
            [
                sys.executable,
                "-m",
                "quadratic_voting.analyze",
                "--input-dir",
                str(export_dir),
                "--out",
                str(out_dir),
                "--overwrite",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(overwritten.returncode, 0, overwritten.stderr)
        self.assertFalse(marker.exists())
        failed = subprocess.run(
            [
                sys.executable,
                "-m",
                "quadratic_voting.analyze",
                "--input-dir",
                str(self.root / "missing"),
                "--out",
                str(out_dir),
                "--overwrite",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(failed.returncode, 1)
        self.assertTrue((out_dir / "timeline.html").exists())
        detail = pq.read_table(out_dir / "snapshot_voter_candidate.parquet").to_pylist()
        self.assertTrue(detail)
        vote_two = next(row for row in detail if row["raw_votes"] == 2)
        self.assertEqual(vote_two["current_credits"], 4)
        self.assertEqual(vote_two["cumulative_before_credits"], 0)
        self.assertEqual(vote_two["cumulative_through_credits"], 4)
        demographics = pq.read_table(
            out_dir / "survivor_demographics.parquet"
        ).to_pylist()
        self.assertTrue(
            all(
                row["second_turn_length"] == len("previous-model-1")
                for row in demographics
            )
        )
        self.assertTrue((out_dir / "stated_preference_agreement.png").exists())
        self.assertEqual(
            {path.name for path in out_dir.glob("*.parquet")},
            {
                "snapshot_voter_candidate.parquet",
                "snapshot_voter_summary.parquet",
                "snapshot_voter_rudeness_summary.parquet",
                "snapshot_candidate_summary.parquet",
                "snapshot_candidate_labels.parquet",
                "snapshot_voter_budget_distribution.parquet",
                "snapshot_budget_utilization.parquet",
                "snapshot_rudeness_summary.parquet",
                "survivor_demographics.parquet",
                "stated_preference_agreement.parquet",
                "stated_preference_agreement_by_rudeness.parquet",
            },
        )
        self.assertEqual(
            {path.name for path in out_dir.glob("*.png")},
            {
                "average_current_votes_credits.png",
                "cumulative_votes_credits_before_through.png",
                "survivor_rudeness_distribution.png",
                "candidate_rudeness_demographics.png",
                "survivor_message_lengths.png",
                "stated_preference_agreement.png",
                "per_voter_current_votes_by_rudeness.png",
                "per_voter_current_credits_by_rudeness.png",
                "per_candidate_current_votes_by_rudeness.png",
                "per_candidate_current_credits_by_rudeness.png",
                "cumulative_vote_totals_before_through_by_rudeness.png",
                "cumulative_credit_totals_before_through_by_rudeness.png",
                "survivor_first_message_length_distribution_by_rudeness.png",
                "survivor_second_message_length_distribution_by_rudeness.png",
                "survivor_total_message_length_distribution_by_rudeness.png",
                "stated_preference_agreement_by_rudeness.png",
                "voter_credit_budget_distribution.png",
            },
        )
        self.assertTrue((out_dir / "timeline.html").exists())
        utilization = pq.read_table(
            out_dir / "snapshot_budget_utilization.parquet"
        ).to_pylist()
        valid = next(row for row in utilization if row["current_spend"] is not None)
        self.assertEqual(
            valid["unspent_credits"], valid["credit_budget"] - valid["current_spend"]
        )
        self.assertEqual(
            valid["utilization_fraction"],
            valid["current_spend"] / valid["credit_budget"],
        )
        timeline = (out_dir / "timeline.html").read_text(encoding="utf-8")
        self.assertIn("Most Votes Kept", timeline)
        self.assertIn("Most Votes Kicked", timeline)
        self.assertIn('type="range"', timeline)
        self.assertIn("round-diagram-", timeline)
        self.assertIn("createElementNS", timeline)
        self.assertIn("quadratic credits", timeline)
        self.assertNotIn(r"<\/script", timeline)
        self.assertIn("candidate-sidebar", timeline)
        self.assertIn("Voter statements and ballot evidence", timeline)
        self.assertIn("Conversation actually shown in this pilot", timeline)
        self.assertIn("Optional raw source annotation provenance", timeline)
        self.assertNotIn(".slice(0,42)", timeline)
        self.assertNotIn("First: ", timeline)
        self.assertNotIn("Second: ", timeline)
        payload = build_timeline_payload(
            export_dir,
            pq.read_table(out_dir / "snapshot_candidate_labels.parquet").to_pylist(),
        )
        for run in payload["runs"]:  # type: ignore[index,attr-defined]
            frames = run["frames"]  # type: ignore[index]
            self.assertEqual(
                [frame["round"] for frame in frames],  # type: ignore[index]
                sorted(frame["round"] for frame in frames),  # type: ignore[index]
            )
            self.assertTrue(
                all("voters" in frame and "candidates" in frame for frame in frames)
            )  # type: ignore[index]
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
