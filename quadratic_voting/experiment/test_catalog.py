"""Tests for ConvAbuse loading, derivation, rendering, and ingestion."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from quadratic_voting.experiment.catalog import (
    DEFAULT_PRESENTATION_TEMPLATE_BODY,
    RudenessDerivationRule,
    ingest_convabuse,
    load_convabuse,
    render_candidate_card,
)
from quadratic_voting.experiment.store import open_sqlite_store
from quadratic_voting.experiment.types import RudenessLabel


FIELDS = (
    "example_no",
    "annotator_id",
    "conv_id",
    "prev_agent",
    "prev_user",
    "agent",
    "user",
    "bot",
    "is_abuse.1",
    "is_abuse.0",
    "is_abuse.-1",
    "is_abuse.-2",
    "is_abuse.-3",
)


def write_fixture(path: Path) -> None:
    groups = (
        ("c-rude", "Agent rude candidate", "Bad user", (-1, -2, 0)),
        ("c-clean", "Agent clean candidate", "Fine user", (0, 0, -1)),
        ("c-tie", "Agent tied candidate", "Maybe user", (-3, 1)),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        index = 0
        for conv_id, agent, user, severities in groups:
            for severity in severities:
                index += 1
                row = dict.fromkeys(FIELDS, "")
                row.update(
                    {
                        "example_no": str(index),
                        "annotator_id": f"a{index}",
                        "conv_id": conv_id,
                        "agent": agent,
                        "user": user,
                        "bot": "bot",
                    }
                )
                for band in (1, 0, -1, -2, -3):
                    row[f"is_abuse.{band}"] = "1" if severity == band else "0"
                writer.writerow(row)
        blank = dict.fromkeys(FIELDS, "0")
        blank.update({"conv_id": "blank", "agent": "ignored", "user": " "})
        writer.writerow(blank)


class CatalogTests(unittest.TestCase):
    def test_grouping_majority_tie_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            write_fixture(path)
            first = load_convabuse(
                path, rule=RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE
            )
            second = load_convabuse(
                path, rule=RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE
            )
        self.assertEqual(first, second)
        self.assertEqual(
            tuple(item.source_row_id for item in first),
            tuple(sorted(item.source_row_id for item in first)),
        )
        labels = {
            item.source_row_id.split(":", 1)[0]: item.rudeness_label for item in first
        }
        self.assertEqual(labels["c-rude"], RudenessLabel.RUDE)
        self.assertEqual(labels["c-clean"], RudenessLabel.NON_RUDE)
        self.assertEqual(labels["c-tie"], RudenessLabel.AMBIGUOUS_TIE)

    def test_render_card_does_not_leak_derived_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            write_fixture(path)
            record = load_convabuse(
                path, rule=RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE
            )[0]
        card = render_candidate_card("C001", record, DEFAULT_PRESENTATION_TEMPLATE_BODY)
        self.assertIn("Candidate C001", card)
        self.assertIn("Agent:", card)
        self.assertIn("User:", card)
        self.assertNotIn("rude", card.casefold())

    def test_missing_columns_and_zero_candidates_are_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("conv_id,user\nc1,\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required CSV columns"):
                load_convabuse(
                    path, rule=RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE
                )

    def test_ingest_records_rule_and_duplicate_version_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "fixture.csv"
            write_fixture(path)
            with open_sqlite_store(root / "qv.sqlite3") as store:
                release_id = ingest_convabuse(
                    store,
                    path,
                    "fixture-v1",
                    RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE,
                )
                source = store.connection.execute(
                    "SELECT source_path FROM dataset_release WHERE release_id=?",
                    (release_id,),
                ).fetchone()[0]
                self.assertIn("rudeness-rule=majority-severity-negative/v2", source)
                policy = store.connection.execute(
                    "SELECT name,version FROM label_policy"
                ).fetchone()
                self.assertEqual(
                    tuple(policy),
                    (
                        "convabuse-rudeness",
                        RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE.value,
                    ),
                )
                self.assertGreater(
                    store.connection.execute(
                        "SELECT COUNT(*) FROM source_annotation"
                    ).fetchone()[0],
                    0,
                )
                with self.assertRaisesRegex(ValueError, "already exists"):
                    ingest_convabuse(
                        store,
                        path,
                        "fixture-v1",
                        RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE,
                    )
                count = store.connection.execute(
                    "SELECT COUNT(*) FROM dataset_release"
                ).fetchone()[0]
            self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
