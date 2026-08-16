"""Tests for ConvAbuse loading, derivation, rendering, and ingestion."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from quadratic_voting.experiment.catalog import (
    DEFAULT_PRESENTATION_TEMPLATE_BODY,
    RudenessDerivationRule,
    _content_digest,
    ingest_convabuse,
    load_convabuse,
    render_candidate_card,
)
from quadratic_voting.experiment.store import CandidateRecord, open_sqlite_store
from quadratic_voting.experiment.transcript import render_transcript
from quadratic_voting.experiment.types import (
    CandidateId,
    ElicitationArm,
    PendingTurn,
    RudenessLabel,
    RunId,
    SetupContext,
    TurnKind,
    VoterRoundView,
    VotingRegime,
)


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
    # Modal per-annotator severity (mode, ties toward more negative) is noted per
    # group so the fixture spans all five ConvAbuse severity levels {1,0,-1,-2,-3}
    # required by the level-stratified default pilot. conv_ids are chosen so
    # "c-clean" remains the lexicographically first source_row_id (several tests
    # rely on records[0] being the clean candidate).
    groups = (
        (
            "c-rude",  # modal severity -2 (votes -1,-2,0 tie -> -2), label RUDE
            "Prior agent rude",
            "Prior user rude",
            "Agent rude candidate",
            "Bad user",
            (-1, -2, 0),
        ),
        (
            "c-clean",  # modal severity 0, label NON_RUDE
            "Prior agent clean",
            "Prior user clean",
            "Agent clean candidate",
            "Fine user",
            (0, 0, -1),
        ),
        (
            "c-tie",  # modal severity -3 (votes -3,1 tie -> -3), label AMBIGUOUS_TIE
            "Prior agent tie",
            "Prior user tie",
            "Agent tied candidate",
            "Maybe user",
            (-3, 1),
        ),
        (
            "c-lvl-pos1",  # modal severity 1, label NON_RUDE
            "Prior agent pos1",
            "Prior user pos1",
            "Agent pos1 candidate",
            "Calm user",
            (1, 1, 0),
        ),
        (
            "c-lvl-neg1",  # modal severity -1, label RUDE
            "Prior agent neg1",
            "Prior user neg1",
            "Agent neg1 candidate",
            "Curt user",
            (-1, -1, 0),
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        index = 0
        for conv_id, prev_agent, prev_user, agent, user, severities in groups:
            for severity in severities:
                index += 1
                row = dict.fromkeys(FIELDS, "")
                row.update(
                    {
                        "example_no": str(index),
                        "annotator_id": f"a{index}",
                        "conv_id": conv_id,
                        "prev_agent": prev_agent,
                        "prev_user": prev_user,
                        "agent": agent,
                        "user": user,
                        "bot": "bot",
                    }
                )
                for band in (1, 0, -1, -2, -3):
                    row[f"is_abuse.{band}"] = "1" if severity == band else "0"
                writer.writerow(row)


class CatalogTests(unittest.TestCase):
    def test_grouping_majority_tie_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            write_fixture(path)
            first = load_convabuse(
                path,
                rule=RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE_COMPLETE_CONTEXT,
            )
            second = load_convabuse(
                path,
                rule=RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE_COMPLETE_CONTEXT,
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

    def test_loader_maps_all_four_source_fields_in_chronological_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "markers.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                row = dict.fromkeys(FIELDS, "0")
                row.update(
                    {
                        "conv_id": "markers",
                        "prev_user": "PREVIOUS-USER-MARKER",
                        "prev_agent": "PREVIOUS-MODEL-MARKER",
                        "user": "CURRENT-USER-MARKER",
                        "agent": "CURRENT-MODEL-MARKER",
                        "is_abuse.0": "1",
                    }
                )
                writer.writerow(row)
            record = load_convabuse(
                path,
                rule=RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE_COMPLETE_CONTEXT,
            )[0]
        self.assertEqual(
            record.turns,
            (
                ("user", "PREVIOUS-USER-MARKER"),
                ("assistant", "PREVIOUS-MODEL-MARKER"),
                ("user", "CURRENT-USER-MARKER"),
                ("assistant", "CURRENT-MODEL-MARKER"),
            ),
        )

    def test_render_card_and_model_visible_setup_preserve_exact_four_message_order(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            write_fixture(path)
            record = load_convabuse(
                path,
                rule=RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE_COMPLETE_CONTEXT,
            )[0]
        card = render_candidate_card("C001", record, DEFAULT_PRESENTATION_TEMPLATE_BODY)
        expected = (
            "User C001\nUser: Prior user clean\nAssistant: Prior agent clean\n"
            "User: Fine user\nAssistant: Agent clean candidate"
        )
        self.assertEqual(card, expected)
        self.assertNotIn("rude", card.casefold())
        messages = render_transcript(
            VoterRoundView(
                SetupContext(
                    RunId("run"),
                    ElicitationArm.ACTION_ONLY,
                    VotingRegime.SUPPORT,
                    100,
                    "",
                    ((CandidateId("C001"), card),),
                ),
                (),
                PendingTurn(1, TurnKind.BALLOT, (CandidateId("C001"),), 0, ()),
            )
        )
        self.assertIn(expected, messages[0].content)
        self.assertLess(
            messages[0].content.index("Prior user clean"),
            messages[0].content.index("Prior agent clean"),
        )

    def test_render_card_rejects_missing_or_misordered_roles(self) -> None:
        valid = CandidateRecord(
            "source",
            RudenessLabel.NON_RUDE,
            (
                ("user", "one"),
                ("assistant", "two"),
                ("user", "three"),
                ("assistant", "four"),
            ),
            "a" * 64,
        )
        with self.assertRaisesRegex(ValueError, "required chronological roles"):
            render_candidate_card(
                "C001",
                valid.__class__(
                    valid.source_row_id,
                    valid.rudeness_label,
                    valid.turns[:3],
                    valid.content_sha256,
                ),
                DEFAULT_PRESENTATION_TEMPLATE_BODY,
            )
        with self.assertRaisesRegex(ValueError, "required chronological roles"):
            render_candidate_card(
                "C001",
                valid.__class__(
                    valid.source_row_id,
                    valid.rudeness_label,
                    (("assistant", "one"), *valid.turns[1:]),
                    valid.content_sha256,
                ),
                DEFAULT_PRESENTATION_TEMPLATE_BODY,
            )

    def test_missing_columns_and_blank_conversation_fields_are_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("conv_id,user\nc1,\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required CSV columns"):
                load_convabuse(
                    path,
                    rule=RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE_COMPLETE_CONTEXT,
                )
            path = Path(directory) / "blank.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                row = dict.fromkeys(FIELDS, "0")
                row.update(
                    {
                        "conv_id": "blank",
                        "prev_agent": "previous model",
                        "prev_user": "previous user",
                        "agent": "current model",
                        "user": " ",
                        "is_abuse.0": "1",
                    }
                )
                writer.writerow(row)
            with self.assertRaisesRegex(
                ValueError, "blank required conversation fields"
            ):
                load_convabuse(
                    path,
                    rule=RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE_COMPLETE_CONTEXT,
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
                    RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE_COMPLETE_CONTEXT,
                )
                source = store.connection.execute(
                    "SELECT source_path FROM dataset_release WHERE release_id=?",
                    (release_id,),
                ).fetchone()[0]
                self.assertIn(
                    "rudeness-rule=majority-severity-negative-complete-context/v3",
                    source,
                )
                policy = store.connection.execute(
                    "SELECT name,version FROM label_policy"
                ).fetchone()
                self.assertEqual(
                    tuple(policy),
                    (
                        "convabuse-rudeness",
                        RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE_COMPLETE_CONTEXT.value,
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
                        RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE_COMPLETE_CONTEXT,
                    )
                count = store.connection.execute(
                    "SELECT COUNT(*) FROM dataset_release"
                ).fetchone()[0]
            self.assertEqual(count, 1)

    def test_store_round_trip_preserves_four_turns_and_content_hashes_every_message(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "fixture.csv"
            write_fixture(path)
            records = load_convabuse(
                path,
                rule=RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE_COMPLETE_CONTEXT,
            )
            mutated_hashes = set()
            for index in range(4):
                turns = list(records[0].turns)
                role, text = turns[index]
                turns[index] = (role, f"{text} changed")
                mutated = CandidateRecord(
                    records[0].source_row_id,
                    records[0].rudeness_label,
                    tuple(turns),
                    records[0].content_sha256,
                )
                # The canonical digest covers each role/text pair in sequence.
                mutated_hashes.add(_content_digest(mutated.turns))
            self.assertNotIn(records[0].content_sha256, mutated_hashes)
            self.assertEqual(len(mutated_hashes), 4)
            with open_sqlite_store(root / "qv.sqlite3") as store:
                release = ingest_convabuse(
                    store,
                    path,
                    "fixture-v3",
                    RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE_COMPLETE_CONTEXT,
                )
                rows = store.connection.execute(
                    "SELECT turn_index,role,text FROM candidate_turn WHERE candidate_id="
                    "(SELECT candidate_id FROM candidate WHERE release_id=? ORDER BY source_row_id LIMIT 1) "
                    "ORDER BY turn_index",
                    (release,),
                ).fetchall()
            self.assertEqual(
                [tuple(row) for row in rows],
                [
                    (index, role, text)
                    for index, (role, text) in enumerate(records[0].turns)
                ],
            )


if __name__ == "__main__":
    unittest.main()
