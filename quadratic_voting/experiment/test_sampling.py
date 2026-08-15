"""Tests for deterministic balanced candidate sampling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quadratic_voting.experiment.sampling import (
    candidates_by_label,
    create_balanced_sample,
)
from quadratic_voting.experiment.store import (
    CandidateRecord,
    SqliteExperimentStore,
    open_sqlite_store,
)
from quadratic_voting.experiment.types import (
    CandidateId,
    ReleaseId,
    RudenessLabel,
    TemplateId,
)


class SamplingTests(unittest.TestCase):
    def _catalog(
        self, path: Path
    ) -> tuple[ReleaseId, TemplateId, SqliteExperimentStore]:
        store = open_sqlite_store(path)
        records = tuple(
            CandidateRecord(
                source_row_id=f"source-{index}",
                rudeness_label=label,
                turns=(("agent", f"agent-{index}"), ("user", f"user-{index}")),
                content_sha256=f"{index:064x}",
            )
            for index, label in enumerate(
                (RudenessLabel.RUDE,) * 6 + (RudenessLabel.NON_RUDE,) * 6,
                start=1,
            )
        )
        release = store.ingest_release("fixture", "v1", "fixture", "f" * 64, records)
        template = store.register_template("candidate-card", "v1", "{candidate_id}")
        store.render_presentations(
            release, template, lambda record: record.source_row_id
        )
        return release, template, store

    def test_same_seed_same_members_and_different_seed_changes_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release, template, store = self._catalog(Path(directory) / "qv.sqlite3")
            grouped = candidates_by_label(store, release)
            first = create_balanced_sample(
                store, release, template, size=6, seed=7, candidates_by_label=grouped
            )
            second = create_balanced_sample(
                store, release, template, size=6, seed=7, candidates_by_label=grouped
            )
            third = create_balanced_sample(
                store, release, template, size=6, seed=8, candidates_by_label=grouped
            )
            memberships = []
            for sample_id in (first, second, third):
                memberships.append(
                    tuple(
                        row[0]
                        for row in store.connection.execute(
                            "SELECT candidate_id FROM candidate_sample_member "
                            "WHERE sample_id=? ORDER BY position",
                            (sample_id,),
                        )
                    )
                )
            store.close()
        self.assertEqual(memberships[0], memberships[1])
        self.assertNotEqual(memberships[0], memberships[2])

    def test_small_stratum_names_counts_and_fix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release, template, store = self._catalog(Path(directory) / "qv.sqlite3")
            with self.assertRaisesRegex(ValueError, "rude=1, non_rude=2") as raised:
                create_balanced_sample(
                    store,
                    release,
                    template,
                    size=4,
                    seed=1,
                    candidates_by_label={
                        RudenessLabel.RUDE: (CandidateId("r1"),),
                        RudenessLabel.NON_RUDE: (
                            CandidateId("n1"),
                            CandidateId("n2"),
                        ),
                    },
                )
            store.close()
        self.assertIn("reduce --size", str(raised.exception))

    def test_row_helper_sorts_typed_groups(self) -> None:
        grouped = candidates_by_label(
            (
                {"candidate_id": "C2", "rudeness_label": "rude"},
                {"candidate_id": "C1", "rudeness_label": "rude"},
                {"candidate_id": "N1", "rudeness_label": "non_rude"},
            )
        )
        self.assertEqual(
            grouped[RudenessLabel.RUDE], (CandidateId("C1"), CandidateId("C2"))
        )

    def test_odd_sample_persists_versioned_extra_stratum_draw(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release, template, store = self._catalog(Path(directory) / "qv.sqlite3")
            sample_id = create_balanced_sample(
                store,
                release,
                template,
                size=5,
                seed=(1 << 64) - 1,
                candidates_by_label=candidates_by_label(store, release),
            )
            draw = store.connection.execute(
                "SELECT * FROM sample_rng_draw WHERE sample_id=?", (sample_id,)
            ).fetchone()
            population = tuple(
                row[0]
                for row in store.connection.execute(
                    "SELECT stratum_value FROM sample_rng_draw_population "
                    "WHERE draw_id=? ORDER BY position",
                    (draw["draw_id"],),
                )
            )
            count = store.connection.execute(
                "SELECT COUNT(*) FROM candidate_sample_member WHERE sample_id=?",
                (sample_id,),
            ).fetchone()[0]
            store.close()
        self.assertEqual(count, 5)
        self.assertEqual(population, ("non_rude", "rude"))
        self.assertEqual(draw["seed_version"], "qv-seed/v1")


if __name__ == "__main__":
    unittest.main()
