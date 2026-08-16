"""Tests for deterministic balanced candidate sampling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from quadratic_voting.experiment.sampling import (
    LEVEL_STRATIFIED_LEVELS,
    candidate_modal_severity,
    candidates_by_label,
    candidates_by_severity_level,
    create_balanced_sample,
    create_level_stratified_sample,
)
from quadratic_voting.experiment.store import (
    CandidateRecord,
    SourceAnnotation,
    SqliteExperimentStore,
    open_sqlite_store,
)
from quadratic_voting.experiment.types import (
    CandidateId,
    ReleaseId,
    RudenessLabel,
    SamplerPolicy,
    TemplateId,
)


_SEVERITY_LABELS = (
    "is_abuse.1",
    "is_abuse.0",
    "is_abuse.-1",
    "is_abuse.-2",
    "is_abuse.-3",
)
_LEVEL_TO_LABEL = {
    1: "is_abuse.1",
    0: "is_abuse.0",
    -1: "is_abuse.-1",
    -2: "is_abuse.-2",
    -3: "is_abuse.-3",
}


def _annotator_rows(
    start_index: int, active_level: int | None
) -> list[dict[str, object]]:
    """One annotator's five one-hot severity rows starting at annotation_index."""
    rows: list[dict[str, object]] = []
    active_label = None if active_level is None else _LEVEL_TO_LABEL[active_level]
    for offset, label in enumerate(_SEVERITY_LABELS):
        rows.append(
            {
                "annotation_index": start_index + offset,
                "source_label": label,
                "source_value": "1" if label == active_label else "0",
            }
        )
    return rows


def _annotations_for_levels(
    active_levels: tuple[int | None, ...],
) -> tuple[SourceAnnotation, ...]:
    """Build SourceAnnotation tuples for a candidate from per-annotator levels."""
    annotations: list[SourceAnnotation] = []
    for annotator_index, level in enumerate(active_levels):
        active_label = None if level is None else _LEVEL_TO_LABEL[level]
        for label in _SEVERITY_LABELS:
            annotations.append(
                SourceAnnotation(
                    annotator_hash=f"{annotator_index:064d}",
                    source_label=label,
                    source_value="1" if label == active_label else "0",
                )
            )
    return tuple(annotations)


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
                (RudenessLabel.RUDE,) * 6
                + (RudenessLabel.NON_RUDE,) * 6
                + (RudenessLabel.AMBIGUOUS_TIE,),
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
                {"candidate_id": "A1", "rudeness_label": "ambiguous_tie"},
            )
        )
        self.assertEqual(
            grouped[RudenessLabel.RUDE], (CandidateId("C1"), CandidateId("C2"))
        )
        self.assertEqual(grouped[RudenessLabel.AMBIGUOUS_TIE], (CandidateId("A1"),))

    def test_balanced_sample_excludes_ambiguous_ties(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release, template, store = self._catalog(Path(directory) / "qv.sqlite3")
            grouped = candidates_by_label(store, release)
            sample_id = create_balanced_sample(
                store,
                release,
                template,
                size=4,
                seed=5,
                candidates_by_label=grouped,
            )
            selected = {
                row[0]
                for row in store.connection.execute(
                    "SELECT candidate_id FROM candidate_sample_member WHERE sample_id=?",
                    (sample_id,),
                )
            }
            store.close()
        self.assertTrue(selected.isdisjoint(grouped[RudenessLabel.AMBIGUOUS_TIE]))

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


class ModalSeverityTests(unittest.TestCase):
    def test_strict_majority_mode(self) -> None:
        rows = _annotator_rows(0, -2) + _annotator_rows(5, -2) + _annotator_rows(10, 0)
        self.assertEqual(candidate_modal_severity(rows), -2)

    def test_tie_breaks_toward_more_severe(self) -> None:
        # One vote for level 1, one for level -1: tie resolves to the more
        # negative (more severe) level.
        rows = _annotator_rows(0, 1) + _annotator_rows(5, -1)
        self.assertEqual(candidate_modal_severity(rows), -1)

    def test_three_way_tie_breaks_to_most_negative(self) -> None:
        rows = _annotator_rows(0, 0) + _annotator_rows(5, -1) + _annotator_rows(10, -3)
        self.assertEqual(candidate_modal_severity(rows), -3)

    def test_malformed_annotators_are_ignored(self) -> None:
        # Annotator with no active severity and one with two active labels are
        # both ignored; only the single valid annotator (level -1) counts.
        rows = _annotator_rows(0, None) + _annotator_rows(5, -1)
        two_active = _annotator_rows(10, -2)
        two_active[0]["source_value"] = "1"  # is_abuse.1 also active -> malformed
        self.assertEqual(candidate_modal_severity(rows + two_active), -1)

    def test_no_valid_annotator_returns_none(self) -> None:
        self.assertIsNone(candidate_modal_severity(_annotator_rows(0, None)))

    def test_non_multiple_of_five_is_actionable(self) -> None:
        rows = _annotator_rows(0, -1)[:3]
        with self.assertRaisesRegex(ValueError, "multiple of the"):
            candidate_modal_severity(rows)


class LevelStratifiedSamplingTests(unittest.TestCase):
    def _catalog(
        self, path: Path, *, extra_neg1: bool = True
    ) -> tuple[ReleaseId, TemplateId, SqliteExperimentStore]:
        store = open_sqlite_store(path)
        # One candidate whose modal severity is each of the five levels, plus a
        # second candidate at level -1 to exercise deterministic per-level draws.
        specs: list[tuple[str, tuple[int | None, ...]]] = [
            ("lvl-pos1", (1, 1, 0)),
            ("lvl-zero", (0, 0, -1)),
            ("lvl-neg1-a", (-1, -1, 0)),
            ("lvl-neg2", (-2, -2, 0)),
            ("lvl-neg3", (-3, -3, 0)),
        ]
        if extra_neg1:
            specs.append(("lvl-neg1-b", (-1, -1, 1)))
        records = tuple(
            CandidateRecord(
                source_row_id=source_id,
                rudeness_label=RudenessLabel.NON_RUDE,
                turns=(("agent", f"agent-{source_id}"), ("user", f"user-{source_id}")),
                content_sha256=f"{index:064x}",
                annotations=_annotations_for_levels(levels),
            )
            for index, (source_id, levels) in enumerate(specs, start=1)
        )
        release = store.ingest_release("fixture", "v1", "fixture", "f" * 64, records)
        template = store.register_template("candidate-card", "v1", "{candidate_id}")
        store.render_presentations(
            release, template, lambda record: record.source_row_id
        )
        return release, template, store

    def test_grouping_assigns_each_candidate_to_its_modal_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release, _template, store = self._catalog(Path(directory) / "qv.sqlite3")
            grouped = candidates_by_severity_level(store, release)
            counts = {
                level: len(grouped.get(level, ())) for level in LEVEL_STRATIFIED_LEVELS
            }
            store.close()
        self.assertEqual(counts, {1: 1, 0: 1, -1: 2, -2: 1, -3: 1})

    def test_draws_one_per_level_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release, template, store = self._catalog(Path(directory) / "qv.sqlite3")
            grouped = candidates_by_severity_level(store, release)
            first = create_level_stratified_sample(
                store, release, template, seed=17, candidates_by_level=grouped
            )
            second = create_level_stratified_sample(
                store, release, template, seed=17, candidates_by_level=grouped
            )

            def members(sample_id: object) -> tuple[str, ...]:
                return tuple(
                    row[0]
                    for row in store.connection.execute(
                        "SELECT candidate_id FROM candidate_sample_member "
                        "WHERE sample_id=? ORDER BY position",
                        (sample_id,),
                    )
                )

            first_members = members(first)
            second_members = members(second)
            policy = store.connection.execute(
                "SELECT sampler_policy,size FROM candidate_sample WHERE sample_id=?",
                (first,),
            ).fetchone()
            # Map each drawn member back to its modal level to prove one-per-level.
            level_of = {
                str(candidate_id): level
                for level, ids in grouped.items()
                for candidate_id in ids
            }
            drawn_levels = tuple(level_of[str(member)] for member in first_members)
            store.close()
        self.assertEqual(first_members, second_members)
        self.assertEqual(len(first_members), 5)
        self.assertEqual(drawn_levels, LEVEL_STRATIFIED_LEVELS)
        self.assertEqual(policy["sampler_policy"], SamplerPolicy.LEVEL_STRATIFIED.value)
        self.assertEqual(policy["size"], 5)

    def test_missing_level_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release, template, store = self._catalog(Path(directory) / "qv.sqlite3")
            grouped = dict(candidates_by_severity_level(store, release))
            grouped[-3] = ()
            with self.assertRaisesRegex(
                ValueError, "severity levels have no candidates"
            ):
                create_level_stratified_sample(
                    store, release, template, seed=1, candidates_by_level=grouped
                )
            store.close()


if __name__ == "__main__":
    unittest.main()
