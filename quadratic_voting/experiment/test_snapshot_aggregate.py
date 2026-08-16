"""Tests for the thin seed-repeat snapshot-dashboard Parquet aggregator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from quadratic_voting.experiment.snapshot_aggregate import (
    SEED_REPEAT_INDEX_COLUMN,
    aggregate_snapshot_tables,
)


def _write_candidate_summary(directory: Path, rows: list[dict[str, object]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("candidate_id", pa.string()), ("survived", pa.int64())])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, directory / "snapshot_candidate_summary.parquet")


def _write_candidate_labels(directory: Path, rows: list[dict[str, object]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("candidate_id", pa.string()), ("rudeness_label", pa.string())])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, directory / "snapshot_candidate_labels.parquet")


class AggregateSnapshotTests(unittest.TestCase):
    def test_concat_adds_seed_repeat_index_and_copies_invariant_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repeat0 = root / "snapshot" / "repeat-0"
            repeat1 = root / "snapshot" / "repeat-1"
            _write_candidate_summary(repeat0, [{"candidate_id": "C1", "survived": 1}])
            _write_candidate_summary(
                repeat1,
                [
                    {"candidate_id": "C1", "survived": 0},
                    {"candidate_id": "C2", "survived": 1},
                ],
            )
            invariant_rows: list[dict[str, object]] = [
                {"candidate_id": "C1", "rudeness_label": "rude"},
                {"candidate_id": "C2", "rudeness_label": "polite"},
            ]
            _write_candidate_labels(repeat0, invariant_rows)
            _write_candidate_labels(repeat1, invariant_rows)

            out_dir = root / "snapshot" / "aggregate"
            manifest = aggregate_snapshot_tables([(0, repeat0), (1, repeat1)], out_dir)

            summary = pq.read_table(
                out_dir / "snapshot_candidate_summary.parquet"
            ).to_pylist()
            labels_table = pq.read_table(out_dir / "snapshot_candidate_labels.parquet")
            labels = labels_table.to_pylist()
            labels_columns = labels_table.column_names
            manifest_ok = all((out_dir / name).is_file() for name in manifest.files)

        # Concatenated per-run table carries every replicate row, tagged.
        self.assertEqual(len(summary), 3)
        self.assertEqual(
            {
                (row["candidate_id"], row["survived"], row[SEED_REPEAT_INDEX_COLUMN])
                for row in summary
            },
            {("C1", 1, 0), ("C1", 0, 1), ("C2", 1, 1)},
        )
        # Invariant candidate-labels table copied verbatim: no seed_repeat_index,
        # no duplication.
        self.assertEqual(labels, invariant_rows)
        self.assertNotIn(SEED_REPEAT_INDEX_COLUMN, labels_columns)
        self.assertTrue(manifest_ok)
        self.assertEqual(manifest.out_dir, out_dir)

    def test_refuses_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repeat0 = root / "repeat-0"
            _write_candidate_summary(repeat0, [{"candidate_id": "C1", "survived": 1}])
            _write_candidate_labels(
                repeat0, [{"candidate_id": "C1", "rudeness_label": "rude"}]
            )
            out_dir = root / "aggregate"
            out_dir.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refused to replace"):
                aggregate_snapshot_tables([(0, repeat0)], out_dir)

    def test_duplicate_repeat_indices_are_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repeat0 = root / "repeat-0"
            repeat0b = root / "repeat-0b"
            _write_candidate_summary(repeat0, [{"candidate_id": "C1", "survived": 1}])
            _write_candidate_summary(repeat0b, [{"candidate_id": "C2", "survived": 0}])
            with self.assertRaisesRegex(ValueError, "not unique"):
                aggregate_snapshot_tables(
                    [(0, repeat0), (0, repeat0b)], root / "aggregate"
                )


if __name__ == "__main__":
    unittest.main()
