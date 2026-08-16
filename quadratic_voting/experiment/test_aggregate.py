"""Tests for the thin seed-repeat Parquet aggregator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from quadratic_voting.experiment.aggregate import (
    SEED_REPEAT_INDEX_COLUMN,
    aggregate_repeats,
)


def _write(directory: Path, name: str, rows: list[dict[str, object]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("run_id", pa.string()), ("value", pa.int64())])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, directory / f"{name}.parquet")


def _write_invariant(directory: Path, rows: list[dict[str, object]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([("candidate_id", pa.string()), ("rudeness_label", pa.string())])
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, directory / "candidate_metadata.parquet")


class AggregateTests(unittest.TestCase):
    def test_concat_adds_seed_repeat_index_and_copies_invariant_tables(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repeat0 = root / "export" / "repeat-0"
            repeat1 = root / "export" / "repeat-1"
            _write(repeat0, "runs", [{"run_id": "r0a", "value": 1}])
            _write(
                repeat1,
                "runs",
                [{"run_id": "r1a", "value": 2}, {"run_id": "r1b", "value": 3}],
            )
            _write(repeat0, "candidate_survival", [{"run_id": "r0a", "value": 9}])
            _write(repeat1, "candidate_survival", [{"run_id": "r1a", "value": 8}])
            # Candidate metadata is identical across replicates (same 5 candidates).
            invariant_rows: list[dict[str, object]] = [
                {"candidate_id": "C1", "rudeness_label": "rude"}
            ]
            _write_invariant(repeat0, invariant_rows)
            _write_invariant(repeat1, invariant_rows)

            out_dir = root / "export" / "aggregate"
            manifest = aggregate_repeats([(0, repeat0), (1, repeat1)], out_dir)

            runs = pq.read_table(out_dir / "runs.parquet").to_pylist()
            survival = pq.read_table(out_dir / "candidate_survival.parquet").to_pylist()
            metadata_table = pq.read_table(out_dir / "candidate_metadata.parquet")
            metadata = metadata_table.to_pylist()
            metadata_columns = metadata_table.column_names
            manifest_ok = all(path.is_file() for path in manifest.files)

        # Concatenated run-scoped table carries every replicate row.
        self.assertEqual(len(runs), 3)
        self.assertEqual(
            {(row["run_id"], row[SEED_REPEAT_INDEX_COLUMN]) for row in runs},
            {("r0a", 0), ("r1a", 1), ("r1b", 1)},
        )
        self.assertEqual(
            {(row["run_id"], row[SEED_REPEAT_INDEX_COLUMN]) for row in survival},
            {("r0a", 0), ("r1a", 1)},
        )
        # Invariant candidate table copied verbatim (no seed_repeat_index, no dupes).
        self.assertEqual(metadata, [{"candidate_id": "C1", "rudeness_label": "rude"}])
        self.assertNotIn(SEED_REPEAT_INDEX_COLUMN, metadata_columns)
        self.assertTrue(manifest_ok)

    def test_missing_table_in_replicate_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repeat0 = root / "repeat-0"
            repeat1 = root / "repeat-1"
            _write(repeat0, "runs", [{"run_id": "r0", "value": 1}])
            _write(repeat0, "candidate_survival", [{"run_id": "r0", "value": 1}])
            _write(repeat1, "runs", [{"run_id": "r1", "value": 1}])
            # repeat1 is missing candidate_survival.parquet
            with self.assertRaisesRegex(ValueError, "missing table candidate_survival"):
                aggregate_repeats([(0, repeat0), (1, repeat1)], root / "aggregate")

    def test_refuses_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repeat0 = root / "repeat-0"
            _write(repeat0, "runs", [{"run_id": "r0", "value": 1}])
            out_dir = root / "aggregate"
            out_dir.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refused to replace"):
                aggregate_repeats([(0, repeat0)], out_dir)

    def test_empty_inputs_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no replicate export directories"):
                aggregate_repeats([], Path(directory) / "aggregate")


if __name__ == "__main__":
    unittest.main()
