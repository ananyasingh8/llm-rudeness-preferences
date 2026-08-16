"""Tests for quadratic_voting.experiment.snapshot_pooled_plots."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from quadratic_voting.experiment.snapshot_pooled_plots import (
    CREDITS_PLOT_FILE,
    POOLED_PARQUET_FILE,
    VOTES_PLOT_FILE,
    render_pooled_snapshot_figures,
)

_SCHEMA = pa.schema(
    [
        ("regime", pa.string()),
        ("rudeness_label", pa.string()),
        ("snapshot_round", pa.int64()),
        ("mean_current_votes", pa.float64()),
        ("mean_current_credits", pa.float64()),
        ("seed_repeat_index", pa.int64()),
    ]
)


def _write_aggregate(aggregate_dir: Path, rows: list[dict[str, object]]) -> None:
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    pq.write_table(table, aggregate_dir / "snapshot_rudeness_summary.parquet")


def _synthetic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    value = 0.0
    for regime in ("support", "opposition"):
        for rudeness_label in ("rude", "non_rude"):
            for snapshot_round in (0, 1):
                for repeat in range(3):
                    value += 1.0
                    rows.append(
                        {
                            "regime": regime,
                            "rudeness_label": rudeness_label,
                            "snapshot_round": snapshot_round,
                            "mean_current_votes": value,
                            "mean_current_credits": value * 10.0,
                            "seed_repeat_index": repeat,
                        }
                    )
    return rows


class RenderPooledSnapshotFiguresTests(unittest.TestCase):
    def test_renders_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aggregate_dir = root / "aggregate"
            out_dir = root / "out"
            _write_aggregate(aggregate_dir, _synthetic_rows())

            produced = render_pooled_snapshot_figures(aggregate_dir, out_dir)

            produced_names = {path.name for path in produced}
            self.assertIn(VOTES_PLOT_FILE, produced_names)
            self.assertIn(CREDITS_PLOT_FILE, produced_names)
            self.assertIn(POOLED_PARQUET_FILE, produced_names)

            votes_path = out_dir / VOTES_PLOT_FILE
            credits_path = out_dir / CREDITS_PLOT_FILE
            parquet_path = out_dir / POOLED_PARQUET_FILE
            self.assertGreater(votes_path.stat().st_size, 0)
            self.assertGreater(credits_path.stat().st_size, 0)
            self.assertGreater(parquet_path.stat().st_size, 0)

            pooled_table = pq.read_table(parquet_path).to_pylist()
            value_columns = {row["value_column"] for row in pooled_table}
            self.assertEqual(
                value_columns, {"mean_current_votes", "mean_current_credits"}
            )
            for row in pooled_table:
                self.assertIn("mean", row)
                self.assertIn("ci_lower", row)
                self.assertIn("ci_upper", row)
                self.assertEqual(row["n_repeats"], 3)

            with self.assertRaises(FileExistsError):
                render_pooled_snapshot_figures(aggregate_dir, out_dir)

    def test_empty_input_renders_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aggregate_dir = root / "aggregate"
            out_dir = root / "out"
            _write_aggregate(aggregate_dir, [])

            produced = render_pooled_snapshot_figures(aggregate_dir, out_dir)

            produced_names = {path.name for path in produced}
            self.assertIn(VOTES_PLOT_FILE, produced_names)
            self.assertIn(CREDITS_PLOT_FILE, produced_names)
            for path in produced:
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
