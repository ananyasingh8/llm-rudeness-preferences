"""Tests for quadratic_voting.experiment.aggregate_dashboard."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from quadratic_voting.experiment.aggregate_dashboard import run_aggregate_dashboard

_SUMMARY_SCHEMA = pa.schema(
    [
        ("regime", pa.string()),
        ("rudeness_label", pa.string()),
        ("snapshot_round", pa.int64()),
        ("mean_current_votes", pa.float64()),
        ("mean_current_credits", pa.float64()),
    ]
)

_LABELS_SCHEMA = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("candidate_label", pa.string()),
    ]
)


def _fake_build_snapshot_tables(
    export_dir: Path, out_dir: Path, *, snapshot_count: int = 3
) -> tuple[Path, ...]:
    """Stand-in for snapshots.build_snapshot_tables: ignores export_dir contents."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # Vary values by which snapshot dir this is (derived from out_dir name suffix)
    salt = float(out_dir.name.rsplit("-", 1)[-1])
    summary_rows = [
        {
            "regime": "support",
            "rudeness_label": "rude",
            "snapshot_round": 0,
            "mean_current_votes": 1.0 + salt,
            "mean_current_credits": 10.0 + salt,
        },
        {
            "regime": "opposition",
            "rudeness_label": "non_rude",
            "snapshot_round": 0,
            "mean_current_votes": 2.0 + salt,
            "mean_current_credits": 20.0 + salt,
        },
    ]
    summary_table = pa.Table.from_pylist(summary_rows, schema=_SUMMARY_SCHEMA)
    summary_path = out_dir / "snapshot_rudeness_summary.parquet"
    pq.write_table(summary_table, summary_path)

    labels_rows = [{"candidate_id": "c1", "candidate_label": "Candidate One"}]
    labels_table = pa.Table.from_pylist(labels_rows, schema=_LABELS_SCHEMA)
    labels_path = out_dir / "snapshot_candidate_labels.parquet"
    pq.write_table(labels_table, labels_path)

    return (summary_path, labels_path)


class RunAggregateDashboardTests(unittest.TestCase):
    def test_discovers_repeats_and_renders_pooled_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export_root = Path(tmp) / "export_root"
            (export_root / "repeat-0").mkdir(parents=True)
            (export_root / "repeat-1").mkdir(parents=True)
            out_dir = Path(tmp) / "out"

            with patch(
                "quadratic_voting.experiment.aggregate_dashboard.build_snapshot_tables",
                side_effect=_fake_build_snapshot_tables,
            ) as mock_build:
                paths = run_aggregate_dashboard(export_root, out_dir)

            self.assertEqual(mock_build.call_count, 2)
            names = {path.name for path in paths}
            self.assertIn("pooled_current_votes_by_rudeness.png", names)
            self.assertIn("pooled_current_credits_by_rudeness.png", names)
            self.assertIn("pooled_rudeness_summary.parquet", names)

            self.assertTrue((out_dir / "pooled_current_votes_by_rudeness.png").exists())
            self.assertTrue(
                (out_dir / "pooled_current_credits_by_rudeness.png").exists()
            )
            self.assertTrue((out_dir / "pooled_rudeness_summary.parquet").exists())

    def test_raises_actionable_error_when_no_repeat_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            export_root = Path(tmp) / "export_root"
            export_root.mkdir()
            (export_root / "not-a-repeat").mkdir()
            out_dir = Path(tmp) / "out"

            with self.assertRaisesRegex(ValueError, "repeat-<i>"):
                run_aggregate_dashboard(export_root, out_dir)


if __name__ == "__main__":
    unittest.main()
