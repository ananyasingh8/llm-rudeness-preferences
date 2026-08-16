"""Tests for :mod:`quadratic_voting.experiment.snapshot_pooled`."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]

from quadratic_voting.experiment.snapshot_pooled import pool_snapshot_metric


def _write_table(directory: Path, name: str, rows: list[dict[str, object]]) -> None:
    arrow_table = pa.Table.from_pylist(rows)
    pq.write_table(arrow_table, directory / f"{name}.parquet")


class TestPoolSnapshotMetric(unittest.TestCase):
    def test_pools_two_groups_over_three_repeats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            aggregate_dir = Path(tmp)
            rows: list[dict[str, object]] = []
            for repeat in range(3):
                for label, base in (("rude", 10.0), ("polite", 20.0)):
                    for round_index in (0, 1):
                        rows.append(
                            {
                                "rudeness_label": label,
                                "snapshot_round": round_index,
                                "mean_current_votes": base + repeat + round_index * 0.5,
                                "seed_repeat_index": repeat,
                            }
                        )
            _write_table(aggregate_dir, "snapshot_metrics", rows)

            records = pool_snapshot_metric(
                aggregate_dir,
                "snapshot_metrics",
                group_keys=["rudeness_label", "snapshot_round"],
                value_column="mean_current_votes",
            )

            by_key = {(r["rudeness_label"], r["snapshot_round"]): r for r in records}
            rude_round0 = by_key[("rude", 0)]
            self.assertEqual(rude_round0["n_repeats"], 3)
            self.assertEqual(rude_round0["df"], 2)
            expected_mean = sum(10.0 + repeat for repeat in range(3)) / 3
            self.assertAlmostEqual(cast(float, rude_round0["mean"]), expected_mean)
            expected_t_crit = float(stats.t.ppf(0.975, 2))
            self.assertAlmostEqual(cast(float, rude_round0["t_crit"]), expected_t_crit)

            polite_round1 = by_key[("polite", 1)]
            self.assertEqual(polite_round1["n_repeats"], 3)
            expected_polite_mean = sum(20.0 + repeat + 0.5 for repeat in range(3)) / 3
            self.assertAlmostEqual(
                cast(float, polite_round1["mean"]), expected_polite_mean
            )

            # Deterministic sorted order by group-key tuple repr.
            keys = [(r["rudeness_label"], r["snapshot_round"]) for r in records]
            self.assertEqual(keys, sorted(keys, key=repr))

    def test_same_repeat_multiple_rows_are_averaged_before_pooling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            aggregate_dir = Path(tmp)
            rows = [
                # Repeat 0 has two rows for the same group -> must collapse to
                # one per-repeat value (average = 15.0) before pooling.
                {
                    "rudeness_label": "rude",
                    "snapshot_round": 0,
                    "mean_current_votes": 10.0,
                    "seed_repeat_index": 0,
                },
                {
                    "rudeness_label": "rude",
                    "snapshot_round": 0,
                    "mean_current_votes": 20.0,
                    "seed_repeat_index": 0,
                },
                {
                    "rudeness_label": "rude",
                    "snapshot_round": 0,
                    "mean_current_votes": 30.0,
                    "seed_repeat_index": 1,
                },
                {
                    "rudeness_label": "rude",
                    "snapshot_round": 0,
                    "mean_current_votes": 40.0,
                    "seed_repeat_index": 2,
                },
            ]
            _write_table(aggregate_dir, "snapshot_metrics", rows)

            records = pool_snapshot_metric(
                aggregate_dir,
                "snapshot_metrics",
                group_keys=["rudeness_label", "snapshot_round"],
                value_column="mean_current_votes",
            )

            self.assertEqual(len(records), 1)
            record = records[0]
            # n_repeats counts distinct repeats (3), not raw rows (4).
            self.assertEqual(record["n_repeats"], 3)
            expected_mean = (15.0 + 30.0 + 40.0) / 3
            self.assertAlmostEqual(cast(float, record["mean"]), expected_mean)

    def test_none_values_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            aggregate_dir = Path(tmp)
            rows = [
                {
                    "rudeness_label": "rude",
                    "snapshot_round": 0,
                    "mean_current_votes": 10.0,
                    "seed_repeat_index": 0,
                },
                {
                    "rudeness_label": "rude",
                    "snapshot_round": 0,
                    "mean_current_votes": None,
                    "seed_repeat_index": 1,
                },
                {
                    "rudeness_label": "rude",
                    "snapshot_round": 0,
                    "mean_current_votes": 30.0,
                    "seed_repeat_index": 2,
                },
                {
                    "rudeness_label": "all_none",
                    "snapshot_round": 0,
                    "mean_current_votes": None,
                    "seed_repeat_index": 0,
                },
            ]
            _write_table(aggregate_dir, "snapshot_metrics", rows)

            records = pool_snapshot_metric(
                aggregate_dir,
                "snapshot_metrics",
                group_keys=["rudeness_label", "snapshot_round"],
                value_column="mean_current_votes",
            )

            # The all-None group must be dropped entirely; the other group
            # keeps only its two non-None repeats.
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["rudeness_label"], "rude")
            self.assertEqual(record["n_repeats"], 2)
            self.assertAlmostEqual(cast(float, record["mean"]), 20.0)

    def test_missing_seed_repeat_index_column_raises_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            aggregate_dir = Path(tmp)
            rows = [
                {
                    "rudeness_label": "rude",
                    "snapshot_round": 0,
                    "mean_current_votes": 10.0,
                }
            ]
            _write_table(aggregate_dir, "snapshot_metrics", rows)

            with self.assertRaises(KeyError) as ctx:
                pool_snapshot_metric(
                    aggregate_dir,
                    "snapshot_metrics",
                    group_keys=["rudeness_label", "snapshot_round"],
                    value_column="mean_current_votes",
                )
            message = str(ctx.exception)
            self.assertIn("seed_repeat_index", message)
            self.assertIn("snapshot_metrics", message)

    def test_missing_table_raises_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            aggregate_dir = Path(tmp)
            with self.assertRaises(FileNotFoundError) as ctx:
                pool_snapshot_metric(
                    aggregate_dir,
                    "does_not_exist",
                    group_keys=["rudeness_label"],
                    value_column="mean_current_votes",
                )
            message = str(ctx.exception)
            self.assertIn("does_not_exist", message)

    def test_missing_group_key_column_raises_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            aggregate_dir = Path(tmp)
            rows = [
                {
                    "rudeness_label": "rude",
                    "mean_current_votes": 10.0,
                    "seed_repeat_index": 0,
                }
            ]
            _write_table(aggregate_dir, "snapshot_metrics", rows)

            with self.assertRaises(KeyError) as ctx:
                pool_snapshot_metric(
                    aggregate_dir,
                    "snapshot_metrics",
                    group_keys=["rudeness_label", "snapshot_round"],
                    value_column="mean_current_votes",
                )
            self.assertIn("snapshot_round", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
