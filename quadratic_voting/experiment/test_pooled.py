"""Tests for across-repeat pooling: t-based SEM, severity mapping, ranking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]

from quadratic_voting.experiment import pooled

_SEVERITY_LABELS = (
    "is_abuse.1",
    "is_abuse.0",
    "is_abuse.-1",
    "is_abuse.-2",
    "is_abuse.-3",
)
_LEVEL_TO_LABEL = dict(zip((1, 0, -1, -2, -3), _SEVERITY_LABELS, strict=True))


def _annotation_rows(
    candidate_id: str, annotators: list[int]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for annotator_index, level in enumerate(annotators):
        active = _LEVEL_TO_LABEL[level]
        for offset, label in enumerate(_SEVERITY_LABELS):
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "annotation_index": annotator_index * 5 + offset,
                    "source_label": label,
                    "source_value": "1" if label == active else "0",
                }
            )
    return rows


def _write(directory: Path, name: str, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), directory / f"{name}.parquet")


class TSemCellTests(unittest.TestCase):
    def test_matches_scipy_for_a_small_sample(self) -> None:
        values = [1.0, 2.0, 3.0, 5.0]
        cell = pooled.t_sem_cell(values)
        self.assertEqual(cell["n_repeats"], 4)
        self.assertEqual(cell["df"], 3)
        self.assertAlmostEqual(float(cell["mean"]), 2.75)  # type: ignore[arg-type]
        self.assertAlmostEqual(float(cell["sem"]), float(stats.sem(values)))  # type: ignore[arg-type]
        self.assertAlmostEqual(
            float(cell["t_crit"]),  # type: ignore[arg-type]
            float(stats.t.ppf(0.975, 3)),
        )
        lower, upper = stats.t.interval(
            0.95, df=3, loc=2.75, scale=float(stats.sem(values))
        )
        self.assertAlmostEqual(float(cell["ci_lower"]), float(lower))  # type: ignore[arg-type]
        self.assertAlmostEqual(float(cell["ci_upper"]), float(upper))  # type: ignore[arg-type]

    def test_single_repeat_has_no_interval(self) -> None:
        cell = pooled.t_sem_cell([4.0])
        self.assertEqual(cell["n_repeats"], 1)
        self.assertEqual(cell["mean"], 4.0)
        self.assertIsNone(cell["sem"])
        self.assertIsNone(cell["ci_lower"])
        self.assertEqual(cell["df"], 0)


class SeverityMappingTests(unittest.TestCase):
    def test_modal_and_tie_toward_more_severe_and_skips_malformed(self) -> None:
        rows = (
            _annotation_rows("A", [1, 1, 0])  # mode 1
            + _annotation_rows("B", [-3])  # single annotator, level -3
            + _annotation_rows("C", [0, -1])  # tie -> more severe (-1)
            + [
                {
                    "candidate_id": "D",
                    "annotation_index": 0,
                    "source_label": "x",
                    "source_value": "1",
                }
            ]  # malformed (not multiple of 5) -> skipped
        )
        levels = pooled.severity_level_by_candidate(rows)
        self.assertEqual(levels["A"], 1)
        self.assertEqual(levels["B"], -3)
        self.assertEqual(levels["C"], -1)
        self.assertNotIn("D", levels)


class AverageRankTests(unittest.TestCase):
    def test_ties_receive_average_rank(self) -> None:
        ranks = pooled._average_ranks({1: 5.0, 0: 5.0, -1: 2.0})
        self.assertEqual(ranks[1], 1.5)  # tie for ranks 1 and 2
        self.assertEqual(ranks[0], 1.5)
        self.assertEqual(ranks[-1], 3.0)


class PooledIntegrationTests(unittest.TestCase):
    def test_pools_survival_and_net_votes_over_generic_repeat_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(
                root,
                "source_annotations",
                _annotation_rows("A", [1]) + _annotation_rows("B", [-3]),
            )
            # Three repeats (generic N=3, not the pilot's 10).
            survival: list[dict[str, object]] = []
            for repeat, a_round, b_round in ((0, 3, 1), (1, 3, 2), (2, 3, 3)):
                survival.append(
                    {
                        "run_id": f"r{repeat}",
                        "regime": "support",
                        "candidate_id": "A",
                        "survival_round": a_round,
                        "seed_repeat_index": repeat,
                    }
                )
                survival.append(
                    {
                        "run_id": f"r{repeat}",
                        "regime": "support",
                        "candidate_id": "B",
                        "survival_round": b_round,
                        "seed_repeat_index": repeat,
                    }
                )
            _write(root, "candidate_survival", survival)
            analysis: list[dict[str, object]] = []
            for repeat in (0, 1, 2):
                for voter in (0, 1):
                    analysis.append(
                        {
                            "candidate_id": "A",
                            "regime": "support",
                            "voter_index": voter,
                            "round_index": 1,
                            "raw_votes": 2,
                            "signed_action": 2,
                            "seed_repeat_index": repeat,
                        }
                    )
                    analysis.append(
                        {
                            "candidate_id": "B",
                            "regime": "support",
                            "voter_index": voter,
                            "round_index": 1,
                            "raw_votes": 1,
                            "signed_action": 1,
                            "seed_repeat_index": repeat,
                        }
                    )
            _write(root, "candidate_analysis", analysis)

            rows = pooled.pooled_by_severity(root)
            survival_a = next(
                r
                for r in rows
                if r["metric"] == "survival_rounds" and r["severity_level"] == 1
            )
            self.assertEqual(survival_a["n_repeats"], 3)
            self.assertEqual(survival_a["df"], 2)
            self.assertAlmostEqual(float(survival_a["mean"]), 3.0)  # type: ignore[arg-type]
            self.assertAlmostEqual(float(survival_a["sem"]), 0.0)  # type: ignore[arg-type]

            survival_b = next(
                r
                for r in rows
                if r["metric"] == "survival_rounds" and r["severity_level"] == -3
            )
            self.assertAlmostEqual(float(survival_b["mean"]), 2.0)  # type: ignore[arg-type]
            self.assertAlmostEqual(
                float(survival_b["t_crit"]),  # type: ignore[arg-type]
                float(stats.t.ppf(0.975, 2)),
            )

            net_a = next(
                r
                for r in rows
                if r["metric"] == "net_signed_votes" and r["severity_level"] == 1
            )
            # each voter casts 2 signed votes; mean over voters = 2 per repeat.
            self.assertAlmostEqual(float(net_a["mean"]), 2.0)  # type: ignore[arg-type]
            self.assertEqual(net_a["n_repeats"], 3)

    def test_rank_records_present_per_regime_and_severity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write(
                root,
                "source_annotations",
                _annotation_rows("A", [1]) + _annotation_rows("B", [-3]),
            )
            analysis: list[dict[str, object]] = []
            for repeat in (0, 1):
                # A gets more votes than B -> A rank 1, B rank 2 each repeat.
                analysis.append(
                    {
                        "candidate_id": "A",
                        "regime": "support",
                        "voter_index": 0,
                        "round_index": 1,
                        "raw_votes": 5,
                        "signed_action": 5,
                        "seed_repeat_index": repeat,
                    }
                )
                analysis.append(
                    {
                        "candidate_id": "B",
                        "regime": "support",
                        "voter_index": 0,
                        "round_index": 1,
                        "raw_votes": 1,
                        "signed_action": 1,
                        "seed_repeat_index": repeat,
                    }
                )
            _write(root, "candidate_analysis", analysis)
            records = pooled.rank_records_by_severity(root)
            support = {
                r["severity_level"]: r for r in records if r["regime"] == "support"
            }
            self.assertEqual(support[1]["median_rank"], [1.0])
            self.assertEqual(support[-3]["median_rank"], [2.0])


if __name__ == "__main__":
    unittest.main()
