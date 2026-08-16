"""Unit tests for the non-GPU surface of quadratic_voting_v2.

Runs without torch or a model: quadratic_voting_v2.main keeps every torch /
transformers import inside the GPU code paths, so the draw, prompt, parsing,
and resume logic here import cleanly on any machine with pandas + pyarrow.

Run from the repo root:
  python3 -m unittest quadratic_voting_v2.test_qv2 -v
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from quadratic_voting_v2 import main as qv
from quadratic_voting_v2.prompts import FRAME_SENTENCES, FRAMES


class TestCandidateDraw(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.eligible = qv.load_eligible()
        cls.rounds = qv.draw_rounds(cls.eligible)

    def test_eligibility_filter(self) -> None:
        self.assertTrue(
            (self.eligible["severity_std"].fillna(0) <= qv.MAX_SEVERITY_STD).all()
        )

    def test_shape_one_per_band_per_round(self) -> None:
        self.assertEqual(len(self.rounds), qv.N_ROUNDS * len(qv.BANDS))
        per_round = self.rounds.groupby("round_index")["band"].apply(sorted)
        for bands in per_round:
            self.assertEqual(bands, sorted(qv.BANDS))

    def test_no_repeats_within_band_across_rounds(self) -> None:
        for band, group in self.rounds.groupby("band"):
            self.assertEqual(
                group["snippet_id"].nunique(), qv.N_ROUNDS, f"band {band}"
            )

    def test_drawn_snippets_respect_std_filter(self) -> None:
        eligible_ids = set(self.eligible["snippet_id"])
        self.assertTrue(set(self.rounds["snippet_id"]) <= eligible_ids)

    def test_deterministic(self) -> None:
        again = qv.draw_rounds(qv.load_eligible())
        pd.testing.assert_frame_equal(self.rounds, again)


class TestQuadraticCost(unittest.TestCase):
    def test_cost_is_sum_of_squares(self) -> None:
        self.assertEqual(
            qv.ballot_cost({"A": 5, "B": 3, "C": 0, "D": 1, "E": 0}), 35
        )

    def test_full_budget_accepted(self) -> None:
        votes, _ = qv.parse_ballot(
            '{"votes": {"A": 10, "B": 0, "C": 0, "D": 0, "E": 0}, "reason": "x"}'
        )
        self.assertEqual(qv.ballot_cost(votes), qv.BUDGET)

    def test_over_budget_rejected(self) -> None:
        with self.assertRaisesRegex(qv.BallotError, "101 credits"):
            qv.parse_ballot(
                '{"votes": {"A": 10, "B": 1, "C": 0, "D": 0, "E": 0},'
                ' "reason": "x"}'
            )

    def test_abstention_accepted(self) -> None:
        votes, _ = qv.parse_ballot(
            '{"votes": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}, "reason": "x"}'
        )
        self.assertEqual(qv.ballot_cost(votes), 0)


class TestBallotParsing(unittest.TestCase):
    def test_valid(self) -> None:
        votes, reason = qv.parse_ballot(
            '{"votes": {"A": 2, "B": 0, "C": 4, "D": 1, "E": 0},'
            ' "reason": "C was hostile"}'
        )
        self.assertEqual(votes, {"A": 2, "B": 0, "C": 4, "D": 1, "E": 0})
        self.assertEqual(reason, "C was hostile")

    def test_code_fenced_json_accepted(self) -> None:
        text = (
            '```json\n{"votes": {"A": 1, "B": 0, "C": 0, "D": 0, "E": 0},'
            ' "reason": "x"}\n```'
        )
        votes, _ = qv.parse_ballot(text)
        self.assertEqual(votes["A"], 1)

    def test_malformed_rejected(self) -> None:
        for bad in (
            "I vote to remove C.",
            '{"votes": [1, 2, 3, 4, 5]}',
            '{"reason": "no votes"}',
            '{"votes": {"A": 1, "B": 0, "C": 0, "D": 0}}',  # missing E
            '{"votes": {"A": 1.5, "B": 0, "C": 0, "D": 0, "E": 0}}',
            '{"votes": {"A": -1, "B": 0, "C": 0, "D": 0, "E": 0}}',
            '{"votes": {"A": true, "B": 0, "C": 0, "D": 0, "E": 0}}',
        ):
            with self.assertRaises(qv.BallotError, msg=bad):
                qv.parse_ballot(bad)

    def test_missing_reason_tolerated(self) -> None:
        _, reason = qv.parse_ballot(
            '{"votes": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}}'
        )
        self.assertEqual(reason, "")


class TestPresentationOrder(unittest.TestCase):
    def test_deterministic(self) -> None:
        for frame in FRAMES:
            self.assertEqual(
                qv.presentation_order(frame, 7, 1, 3),
                qv.presentation_order(frame, 7, 1, 3),
                frame,
            )

    def test_is_a_permutation(self) -> None:
        order = qv.presentation_order("remove", 0, 0, 0)
        self.assertEqual(sorted(order), list(range(len(qv.BANDS))))

    def test_varies_across_ballots_and_frames(self) -> None:
        orders = {
            qv.presentation_order(frame, round_index, voter, repeat)
            for frame in FRAMES
            for round_index in range(5)
            for voter in range(qv.N_VOTERS)
            for repeat in range(qv.N_REPEATS)
        }
        self.assertGreater(len(orders), 1)

    def test_ballot_text_differs_only_in_frame_sentence(self) -> None:
        rounds = pd.DataFrame(
            [
                {
                    "round_index": 0,
                    "band": band,
                    "snippet_id": f"s{band}",
                    "prev_agent": "hi",
                    "prev_user": "hello",
                    "agent": "how are you",
                    "user": f"user text {band}",
                }
                for band in qv.BANDS
            ]
        )
        texts = {}
        for frame in FRAMES:
            # Same (round, voter, repeat) so only frame-derived parts differ;
            # normalize the frame-seeded card order away by fixing it.
            assignment = qv.assign_letters(rounds, "remove", 0, 0, 0)
            texts[frame] = qv.build_ballot_text(frame, assignment)
        stripped = {
            frame: text.replace(FRAME_SENTENCES[frame], "<FRAME>")
            for frame, text in texts.items()
        }
        self.assertEqual(stripped["remove"], stripped["keep"])


class TestResume(unittest.TestCase):
    def test_done_keys_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for round_index, voter, repeat in [(0, 0, 0), (0, 0, 1), (3, 2, 4)]:
                qv.append_ballot(
                    run_dir,
                    {
                        "frame": "remove",
                        "round_index": round_index,
                        "voter": voter,
                        "repeat": repeat,
                        "presentation_order": "{}",
                        "votes_by_snippet": "",
                        **{
                            qv.BAND_VOTE_COLUMNS[band]: ""
                            for band in qv.BANDS
                        },
                        "credits_spent": "",
                        "valid": False,
                        "failure_reason": "test",
                        "retry_count": 3,
                        "raw_response": "x",
                    },
                )
            done = qv.load_done(run_dir)
            self.assertEqual(done, {"0|0|0", "0|0|1", "3|2|4"})
            todo = [
                task
                for task in qv.build_tasks(limit=None)
                if qv.task_key(*task) not in done
            ]
            self.assertEqual(len(todo), qv.N_ROUNDS * qv.N_VOTERS * qv.N_REPEATS - 3)
            self.assertNotIn((0, 0, 0), todo)
            self.assertIn((0, 0, 2), todo)

    def test_empty_run_dir_has_no_done_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(qv.load_done(Path(tmp)), set())

    def test_ballot_csv_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            assignment = {
                letter: {"snippet_id": f"s{i}", "band": qv.BANDS[i]}
                for i, letter in enumerate(qv.LETTERS)
            }
            votes = {"A": 3, "B": 0, "C": 2, "D": 0, "E": 1}
            qv.append_ballot(
                run_dir,
                qv.ballot_row(
                    "remove", 4, 1, 2, assignment, votes, "", 1, "raw"
                ),
            )
            with (run_dir / qv.BALLOTS_FILE).open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["valid"], "True")
            self.assertEqual(int(row["credits_spent"]), 14)
            self.assertEqual(
                json.loads(row["votes_by_snippet"]),
                {"s0": 3, "s1": 0, "s2": 2, "s3": 0, "s4": 1},
            )
            self.assertEqual(int(row["votes_band_p1"]), 3)  # A holds band +1
            self.assertEqual(int(row["votes_band_m3"]), 1)  # E holds band -3


if __name__ == "__main__":
    unittest.main()
