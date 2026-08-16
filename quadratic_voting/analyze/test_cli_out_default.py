"""Hermetic unit tests for the analyze CLI's --out default resolution."""

from __future__ import annotations

import unittest
from pathlib import Path

from quadratic_voting.analyze.__main__ import _resolve_out


class ResolveOutTests(unittest.TestCase):
    def test_explicit_out_returned_unchanged_aggregate(self) -> None:
        input_dir = Path("/tmp/example-input")
        explicit = Path("/tmp/explicit-out")
        self.assertEqual(_resolve_out(input_dir, explicit, aggregate=True), explicit)

    def test_explicit_out_returned_unchanged_single(self) -> None:
        input_dir = Path("/tmp/example-input")
        explicit = Path("/tmp/explicit-out")
        self.assertEqual(_resolve_out(input_dir, explicit, aggregate=False), explicit)

    def test_default_aggregate_mode(self) -> None:
        input_dir = Path("/tmp/example-input")
        self.assertEqual(
            _resolve_out(input_dir, None, aggregate=True),
            input_dir / "aggregate-dashboard",
        )

    def test_default_single_export_mode(self) -> None:
        input_dir = Path("/tmp/example-input")
        self.assertEqual(
            _resolve_out(input_dir, None, aggregate=False),
            input_dir / "analysis",
        )


if __name__ == "__main__":
    unittest.main()
