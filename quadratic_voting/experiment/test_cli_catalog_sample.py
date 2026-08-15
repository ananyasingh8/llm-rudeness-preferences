"""Production CLI-path integration tests for catalog and sample commands."""

from __future__ import annotations

import argparse
import contextlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path

from quadratic_voting.experiment.cli import catalog_cmds, sample_cmds
from quadratic_voting.experiment.store import acquire_writer_lock
from quadratic_voting.experiment.test_catalog import write_fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    subparsers = parser.add_subparsers(required=True)
    catalog_cmds.register(subparsers)
    sample_cmds.register(subparsers)
    return parser


def invoke(parser: argparse.ArgumentParser, argv: list[str]) -> str:
    args = parser.parse_args(argv)
    output = io.StringIO()
    with acquire_writer_lock(args.db) as lock, contextlib.redirect_stdout(output):
        args.writer_lock = lock
        result = args.handler(args)
    if result != 0:
        raise AssertionError(f"handler returned {result}")
    return output.getvalue().strip()


class CatalogSampleCliTests(unittest.TestCase):
    def test_nested_sample_parser_accepts_odd_sizes_at_least_two(self) -> None:
        args = build_parser().parse_args(
            [
                "--db",
                "fixture.sqlite3",
                "sample",
                "create",
                "--release-id",
                "release",
                "--template-id",
                "template",
                "--size",
                "3",
                "--seed",
                "1",
            ]
        )
        self.assertEqual(args.size, 3)

    def test_full_ingest_template_sample_freeze_validate_flow(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "qv.sqlite3"
            csv_path = root / "fixture.csv"
            artifact = root / "sample.json"
            write_fixture(csv_path)
            parser = build_parser()
            ingest_output = invoke(
                parser,
                [
                    "--db",
                    str(db),
                    "catalog",
                    "ingest",
                    "--dataset-path",
                    str(csv_path),
                    "--dataset-version",
                    "fixture-v1",
                ],
            )
            release_id = ingest_output.removeprefix("release_id=")
            template_output = invoke(parser, ["--db", str(db), "template", "register"])
            self.assertIn("final-result=", template_output)
            connection = sqlite3.connect(db)
            template_id = connection.execute(
                "SELECT template_id FROM presentation_template WHERE name='candidate-card' "
                "AND version='v1'"
            ).fetchone()[0]
            connection.close()
            sample_output = invoke(
                parser,
                [
                    "--db",
                    str(db),
                    "sample",
                    "create",
                    "--release-id",
                    release_id,
                    "--template-id",
                    template_id,
                    "--size",
                    "2",
                    "--seed",
                    "41",
                ],
            )
            self.assertIn("status=DRAFT", sample_output)
            sample_id = sample_output.split()[0].removeprefix("sample_id=")
            freeze_output = invoke(
                parser,
                [
                    "--db",
                    str(db),
                    "sample",
                    "freeze",
                    "--sample-id",
                    sample_id,
                    "--out",
                    str(artifact),
                ],
            )
            self.assertIn(f"artifact={artifact}", freeze_output)
            self.assertTrue(Path(f"{artifact}.provenance.json").is_file())
            validate_output = invoke(
                parser,
                [
                    "--db",
                    str(db),
                    "sample",
                    "verify",
                    "--sample-id",
                    sample_id,
                    "--artifact",
                    str(artifact),
                ],
            )
            self.assertEqual(validate_output, f"sample_id={sample_id}")

    def test_reingest_same_version_surfaces_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "qv.sqlite3"
            csv_path = root / "fixture.csv"
            write_fixture(csv_path)
            parser = build_parser()
            argv = [
                "--db",
                str(db),
                "catalog",
                "ingest",
                "--dataset-path",
                str(csv_path),
                "--dataset-version",
                "fixture-v1",
            ]
            invoke(parser, argv)
            with self.assertRaisesRegex(ValueError, "Choose a new --dataset-version"):
                invoke(parser, argv)


if __name__ == "__main__":
    unittest.main()
