"""Typed dispatcher for the quadratic-voting experiment CLI."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from pathlib import Path

from llm_runtime import ModelRouteError
from llm_runtime.transformers import TransformersRuntimeError
from quadratic_voting.experiment.cli import (
    catalog_cmds,
    export_cmds,
    pipeline_cmds,
    run_cmds,
    sample_cmds,
)
from quadratic_voting.experiment.store import acquire_writer_lock
from quadratic_voting.experiment.types import SamplingProfile, VoterGenerator

GeneratorFactory = Callable[[SamplingProfile], VoterGenerator]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quadratic_voting.experiment.cli",
        description="Run and audit resumable quadratic-voting experiments.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("quadratic_voting/data/qv.sqlite3"),
        help="SQLite experiment database (default: %(default)s)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    migrate = subparsers.add_parser(
        "migrate", help="create or validate the current SQLite schema"
    )
    migrate.set_defaults(handler=_migrate, mutates_db=True)
    catalog_cmds.register(subparsers)
    sample_cmds.register(subparsers)
    run_cmds.register(subparsers)
    export_cmds.register(subparsers)
    pipeline_cmds.register(subparsers)
    return parser


def _migrate(args: argparse.Namespace) -> int:
    """Open and close the store while the dispatcher holds the common lock."""
    from quadratic_voting.experiment.store import open_sqlite_store

    with open_sqlite_store(
        args.db,
        writer_lock=args.writer_lock,
        require_writer_lock=True,
    ):
        pass
    print(f"db={args.db} schema=current")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    generator_factory: GeneratorFactory | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    args.generator_factory = generator_factory
    try:
        if args.command == "run" and run_cmds.complete_without_writer_lock(args):
            return 0
        lock = (
            acquire_writer_lock(args.db, command=args.command)
            if getattr(args, "mutates_db", False)
            else nullcontext()
        )
        with lock as writer_lock:
            args.writer_lock = writer_lock
            return int(args.handler(args))
    except (
        ModelRouteError,
        TransformersRuntimeError,
        ValueError,
        RuntimeError,
        sqlite3.Error,
        OSError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
