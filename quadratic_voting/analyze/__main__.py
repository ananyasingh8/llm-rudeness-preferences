"""Run deterministic snapshot analytics over an existing Parquet export."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

from quadratic_voting.experiment.snapshots import (
    DEFAULT_SNAPSHOT_COUNT,
    build_snapshot_tables,
    render_snapshot_figures,
)
from quadratic_voting.experiment.timeline_flow import render_timeline_html


def _confirm_replacement(out: Path, *, overwrite: bool) -> None:
    if not out.exists() or overwrite:
        return
    if not sys.stdin.isatty():
        raise FileExistsError(
            f"analysis output already exists: {out}; non-interactive replacement "
            "requires --overwrite"
        )
    try:
        answer = input(f"Analysis output already exists at {out}. Replace it? [y/N] ")
    except EOFError as error:
        raise FileExistsError(
            f"analysis output already exists: {out}; confirmation input ended before "
            "replacement was approved. Re-run and answer yes, or pass --overwrite"
        ) from error
    if answer.strip().lower() not in {"y", "yes"}:
        raise FileExistsError(
            f"analysis output replacement cancelled: {out} remains unchanged"
        )


def _publish(staging: Path, out: Path) -> None:
    if not out.exists():
        os.replace(staging, out)
        return
    backup = Path(tempfile.mkdtemp(prefix=f".{out.name}.backup-", dir=out.parent))
    backup.rmdir()
    os.replace(out, backup)
    try:
        os.replace(staging, out)
    except OSError as publish_error:
        try:
            os.replace(backup, out)
        except OSError as restore_error:
            raise OSError(
                f"failed to publish analysis at {out} and could not restore the prior "
                f"artifact; recover it from {backup}"
            ) from restore_error
        raise publish_error
    if backup.is_dir():
        shutil.rmtree(backup)
    else:
        backup.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a quadratic-voting Parquet export without a model or GPU."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output without interactive confirmation",
    )
    parser.add_argument("--snapshot-count", type=int, default=DEFAULT_SNAPSHOT_COUNT)
    args = parser.parse_args(argv)
    try:
        _confirm_replacement(args.out, overwrite=args.overwrite)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{args.out.name}.staging-", dir=args.out.parent)
        )
        table_paths = build_snapshot_tables(
            args.input_dir, staging, snapshot_count=args.snapshot_count
        )
        paths = (
            *table_paths,
            *render_snapshot_figures(staging),
            render_timeline_html(args.input_dir, staging),
        )
        _publish(staging, args.out)
        paths = tuple(args.out / path.name for path in paths)
    except (OSError, ValueError) as error:
        if "staging" in locals():
            shutil.rmtree(staging, ignore_errors=True)
        print(f"error: {error}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
