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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a quadratic-voting Parquet export without a model or GPU."
    )
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--snapshot-count", type=int, default=DEFAULT_SNAPSHOT_COUNT)
    args = parser.parse_args(argv)
    try:
        if args.out.exists():
            raise FileExistsError(
                f"analysis output directory already exists: {args.out}; refuse to overwrite "
                "an existing analytics artifact. Choose a new --out path."
            )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{args.out.name}.staging-", dir=args.out.parent)
        )
        table_paths = build_snapshot_tables(
            args.export_dir, staging, snapshot_count=args.snapshot_count
        )
        paths = (
            *table_paths,
            *render_snapshot_figures(staging),
            render_timeline_html(args.export_dir, staging),
        )
        os.replace(staging, args.out)
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
