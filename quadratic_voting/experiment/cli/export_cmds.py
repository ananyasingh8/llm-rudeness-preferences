"""Register Parquet export and static-plot CLI commands."""

from __future__ import annotations

from argparse import _SubParsersAction
from pathlib import Path
from typing import Any
from contextlib import AbstractContextManager

from quadratic_voting.experiment.export import export_parquet
from quadratic_voting.experiment.plots import render_plots
from quadratic_voting.experiment.store import open_sqlite_store


class _ArtifactLock(AbstractContextManager["_ArtifactLock"]):
    def __init__(self, output: Path) -> None:
        self._output = output
        self._stream: Any = None

    def __enter__(self) -> "_ArtifactLock":
        try:
            import fcntl
        except ImportError as error:  # pragma: no cover - non-POSIX only
            raise RuntimeError(
                "Plot artifact locking failed because fcntl is unavailable in "
                "cli.export_cmds._ArtifactLock before output creation. Run plotting on "
                "a POSIX system with flock support."
            ) from error
        self._output.parent.mkdir(parents=True, exist_ok=True)
        path = self._output.parent / f".{self._output.name}.plot.lock"
        self._stream = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._stream.close()
            raise RuntimeError(
                f"Plot creation refused {self._output} because another process holds "
                f"{path}. Lock acquisition failed before rendering, so output was not "
                "interleaved. Wait for that process to finish and retry."
            ) from error
        return self

    def __exit__(self, *exc: object) -> None:
        if self._stream is not None:
            self._stream.close()


def _export(args: Any) -> int:
    with open_sqlite_store(
        Path(args.db), writer_lock=args.writer_lock, require_writer_lock=True
    ) as store:
        manifest = export_parquet(
            store, Path(args.out), matched_set_id=str(args.matched_set)
        )
    for path in manifest.files:
        print(path)
    return 0


def _plot(args: Any) -> int:
    output = Path(args.out)
    with _ArtifactLock(output):
        for path in render_plots(Path(args.export_dir), output):
            print(path)
    return 0


def register(subparsers: _SubParsersAction[Any]) -> None:
    """Add whole-database ``export`` and Parquet-backed ``plot`` commands."""
    export_parser = subparsers.add_parser("export", help="export the SQLite database")
    export_parser.add_argument("--matched-set", required=True)
    export_parser.add_argument("--out", type=Path, required=True)
    export_parser.set_defaults(handler=_export, mutates_db=True)

    plot_parser = subparsers.add_parser(
        "plot", help="render static plots and the timeline.html from Parquet"
    )
    plot_parser.add_argument("--export-dir", type=Path, required=True)
    plot_parser.add_argument("--out", type=Path, required=True)
    plot_parser.set_defaults(handler=_plot)
