"""End-to-end orchestration for the pooled-across-repeats snapshot dashboard.

Wires the three existing building blocks together for a directory of
per-repeat exports (``export_root/repeat-<i>``):

1. ``quadratic_voting.experiment.snapshots.build_snapshot_tables`` — builds
   snapshot tables for one single-matched-set export.
2. ``quadratic_voting.experiment.snapshot_aggregate.aggregate_snapshot_tables``
   — concatenates the per-repeat snapshot tables, tagging every row with its
   ``seed_repeat_index``.
3. ``quadratic_voting.experiment.snapshot_pooled_plots.render_pooled_snapshot_figures``
   — renders the pooled figures and summary Parquet from the aggregate.

This module performs no statistics of its own; it only discovers repeat
directories, stages intermediate per-repeat and aggregate tables in a
temporary working directory, and cleans that working directory up afterward.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path

from quadratic_voting.experiment.snapshot_aggregate import aggregate_snapshot_tables
from quadratic_voting.experiment.snapshot_pooled_plots import (
    render_pooled_snapshot_figures,
)
from quadratic_voting.experiment.snapshots import build_snapshot_tables

__all__ = ["run_aggregate_dashboard"]

_REPEAT_DIR_PATTERN = re.compile(r"^repeat-(\d+)$")


def _discover_repeat_dirs(export_root: Path) -> list[tuple[int, Path]]:
    if not export_root.is_dir():
        raise ValueError(
            f"aggregate dashboard failed because export root {export_root} does not "
            "exist or is not a directory. Validation failed in "
            "experiment.aggregate_dashboard.run_aggregate_dashboard before building "
            "any snapshot tables. Pass the directory that contains repeat-<i> "
            "subdirectories and retry."
        )
    discovered: list[tuple[int, Path]] = []
    for child in export_root.iterdir():
        if not child.is_dir():
            continue
        match = _REPEAT_DIR_PATTERN.match(child.name)
        if match is None:
            continue
        discovered.append((int(match.group(1)), child))
    if not discovered:
        raise ValueError(
            f"aggregate dashboard failed because export root {export_root} contains "
            "no repeat-<i> subdirectories. Validation failed in "
            "experiment.aggregate_dashboard.run_aggregate_dashboard before building "
            "any snapshot tables. Export at least one seed-repeat replicate to "
            f"{export_root}/repeat-0 (etc.) and retry."
        )
    discovered.sort(key=lambda item: item[0])
    return discovered


def run_aggregate_dashboard(export_root: Path, out_dir: Path) -> tuple[Path, ...]:
    """Build the pooled-across-repeats snapshot dashboard from raw exports.

    ``export_root`` must contain one ``repeat-<i>`` subdirectory per
    seed-repeat replicate export (as produced by
    ``experiment.export.export_parquet``). For each replicate, builds
    snapshot tables via ``snapshots.build_snapshot_tables``, concatenates
    them across replicates via
    ``snapshot_aggregate.aggregate_snapshot_tables``, and renders the pooled
    figures via ``snapshot_pooled_plots.render_pooled_snapshot_figures`` into
    ``out_dir`` (published atomically by that function; refuses to overwrite
    an existing ``out_dir``).

    All intermediate per-repeat snapshot tables and the concatenated
    aggregate are staged in a temporary working directory that is removed
    before returning, regardless of success or failure.

    Returns the tuple of published Paths under ``out_dir``.
    """
    repeat_dirs = _discover_repeat_dirs(export_root)
    work_dir = Path(tempfile.mkdtemp(prefix="aggregate-dashboard-"))
    try:
        snapshot_inputs: list[tuple[int, Path]] = []
        for index, repeat_dir in repeat_dirs:
            snapshot_dir = work_dir / f"snap-{index}"
            build_snapshot_tables(repeat_dir, snapshot_dir)
            snapshot_inputs.append((index, snapshot_dir))
        aggregate_dir = work_dir / "aggregate"
        aggregate_snapshot_tables(snapshot_inputs, aggregate_dir)
        return render_pooled_snapshot_figures(aggregate_dir, out_dir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
