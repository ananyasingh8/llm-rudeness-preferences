"""Thin concatenating aggregator over per-replicate snapshot-dashboard exports.

``quadratic_voting.experiment.snapshots.build_snapshot_tables`` writes one
snapshot-dashboard Parquet table set per seed-repeat replicate. This module
concatenates those tables across replicates into a single aggregate directory,
tagging every row with its ``seed_repeat_index`` so downstream dashboard code
can distinguish replicates. The candidate-invariant table
(``snapshot_candidate_labels``, identical across replicates because every
replicate reuses the same candidate set) is copied verbatim from the first
replicate instead of being concatenated. The aggregator does not touch the
database and never mutates the per-replicate snapshot exports.

This mirrors ``quadratic_voting.experiment.aggregate.aggregate_repeats``, which
performs the same concatenation for the export tables produced by
``experiment.export.export_parquet``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

SEED_REPEAT_INDEX_COLUMN = "seed_repeat_index"

# The candidate-labels table is identical across replicates (every replicate
# reuses the same five candidates), so it is copied unchanged from the first
# replicate rather than concatenated. Concatenating it would duplicate rows and
# corrupt per-candidate joins in the snapshot dashboard.
_INVARIANT_TABLES: frozenset[str] = frozenset({"snapshot_candidate_labels"})


@dataclass(frozen=True, slots=True)
class AggregateSnapshotManifest:
    out_dir: Path
    files: tuple[str, ...]


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def aggregate_snapshot_tables(
    inputs: Sequence[tuple[int, Path]], out_dir: Path
) -> AggregateSnapshotManifest:
    """Concatenate per-replicate snapshot Parquet tables into one aggregate directory.

    ``inputs`` pairs each replicate's ``seed_repeat_index`` with its snapshot
    output directory (as produced by
    ``experiment.snapshots.build_snapshot_tables``). Tables from the first
    replicate define the table set; each is concatenated across all replicates
    with a ``seed_repeat_index`` column appended, except
    ``snapshot_candidate_labels`` which is copied verbatim from the first
    replicate (candidate-invariant, no ``seed_repeat_index`` column). The result
    is published atomically; the destination must not already exist.
    """
    ordered = sorted(inputs, key=lambda item: item[0])
    if not ordered:
        raise ValueError(
            "Snapshot aggregation failed because no replicate snapshot directories "
            "were supplied. Validation failed in "
            "experiment.snapshot_aggregate.aggregate_snapshot_tables before reading "
            "any Parquet, so no aggregate was written. Build snapshot tables for at "
            "least one seed-repeat replicate and retry."
        )
    indices = [index for index, _path in ordered]
    if len(set(indices)) != len(indices):
        raise ValueError(
            "Snapshot aggregation failed because replicate indices are not unique: "
            f"{indices}. Validation failed in "
            "experiment.snapshot_aggregate.aggregate_snapshot_tables before reading "
            "Parquet, so no aggregate was written. Supply one directory per distinct "
            "seed_repeat_index and retry."
        )
    for index, directory in ordered:
        if not directory.is_dir():
            raise ValueError(
                f"Snapshot aggregation failed because replicate {index} snapshot "
                f"directory {directory} does not exist. Validation failed in "
                "experiment.snapshot_aggregate.aggregate_snapshot_tables before "
                "reading Parquet, so no aggregate was written. Run "
                "experiment.snapshots.build_snapshot_tables for that replicate and "
                "retry."
            )
    base_dir = ordered[0][1]
    table_names = sorted(path.stem for path in base_dir.glob("*.parquet"))
    if not table_names:
        raise ValueError(
            f"Snapshot aggregation failed because the first replicate snapshot "
            f"directory {base_dir} has no Parquet tables. Validation failed in "
            "experiment.snapshot_aggregate.aggregate_snapshot_tables before "
            "writing, so no aggregate was written. Re-run "
            "experiment.snapshots.build_snapshot_tables for that replicate and "
            "retry."
        )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=out_dir.parent)
    )
    files: list[Path] = []
    try:
        for name in table_names:
            destination = staging / f"{name}.parquet"
            if name in _INVARIANT_TABLES:
                shutil.copyfile(base_dir / f"{name}.parquet", destination)
                files.append(destination)
                continue
            tables: list[pa.Table] = []
            for index, directory in ordered:
                source = directory / f"{name}.parquet"
                if not source.is_file():
                    raise ValueError(
                        f"Snapshot aggregation failed because replicate {index} is "
                        f"missing table {name}.parquet at {source}. Aggregation "
                        "stopped in "
                        "experiment.snapshot_aggregate.aggregate_snapshot_tables "
                        "before writing, so no aggregate was published. Every "
                        "replicate must export the same table set produced by "
                        "experiment.snapshots.build_snapshot_tables; re-run the "
                        "incomplete replicate and retry."
                    )
                table = pq.read_table(source)
                table = table.append_column(
                    SEED_REPEAT_INDEX_COLUMN,
                    pa.array([index] * table.num_rows, pa.int64()),
                )
                tables.append(table)
            combined = pa.concat_tables(tables)
            pq.write_table(
                combined,
                destination,
                compression="zstd",
                version="2.6",
                write_statistics=True,
            )
            files.append(destination)

        for path in files:
            _fsync_file(path)
        _fsync_directory(staging)
        if out_dir.exists():
            raise FileExistsError(
                f"Snapshot aggregation refused to replace existing directory "
                f"{out_dir}. Atomic publication stopped in "
                "experiment.snapshot_aggregate.aggregate_snapshot_tables while the "
                "complete aggregate remained staged; choose a new output directory "
                "or remove the old aggregate explicitly, then retry."
            )
        os.replace(staging, out_dir)
        _fsync_directory(out_dir.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return AggregateSnapshotManifest(out_dir, tuple(path.name for path in files))
