"""Thin concatenating aggregator over per-replicate seed-repeat exports.

The default pilot samples one five-candidate set, then runs ``repeat`` replicate
matched-sets that reuse those candidates under distinct master seeds. Each
replicate is exported independently with the unchanged single-matched-set
exporter into ``export/repeat-<i>/``. This module concatenates the run- and
analysis-scoped Parquet tables across replicates into ``export/aggregate/``,
tagging every row with its ``seed_repeat_index`` so downstream plots and analysis
can distinguish replicates. Candidate-invariant tables (identical across
replicates because the same five candidates are reused) are copied verbatim from
the first replicate. The aggregator does not touch the database and never mutates
the per-replicate exports.

Note: non-candidate provenance tables (e.g. ``model_definitions``,
``experiment_configurations``) are concatenated like any other run-scoped table,
so each such row is repeated once per replicate and tagged by
``seed_repeat_index``. This is intentional for a thin aggregator and is not
corruption, but consumers that expect a single provenance row must de-duplicate
(e.g. group by ``seed_repeat_index`` or take the first replicate).
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

# Candidate-scoped tables are identical across replicates (the same five
# candidates are reused), so they are copied unchanged from the first replicate
# rather than concatenated. Concatenating them would duplicate rows and corrupt
# per-candidate timeline aggregations.
_INVARIANT_TABLES: frozenset[str] = frozenset(
    {
        "candidate_metadata",
        "source_annotations",
        "candidate_presentations",
        "candidate_source_turns",
    }
)


@dataclass(frozen=True, slots=True)
class AggregateManifest:
    out_dir: Path
    files: tuple[Path, ...]


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def aggregate_repeats(
    inputs: Sequence[tuple[int, Path]], out_dir: Path
) -> AggregateManifest:
    """Concatenate per-replicate Parquet exports into one aggregate directory.

    ``inputs`` pairs each replicate's ``seed_repeat_index`` with its export
    directory. Run- and analysis-scoped tables from the first replicate define the
    table set; each is concatenated across all replicates with a
    ``seed_repeat_index`` column appended. Invariant candidate tables are copied
    from the first replicate. The result is published atomically; the destination
    must not already exist.
    """
    ordered = sorted(inputs, key=lambda item: item[0])
    if not ordered:
        raise ValueError(
            "Seed-repeat aggregation failed because no replicate export directories were "
            "supplied. Validation failed in experiment.aggregate.aggregate_repeats before "
            "reading any Parquet, so no aggregate was written. Export at least one replicate "
            "to export/repeat-<i>/ and retry."
        )
    indices = [index for index, _path in ordered]
    if len(set(indices)) != len(indices):
        raise ValueError(
            "Seed-repeat aggregation failed because replicate indices are not unique: "
            f"{indices}. Validation failed in experiment.aggregate.aggregate_repeats before "
            "reading Parquet, so no aggregate was written. Supply one directory per distinct "
            "seed_repeat_index and retry."
        )
    for index, directory in ordered:
        if not directory.is_dir():
            raise ValueError(
                f"Seed-repeat aggregation failed because replicate {index} export directory "
                f"{directory} does not exist. Validation failed in "
                "experiment.aggregate.aggregate_repeats before reading Parquet, so no "
                "aggregate was written. Complete the replicate export and retry."
            )
    base_dir = ordered[0][1]
    table_names = sorted(path.stem for path in base_dir.glob("*.parquet"))
    if not table_names:
        raise ValueError(
            f"Seed-repeat aggregation failed because the first replicate export {base_dir} "
            "has no Parquet tables. Validation failed in "
            "experiment.aggregate.aggregate_repeats before writing, so no aggregate was "
            "written. Re-export the replicate and retry."
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
                        f"Seed-repeat aggregation failed because replicate {index} is "
                        f"missing table {name}.parquet at {source}. Aggregation stopped in "
                        "experiment.aggregate.aggregate_repeats before writing, so no "
                        "aggregate was published. Every replicate must export the same table "
                        "set produced by experiment.export.export_parquet; re-export the "
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
                f"Seed-repeat aggregation refused to replace existing directory {out_dir}. "
                "Atomic publication stopped in experiment.aggregate.aggregate_repeats while "
                "the complete aggregate remained staged; choose a new output directory or "
                "remove the old aggregate explicitly, then retry."
            )
        os.replace(staging, out_dir)
        _fsync_directory(out_dir.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return AggregateManifest(out_dir, tuple(out_dir / path.name for path in files))
