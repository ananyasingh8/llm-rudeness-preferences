"""Pool a numeric snapshot metric across seed-repeats from an aggregated table.

``quadratic_voting.experiment.snapshot_aggregate.aggregate_snapshot_tables``
concatenates per-replicate snapshot-dashboard Parquet tables into one aggregate
directory, tagging every row with its ``seed_repeat_index``. This module reads
one such aggregated table and pools a numeric column across the repeat axis,
grouped by an arbitrary set of key columns, reusing the existing t-based SEM
helper (:func:`quadratic_voting.experiment.pooled.t_sem_cell`) rather than
reimplementing statistics.

Consistent with the two-stage collapse described in
``quadratic_voting/METHODS.md``: each seed-repeat contributes exactly ONE value
per group to the pool. If a group has multiple rows for the same
``seed_repeat_index`` (e.g. multiple candidates or voters folded into one
group), those rows are first averaged into a single per-repeat value before
pooling across repeats.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from quadratic_voting.experiment.snapshot_aggregate import SEED_REPEAT_INDEX_COLUMN
from quadratic_voting.experiment import pooled

__all__ = ["pool_snapshot_metric"]


def pool_snapshot_metric(
    aggregate_dir: Path,
    table: str,
    *,
    group_keys: Sequence[str],
    value_column: str,
) -> list[dict[str, object]]:
    """Pool ``value_column`` across seed-repeats, grouped by ``group_keys``.

    Reads ``aggregate_dir / f"{table}.parquet"`` (a table produced by
    ``snapshot_aggregate.aggregate_snapshot_tables``, which carries a
    ``seed_repeat_index`` column). Rows are grouped by the tuple of
    ``group_keys`` values. Within each group, rows are further split by
    ``seed_repeat_index``; multiple rows for the same repeat are averaged into
    one per-repeat value (a two-stage collapse consistent with
    ``quadratic_voting/METHODS.md``) so that each repeat contributes exactly one
    value to the pool. Rows whose ``value_column`` is ``None`` are skipped.

    For each group, the per-repeat values are pooled with
    ``pooled.t_sem_cell``, which returns ``{n_repeats, mean, sem, df, t_crit,
    ci_lower, ci_upper}``. Each returned record is
    ``{**group_key_values, "value_column": value_column, **t_sem_cell(...)}``.

    Groups with zero surviving values (e.g. all ``None``) are omitted. Records
    are returned in a deterministic order: sorted by the group-key tuple, using
    ``repr`` as the sort key to keep the ordering stable across mixed value
    types (e.g. ints and strings together).

    Raises a ``FileNotFoundError`` if the table's Parquet file does not exist,
    and a ``KeyError`` if ``seed_repeat_index``, any ``group_keys`` column, or
    ``value_column`` is absent from the table's schema.
    """
    table_path = aggregate_dir / f"{table}.parquet"
    if not table_path.exists():
        raise FileNotFoundError(
            f"Snapshot pooling failed because the aggregated table "
            f"'{table}.parquet' was not found under {aggregate_dir}. This "
            f"happens when snapshot_pooled.pool_snapshot_metric is called "
            f"before experiment.snapshot_aggregate.aggregate_snapshot_tables "
            f"has been run for this aggregate directory, or when 'table' "
            f"({table!r}) is misspelled. Run the snapshot aggregation step "
            f"first, or pass the correct table name, and retry."
        )

    parquet_table = pq.read_table(table_path)
    column_names = set(parquet_table.column_names)

    required_columns = {SEED_REPEAT_INDEX_COLUMN, value_column, *group_keys}
    missing = sorted(required_columns - column_names)
    if missing:
        raise KeyError(
            f"Snapshot pooling failed because the aggregated table "
            f"'{table}.parquet' under {aggregate_dir} is missing required "
            f"column(s) {missing!r}. This happens when 'group_keys' or "
            f"'value_column' do not match the table's actual schema (available "
            f"columns: {sorted(column_names)!r}), or when the table was "
            f"aggregated without a 'seed_repeat_index' column (e.g. it is a "
            f"candidate-invariant table copied verbatim, not a per-repeat "
            f"table). Pass column names that exist in the table, or pool a "
            f"table that carries 'seed_repeat_index', and retry."
        )

    rows = parquet_table.to_pylist()

    # (group_key_tuple) -> (seed_repeat_index) -> list of raw values for that repeat
    per_group_per_repeat: dict[tuple[object, ...], dict[int, list[float]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    group_key_values: dict[tuple[object, ...], dict[str, object]] = {}

    for row in rows:
        value = row[value_column]
        if value is None:
            continue
        group_key = tuple(row[key] for key in group_keys)
        repeat_index = int(row[SEED_REPEAT_INDEX_COLUMN])
        per_group_per_repeat[group_key][repeat_index].append(float(value))
        group_key_values.setdefault(
            group_key, dict(zip(group_keys, group_key, strict=True))
        )

    records: list[dict[str, object]] = []
    for group_key in sorted(per_group_per_repeat, key=repr):
        per_repeat = per_group_per_repeat[group_key]
        # Two-stage collapse: average multiple rows within the same repeat into
        # a single per-repeat value before pooling across repeats.
        per_repeat_values = [
            sum(values) / len(values) for values in per_repeat.values()
        ]
        if not per_repeat_values:
            continue
        records.append(
            {
                **group_key_values[group_key],
                "value_column": value_column,
                **pooled.t_sem_cell(per_repeat_values),
            }
        )

    return records
