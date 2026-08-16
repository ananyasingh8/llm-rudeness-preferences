"""Aggregate (pooled-across-repeats) dashboard figures for snapshot rudeness data.

Renders the pooled snapshot-rudeness figures from an aggregated snapshot
directory (the output of
``quadratic_voting.experiment.snapshot_aggregate.aggregate_snapshot_tables``).
Reuses ``quadratic_voting.experiment.snapshot_pooled.pool_snapshot_metric`` for
all statistics (t-based SEM across seed-repeats); no new statistics are
computed here.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pyarrow as pa  # type: ignore[import-untyped]  # noqa: E402
import pyarrow.parquet as pq  # type: ignore[import-untyped]  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from quadratic_voting.experiment.snapshot_pooled import pool_snapshot_metric  # noqa: E402

__all__ = ["render_pooled_snapshot_figures"]

RUDENESS_PALETTE: dict[str, str] = {
    "rude": "#B2182B",
    "non_rude": "#2166AC",
    "ambiguous_tie": "#79706E",
}
REGIME_LINESTYLE: dict[str, str] = {"support": "-", "opposition": "--"}
RUDENESS_ORDER: tuple[str, ...] = ("rude", "ambiguous_tie", "non_rude")
REGIME_ORDER: tuple[str, ...] = ("support", "opposition")

SNAPSHOT_RUDENESS_TABLE = "snapshot_rudeness_summary"
GROUP_KEYS: tuple[str, ...] = ("regime", "rudeness_label", "snapshot_round")

VOTES_VALUE_COLUMN = "mean_current_votes"
CREDITS_VALUE_COLUMN = "mean_current_credits"

VOTES_PLOT_FILE = "pooled_current_votes_by_rudeness.png"
CREDITS_PLOT_FILE = "pooled_current_credits_by_rudeness.png"
POOLED_PARQUET_FILE = "pooled_rudeness_summary.parquet"

_POOLED_PARQUET_SCHEMA = pa.schema(
    [
        ("regime", pa.string()),
        ("rudeness_label", pa.string()),
        ("snapshot_round", pa.int64()),
        ("value_column", pa.string()),
        ("n_repeats", pa.int64()),
        ("mean", pa.float64()),
        ("sem", pa.float64()),
        ("df", pa.int64()),
        ("t_crit", pa.float64()),
        ("ci_lower", pa.float64()),
        ("ci_upper", pa.float64()),
    ]
)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(
            f"snapshot_pooled_plots expected an integer-compatible value, got {value!r}"
        )
    return int(value)


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(
            f"snapshot_pooled_plots expected a numeric value, got {value!r}"
        )
    return float(value)


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise TypeError(
        f"snapshot_pooled_plots expected a list-valued field, got {value!r}"
    )


def _series_for_records(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build one plottable line per (rudeness_label, regime), ordered."""
    by_key: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for record in records:
        rudeness_label = str(record["rudeness_label"])
        regime = str(record["regime"])
        by_key.setdefault((rudeness_label, regime), []).append(record)

    series: list[dict[str, object]] = []
    for rudeness_label in RUDENESS_ORDER:
        for regime in REGIME_ORDER:
            key = (rudeness_label, regime)
            group_records = by_key.get(key)
            if not group_records:
                continue
            ordered = sorted(
                group_records, key=lambda row: _as_int(row["snapshot_round"])
            )
            rounds = [_as_int(row["snapshot_round"]) for row in ordered]
            means = [_as_float(row["mean"]) for row in ordered]
            errors: list[float] = []
            for row in ordered:
                if row["sem"] is not None and row["t_crit"] is not None:
                    errors.append(_as_float(row["t_crit"]) * _as_float(row["sem"]))
                else:
                    errors.append(0.0)
            series.append(
                {
                    "rudeness_label": rudeness_label,
                    "regime": regime,
                    "x": rounds,
                    "y": means,
                    "yerr": errors,
                    "color": RUDENESS_PALETTE.get(rudeness_label, "#79706E"),
                    "linestyle": REGIME_LINESTYLE.get(regime, "-"),
                }
            )
    return series


def _rudeness_regime_legends(
    axis: plt.Axes, rudeness_labels: Sequence[str], regimes: Sequence[str]
) -> None:
    """Attach two proxy legends: color -> rudeness_label, linestyle -> regime."""
    color_handles = [
        Line2D(
            [0],
            [0],
            color=RUDENESS_PALETTE.get(label, "#79706E"),
            marker="o",
            linestyle="",
            label=label,
        )
        for label in rudeness_labels
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=REGIME_LINESTYLE.get(regime, "-"),  # type: ignore[arg-type]
            label=regime,
        )
        for regime in regimes
        if regime in REGIME_LINESTYLE
    ]
    if color_handles:
        rudeness_legend = axis.legend(
            handles=color_handles,
            title="rudeness_label",
            fontsize="x-small",
            loc="upper right",
        )
        axis.add_artist(rudeness_legend)
    if style_handles:
        axis.legend(
            handles=style_handles,
            title="regime",
            fontsize="x-small",
            loc="lower right",
        )


def _render_pooled_metric(
    records: Sequence[Mapping[str, object]], *, title: str, y_label: str
) -> Figure:
    figure, axis = plt.subplots(figsize=(8, 5))
    series = _series_for_records(records)
    if series:
        rudeness_labels: list[str] = []
        regimes: list[str] = []
        for item in series:
            x = [_as_float(value) for value in _as_sequence(item["x"])]
            y = [_as_float(value) for value in _as_sequence(item["y"])]
            yerr = [_as_float(value) for value in _as_sequence(item["yerr"])]
            axis.errorbar(
                x,
                y,
                yerr=yerr,
                color=str(item["color"]),
                linestyle=str(item["linestyle"]),
                marker="o",
                capsize=4,
            )
            label = str(item["rudeness_label"])
            if label not in rudeness_labels:
                rudeness_labels.append(label)
            regime = str(item["regime"])
            if regime not in regimes:
                regimes.append(regime)
        ordered_labels = [label for label in RUDENESS_ORDER if label in rudeness_labels]
        ordered_regimes = [regime for regime in REGIME_ORDER if regime in regimes]
        _rudeness_regime_legends(axis, ordered_labels, ordered_regimes)
    else:
        axis.text(
            0.5,
            0.5,
            "No pooled rudeness data",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
    axis.set_title(title)
    axis.set_xlabel("Snapshot round")
    axis.set_ylabel(y_label)
    return figure


def render_pooled_snapshot_figures(
    aggregate_dir: Path, out_dir: Path
) -> tuple[Path, ...]:
    """Render pooled snapshot-rudeness figures and their summary Parquet.

    Reads ``aggregate_dir`` (the output of
    ``snapshot_aggregate.aggregate_snapshot_tables``) and pools
    ``mean_current_votes`` and ``mean_current_credits`` from the
    ``snapshot_rudeness_summary`` table across seed-repeats, grouped by
    ``(regime, rudeness_label, snapshot_round)``, using
    ``snapshot_pooled.pool_snapshot_metric``.

    Writes two PNGs (``pooled_current_votes_by_rudeness.png`` and
    ``pooled_current_credits_by_rudeness.png``) plus a
    ``pooled_rudeness_summary.parquet`` file containing the concatenated
    pooled records for both value columns, atomically publishing everything
    under ``out_dir`` (staged in a temp directory, then ``os.replace``d into
    place). Refuses to overwrite an existing ``out_dir``.

    Each figure uses a single axis with color = ``rudeness_label``
    (``RUDENESS_PALETTE``) and linestyle = ``regime``
    (``REGIME_LINESTYLE``). If the pooled data is empty, an annotated
    placeholder axis is rendered instead of raising.

    Returns the tuple of published Paths (parquet first, then the two PNGs).

    Raises ``FileExistsError`` if ``out_dir`` already exists, and propagates
    ``FileNotFoundError``/``KeyError`` from ``pool_snapshot_metric`` if the
    aggregated table or its expected columns are missing.
    """
    votes_records = pool_snapshot_metric(
        aggregate_dir,
        SNAPSHOT_RUDENESS_TABLE,
        group_keys=GROUP_KEYS,
        value_column=VOTES_VALUE_COLUMN,
    )
    credits_records = pool_snapshot_metric(
        aggregate_dir,
        SNAPSHOT_RUDENESS_TABLE,
        group_keys=GROUP_KEYS,
        value_column=CREDITS_VALUE_COLUMN,
    )

    votes_figure = _render_pooled_metric(
        votes_records,
        title="Mean current votes by rudeness per round (95% t\u00b7SEM)",
        y_label="Mean current votes",
    )
    credits_figure = _render_pooled_metric(
        credits_records,
        title="Mean current credits by rudeness per round (95% t\u00b7SEM)",
        y_label="Mean current credits",
    )

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=out_dir.parent)
    )
    produced: list[Path] = []
    figures = ((VOTES_PLOT_FILE, votes_figure), (CREDITS_PLOT_FILE, credits_figure))
    try:
        pooled_path = staging / POOLED_PARQUET_FILE
        pq.write_table(
            pa.Table.from_pylist(
                [*votes_records, *credits_records], schema=_POOLED_PARQUET_SCHEMA
            ),
            pooled_path,
        )
        produced.append(pooled_path)
        for filename, figure in figures:
            path = staging / filename
            figure.tight_layout()
            figure.savefig(
                path,
                dpi=120,
                bbox_inches="tight",
                metadata={"Software": "quadratic-voting"},
            )
            produced.append(path)
        for path in produced:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if out_dir.exists():
            raise FileExistsError(
                f"Pooled snapshot figure publication refused to replace existing "
                f"directory {out_dir}; the new PNGs and parquet remain "
                f"unpublished. Choose a new output directory or explicitly "
                f"remove the old complete artifact, then retry."
            )
        os.replace(staging, out_dir)
        parent_descriptor = os.open(out_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        for _, figure in figures:
            plt.close(figure)
    return tuple(out_dir / path.name for path in produced)
