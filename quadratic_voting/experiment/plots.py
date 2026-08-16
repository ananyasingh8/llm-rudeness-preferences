"""Semantic-deterministic headless plots over analysis Parquet exports."""

from __future__ import annotations

import json
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

from quadratic_voting.experiment import pooled  # noqa: E402
from quadratic_voting.experiment.timeline_flow import render_export_timeline  # noqa: E402


PLOT_MANIFEST_VERSION = "qv-plot-manifest/v1"
PLOT_STYLE = "qv-static/v1"
PALETTE = {
    "action-only": "#4C78A8",
    "statement-then-action": "#F58518",
    "action-then-statement": "#54A24B",
    "error": "#E45756",
    "neutral": "#79706E",
}
ARM_ORDER = ("action-only", "statement-then-action", "action-then-statement")
REGIME_ORDER = ("support", "opposition")
PLOT_FILES = (
    "preference_action_agreement.png",
    "candidate_survival.png",
    "run_quality.png",
    "round_trajectories.png",
)
POOLED_PARQUET_FILE = "pooled_by_severity.parquet"
POOLED_REGIME_COLOR = {"support": "#4C78A8", "opposition": "#E45756"}
# One color per severity level for the ranking bump chart.
SEVERITY_COLOR = {
    1: "#54A24B",
    0: "#79706E",
    -1: "#F58518",
    -2: "#E45756",
    -3: "#B279A2",
}
# (metric value, output filename, title, y-axis label) for the pooled bar plots.
POOLED_METRIC_SPECS = (
    (
        pooled.PooledMetric.SURVIVAL_ROUNDS.value,
        "survival_by_severity.png",
        "Candidate survival by severity level",
        "Mean rounds survived (mean ± 95% t·SEM over repeats)",
    ),
    (
        pooled.PooledMetric.NET_SIGNED_VOTES.value,
        "net_votes_by_severity.png",
        "Net signed votes by severity level",
        "Mean net signed votes  (+ keep / − remove)",
    ),
)
RANKING_FILE = "ranking_over_rounds.png"
VOTE_SHARE_FILE = "vote_share_by_severity.png"
SEVERITY_AXIS_LABEL = "Severity level  (1 = least rude … −3 = most rude)"


def _safe_vote_share(export_dir: Path) -> list[dict[str, object]]:
    try:
        return pooled.vote_share_by_severity(export_dir)
    except FileNotFoundError:
        return []


RELIABILITY_METRICS = (
    "invalid_attempts",
    "correction_attempts",
    "abstentions",
    "invalid_missing_statements",
    "runtime_failures",
    "interruptions",
)
RELIABILITY_COLORS = ("#E45756", "#F58518", "#79706E", "#B279A2", "#9D755D", "#BAB0AC")
REGIME_TICK_COLOR = {"support": "#2E5A88", "opposition": "#A22F2E"}
_REGIME_SHORT = {"support": "sup", "opposition": "opp"}


def _quality_label(row: Mapping[str, object]) -> str:
    repeat = _as_int(row.get("seed_repeat_index", 0))
    regime = str(row["regime"])
    return f"r{repeat} {_REGIME_SHORT.get(regime, regime)}"


def _error_code_breakdown(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[object]]:
    """Aggregate invalid_attempts_by_error_code (map<string,int>) across runs."""
    totals: dict[str, int] = {}
    for row in rows:
        entries = row.get("invalid_attempts_by_error_code") or []
        for entry in _as_sequence(entries):
            code, count = _as_sequence(entry)  # (code, count) tuple from the map
            totals[str(code)] = totals.get(str(code), 0) + _as_int(count)
    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    return {
        "labels": [code for code, _ in ordered],
        "counts": [count for _, count in ordered],
    }


_POOLED_PARQUET_SCHEMA = pa.schema(
    [
        ("analysis_version", pa.string()),
        ("estimator", pa.string()),
        ("ci_level", pa.float64()),
        ("metric", pa.string()),
        ("severity_level", pa.int64()),
        ("regime", pa.string()),
        ("n_repeats", pa.int64()),
        ("mean", pa.float64()),
        ("sem", pa.float64()),
        ("df", pa.int64()),
        ("t_crit", pa.float64()),
        ("ci_lower", pa.float64()),
        ("ci_upper", pa.float64()),
    ]
)


def _safe_pooled_rows(export_dir: Path) -> list[dict[str, object]]:
    try:
        return [dict(row) for row in pooled.pooled_by_severity(export_dir)]
    except FileNotFoundError:
        return []


def _safe_rank_records(export_dir: Path) -> list[dict[str, object]]:
    try:
        return pooled.rank_records_by_severity(export_dir)
    except FileNotFoundError:
        return []


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(f"plots expected an integer-compatible value, got {value!r}")
    return int(value)


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(f"plots expected a numeric value, got {value!r}")
    return float(value)


def _as_sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise TypeError(f"plots expected a list-valued field, got {value!r}")


def _rows(export_dir: Path, name: str) -> list[dict[str, object]]:
    return pq.read_table(export_dir / f"{name}.parquet").to_pylist()


def build_plot_manifest(export_dir: Path) -> dict[str, object]:
    """Build the normalized data/order/label/style contract used by rendering."""
    agreement_rows = [
        row
        for row in _rows(export_dir, "preference_action_summary")
        if row["scope"] == "overall" and row["mean_spearman_rho"] is not None
    ]
    agreement_by_category = {
        (
            str(row["matched_set_id"]),
            _as_int(row["round_index"]),
            str(row["arm"]),
            str(row["regime"]),
        ): _as_float(row["mean_spearman_rho"])
        for row in agreement_rows
    }
    agreement_categories = sorted(
        agreement_by_category,
        key=lambda item: (
            item[0],
            item[1],
            ARM_ORDER.index(item[2]),
            REGIME_ORDER.index(item[3]),
        ),
    )

    survival_rows = sorted(
        _rows(export_dir, "candidate_survival"),
        key=lambda row: (str(row["run_id"]), str(row["candidate_id"])),
    )
    quality_rows = sorted(
        _rows(export_dir, "run_quality"),
        key=lambda row: (
            _as_int(row.get("seed_repeat_index", 0)),
            REGIME_ORDER.index(str(row["regime"]))
            if str(row["regime"]) in REGIME_ORDER
            else len(REGIME_ORDER),
            str(row["run_id"]),
        ),
    )
    trajectory_rows = sorted(
        _rows(export_dir, "round_trajectories"),
        key=lambda row: (str(row["run_id"]), _as_int(row["round_index"])),
    )
    return {
        "version": PLOT_MANIFEST_VERSION,
        "style": PLOT_STYLE,
        "palette": PALETTE,
        "pooled_by_severity": _safe_pooled_rows(export_dir),
        "rank_trajectories": _safe_rank_records(export_dir),
        "vote_share_by_severity": _safe_vote_share(export_dir),
        "plots": {
            "preference_action_agreement": {
                "categories": [
                    f"{matched}\nr{round_index}\n{arm}\n{regime}"
                    for matched, round_index, arm, regime in agreement_categories
                ],
                "values": [
                    agreement_by_category[item] for item in agreement_categories
                ],
                "colors": [PALETTE[arm] for _, _, arm, _ in agreement_categories],
                "title": "Preference–action agreement (joint arms only)",
                "y_label": "Mean within-voter-round Spearman rho",
                "y_limits": [-1.0, 1.0],
                "estimand": "association",
            },
            "candidate_survival": {
                "series": [
                    {
                        "label": f"{row['run_id']}:{row['candidate_id']}",
                        "x": [
                            _as_int(value)
                            for value in _as_sequence(row["round_indices"])
                        ],
                        "y": [1] * len(_as_sequence(row["round_indices"])),
                        "color": PALETTE[str(row["arm"])],
                        "linestyle": "-" if row["regime"] == "support" else "--",
                    }
                    for row in survival_rows
                ],
                "title": "Candidate survival curves",
                "x_label": "Round",
                "y_label": "Active in round",
                "y_limits": [0.0, 1.05],
            },
            "run_quality": {
                "categories": [_quality_label(row) for row in quality_rows],
                "regimes": [str(row["regime"]) for row in quality_rows],
                "reliability_metrics": {
                    metric: [_as_int(row[metric]) for row in quality_rows]
                    for metric in RELIABILITY_METRICS
                },
                "error_codes": _error_code_breakdown(quality_rows),
                "titles": [
                    "Reliability per run (repeat · regime)",
                    "Invalid-attempt error codes (all runs)",
                ],
            },
            "round_trajectories": {
                "series": [
                    {
                        "run_id": run_id,
                        "arm": rows[0]["arm"],
                        "regime": rows[0]["regime"],
                        "x": [_as_int(row["round_index"]) for row in rows],
                        "active_pool_size": [
                            _as_int(row["active_pool_size"]) for row in rows
                        ],
                        "total_votes": [_as_int(row["total_votes"]) for row in rows],
                        "max_votes": [_as_int(row["max_votes"]) for row in rows],
                        "color": PALETTE[str(rows[0]["arm"])],
                    }
                    for run_id, rows in _group_trajectories(trajectory_rows)
                ],
                "titles": ["Active pool size", "Allocation distribution"],
                "x_label": "Round",
            },
        },
    }


def _group_trajectories(
    rows: Sequence[Mapping[str, object]],
) -> list[tuple[str, list[Mapping[str, object]]]]:
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["run_id"]), []).append(row)
    return [(run_id, groups[run_id]) for run_id in sorted(groups)]


def _empty(axis: plt.Axes, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def _pooled_bar_figure(
    pooled_rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    title: str,
    y_label: str,
) -> Figure:
    """Grouped bars per severity level, one group member per regime, with t·SEM."""
    figure, axis = plt.subplots(figsize=(8, 4))
    rows = [row for row in pooled_rows if str(row["metric"]) == metric]
    levels = [
        level
        for level in pooled.SEVERITY_ORDER
        if any(_as_int(row["severity_level"]) == level for row in rows)
    ]
    regimes = [
        regime
        for regime in pooled.REGIME_ORDER
        if any(str(row["regime"]) == regime for row in rows)
    ]
    if not levels or not regimes:
        _empty(
            axis, "No pooled data (needs >=1 repeat with severity-mapped candidates)"
        )
        axis.set_title(title)
        axis.set_ylabel(y_label)
        return figure
    positions = list(range(len(levels)))
    width = 0.8 / len(regimes)
    for regime_index, regime in enumerate(regimes):
        by_level = {
            _as_int(row["severity_level"]): row
            for row in rows
            if str(row["regime"]) == regime
        }
        means: list[float] = []
        errors: list[float] = []
        for level in levels:
            row = by_level.get(level)
            means.append(_as_float(row["mean"]) if row else float("nan"))
            if row and row["sem"] is not None and row["t_crit"] is not None:
                errors.append(_as_float(row["t_crit"]) * _as_float(row["sem"]))
            else:
                errors.append(0.0)
        offsets = [
            position + (regime_index - (len(regimes) - 1) / 2) * width
            for position in positions
        ]
        axis.bar(
            offsets,
            means,
            width,
            yerr=errors,
            capsize=4,
            label=regime,
            color=POOLED_REGIME_COLOR.get(regime, PALETTE["neutral"]),
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, [str(level) for level in levels])
    axis.set_xlabel(SEVERITY_AXIS_LABEL)
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.legend(fontsize="small", title="regime")
    return figure


def _ranking_figure(rank_records: Sequence[Mapping[str, object]]) -> Figure:
    """Bump chart: median rank over rounds per severity level, faceted by regime."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    facet = dict(zip(pooled.REGIME_ORDER, axes, strict=True))
    for regime, axis in facet.items():
        records = sorted(
            (row for row in rank_records if str(row["regime"]) == regime),
            key=lambda row: pooled.SEVERITY_ORDER.index(_as_int(row["severity_level"])),
        )
        if not records:
            _empty(axis, f"No ranking data ({regime})")
            axis.set_title(f"{regime.capitalize()} regime")
            continue
        for row in records:
            level = _as_int(row["severity_level"])
            axis.plot(
                [_as_float(value) for value in _as_sequence(row["rounds"])],
                [_as_float(value) for value in _as_sequence(row["median_rank"])],
                marker="o",
                label=str(level),
                color=SEVERITY_COLOR.get(level, PALETTE["neutral"]),
            )
        axis.set_title(f"{regime.capitalize()} regime")
        axis.set_xlabel("Round")
        axis.legend(fontsize="x-small", title="severity", ncol=2)
    axes[0].set_ylabel("Median rank  (1 = most votes that round)")
    for axis in axes:
        if not axis.has_data():
            continue
        axis.invert_yaxis()
    figure.suptitle("Candidate ranking over rounds (median across repeats)")
    return figure


def _vote_share_figure(records: Sequence[Mapping[str, object]]) -> Figure:
    """Mean vote share per round per severity level, faceted by regime, t·SEM bars."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    facet = dict(zip(pooled.REGIME_ORDER, axes, strict=True))
    for regime, axis in facet.items():
        regime_records = sorted(
            (row for row in records if str(row["regime"]) == regime),
            key=lambda row: pooled.SEVERITY_ORDER.index(_as_int(row["severity_level"])),
        )
        if not regime_records:
            _empty(axis, f"No vote-share data ({regime})")
            axis.set_title(f"{regime.capitalize()} regime")
            continue
        for row in regime_records:
            level = _as_int(row["severity_level"])
            axis.errorbar(
                [_as_float(value) for value in _as_sequence(row["rounds"])],
                [_as_float(value) for value in _as_sequence(row["mean"])],
                yerr=[_as_float(value) for value in _as_sequence(row["err"])],
                marker="o",
                capsize=3,
                label=str(level),
                color=SEVERITY_COLOR.get(level, PALETTE["neutral"]),
            )
        axis.set_title(f"{regime.capitalize()} regime")
        axis.set_xlabel("Round")
        axis.set_ylim(0.0, 1.0)
        axis.legend(fontsize="x-small", title="severity", ncol=2)
    axes[0].set_ylabel("Mean vote share  (95% t·SEM over repeats)")
    figure.suptitle("Mean vote share by severity level per round")
    return figure


def build_plot_figures(
    manifest: Mapping[str, object],
) -> tuple[tuple[str, Figure], ...]:
    """Create inspectable Matplotlib artists directly from a plot manifest."""
    plots = manifest["plots"]
    assert isinstance(plots, Mapping)
    figures: list[tuple[str, Figure]] = []

    agreement = plots["preference_action_agreement"]
    assert isinstance(agreement, Mapping)
    figure, axis = plt.subplots(figsize=(8, 4))
    categories = [str(value) for value in _as_sequence(agreement["categories"])]
    values = [_as_float(value) for value in _as_sequence(agreement["values"])]
    if values:
        axis.bar(categories, values, color=list(_as_sequence(agreement["colors"])))
        axis.axhline(0, color="black", linewidth=0.8)
        axis.tick_params(axis="x", labelrotation=15)
    else:
        _empty(axis, "No estimable joint-arm agreement")
    axis.set_title(str(agreement["title"]))
    axis.set_ylabel(str(agreement["y_label"]))
    axis.set_ylim(*[_as_float(value) for value in _as_sequence(agreement["y_limits"])])
    figures.append((PLOT_FILES[0], figure))

    survival = plots["candidate_survival"]
    assert isinstance(survival, Mapping)
    figure, axis = plt.subplots(figsize=(8, 4))
    series = list(_as_sequence(survival["series"]))
    if series:
        for item in series:
            assert isinstance(item, Mapping)
            axis.step(
                item["x"],
                item["y"],
                where="post",
                alpha=0.65,
                label=str(item["label"]),
                color=str(item["color"]),
                linestyle=str(item["linestyle"]),
            )
    else:
        _empty(axis, "No candidate survival data")
    axis.set_title(str(survival["title"]))
    axis.set_xlabel(str(survival["x_label"]))
    axis.set_ylabel(str(survival["y_label"]))
    axis.set_ylim(*[_as_float(value) for value in _as_sequence(survival["y_limits"])])
    figures.append((PLOT_FILES[1], figure))

    quality = plots["run_quality"]
    assert isinstance(quality, Mapping)
    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]}
    )
    quality_categories = [str(value) for value in _as_sequence(quality["categories"])]
    if quality_categories:
        reliability = quality["reliability_metrics"]
        assert isinstance(reliability, Mapping)
        positions = list(range(len(quality_categories)))
        bottom: list[float] = [0.0] * len(quality_categories)
        for metric, color in zip(RELIABILITY_METRICS, RELIABILITY_COLORS, strict=True):
            values = [_as_int(value) for value in _as_sequence(reliability[metric])]
            left.bar(
                positions,
                values,
                bottom=bottom,
                label=str(metric).replace("_", " "),
                color=color,
            )
            bottom = [a + b for a, b in zip(bottom, values, strict=True)]
        left.set_xticks(positions, quality_categories, rotation=90)
        left.set_ylabel("Count")
        left.set_xlabel("Run  (repeat · regime)")
        left.legend(
            fontsize="x-small", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22)
        )
        regimes = [str(value) for value in _as_sequence(quality["regimes"])]
        for tick, regime in zip(left.get_xticklabels(), regimes, strict=True):
            tick.set_color(REGIME_TICK_COLOR.get(regime, "#000000"))
        error_codes = quality["error_codes"]
        assert isinstance(error_codes, Mapping)
        code_labels = [str(value) for value in _as_sequence(error_codes["labels"])]
        code_counts = [_as_int(value) for value in _as_sequence(error_codes["counts"])]
        if code_labels:
            right.barh(code_labels, code_counts, color="#4C78A8")
            right.invert_yaxis()
            right.set_xlabel("Invalid attempts (all runs)")
            for index, count in enumerate(code_counts):
                right.text(count, index, f" {count}", va="center", fontsize="x-small")
        else:
            _empty(right, "No invalid-attempt errors")
    else:
        _empty(left, "No run-quality data")
        _empty(right, "No error-code data")
    titles = _as_sequence(quality["titles"])
    left.set_title(str(titles[0]))
    right.set_title(str(titles[1]))
    figures.append((PLOT_FILES[2], figure))

    trajectories = plots["round_trajectories"]
    assert isinstance(trajectories, Mapping)
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    trajectory_series = list(_as_sequence(trajectories["series"]))
    if trajectory_series:
        for item in trajectory_series:
            assert isinstance(item, Mapping)
            linestyle = "-" if item["regime"] == "support" else "--"
            left.plot(
                item["x"],
                item["active_pool_size"],
                color=item["color"],
                linestyle=linestyle,
                label=str(item["run_id"]),
            )
            right.plot(
                item["x"],
                item["total_votes"],
                color=item["color"],
                linestyle=linestyle,
                label=f"{item['run_id']} total",
            )
            right.plot(
                item["x"],
                item["max_votes"],
                color=item["color"],
                linestyle=":",
                label=f"{item['run_id']} max",
            )
        left.legend(fontsize="x-small")
        right.legend(fontsize="x-small")
    else:
        _empty(left, "No pool trajectories")
        _empty(right, "No allocation trajectories")
    titles = _as_sequence(trajectories["titles"])
    left.set_title(str(titles[0]))
    right.set_title(str(titles[1]))
    left.set_xlabel(str(trajectories["x_label"]))
    right.set_xlabel(str(trajectories["x_label"]))
    figures.append((PLOT_FILES[3], figure))

    pooled_rows = manifest.get("pooled_by_severity", [])
    assert isinstance(pooled_rows, Sequence)
    for metric, filename, title, y_label in POOLED_METRIC_SPECS:
        figures.append(
            (
                filename,
                _pooled_bar_figure(
                    pooled_rows, metric=metric, title=title, y_label=y_label
                ),
            )
        )
    rank_records = manifest.get("rank_trajectories", [])
    assert isinstance(rank_records, Sequence)
    figures.append((RANKING_FILE, _ranking_figure(rank_records)))
    vote_share = manifest.get("vote_share_by_severity", [])
    assert isinstance(vote_share, Sequence)
    figures.append((VOTE_SHARE_FILE, _vote_share_figure(vote_share)))
    return tuple(figures)


def render_plots(export_dir: Path, out_dir: Path) -> tuple[Path, ...]:
    """Write four PNGs, the self-contained timeline, and the semantic manifest."""
    manifest = build_plot_manifest(export_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=out_dir.parent)
    )
    produced: list[Path] = []
    figures = build_plot_figures(manifest)
    try:
        manifest_path = staging / "plot-manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        pooled_rows = manifest["pooled_by_severity"]
        assert isinstance(pooled_rows, list)
        pooled_path = staging / POOLED_PARQUET_FILE
        pq.write_table(
            pa.Table.from_pylist(pooled_rows, schema=_POOLED_PARQUET_SCHEMA),
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
            plt.close(figure)
            produced.append(path)
        produced.append(render_export_timeline(export_dir, staging / "timeline.html"))
        for path in (manifest_path, *produced):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if out_dir.exists():
            raise FileExistsError(
                f"Plot publication refused to replace existing directory {out_dir}; the new "
                "semantic manifest and images remain unpublished. Choose a new output directory "
                "or explicitly remove the old complete artifact, then retry."
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
