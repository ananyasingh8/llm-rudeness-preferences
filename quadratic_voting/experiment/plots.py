"""Semantic-deterministic headless plots over analysis Parquet exports."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # type: ignore[import-untyped]  # noqa: E402
import pyarrow.parquet as pq  # type: ignore[import-untyped]  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

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
NEUTRAL_COLOR = "#79706E"

# Shared severity/regime style. Severity is encoded as *color* (blue = least
# rude … red = most rude); regime is encoded as *linestyle*. Palette keys are
# ints in code; the manifest exposes STRING keys so it round-trips through JSON.
SEVERITY_PALETTE: dict[int, str] = {
    1: "#2166AC",
    0: "#67A9CF",
    -1: "#F4A582",
    -2: "#D6604D",
    -3: "#B2182B",
}
REGIME_LINESTYLE: dict[str, str] = {"support": "-", "opposition": "--"}
# Regime marker for the per-run candidate-survival scatter (secondary cue to the
# x-axis run labels; color still carries severity).
REGIME_MARKER: dict[str, str] = {"support": "o", "opposition": "s"}
_REGIME_SHORT = {"support": "sup", "opposition": "opp"}

PLOT_FILES = (
    "preference_action_agreement.png",
    "run_quality.png",
    "vote_share_by_severity.png",
    "net_votes_by_severity.png",
    "candidate_survival.png",
    "round_trajectories.png",
)
POOLED_PARQUET_FILE = "pooled_by_severity.parquet"
SEVERITY_AXIS_LABEL = "Severity level  (1 = least rude … −3 = most rude)"

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


def _severity_color(level: int) -> str:
    return SEVERITY_PALETTE.get(level, NEUTRAL_COLOR)


def _severity_palette_manifest() -> dict[str, str]:
    """String-keyed severity palette for the JSON-round-trip-safe manifest."""
    return {str(level): color for level, color in SEVERITY_PALETTE.items()}


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


def _safe_vote_share(export_dir: Path) -> list[dict[str, object]]:
    try:
        return pooled.vote_share_by_severity(export_dir)
    except FileNotFoundError:
        return []


def _safe_severity_by_candidate(export_dir: Path) -> dict[str, int]:
    try:
        rows = _rows(export_dir, "source_annotations")
    except FileNotFoundError:
        return {}
    return pooled.severity_level_by_candidate(rows)


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


def _mean_sem(values: Sequence[float]) -> tuple[float, float]:
    """Mean and standard error of the mean (0 when fewer than two values)."""
    array = np.asarray(values, dtype=float)
    n = int(array.size)
    mean = float(array.mean())
    if n < 2:
        return mean, 0.0
    return mean, float(array.std(ddof=1) / np.sqrt(n))


def _repeat_index(row: Mapping[str, object]) -> int:
    return _as_int(row.get("seed_repeat_index", 0))


def _agreement_section(export_dir: Path) -> dict[str, object]:
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
    return {
        "categories": [
            f"{matched}\nr{round_index}\n{arm}\n{regime}"
            for matched, round_index, arm, regime in agreement_categories
        ],
        "values": [agreement_by_category[item] for item in agreement_categories],
        "colors": [PALETTE[arm] for _, _, arm, _ in agreement_categories],
        "title": "Preference–action agreement (joint arms only)",
        "y_label": "Mean within-voter-round Spearman rho",
        "y_limits": [-1.0, 1.0],
        "estimand": "association",
    }


def _run_quality_section(export_dir: Path) -> dict[str, object]:
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
    return {
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
    }


def _vote_share_section(records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """One line per (severity_level, regime): color = severity, style = regime."""
    series: list[dict[str, object]] = []
    for row in records:
        level = _as_int(row["severity_level"])
        regime = str(row["regime"])
        series.append(
            {
                "severity_level": level,
                "regime": regime,
                "x": [_as_float(value) for value in _as_sequence(row["rounds"])],
                "y": [_as_float(value) for value in _as_sequence(row["mean"])],
                "ci_lower": [
                    _as_float(value) for value in _as_sequence(row["ci_lower"])
                ],
                "ci_upper": [
                    _as_float(value) for value in _as_sequence(row["ci_upper"])
                ],
                "color": _severity_color(level),
                "linestyle": REGIME_LINESTYLE.get(regime, "-"),
            }
        )
    return {
        "series": series,
        "title": "Mean vote share by severity per round",
        "x_label": "Round",
        "y_label": "Mean vote share (95% t·SEM)",
        "y_limits": [0.0, 1.0],
    }


def _pooled_severity_series(
    pooled_rows: Sequence[Mapping[str, object]], metric: str
) -> dict[str, object]:
    """Per-regime errorbar series over severity levels (color = severity)."""
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
    series: list[dict[str, object]] = []
    for regime in regimes:
        by_level = {
            _as_int(row["severity_level"]): row
            for row in rows
            if str(row["regime"]) == regime
        }
        level_indices: list[int] = []
        means: list[float] = []
        errors: list[float] = []
        point_colors: list[str] = []
        for index, level in enumerate(levels):
            row = by_level.get(level)
            if row is None:
                continue
            level_indices.append(index)
            means.append(_as_float(row["mean"]))
            if row["sem"] is not None and row["t_crit"] is not None:
                errors.append(_as_float(row["t_crit"]) * _as_float(row["sem"]))
            else:
                errors.append(0.0)
            point_colors.append(_severity_color(level))
        series.append(
            {
                "regime": regime,
                "linestyle": REGIME_LINESTYLE.get(regime, "-"),
                "level_indices": level_indices,
                "means": means,
                "errors": errors,
                "point_colors": point_colors,
            }
        )
    return {
        "levels": list(levels),
        "level_labels": [str(level) for level in levels],
        "series": series,
    }


def _net_votes_section(
    pooled_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    section = _pooled_severity_series(
        pooled_rows, pooled.PooledMetric.NET_SIGNED_VOTES.value
    )
    section.update(
        {
            "title": "Net signed votes by severity level",
            "x_label": SEVERITY_AXIS_LABEL,
            "y_label": "Mean net signed votes  (+ keep / − remove)",
        }
    )
    return section


def _candidate_survival_section(
    pooled_rows: Sequence[Mapping[str, object]], export_dir: Path
) -> dict[str, object]:
    pooled_panel = _pooled_severity_series(
        pooled_rows, pooled.PooledMetric.SURVIVAL_ROUNDS.value
    )
    pooled_panel.update(
        {
            "title": "Rounds survived by severity",
            "x_label": SEVERITY_AXIS_LABEL,
            "y_label": "Mean rounds survived (95% t·SEM)",
        }
    )
    return {
        "pooled": pooled_panel,
        "per_run": _survival_per_run_panel(export_dir),
    }


def _survival_per_run_panel(export_dir: Path) -> dict[str, object]:
    """Per-run survival scatter: x = readable run label, color = severity."""
    level_by_candidate = _safe_severity_by_candidate(export_dir)
    survival_rows = _rows(export_dir, "candidate_survival")
    run_key: dict[str, tuple[int, str]] = {}
    for row in survival_rows:
        run_id = str(row["run_id"])
        run_key[run_id] = (_repeat_index(row), str(row["regime"]))
    ordered_runs = sorted(run_key, key=lambda run: (run_key[run][0], run_key[run][1]))
    run_index = {run_id: index for index, run_id in enumerate(ordered_runs)}
    labels = [
        f"r{run_key[run_id][0]} {_REGIME_SHORT.get(run_key[run_id][1], run_key[run_id][1])}"
        for run_id in ordered_runs
    ]
    points: list[dict[str, object]] = []
    for row in survival_rows:
        candidate_id = str(row["candidate_id"])
        level = level_by_candidate.get(candidate_id)
        if level is None:
            continue
        regime = str(row["regime"])
        points.append(
            {
                "x": run_index[str(row["run_id"])],
                "y": _as_int(row["survival_round"]),
                "color": _severity_color(level),
                "marker": REGIME_MARKER.get(regime, "o"),
            }
        )
    return {
        "labels": labels,
        "points": points,
        "title": "Survival per run",
        "x_label": "Run  (repeat · regime)",
        "y_label": "Rounds survived",
    }


def _round_trajectories_section(export_dir: Path) -> dict[str, object]:
    rows = sorted(
        _rows(export_dir, "round_trajectories"),
        key=lambda row: (str(row["run_id"]), _as_int(row["round_index"])),
    )
    return {
        "pooled": _trajectories_pooled_panel(rows),
        "per_run": _trajectories_per_run_panel(rows),
    }


def _trajectories_pooled_panel(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Mean total_votes per (regime, round) across repeats, ± SEM band."""
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        regime = str(row["regime"])
        rnd = _as_int(row["round_index"])
        grouped[(regime, rnd)].append(_as_float(row["total_votes"]))
    series: list[dict[str, object]] = []
    for regime in REGIME_ORDER:
        rounds_seen = sorted(rnd for (rg, rnd) in grouped if rg == regime)
        if not rounds_seen:
            continue
        xs: list[float] = []
        means: list[float] = []
        lowers: list[float] = []
        uppers: list[float] = []
        for rnd in rounds_seen:
            mean, sem = _mean_sem(grouped[(regime, rnd)])
            xs.append(float(rnd))
            means.append(mean)
            lowers.append(mean - sem)
            uppers.append(mean + sem)
        series.append(
            {
                "regime": regime,
                "linestyle": REGIME_LINESTYLE.get(regime, "-"),
                "x": xs,
                "mean": means,
                "lower": lowers,
                "upper": uppers,
            }
        )
    return {
        "series": series,
        "color": NEUTRAL_COLOR,
        "title": "Mean total votes by regime",
        "x_label": "Round",
        "y_label": "Mean total votes (±SEM)",
    }


def _trajectories_per_run_panel(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Per-run total_votes; linestyle = regime, color = repeat gradient."""
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["run_id"])].append(row)
    repeats = sorted({_repeat_index(row) for row in rows})
    repeat_color = _repeat_color_map(repeats)
    series: list[dict[str, object]] = []
    for run_id in sorted(groups):
        run_rows = sorted(groups[run_id], key=lambda row: _as_int(row["round_index"]))
        regime = str(run_rows[0]["regime"])
        repeat = _repeat_index(run_rows[0])
        series.append(
            {
                "run_id": run_id,
                "regime": regime,
                "seed_repeat_index": repeat,
                "x": [_as_int(row["round_index"]) for row in run_rows],
                "total_votes": [_as_int(row["total_votes"]) for row in run_rows],
                "linestyle": REGIME_LINESTYLE.get(regime, "-"),
                "color": repeat_color[repeat],
            }
        )
    return {
        "series": series,
        "repeat_colors": {str(repeat): repeat_color[repeat] for repeat in repeats},
        "title": "Total votes per run",
        "x_label": "Round",
        "y_label": "Total votes",
    }


def _repeat_color_map(repeats: Sequence[int]) -> dict[int, str]:
    """Map each distinct repeat to a viridis-gradient hex color (deterministic)."""
    cmap = plt.get_cmap("viridis")
    count = len(repeats)
    colors: dict[int, str] = {}
    for index, repeat in enumerate(repeats):
        fraction = 0.0 if count < 2 else index / (count - 1)
        colors[repeat] = to_hex(cmap(fraction))
    return colors


def build_plot_manifest(export_dir: Path) -> dict[str, object]:
    """Build the normalized data/order/label/style contract used by rendering."""
    pooled_rows = _safe_pooled_rows(export_dir)
    return {
        "version": PLOT_MANIFEST_VERSION,
        "style": PLOT_STYLE,
        "palette": PALETTE,
        "severity_palette": _severity_palette_manifest(),
        "regime_linestyle": dict(REGIME_LINESTYLE),
        "pooled_by_severity": pooled_rows,
        "plots": {
            "preference_action_agreement": _agreement_section(export_dir),
            "run_quality": _run_quality_section(export_dir),
            "vote_share_by_severity": _vote_share_section(_safe_vote_share(export_dir)),
            "net_votes_by_severity": _net_votes_section(pooled_rows),
            "candidate_survival": _candidate_survival_section(pooled_rows, export_dir),
            "round_trajectories": _round_trajectories_section(export_dir),
        },
    }


def _empty(axis: plt.Axes, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_xticks([])
    axis.set_yticks([])


def _severity_regime_legends(
    axis: plt.Axes,
    manifest: Mapping[str, object],
    levels: Sequence[int],
    regimes: Sequence[str],
) -> None:
    """Attach two proxy legends: color → severity level, linestyle → regime."""
    palette = manifest["severity_palette"]
    assert isinstance(palette, Mapping)
    linestyles = manifest["regime_linestyle"]
    assert isinstance(linestyles, Mapping)
    color_handles = [
        Line2D(
            [0],
            [0],
            color=str(palette[str(level)]),
            marker="o",
            linestyle="",
            label=str(level),
        )
        for level in levels
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=str(linestyles[regime]),  # type: ignore[arg-type]
            label=regime,
        )
        for regime in regimes
        if regime in linestyles
    ]
    if color_handles:
        severity_legend = axis.legend(
            handles=color_handles,
            title="severity",
            fontsize="x-small",
            ncol=2,
            loc="upper right",
        )
        axis.add_artist(severity_legend)
    if style_handles:
        axis.legend(
            handles=style_handles,
            title="regime",
            fontsize="x-small",
            loc="lower right",
        )


def _render_agreement(agreement: Mapping[str, object]) -> Figure:
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
    return figure


def _render_run_quality(quality: Mapping[str, object]) -> Figure:
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
    return figure


def _render_vote_share(
    vote_share: Mapping[str, object], manifest: Mapping[str, object]
) -> Figure:
    figure, axis = plt.subplots(figsize=(9, 5))
    series = [item for item in _as_sequence(vote_share["series"])]
    if series:
        levels: list[int] = []
        regimes: list[str] = []
        for item in series:
            assert isinstance(item, Mapping)
            x = [_as_float(value) for value in _as_sequence(item["x"])]
            y = [_as_float(value) for value in _as_sequence(item["y"])]
            lower = [_as_float(value) for value in _as_sequence(item["ci_lower"])]
            upper = [_as_float(value) for value in _as_sequence(item["ci_upper"])]
            color = str(item["color"])
            axis.plot(x, y, marker="o", color=color, linestyle=str(item["linestyle"]))
            axis.fill_between(x, lower, upper, color=color, alpha=0.15)
            level = _as_int(item["severity_level"])
            if level not in levels:
                levels.append(level)
            regime = str(item["regime"])
            if regime not in regimes:
                regimes.append(regime)
        ordered_levels = [lvl for lvl in pooled.SEVERITY_ORDER if lvl in levels]
        ordered_regimes = [rg for rg in REGIME_ORDER if rg in regimes]
        _severity_regime_legends(axis, manifest, ordered_levels, ordered_regimes)
    else:
        _empty(axis, "No vote-share data")
    axis.set_title(str(vote_share["title"]))
    axis.set_xlabel(str(vote_share["x_label"]))
    axis.set_ylabel(str(vote_share["y_label"]))
    axis.set_ylim(*[_as_float(value) for value in _as_sequence(vote_share["y_limits"])])
    return figure


def _render_pooled_severity(
    axis: plt.Axes,
    panel: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    empty_message: str,
) -> None:
    """Shared renderer for a per-regime errorbar-over-severity panel."""
    levels = [_as_int(value) for value in _as_sequence(panel["levels"])]
    level_labels = [str(value) for value in _as_sequence(panel["level_labels"])]
    series = [item for item in _as_sequence(panel["series"])]
    if levels and series:
        positions = list(range(len(levels)))
        regimes: list[str] = []
        for item in series:
            assert isinstance(item, Mapping)
            indices = [_as_int(value) for value in _as_sequence(item["level_indices"])]
            means = [_as_float(value) for value in _as_sequence(item["means"])]
            errors = [_as_float(value) for value in _as_sequence(item["errors"])]
            colors = [str(value) for value in _as_sequence(item["point_colors"])]
            axis.errorbar(
                indices,
                means,
                yerr=errors,
                linestyle=str(item["linestyle"]),
                color="#555555",
                marker="",
                capsize=4,
                zorder=1,
            )
            axis.scatter(indices, means, c=colors, marker="o", zorder=2)
            regime = str(item["regime"])
            if regime not in regimes:
                regimes.append(regime)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(positions, level_labels)
        ordered_regimes = [rg for rg in REGIME_ORDER if rg in regimes]
        _severity_regime_legends(axis, manifest, levels, ordered_regimes)
    else:
        _empty(axis, empty_message)
    axis.set_xlabel(str(panel["x_label"]))
    axis.set_ylabel(str(panel["y_label"]))
    axis.set_title(str(panel["title"]))


def _render_net_votes(
    net_votes: Mapping[str, object], manifest: Mapping[str, object]
) -> Figure:
    figure, axis = plt.subplots(figsize=(8, 5))
    _render_pooled_severity(
        axis,
        net_votes,
        manifest,
        empty_message="No pooled data (needs >=1 repeat with severity-mapped candidates)",
    )
    return figure


def _render_candidate_survival(
    survival: Mapping[str, object], manifest: Mapping[str, object]
) -> Figure:
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5))
    pooled_panel = survival["pooled"]
    assert isinstance(pooled_panel, Mapping)
    _render_pooled_severity(
        left,
        pooled_panel,
        manifest,
        empty_message="No pooled data (needs >=1 repeat with severity-mapped candidates)",
    )
    per_run = survival["per_run"]
    assert isinstance(per_run, Mapping)
    labels = [str(value) for value in _as_sequence(per_run["labels"])]
    points = [item for item in _as_sequence(per_run["points"])]
    if labels and points:
        for item in points:
            assert isinstance(item, Mapping)
            right.scatter(
                [_as_int(item["x"])],
                [_as_int(item["y"])],
                color=str(item["color"]),
                marker=str(item["marker"]),
                s=36,
                edgecolors="black",
                linewidths=0.3,
            )
        right.set_xticks(list(range(len(labels))), labels, rotation=90)
        palette = manifest["severity_palette"]
        assert isinstance(palette, Mapping)
        color_handles = [
            Line2D(
                [0],
                [0],
                color=str(palette[str(level)]),
                marker="o",
                linestyle="",
                label=str(level),
            )
            for level in pooled.SEVERITY_ORDER
        ]
        right.legend(
            handles=color_handles, title="severity", fontsize="x-small", ncol=2
        )
    else:
        _empty(right, "No per-run survival data")
    right.set_title(str(per_run["title"]))
    right.set_xlabel(str(per_run["x_label"]))
    right.set_ylabel(str(per_run["y_label"]))
    return figure


def _render_round_trajectories(trajectories: Mapping[str, object]) -> Figure:
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    pooled_panel = trajectories["pooled"]
    assert isinstance(pooled_panel, Mapping)
    pooled_series = [item for item in _as_sequence(pooled_panel["series"])]
    color = str(pooled_panel["color"])
    if pooled_series:
        for item in pooled_series:
            assert isinstance(item, Mapping)
            x = [_as_float(value) for value in _as_sequence(item["x"])]
            mean = [_as_float(value) for value in _as_sequence(item["mean"])]
            lower = [_as_float(value) for value in _as_sequence(item["lower"])]
            upper = [_as_float(value) for value in _as_sequence(item["upper"])]
            left.plot(
                x,
                mean,
                color=color,
                linestyle=str(item["linestyle"]),
                marker="o",
                label=str(item["regime"]),
            )
            left.fill_between(x, lower, upper, color=color, alpha=0.15)
        left.legend(fontsize="x-small", title="regime")
    else:
        _empty(left, "No pool trajectories")
    left.set_title(str(pooled_panel["title"]))
    left.set_xlabel(str(pooled_panel["x_label"]))
    left.set_ylabel(str(pooled_panel["y_label"]))

    per_run = trajectories["per_run"]
    assert isinstance(per_run, Mapping)
    run_series = [item for item in _as_sequence(per_run["series"])]
    if run_series:
        for item in run_series:
            assert isinstance(item, Mapping)
            right.plot(
                [_as_int(value) for value in _as_sequence(item["x"])],
                [_as_int(value) for value in _as_sequence(item["total_votes"])],
                color=str(item["color"]),
                linestyle=str(item["linestyle"]),
                alpha=0.8,
            )
        repeat_colors = per_run["repeat_colors"]
        assert isinstance(repeat_colors, Mapping)
        repeat_handles = [
            Line2D([0], [0], color=str(color), label=f"repeat {repeat}")
            for repeat, color in repeat_colors.items()
        ]
        if repeat_handles:
            right.legend(handles=repeat_handles, fontsize="x-small", title="repeat")
    else:
        _empty(right, "No per-run trajectories")
    right.set_title(str(per_run["title"]))
    right.set_xlabel(str(per_run["x_label"]))
    right.set_ylabel(str(per_run["y_label"]))
    return figure


def build_plot_figures(
    manifest: Mapping[str, object],
) -> tuple[tuple[str, Figure], ...]:
    """Create inspectable Matplotlib artists directly from a plot manifest."""
    plots = manifest["plots"]
    assert isinstance(plots, Mapping)

    agreement = plots["preference_action_agreement"]
    assert isinstance(agreement, Mapping)
    quality = plots["run_quality"]
    assert isinstance(quality, Mapping)
    vote_share = plots["vote_share_by_severity"]
    assert isinstance(vote_share, Mapping)
    net_votes = plots["net_votes_by_severity"]
    assert isinstance(net_votes, Mapping)
    survival = plots["candidate_survival"]
    assert isinstance(survival, Mapping)
    trajectories = plots["round_trajectories"]
    assert isinstance(trajectories, Mapping)

    return (
        (PLOT_FILES[0], _render_agreement(agreement)),
        (PLOT_FILES[1], _render_run_quality(quality)),
        (PLOT_FILES[2], _render_vote_share(vote_share, manifest)),
        (PLOT_FILES[3], _render_net_votes(net_votes, manifest)),
        (PLOT_FILES[4], _render_candidate_survival(survival, manifest)),
        (PLOT_FILES[5], _render_round_trajectories(trajectories)),
    )


def render_plots(export_dir: Path, out_dir: Path) -> tuple[Path, ...]:
    """Write the PNGs, the self-contained timeline, and the semantic manifest."""
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
