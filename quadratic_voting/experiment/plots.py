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
import pyarrow.parquet as pq  # type: ignore[import-untyped]  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

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
        _rows(export_dir, "run_quality"), key=lambda row: str(row["run_id"])
    )
    trajectory_rows = sorted(
        _rows(export_dir, "round_trajectories"),
        key=lambda row: (str(row["run_id"]), _as_int(row["round_index"])),
    )
    return {
        "version": PLOT_MANIFEST_VERSION,
        "style": PLOT_STYLE,
        "palette": PALETTE,
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
                "categories": [str(row["run_id"]) for row in quality_rows],
                "failure_metrics": {
                    metric: [_as_int(row[metric]) for row in quality_rows]
                    for metric in (
                        "invalid_attempts",
                        "correction_attempts",
                        "abstentions",
                        "runtime_failures",
                    )
                },
                "token_metrics": {
                    metric: [_as_int(row[metric]) for row in quality_rows]
                    for metric in ("prompt_tokens", "completion_tokens")
                },
                "titles": ["Failures, retries, and abstentions", "Token totals"],
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
    figure, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    quality_categories = [str(value) for value in _as_sequence(quality["categories"])]
    if quality_categories:
        failures = quality["failure_metrics"]
        tokens = quality["token_metrics"]
        assert isinstance(failures, Mapping) and isinstance(tokens, Mapping)
        bottom: list[float] = [0.0] * len(quality_categories)
        for metric, color in zip(
            failures, ("#E45756", "#F58518", "#79706E", "#B279A2"), strict=True
        ):
            values = [_as_int(value) for value in _as_sequence(failures[metric])]
            left.bar(
                quality_categories,
                values,
                bottom=bottom,
                label=str(metric).replace("_", " "),
                color=color,
            )
            bottom = [a + b for a, b in zip(bottom, values, strict=True)]
        positions = list(range(len(quality_categories)))
        width = 0.35
        right.bar(
            [p - width / 2 for p in positions],
            tokens["prompt_tokens"],
            width,
            label="prompt",
            color="#4C78A8",
        )
        right.bar(
            [p + width / 2 for p in positions],
            tokens["completion_tokens"],
            width,
            label="completion",
            color="#54A24B",
        )
        right.set_xticks(positions, quality_categories, rotation=45)
        left.tick_params(axis="x", labelrotation=45)
        left.legend(fontsize="small")
        right.legend(fontsize="small")
    else:
        _empty(left, "No run-quality data")
        _empty(right, "No token data")
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
        for filename, figure in figures:
            path = staging / filename
            figure.tight_layout()
            figure.savefig(path, dpi=120, metadata={"Software": "quadratic-voting"})
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
