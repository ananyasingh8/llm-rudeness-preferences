"""Analysis for bail-steering runs: bail rates per steered emotion.

Reads a run folder written by bail_steering.main and writes summary.csv plus
charts into <run>/analysis/. Needs only stdlib + matplotlib:

  python -m bail_steering.analyze            # latest run
  python -m bail_steering.analyze --run PATH
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from bail.prompts.bail_methods import GREEN, SHUFFLE, remove_thinking

# Mirrored from bail_steering.main, which cannot be imported here: it pulls
# GPU-only dependencies absent on analysis-only machines.
RESULTS_DIR = Path(__file__).parent / "results"
RESPONSES_FILE = "responses.csv"
BASELINE_CONDITION = "baseline"
TOOL_CALL_MARKERS = (
    '"name": "switchconversation_tool"',
    "call:switchconversation_tool",
)
from emotion_probing.analyze.common import (
    BASELINE,
    DELTA_NEGATIVE,
    DELTA_POSITIVE,
    GRID,
    INK,
    MUTED,
    SERIES_1,
    SERIES_2,
    SERIES_3,
    SURFACE,
    get_matplotlib,
    save_figure,
    styled_axes,
)

GROUP_ORDER = ("friendly", "neutral", "rude")
SEVERITY_ORDER = ("1", "0", "-1", "-2", "-3")
SEVERITY_TICKS = (
    "1\nfriendly",
    "0\nneutral",
    "-1\nmild",
    "-2\nstrong",
    "-3\nvery strong",
)
# The three bail elicitation variants: (phase, ordering, label, color).
BAIL_METHODS = (
    ("prompt", "bail_first", "prompt — bail option first", SERIES_1),
    ("prompt", "continue_first", "prompt — continue option first", SERIES_2),
    ("tool", "", "tool call", SERIES_3),
)


def load_condition_lists(run_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """(ordered conditions, risers, fallers) from the run's own run_info."""
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    risers = list(info.get("risers") or [])
    fallers = list(info.get("fallers") or [])
    return [BASELINE_CONDITION] + risers + fallers, risers, fallers


def find_run_dir(explicit: str | None) -> Path:
    if explicit is not None:
        run_dir = Path(explicit)
        if not (run_dir / "run_info.json").exists():
            raise SystemExit(f"error: {run_dir} is not a run folder.")
        return run_dir
    if RESULTS_DIR.exists():
        runs = [
            path
            for path in RESULTS_DIR.iterdir()
            if path.is_dir() and (path / "run_info.json").exists()
        ]
        if runs:
            return max(
                runs,
                key=lambda p: json.loads(
                    (p / "run_info.json").read_text(encoding="utf-8")
                ).get("started", ""),
            )
    raise SystemExit(
        "error: no run folders found. Run "
        "`uv run python -m bail_steering.main run` first."
    )


def load_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / RESPONSES_FILE
    if not path.exists():
        raise SystemExit(f"error: {path} not found (run has produced nothing yet).")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"error: {path} has no rows yet.")
    return rows


def is_bail(row: dict[str, str]) -> bool | None:
    """True = bail, False = continue, None = unparseable / not applicable."""
    if row["phase"] == "prompt":
        if row["wellbeing"] == SHUFFLE:
            return True
        if row["wellbeing"] == GREEN:
            return False
        return None
    # Recompute tool calls from the raw text rather than trusting the stored
    # flag: early runs only scanned for the OpenAI-style JSON shape and
    # missed Gemma's "call:<name>{...}" syntax.
    cleaned = remove_thinking(row["response_text"])
    return any(marker in cleaned for marker in TOOL_CALL_MARKERS)


def rate_stats(rows: list[dict[str, str]]) -> dict[str, float]:
    """n (parsed), bail rate, and its standard error over the given rows."""
    verdicts = [v for v in (is_bail(row) for row in rows) if v is not None]
    n = len(verdicts)
    rate = sum(verdicts) / n if n else 0.0
    sem = math.sqrt(rate * (1 - rate) / n) if n else 0.0
    return {"n_total": len(rows), "n_parsed": n, "rate": rate, "sem": sem}


def _method_rates(
    rows: list[dict[str, str]], phase: str, ordering: str
) -> tuple[list[float], list[float]]:
    """Bail rate and SEM per severity band for one elicitation method.

    Aggregates over however many samples exist per conversation (runs with
    repeated randomized samples simply contribute more rows per cell; the
    SEM treats them as independent generations).
    """
    rates, sems = [], []
    for band in SEVERITY_ORDER:
        subset = [
            r
            for r in rows
            if r["phase"] == phase
            and (phase == "tool" or r["ordering"] == ordering)
            and r["abuse_severity"] == band
        ]
        stats = rate_stats(subset)
        rates.append(stats["rate"] if stats["n_parsed"] else float("nan"))
        sems.append(stats["sem"])
    return rates, sems


def severity_chart(
    plt, rows: list[dict[str, str]], condition: str, path: Path
) -> None:
    """Bail rate per severity band for one condition, one line per method.

    When baseline data exists, each method also gets a dotted line showing
    the unsteered baseline rate for direct comparison.
    """
    condition_rows = [r for r in rows if r["condition"] == condition]
    baseline_rows = [r for r in rows if r["condition"] == BASELINE_CONDITION]
    show_baseline = condition != BASELINE_CONDITION and bool(baseline_rows)
    figure, axes = styled_axes(plt, 7.0, 4.4)
    x = list(range(len(SEVERITY_ORDER)))
    for phase, ordering, label, color in BAIL_METHODS:
        if show_baseline:
            base_rates, _ = _method_rates(baseline_rows, phase, ordering)
            axes.plot(
                x,
                base_rates,
                color=color,
                linewidth=1.5,
                linestyle=(0, (3, 3)),
                alpha=0.85,
            )
        rates, sems = _method_rates(condition_rows, phase, ordering)
        axes.errorbar(
            x,
            rates,
            yerr=sems,
            color=color,
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            ecolor=MUTED,
            elinewidth=1,
            label=label,
        )
    if show_baseline:
        axes.plot(
            [], [], color=MUTED, linewidth=1.5, linestyle=(0, (3, 3)),
            label="dotted: unsteered baseline",
        )
    axes.set_xticks(x, SEVERITY_TICKS)
    axes.set_xlim(-0.35, len(SEVERITY_ORDER) - 0.65)
    axes.set_ylim(bottom=0)
    axes.yaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_xlabel("ConvAbuse severity band", color=MUTED)
    axes.set_ylabel("bail rate", color=MUTED)
    axes.legend(frameon=False, labelcolor=INK, loc="upper left", fontsize=9)
    steering = (
        "no steering"
        if condition == BASELINE_CONDITION
        else f"steering: {condition}"
    )
    axes.set_title(
        f"Bail rate by conversation severity — {steering}",
        color=INK,
        loc="left",
        pad=12,
    )
    save_figure(plt, figure, path)


def condition_color(condition: str, risers: list[str], fallers: list[str]) -> str:
    if condition in risers:
        return DELTA_POSITIVE
    if condition in fallers:
        return DELTA_NEGATIVE
    return BASELINE


def rate_chart(
    plt, stats, phase: str, path: Path, order: list[str],
    risers: list[str], fallers: list[str],
) -> None:
    """Horizontal bars of bail rate per condition, baseline rate as a line."""
    present = [c for c in order if c in stats]
    labels = [f"{c} (n={stats[c]['n_parsed']})" for c in present]
    figure, axes = styled_axes(plt, 7.5, 0.34 * len(present) + 1.2)
    axes.barh(
        labels,
        [stats[c]["rate"] for c in present],
        xerr=[stats[c]["sem"] for c in present],
        height=0.7,
        color=[condition_color(c, risers, fallers) for c in present],
        error_kw={"ecolor": MUTED, "elinewidth": 1},
    )
    axes.invert_yaxis()  # baseline first, then risers, then fallers
    if BASELINE_CONDITION in stats:
        axes.axvline(
            stats[BASELINE_CONDITION]["rate"], color=BASELINE, linewidth=1
        )
    axes.xaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_xlabel("bail rate", color=MUTED)
    axes.set_title(
        f"Bail rate by steered emotion — {phase} method\n"
        "red = rude-riser steered +0.1, blue = faller steered −0.1; "
        "line = unsteered baseline",
        color=INK,
        loc="left",
        pad=12,
    )
    save_figure(plt, figure, path)


def group_heatmap(
    plt, rows: list[dict[str, str]], path: Path, order: list[str]
) -> None:
    """Conditions x conversation groups, prompt-method bail rate per cell."""
    prompt_rows = [r for r in rows if r["phase"] == "prompt"]
    present = sorted(
        {r["condition"] for r in prompt_rows},
        key=lambda c: order.index(c) if c in order else len(order),
    )
    if not present:
        return
    grid = []
    for condition in present:
        row_vals = []
        for group in GROUP_ORDER:
            subset = [
                r
                for r in prompt_rows
                if r["condition"] == condition and r["group"] == group
            ]
            row_vals.append(rate_stats(subset)["rate"] if subset else float("nan"))
        grid.append(row_vals)
    figure, axes = styled_axes(plt, 5.5, 0.34 * len(present) + 1.4)
    image = axes.imshow(
        grid, aspect="auto", cmap="Blues", vmin=0, vmax=max(max(g) for g in grid)
    )
    axes.set_xticks(range(len(GROUP_ORDER)), GROUP_ORDER)
    axes.set_yticks(range(len(present)), present)
    for i, row_vals in enumerate(grid):
        for j, value in enumerate(row_vals):
            if not math.isnan(value):
                axes.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=INK,
                    fontsize=7,
                    bbox={"facecolor": SURFACE, "alpha": 0.55, "pad": 1, "lw": 0},
                )
    figure.colorbar(image, ax=axes, label="bail rate")
    axes.set_title(
        "Prompt-method bail rate: steered emotion × conversation group",
        color=INK,
        loc="left",
        pad=12,
    )
    save_figure(plt, figure, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="run folder (default: latest)")
    args = parser.parse_args()
    run_dir = find_run_dir(args.run)
    rows = load_rows(run_dir)
    order, risers, fallers = load_condition_lists(run_dir)
    out_dir = run_dir / "analysis"
    out_dir.mkdir(exist_ok=True)
    print(f"Analyzing {run_dir} ({len(rows)} generations)")

    # Per condition x phase stats, plus per-ordering rates for the prompt method.
    summary = []
    stats_by_phase: dict[str, dict[str, dict]] = {"prompt": {}, "tool": {}}
    for phase in ("prompt", "tool"):
        phase_rows = [r for r in rows if r["phase"] == phase]
        baseline = rate_stats(
            [r for r in phase_rows if r["condition"] == BASELINE_CONDITION]
        )
        for condition in order:
            subset = [r for r in phase_rows if r["condition"] == condition]
            if not subset:
                continue
            stats = rate_stats(subset)
            stats_by_phase[phase][condition] = stats
            entry = {
                "condition": condition,
                "phase": phase,
                "n_total": stats["n_total"],
                "n_parsed": stats["n_parsed"],
                "bail_rate": round(stats["rate"], 4),
                "sem": round(stats["sem"], 4),
                "delta_vs_baseline": round(stats["rate"] - baseline["rate"], 4)
                if baseline["n_parsed"]
                else "",
            }
            if phase == "prompt":
                for ordering in ("bail_first", "continue_first"):
                    ordered = [r for r in subset if r["ordering"] == ordering]
                    entry[f"rate_{ordering}"] = (
                        round(rate_stats(ordered)["rate"], 4) if ordered else ""
                    )
            summary.append(entry)

    columns = [
        "condition",
        "phase",
        "n_total",
        "n_parsed",
        "bail_rate",
        "sem",
        "delta_vs_baseline",
        "rate_bail_first",
        "rate_continue_first",
    ]
    with (out_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summary written to {out_dir / 'summary.csv'}")

    plt = get_matplotlib()
    if plt is None:
        return
    # Per-condition severity profiles: the three bail methods across bands.
    present = sorted(
        {r["condition"] for r in rows},
        key=lambda c: order.index(c) if c in order else len(order),
    )
    for condition in present:
        severity_chart(
            plt, rows, condition,
            out_dir / "conditions" / f"{condition.replace(' ', '_')}_by_severity.png",
        )

    if stats_by_phase["prompt"]:
        rate_chart(
            plt, stats_by_phase["prompt"], "prompt",
            out_dir / "bail_rate_prompt.png", order, risers, fallers,
        )
        group_heatmap(plt, rows, out_dir / "bail_rate_by_group.png", order)
    if stats_by_phase["tool"]:
        rate_chart(
            plt, stats_by_phase["tool"], "tool",
            out_dir / "bail_rate_tool.png", order, risers, fallers,
        )


if __name__ == "__main__":
    main()
