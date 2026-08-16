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

from bail.prompts.bail_methods import GREEN, SHUFFLE
from bail_steering.main import (
    BASELINE_CONDITION,
    CONDITIONS,
    RESPONSES_FILE,
    RESULTS_DIR,
    STEER_FALLERS,
    STEER_RISERS,
)
from emotion_probing.analyze.common import (
    BASELINE,
    DELTA_NEGATIVE,
    DELTA_POSITIVE,
    GRID,
    INK,
    MUTED,
    SURFACE,
    get_matplotlib,
    save_figure,
    styled_axes,
)

GROUP_ORDER = ("friendly", "neutral", "rude")


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
    return row["tool_called"] == "True"


def rate_stats(rows: list[dict[str, str]]) -> dict[str, float]:
    """n (parsed), bail rate, and its standard error over the given rows."""
    verdicts = [v for v in (is_bail(row) for row in rows) if v is not None]
    n = len(verdicts)
    rate = sum(verdicts) / n if n else 0.0
    sem = math.sqrt(rate * (1 - rate) / n) if n else 0.0
    return {"n_total": len(rows), "n_parsed": n, "rate": rate, "sem": sem}


def condition_color(condition: str) -> str:
    if condition in STEER_RISERS:
        return DELTA_POSITIVE
    if condition in STEER_FALLERS:
        return DELTA_NEGATIVE
    return BASELINE


def rate_chart(plt, stats, phase: str, path: Path) -> None:
    """Horizontal bars of bail rate per condition, baseline rate as a line."""
    present = [c for c in CONDITIONS if c in stats]
    labels = [f"{c} (n={stats[c]['n_parsed']})" for c in present]
    figure, axes = styled_axes(plt, 7.5, 0.34 * len(present) + 1.2)
    axes.barh(
        labels,
        [stats[c]["rate"] for c in present],
        xerr=[stats[c]["sem"] for c in present],
        height=0.7,
        color=[condition_color(c) for c in present],
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


def group_heatmap(plt, rows: list[dict[str, str]], path: Path) -> None:
    """Conditions x conversation groups, prompt-method bail rate per cell."""
    prompt_rows = [r for r in rows if r["phase"] == "prompt"]
    present = sorted(
        {r["condition"] for r in prompt_rows}, key=list(CONDITIONS).index
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
        for condition in CONDITIONS:
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
    if stats_by_phase["prompt"]:
        rate_chart(
            plt, stats_by_phase["prompt"], "prompt", out_dir / "bail_rate_prompt.png"
        )
        group_heatmap(plt, rows, out_dir / "bail_rate_by_group.png")
    if stats_by_phase["tool"]:
        rate_chart(
            plt, stats_by_phase["tool"], "tool", out_dir / "bail_rate_tool.png"
        )


if __name__ == "__main__":
    main()
