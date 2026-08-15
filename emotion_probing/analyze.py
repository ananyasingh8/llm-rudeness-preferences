"""Summarize how emotion scores shift between normal and rude prompts.

Reads results/scores.csv (written by `emotion_probing.main run`), pairs the
normal and rude row for each BailBench prompt, and reports the per-emotion
delta (rude score minus normal score) averaged over all complete pairs. A
positive delta means that emotion's representation activates more strongly
when the user is rude.

Because angry / hostile / frustrated are not reliably separable directions in
a 2B model, they are additionally reported as one combined "hostility cluster",
which is the primary signal for the rudeness-dispreference question.

Usage:
    uv run python -m emotion_probing.analyze
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "results" / "scores.csv"
ANALYSIS_FILE = Path(__file__).parent / "results" / "analysis.csv"
FIGURES_DIR = Path(__file__).parent / "results" / "figures"
HOSTILITY_CLUSTER = ("angry", "hostile", "frustrated")

# Chart colors (project dataviz palette, light mode, CVD-validated).
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
DELTA_POSITIVE = "#e34948"  # rude scores higher than normal
DELTA_NEGATIVE = "#2a78d6"  # rude scores lower than normal


PairScores = dict[str, dict[str, dict[str, float]]]


def load_pairs() -> tuple[list[str], PairScores, dict[str, str]]:
    """Load scores.csv into {bailbench_id: {condition: {emotion: score}}}.

    Also returns the emotion names (in file order) and each prompt's
    rudeness_name for the per-type breakdown.
    """
    if not RESULTS_FILE.exists():
        raise SystemExit(
            f"error: {RESULTS_FILE} not found. Run "
            "`uv run python -m emotion_probing.main run` first."
        )
    with RESULTS_FILE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        emotions = [
            column.removeprefix("score_")
            for column in reader.fieldnames or []
            if column.startswith("score_")
        ]
        pairs: dict[str, dict[str, dict[str, float]]] = {}
        rudeness: dict[str, str] = {}
        for row in reader:
            scores = {name: float(row[f"score_{name}"]) for name in emotions}
            pairs.setdefault(row["bailbench_id"], {})[row["condition"]] = scores
            rudeness[row["bailbench_id"]] = row["rudeness_name"]
    return emotions, pairs, rudeness


def describe(deltas: list[float]) -> dict[str, float]:
    """Mean, standard deviation, standard error, and share > 0 for deltas."""
    n = len(deltas)
    mean = sum(deltas) / n
    variance = sum((d - mean) ** 2 for d in deltas) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    return {
        "n_pairs": n,
        "mean_delta": mean,
        "std_delta": std,
        "sem": std / math.sqrt(n) if n else 0.0,
        "frac_rude_higher": sum(1 for d in deltas if d > 0) / n,
    }


def save_figures(
    emotion_rows: list[dict[str, object]],
    cluster_deltas: list[float],
    by_type: dict[str, list[float]],
) -> None:
    """Write the summary charts to results/figures/ (skipped without matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed, so charts were skipped. Run "
            "`uv lock` then `uv sync` to install it, and rerun analyze."
        )
        return
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def styled_axes(width: float, height: float) -> tuple[object, object]:
        figure, axes = plt.subplots(figsize=(width, height), dpi=200)
        figure.patch.set_facecolor(SURFACE)
        axes.set_facecolor(SURFACE)
        for spine in axes.spines.values():
            spine.set_visible(False)
        axes.tick_params(colors=MUTED, labelcolor=INK, length=0)
        axes.set_axisbelow(True)
        return figure, axes

    def save(figure: object, name: str) -> None:
        figure.savefig(FIGURES_DIR / name, bbox_inches="tight", facecolor=SURFACE)
        plt.close(figure)
        print(f"Chart written to {FIGURES_DIR / name}")

    # 1. Diverging bars: mean delta per emotion (position and color carry sign).
    labels = [str(row["emotion"]) for row in emotion_rows][::-1]
    means = [float(row["mean_delta"]) for row in emotion_rows][::-1]
    sems = [float(row["sem"]) for row in emotion_rows][::-1]
    figure, axes = styled_axes(7, 7.5)
    axes.barh(
        labels,
        means,
        xerr=sems,
        height=0.7,
        color=[DELTA_POSITIVE if mean > 0 else DELTA_NEGATIVE for mean in means],
        error_kw={"ecolor": MUTED, "elinewidth": 1},
    )
    axes.axvline(0, color=BASELINE, linewidth=1)
    axes.xaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_xlabel("mean Δ cosine score (rude − normal)", color=MUTED)
    axes.set_title(
        "Emotion-vector shift under rude prompts", color=INK, loc="left", pad=12
    )
    save(figure, "emotion_deltas.png")

    # 2. Hostility-cluster delta by Culpeper rudeness type.
    type_stats = sorted(
        ((name, describe(deltas)) for name, deltas in by_type.items()),
        key=lambda item: item[1]["mean_delta"],
    )
    type_labels = [f"{name} (n={stats['n_pairs']})" for name, stats in type_stats]
    type_means = [stats["mean_delta"] for _, stats in type_stats]
    type_sems = [stats["sem"] for _, stats in type_stats]
    figure, axes = styled_axes(7.5, 5)
    axes.barh(
        type_labels,
        type_means,
        xerr=type_sems,
        height=0.7,
        color=[DELTA_POSITIVE if mean > 0 else DELTA_NEGATIVE for mean in type_means],
        error_kw={"ecolor": MUTED, "elinewidth": 1},
    )
    axes.axvline(0, color=BASELINE, linewidth=1)
    axes.xaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_xlabel(
        "mean hostility-cluster Δ (angry+hostile+frustrated)", color=MUTED
    )
    axes.set_title(
        "Hostility shift by rudeness type", color=INK, loc="left", pad=12
    )
    save(figure, "hostility_by_rudeness_type.png")

    # 3. Distribution of per-pair hostility deltas (is the shift broad or outliers?).
    mean_delta = sum(cluster_deltas) / len(cluster_deltas)
    figure, axes = styled_axes(7, 4)
    axes.hist(cluster_deltas, bins=40, color=DELTA_NEGATIVE)
    axes.axvline(0, color=BASELINE, linewidth=1)
    axes.axvline(mean_delta, color=INK, linewidth=1.2)
    axes.annotate(
        f"mean {mean_delta:+.4f}",
        xy=(mean_delta, 0.95),
        xycoords=("data", "axes fraction"),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK,
    )
    axes.yaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_xlabel("per-pair hostility-cluster Δ (rude − normal)", color=MUTED)
    axes.set_ylabel("prompt pairs", color=MUTED)
    axes.set_title(
        "Distribution of hostility shifts", color=INK, loc="left", pad=12
    )
    save(figure, "hostility_delta_distribution.png")


def main() -> int:
    """Compute and print the rude-minus-normal delta summary."""
    emotions, pairs, rudeness = load_pairs()
    complete = {
        prompt_id: conditions
        for prompt_id, conditions in pairs.items()
        if "normal" in conditions and "rude" in conditions
    }
    if not complete:
        print(
            "error: no complete normal/rude pairs in the results yet.",
            file=sys.stderr,
        )
        return 1
    print(f"{len(complete)} complete prompt pairs\n")

    deltas_by_emotion = {
        name: [c["rude"][name] - c["normal"][name] for c in complete.values()]
        for name in emotions
    }
    cluster_deltas = {
        prompt_id: sum(c["rude"][e] - c["normal"][e] for e in HOSTILITY_CLUSTER)
        / len(HOSTILITY_CLUSTER)
        for prompt_id, c in complete.items()
    }

    rows = [
        {"emotion": name} | describe(deltas)
        for name, deltas in deltas_by_emotion.items()
    ]
    rows.sort(key=lambda row: row["mean_delta"], reverse=True)
    rows.append(
        {"emotion": "HOSTILITY_CLUSTER"} | describe(list(cluster_deltas.values()))
    )

    header = f"{'emotion':<20}{'mean delta':>12}{'sem':>10}{'% rude higher':>15}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['emotion']:<20}{row['mean_delta']:>+12.4f}{row['sem']:>10.4f}"
            f"{row['frac_rude_higher']:>14.0%}"
        )

    print("\nHostility cluster (angry+hostile+frustrated) by rudeness type:")
    by_type: dict[str, list[float]] = {}
    for prompt_id, delta in cluster_deltas.items():
        by_type.setdefault(rudeness[prompt_id], []).append(delta)
    ordered = sorted(by_type.items(), key=lambda item: -sum(item[1]) / len(item[1]))
    for name, deltas in ordered:
        stats = describe(deltas)
        print(f"  {name:<42}{stats['mean_delta']:>+8.4f}  (n={stats['n_pairs']})")

    save_figures(rows[:-1], list(cluster_deltas.values()), by_type)

    with ANALYSIS_FILE.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "emotion",
                "n_pairs",
                "mean_delta",
                "std_delta",
                "sem",
                "frac_rude_higher",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary written to {ANALYSIS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
