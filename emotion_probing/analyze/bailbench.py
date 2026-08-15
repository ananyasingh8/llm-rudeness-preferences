"""BailBench run analysis: paired rude-minus-normal deltas."""

from __future__ import annotations

import csv
from pathlib import Path

from emotion_probing.analyze.common import (
    BASELINE,
    GRID,
    INK,
    MUTED,
    SERIES_1,
    describe,
    diverging_barh,
    get_matplotlib,
    load_scores,
    save_figure,
    styled_axes,
)

# The EmotionScope trio treated as one direction at 2B scale.
HOSTILITY_CLUSTER = ("angry", "hostile", "frustrated")


def analyze_bailbench(run_dir: Path) -> None:
    """Paired delta analysis for a bailbench run."""
    emotions, raw_rows = load_scores(run_dir)
    # results/run1 predates the per-run layout and named the id column
    # bailbench_id; newer runs use example_id.
    id_column = "example_id" if "example_id" in raw_rows[0] else "bailbench_id"
    pairs: dict[str, dict[str, dict[str, float]]] = {}
    rudeness: dict[str, str] = {}
    for row in raw_rows:
        scores = {name: float(row[f"score_{name}"]) for name in emotions}
        pairs.setdefault(row[id_column], {})[row["condition"]] = scores
        rudeness[row[id_column]] = row["rudeness_name"]
    complete = {
        example: conditions
        for example, conditions in pairs.items()
        if "normal" in conditions and "rude" in conditions
    }
    if not complete:
        raise SystemExit("error: no complete normal/rude pairs in this run yet.")
    print(f"{len(complete)} complete prompt pairs\n")

    deltas_by_emotion = {
        name: [c["rude"][name] - c["normal"][name] for c in complete.values()]
        for name in emotions
    }
    cluster_deltas = {
        example: sum(c["rude"][e] - c["normal"][e] for e in HOSTILITY_CLUSTER)
        / len(HOSTILITY_CLUSTER)
        for example, c in complete.items()
    }

    rows = [
        {"emotion": name} | describe(deltas)
        for name, deltas in deltas_by_emotion.items()
    ]
    rows.sort(key=lambda row: row["mean"], reverse=True)
    rows.append(
        {"emotion": "HOSTILITY_CLUSTER"} | describe(list(cluster_deltas.values()))
    )

    header = f"{'emotion':<20}{'mean delta':>12}{'sem':>10}{'% rude higher':>15}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['emotion']:<20}{row['mean']:>+12.4f}{row['sem']:>10.4f}"
            f"{row['frac_positive']:>14.0%}"
        )

    print("\nHostility cluster by rudeness type:")
    by_type: dict[str, list[float]] = {}
    for example, delta in cluster_deltas.items():
        by_type.setdefault(rudeness[example], []).append(delta)
    ordered = sorted(by_type.items(), key=lambda item: -sum(item[1]) / len(item[1]))
    for name, deltas in ordered:
        stats = describe(deltas)
        print(f"  {name:<42}{stats['mean']:>+8.4f}  (n={int(stats['n'])})")

    with (run_dir / "analysis.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["emotion", "n", "mean", "std", "sem", "frac_positive"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary written to {run_dir / 'analysis.csv'}")

    plt = get_matplotlib()
    if plt is None:
        return
    figures_dir = run_dir / "figures"
    emotion_rows = rows[:-1]
    diverging_barh(
        plt,
        [str(r["emotion"]) for r in emotion_rows][::-1],
        [float(r["mean"]) for r in emotion_rows][::-1],
        [float(r["sem"]) for r in emotion_rows][::-1],
        "mean Δ cosine score (rude − normal)",
        "Emotion-vector shift under rude prompts",
        figures_dir / "emotion_deltas.png",
        7.5,
    )
    type_stats = sorted(
        ((name, describe(deltas)) for name, deltas in by_type.items()),
        key=lambda item: item[1]["mean"],
    )
    diverging_barh(
        plt,
        [f"{name} (n={int(s['n'])})" for name, s in type_stats],
        [s["mean"] for _, s in type_stats],
        [s["sem"] for _, s in type_stats],
        "mean hostility-cluster Δ (angry+hostile+frustrated)",
        "Hostility shift by rudeness type",
        figures_dir / "hostility_by_rudeness_type.png",
        5,
    )
    deltas = list(cluster_deltas.values())
    mean_delta = sum(deltas) / len(deltas)
    figure, axes = styled_axes(plt, 7, 4)
    axes.hist(deltas, bins=40, color=SERIES_1)
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
    axes.set_title("Distribution of hostility shifts", color=INK, loc="left", pad=12)
    save_figure(plt, figure, figures_dir / "hostility_delta_distribution.png")
