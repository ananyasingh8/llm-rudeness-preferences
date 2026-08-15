"""Summarize a probing run: tables, analysis.csv, and charts.

Reads a run folder produced by `emotion_probing.main run` (the latest one by
default, or --run PATH), dispatches on the experiment recorded in its
run_info.json, and writes analysis.csv plus charts into that same folder.

- **bailbench** runs are paired: for each prompt, delta = rude score minus
  normal score, per emotion. Angry/hostile/frustrated are reported as one
  hostility cluster (not separable directions in a 2B model).
- **convabuse** runs are between-groups: examples are grouped by the
  human-annotated labels (majority abusive vs not, severity band, target,
  type, directness) and per-emotion shifts are measured against the
  non-abusive group. The 171 emotions are summarized through the gemotions
  cluster analysis (clusters.json, saved into the run folder at run time).

Usage:
    uv run python -m emotion_probing.analyze [--run PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# Chart colors (project dataviz palette, light mode, CVD-validated).
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
DELTA_POSITIVE = "#e34948"  # abusive/rude scores higher
DELTA_NEGATIVE = "#2a78d6"  # abusive/rude scores lower
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"

# bailbench-2b: the EmotionScope trio treated as one direction at 2B scale.
HOSTILITY_CLUSTER = ("angry", "hostile", "frustrated")

# convabuse-31b: readable names for the gemotions clusters, identified by a
# distinctive member (cluster ids in the analysis file are arbitrary numbers).
CLUSTER_NAME_BY_MEMBER = {
    "angry": "Anger/Hostility",
    "happy": "Positive/Joy",
    "afraid": "Fear/Anxiety",
    "depressed": "Sadness/Despair",
    "amazed": "Surprise/Confusion",
    "guilty": "Shame/Guilt",
    "tired": "Fatigue",
    "defiant": "Defiance/Spite",
    "calm": "Calm/Serenity",
    "compassionate": "Compassion",
    "embarrassed": "Embarrassment",
    "docile": "Passive",
    "paranoid": "Suspicion",
    "nostalgic": "Nostalgia",
    "alert": "Alertness",
}
SEVERITY_BAND_ORDER = (1, 0, -1, -2, -3)
SEVERITY_BAND_LABELS = {
    1: "1\nnot abusive",
    0: "0\nambiguous",
    -1: "-1\nmild",
    -2: "-2\nstrong",
    -3: "-3\nvery strong",
}
MIN_GROUP_SIZE = 5


def find_run_dir(explicit: str | None) -> Path:
    """Resolve the run folder to analyze (--run PATH or the latest run)."""
    if explicit is not None:
        run_dir = Path(explicit)
        if not (run_dir / "run_info.json").exists():
            raise SystemExit(f"error: {run_dir} is not a run folder.")
        return run_dir
    if RESULTS_DIR.exists():
        runs = sorted(
            path
            for path in RESULTS_DIR.iterdir()
            if path.is_dir() and (path / "run_info.json").exists()
        )
        if runs:
            return runs[-1]
    raise SystemExit(
        "error: no run folders found. Run "
        "`uv run python -m emotion_probing.main run` first."
    )


def load_scores(run_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Load scores.csv, returning emotion names and raw rows."""
    scores_file = run_dir / "scores.csv"
    if not scores_file.exists():
        raise SystemExit(f"error: {scores_file} not found.")
    with scores_file.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        emotions = [
            column.removeprefix("score_")
            for column in reader.fieldnames or []
            if column.startswith("score_")
        ]
        rows = list(reader)
    if not rows:
        raise SystemExit(f"error: {scores_file} has no scored rows yet.")
    return emotions, rows


def describe(values: list[float]) -> dict[str, float]:
    """Mean, standard deviation, standard error, and share > 0."""
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": std / math.sqrt(n) if n else 0.0,
        "frac_positive": sum(1 for v in values if v > 0) / n,
    }


def _matplotlib():
    """Import matplotlib for chart output, or None with a helpful message."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print(
            "matplotlib is not installed, so charts were skipped. Run "
            "`uv lock` then `uv sync` to install it, and rerun analyze."
        )
        return None


def _styled_axes(plt, width: float, height: float):
    figure, axes = plt.subplots(figsize=(width, height), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(colors=MUTED, labelcolor=INK, length=0)
    axes.set_axisbelow(True)
    return figure, axes


def _save(plt, figure, figures_dir: Path, name: str) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures_dir / name, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    print(f"Chart written to {figures_dir / name}")


def _diverging_barh(
    plt,
    labels: list[str],
    means: list[float],
    sems: list[float],
    xlabel: str,
    title: str,
    figures_dir: Path,
    name: str,
    height: float,
) -> None:
    """Horizontal diverging bars around zero; position and color carry sign."""
    figure, axes = _styled_axes(plt, 7.5, height)
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
    axes.set_xlabel(xlabel, color=MUTED)
    axes.set_title(title, color=INK, loc="left", pad=12)
    _save(plt, figure, figures_dir, name)


def _group_bars(
    plt,
    labels: list[str],
    stats: list[dict[str, float]],
    ylabel: str,
    title: str,
    figures_dir: Path,
    name: str,
) -> None:
    """Vertical bars of one measure across labeled groups (single series)."""
    figure, axes = _styled_axes(plt, 6.5, 4.2)
    ticks = [f"{label}\n(n={int(s['n'])})" for label, s in zip(labels, stats)]
    axes.bar(
        ticks,
        [s["mean"] for s in stats],
        yerr=[s["sem"] for s in stats],
        width=0.6,
        color=SERIES_1,
        error_kw={"ecolor": MUTED, "elinewidth": 1},
    )
    axes.yaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_ylabel(ylabel, color=MUTED)
    axes.set_title(title, color=INK, loc="left", pad=12)
    _save(plt, figure, figures_dir, name)


# --------------------------------------------------------------------------
# bailbench (paired rude-minus-normal deltas)
# --------------------------------------------------------------------------


def analyze_bailbench(run_dir: Path) -> None:
    """Paired delta analysis for a bailbench run."""
    emotions, raw_rows = load_scores(run_dir)
    pairs: dict[str, dict[str, dict[str, float]]] = {}
    rudeness: dict[str, str] = {}
    for row in raw_rows:
        scores = {name: float(row[f"score_{name}"]) for name in emotions}
        pairs.setdefault(row["example_id"], {})[row["condition"]] = scores
        rudeness[row["example_id"]] = row["rudeness_name"]
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

    plt = _matplotlib()
    if plt is None:
        return
    figures_dir = run_dir / "figures"
    emotion_rows = rows[:-1]
    _diverging_barh(
        plt,
        [str(r["emotion"]) for r in emotion_rows][::-1],
        [float(r["mean"]) for r in emotion_rows][::-1],
        [float(r["sem"]) for r in emotion_rows][::-1],
        "mean Δ cosine score (rude − normal)",
        "Emotion-vector shift under rude prompts",
        figures_dir,
        "emotion_deltas.png",
        7.5,
    )
    type_stats = sorted(
        ((name, describe(deltas)) for name, deltas in by_type.items()),
        key=lambda item: item[1]["mean"],
    )
    _diverging_barh(
        plt,
        [f"{name} (n={int(s['n'])})" for name, s in type_stats],
        [s["mean"] for _, s in type_stats],
        [s["sem"] for _, s in type_stats],
        "mean hostility-cluster Δ (angry+hostile+frustrated)",
        "Hostility shift by rudeness type",
        figures_dir,
        "hostility_by_rudeness_type.png",
        5,
    )
    deltas = list(cluster_deltas.values())
    mean_delta = sum(deltas) / len(deltas)
    figure, axes = _styled_axes(plt, 7, 4)
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
    _save(plt, figure, figures_dir, "hostility_delta_distribution.png")


# --------------------------------------------------------------------------
# convabuse (between-groups shifts against the non-abusive baseline)
# --------------------------------------------------------------------------


def _load_clusters(run_dir: Path, emotions: list[str]) -> dict[str, list[str]]:
    """Load gemotions clusters as {readable name: member emotions}."""
    clusters_file = run_dir / "clusters.json"
    if not clusters_file.exists():
        raise SystemExit(
            f"error: {clusters_file} not found; it is written by the runner."
        )
    saved = json.loads(clusters_file.read_text(encoding="utf-8"))
    named: dict[str, list[str]] = {}
    for cluster_id, members in saved["clusters"].items():
        members = [m for m in members if m in emotions]
        name = next(
            (
                CLUSTER_NAME_BY_MEMBER[m]
                for m in members
                if m in CLUSTER_NAME_BY_MEMBER
            ),
            f"cluster {cluster_id}",
        )
        named[name] = members
    return named


def _cluster_score(row_scores: dict[str, float], members: list[str]) -> float:
    return sum(row_scores[m] for m in members) / len(members)


def analyze_convabuse(run_dir: Path) -> None:
    """Between-groups analysis for a convabuse run."""
    emotions, raw_rows = load_scores(run_dir)
    clusters = _load_clusters(run_dir, emotions)
    cluster_of = {m: name for name, members in clusters.items() for m in members}

    examples = []
    for row in raw_rows:
        examples.append(
            {
                "scores": {n: float(row[f"score_{n}"]) for n in emotions},
                "band": int(row["severity_band"]),
                "abusive": row["abusive_majority"] == "True",
                "system": float(row["target_system_frac"]) >= 0.5,
                "explicit": float(row["direction_explicit_frac"]) >= 0.5,
                "implicit": float(row["direction_implicit_frac"]) >= 0.5,
                "types": {
                    flag: float(row[f"type_{flag}_frac"]) >= 0.5
                    for flag in (
                        "ableism",
                        "homophobic",
                        "intellectual",
                        "racist",
                        "sexist",
                        "sex_harassment",
                        "transphobic",
                    )
                },
            }
        )
    abusive = [e for e in examples if e["abusive"]]
    normal = [e for e in examples if not e["abusive"]]
    if len(abusive) < MIN_GROUP_SIZE or len(normal) < MIN_GROUP_SIZE:
        raise SystemExit(
            "error: not enough scored examples per group yet "
            f"({len(abusive)} abusive / {len(normal)} non-abusive)."
        )
    print(
        f"{len(examples)} examples: {len(abusive)} abusive (majority vote), "
        f"{len(normal)} non-abusive\n"
    )

    def group_emotion_stats(group, name):
        return describe([e["scores"][name] for e in group])

    shifts = []
    for name in emotions:
        stat_a = group_emotion_stats(abusive, name)
        stat_n = group_emotion_stats(normal, name)
        shifts.append(
            {
                "emotion": name,
                "cluster": cluster_of.get(name, "unclustered"),
                "n_abusive": stat_a["n"],
                "n_normal": stat_n["n"],
                "mean_abusive": stat_a["mean"],
                "mean_normal": stat_n["mean"],
                "shift": stat_a["mean"] - stat_n["mean"],
                "sem_shift": math.sqrt(stat_a["sem"] ** 2 + stat_n["sem"] ** 2),
            }
        )
    shifts.sort(key=lambda s: s["shift"], reverse=True)

    print("Top 10 emotions rising under abuse:")
    for s in shifts[:10]:
        print(f"  {s['emotion']:<18}{s['shift']:>+8.4f}  ({s['cluster']})")
    print("Top 10 emotions falling under abuse:")
    for s in shifts[:-11:-1]:
        print(f"  {s['emotion']:<18}{s['shift']:>+8.4f}  ({s['cluster']})")

    cluster_stats = []
    for name, members in sorted(clusters.items()):
        shift_a = [
            _cluster_score(e["scores"], members) for e in abusive
        ]
        shift_n = [
            _cluster_score(e["scores"], members) for e in normal
        ]
        stat_a, stat_n = describe(shift_a), describe(shift_n)
        cluster_stats.append(
            {
                "cluster": f"{name} ({len(members)})",
                "shift": stat_a["mean"] - stat_n["mean"],
                "sem": math.sqrt(stat_a["sem"] ** 2 + stat_n["sem"] ** 2),
            }
        )
    cluster_stats.sort(key=lambda c: c["shift"])
    print("\nCluster shifts (abusive − non-abusive):")
    for c in reversed(cluster_stats):
        print(f"  {c['cluster']:<28}{c['shift']:>+8.4f}")

    with (run_dir / "analysis.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "emotion",
                "cluster",
                "n_abusive",
                "n_normal",
                "mean_abusive",
                "mean_normal",
                "shift",
                "sem_shift",
            ],
        )
        writer.writeheader()
        writer.writerows(shifts)
    print(f"\nSummary written to {run_dir / 'analysis.csv'}")

    plt = _matplotlib()
    if plt is None:
        return
    figures_dir = run_dir / "figures"
    hostility = clusters.get("Anger/Hostility")
    joy = clusters.get("Positive/Joy")

    # 1. Severity trend for the two headline clusters.
    if hostility and joy:
        figure, axes = _styled_axes(plt, 7, 4.5)
        bands = [
            band
            for band in SEVERITY_BAND_ORDER
            if sum(1 for e in examples if e["band"] == band) >= MIN_GROUP_SIZE
        ]
        for members, color, label in (
            (hostility, SERIES_1, "Anger/Hostility"),
            (joy, SERIES_2, "Positive/Joy"),
        ):
            stats = [
                describe(
                    [
                        _cluster_score(e["scores"], members)
                        for e in examples
                        if e["band"] == band
                    ]
                )
                for band in bands
            ]
            positions = range(len(bands))
            axes.errorbar(
                positions,
                [s["mean"] for s in stats],
                yerr=[s["sem"] for s in stats],
                color=color,
                linewidth=2,
                marker="o",
                markersize=5,
                capsize=3,
                label=label,
            )
        axes.set_xticks(range(len(bands)))
        axes.set_xticklabels([SEVERITY_BAND_LABELS[b] for b in bands])
        axes.yaxis.grid(True, color=GRID, linewidth=0.8)
        axes.set_ylabel("mean cluster cosine score", color=MUTED)
        axes.set_title(
            "Emotion-cluster activation by abuse severity", color=INK,
            loc="left", pad=12,
        )
        axes.legend(frameon=False, labelcolor=INK)
        _save(plt, figure, figures_dir, "severity_trend.png")

    # 2. Per-cluster shift, abusive vs non-abusive.
    _diverging_barh(
        plt,
        [c["cluster"] for c in cluster_stats],
        [c["shift"] for c in cluster_stats],
        [c["sem"] for c in cluster_stats],
        "mean cluster score shift (abusive − non-abusive)",
        "Emotion-cluster shift under abuse",
        figures_dir,
        "cluster_shifts.png",
        5.5,
    )

    # 3. Top movers, labeled with their cluster.
    movers = shifts[:10] + shifts[-10:]
    movers.sort(key=lambda s: s["shift"])
    _diverging_barh(
        plt,
        [f"{s['emotion']} — {s['cluster']}" for s in movers],
        [s["shift"] for s in movers],
        [s["sem_shift"] for s in movers],
        "mean score shift (abusive − non-abusive)",
        "Top emotion movers under abuse",
        figures_dir,
        "top_movers.png",
        6.5,
    )

    # 4. Abuse target: is system-directed abuse special?
    if hostility:
        groups = [
            ("not abusive", normal),
            ("abusive,\nother target", [e for e in abusive if not e["system"]]),
            ("abusive,\nat the system", [e for e in abusive if e["system"]]),
        ]
        groups = [(label, g) for label, g in groups if len(g) >= MIN_GROUP_SIZE]
        _group_bars(
            plt,
            [label for label, _ in groups],
            [
                describe([_cluster_score(e["scores"], hostility) for e in g])
                for _, g in groups
            ],
            "mean Anger/Hostility cluster score",
            "Hostility activation by abuse target",
            figures_dir,
            "target_comparison.png",
        )

        # 5. Abuse type breakdown (shift vs non-abusive baseline).
        baseline = describe(
            [_cluster_score(e["scores"], hostility) for e in normal]
        )
        type_rows = []
        for flag, label in (
            ("sex_harassment", "sexual harassment"),
            ("intellectual", "intellectual"),
            ("sexist", "sexist"),
            ("homophobic", "homophobic"),
            ("racist", "racist"),
            ("transphobic", "transphobic"),
            ("ableism", "ableism"),
        ):
            group = [e for e in examples if e["types"][flag]]
            if len(group) < MIN_GROUP_SIZE:
                continue
            stats = describe(
                [_cluster_score(e["scores"], hostility) for e in group]
            )
            type_rows.append(
                {
                    "label": f"{label} (n={int(stats['n'])})",
                    "shift": stats["mean"] - baseline["mean"],
                    "sem": math.sqrt(stats["sem"] ** 2 + baseline["sem"] ** 2),
                }
            )
        type_rows.sort(key=lambda r: r["shift"])
        if type_rows:
            _diverging_barh(
                plt,
                [r["label"] for r in type_rows],
                [r["shift"] for r in type_rows],
                [r["sem"] for r in type_rows],
                "Anger/Hostility shift vs non-abusive baseline",
                "Hostility shift by abuse type",
                figures_dir,
                "type_breakdown.png",
                4,
            )

        # 6. Explicit vs implicit abuse.
        direction_groups = [
            ("not abusive", normal),
            ("implicit abuse", [e for e in abusive if e["implicit"]]),
            ("explicit abuse", [e for e in abusive if e["explicit"]]),
        ]
        direction_groups = [
            (label, g) for label, g in direction_groups if len(g) >= MIN_GROUP_SIZE
        ]
        _group_bars(
            plt,
            [label for label, _ in direction_groups],
            [
                describe([_cluster_score(e["scores"], hostility) for e in g])
                for _, g in direction_groups
            ],
            "mean Anger/Hostility cluster score",
            "Hostility activation by abuse directness",
            figures_dir,
            "direction_comparison.png",
        )


def main() -> int:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description="Analyze a probing run.")
    parser.add_argument(
        "--run",
        default=None,
        help="run folder to analyze (default: the latest under results/)",
    )
    args = parser.parse_args()
    run_dir = find_run_dir(args.run)
    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    print(f"Analyzing {run_dir.name} ({run_info['dataset']})\n")
    if run_info["dataset"] == "bailbench":
        analyze_bailbench(run_dir)
    elif run_info["dataset"] == "convabuse":
        analyze_convabuse(run_dir)
    else:
        print(f"error: unknown dataset {run_info['dataset']!r}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
