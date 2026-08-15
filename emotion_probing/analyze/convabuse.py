"""ConvAbuse run analysis: severity-band figures, maps, overview, breakdowns.

All severity shifts are measured against BASELINE_BAND (currently band 0,
"ambiguous"): that band shows its raw resting profile, and every other band —
including band 1 — is compared to it. The breakdown figures (target / type /
directness) instead use the majority-vote abusive flag, since those labels
only exist for abusive examples.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from emotion_probing.analyze.common import (
    DELTA_NEGATIVE,
    DELTA_POSITIVE,
    GRID,
    INK,
    MIN_GROUP_SIZE,
    MUTED,
    NEUTRAL_MID,
    SERIES_1,
    SERIES_2,
    SEVERITY_BAND_LABELS,
    SEVERITY_BAND_ORDER,
    SEVERITY_RAMP,
    describe,
    diverging_barh,
    get_matplotlib,
    group_bars,
    load_clusters,
    load_pca_coordinates,
    load_scores,
    save_figure,
    sem_diff,
    styled_axes,
)
from emotion_probing.analyze.maps import cluster_map, pca_map

TOP_N = 10
# All severity shifts are measured against this band.
BASELINE_BAND = 0
# A synthetic extra band: all abusive-band examples pooled together. Kept
# disjoint from BASELINE_BAND so its shift vs the baseline is a clean
# independent-groups comparison.
POOLED_BAND = "avg"
POOLED_MEMBERS = (-1, -2, -3)
BAND_LABELS = {**SEVERITY_BAND_LABELS, POOLED_BAND: "avg (-1 to -3)"}
ABUSE_TYPES = (
    ("sex_harassment", "sexual harassment"),
    ("intellectual", "intellectual"),
    ("sexist", "sexist"),
    ("homophobic", "homophobic"),
    ("racist", "racist"),
    ("transphobic", "transphobic"),
    ("ableism", "ableism"),
)


def _parse_examples(emotions: list[str], raw_rows: list[dict[str, str]]):
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
                    for flag, _ in ABUSE_TYPES
                },
            }
        )
    return examples


def _cluster_score(scores: dict[str, float], members: list[str]) -> float:
    return sum(scores[m] for m in members) / len(members)


def analyze_convabuse(run_dir: Path) -> None:
    """Produce the full convabuse figure suite and analysis.csv."""
    emotions, raw_rows = load_scores(run_dir)
    clusters = load_clusters(run_dir, emotions)
    cluster_of = {m: name for name, members in clusters.items() for m in members}
    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    coordinates = load_pca_coordinates(int(run_info["probe_layer"]))
    examples = _parse_examples(emotions, raw_rows)

    band_groups: dict[object, list] = {
        band: [e for e in examples if e["band"] == band]
        for band in SEVERITY_BAND_ORDER
    }
    band_groups[POOLED_BAND] = [
        e for e in examples if e["band"] in POOLED_MEMBERS
    ]
    bands = [
        b
        for b in (*SEVERITY_BAND_ORDER, POOLED_BAND)
        if len(band_groups[b]) >= MIN_GROUP_SIZE
    ]
    if BASELINE_BAND not in bands:
        raise SystemExit(
            f"error: not enough baseline (band {BASELINE_BAND}) examples yet."
        )
    for band in SEVERITY_BAND_ORDER:
        if band not in bands:
            print(f"note: band {band} has < {MIN_GROUP_SIZE} examples; skipped.")
    print(
        "examples per band: "
        + ", ".join(f"{b}: {len(band_groups[b])}" for b in bands)
    )

    band_stats = {
        band: {
            name: describe([e["scores"][name] for e in band_groups[band]])
            for name in emotions
        }
        for band in bands
    }
    baseline = band_stats[BASELINE_BAND]

    def shift(band: int, name: str) -> float:
        return band_stats[band][name]["mean"] - baseline[name]["mean"]

    # ---- printed summary + analysis.csv (sorted by the -3 vs 1 shift) ------
    real_bands = [b for b in bands if b != POOLED_BAND]
    focus_band = -3 if -3 in bands else real_bands[-1]
    order = sorted(emotions, key=lambda n: shift(focus_band, n), reverse=True)
    print(
        f"\nTop {TOP_N} rising emotions "
        f"(band {focus_band} vs band {BASELINE_BAND}):"
    )
    for name in order[:TOP_N]:
        print(
            f"  {name:<18}{shift(focus_band, name):>+8.4f}"
            f"  ({cluster_of.get(name, '?')})"
        )
    print(
        f"Top {TOP_N} falling emotions "
        f"(band {focus_band} vs band {BASELINE_BAND}):"
    )
    for name in order[:-TOP_N - 1:-1]:
        print(
            f"  {name:<18}{shift(focus_band, name):>+8.4f}"
            f"  ({cluster_of.get(name, '?')})"
        )
    print(
        f"\nCluster mean shifts (band {focus_band} vs band {BASELINE_BAND}):"
    )
    cluster_shifts = sorted(
        (
            (name, sum(shift(focus_band, m) for m in members) / len(members))
            for name, members in clusters.items()
        ),
        key=lambda item: -item[1],
    )
    for name, value in cluster_shifts:
        print(f"  {name:<28}{value:>+8.4f}")

    with (run_dir / "analysis.csv").open("w", encoding="utf-8", newline="") as f:
        columns = (
            ["emotion", "cluster"]
            + [f"mean_band_{b}" for b in bands]
            + [f"shift_band_{b}" for b in bands if b != BASELINE_BAND]
            + [f"sem_shift_band_{focus_band}"]
        )
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for name in order:
            row: dict[str, object] = {
                "emotion": name,
                "cluster": cluster_of.get(name, "unclustered"),
            }
            for band in bands:
                row[f"mean_band_{band}"] = round(band_stats[band][name]["mean"], 6)
                if band != BASELINE_BAND:
                    row[f"shift_band_{band}"] = round(shift(band, name), 6)
            row[f"sem_shift_band_{focus_band}"] = round(
                sem_diff(band_stats[focus_band][name], baseline[name]), 6
            )
            writer.writerow(row)
    print(f"\nSummary written to {run_dir / 'analysis.csv'}")

    plt = get_matplotlib()
    if plt is None:
        return
    figures = run_dir / "figures"

    # ---- figure set 1: per severity band ------------------------------------
    for band in bands:
        band_dir = figures / "bands"
        label = BAND_LABELS[band]
        if band == BASELINE_BAND:
            # Baseline band: the raw resting profile, not a shift.
            top = sorted(
                emotions, key=lambda n: baseline[n]["mean"], reverse=True
            )[:TOP_N]
            figure, axes = styled_axes(plt, 7.5, 4.5)
            axes.barh(
                [f"{n} — {cluster_of.get(n, '?')}" for n in top][::-1],
                [baseline[n]["mean"] for n in top][::-1],
                xerr=[baseline[n]["sem"] for n in top][::-1],
                height=0.7,
                color=SERIES_1,
                error_kw={"ecolor": MUTED, "elinewidth": 1},
            )
            axes.xaxis.grid(True, color=GRID, linewidth=0.8)
            axes.set_xlabel("mean cosine score", color=MUTED)
            axes.set_title(
                f"Band {label}: baseline emotion profile (raw top {TOP_N})",
                color=INK, loc="left", pad=12,
            )
            save_figure(plt, figure, band_dir / f"band_{band}_top10.png")
            highlights = {name: SERIES_1 for name in top}
            map_title = f"Band {label}: baseline top {TOP_N}"
        else:
            band_order = sorted(
                emotions, key=lambda n: shift(band, n), reverse=True
            )
            band_risers = band_order[:TOP_N]
            band_fallers = band_order[-TOP_N:]
            movers = sorted(band_risers + band_fallers, key=lambda n: shift(band, n))
            diverging_barh(
                plt,
                [f"{n} — {cluster_of.get(n, '?')}" for n in movers],
                [shift(band, n) for n in movers],
                [sem_diff(band_stats[band][n], baseline[n]) for n in movers],
                f"mean score shift vs band {BAND_LABELS[BASELINE_BAND]}",
                f"Band {label}: top rising and falling emotions",
                band_dir / f"band_{band}_movers.png",
                6.5,
            )
            highlights = {name: DELTA_POSITIVE for name in band_risers} | {
                name: DELTA_NEGATIVE for name in band_fallers
            }
            map_title = f"Band {label}: biggest shifts vs band {BASELINE_BAND}"
        cluster_map(
            plt, coordinates, clusters, highlights,
            f"{map_title} — cluster view",
            band_dir / f"band_{band}_cluster_map.png",
        )
        pca_map(
            plt, coordinates, highlights,
            f"{map_title} — valence/disposition view",
            band_dir / f"band_{band}_pca_map.png",
        )

    # ---- figure set 2: most severe band vs baseline band comparison ---------
    if focus_band != BASELINE_BAND:
        comparison = figures / "comparison"
        risers = order[:TOP_N]
        fallers = order[:-TOP_N - 1:-1]
        movers = sorted(risers + fallers, key=lambda n: shift(focus_band, n))
        diverging_barh(
            plt,
            [f"{n} — {cluster_of.get(n, '?')}" for n in movers],
            [shift(focus_band, n) for n in movers],
            [sem_diff(band_stats[focus_band][n], baseline[n]) for n in movers],
            f"mean score shift (band {focus_band} − band {BASELINE_BAND})",
            f"Biggest emotion shifts: band {focus_band} vs band {BASELINE_BAND}",
            comparison / "top_movers.png",
            6.5,
        )
        highlights = {name: DELTA_POSITIVE for name in risers} | {
            name: DELTA_NEGATIVE for name in fallers
        }
        title = f"Band {focus_band} vs band {BASELINE_BAND}: biggest shifts"
        cluster_map(
            plt, coordinates, clusters, highlights,
            f"{title} — cluster view", comparison / "cluster_map.png",
        )
        pca_map(
            plt, coordinates, highlights,
            f"{title} — valence/disposition view", comparison / "pca_map.png",
        )

    # ---- figure set 3: all-emotion overview ---------------------------------
    overview = figures / "overview"

    # Heatmap: every emotion x severity band, cell = shift vs band 1.
    data = [[shift(band, name) for band in bands] for name in order]
    peak = max(abs(value) for row in data for value in row) or 1.0
    from matplotlib.colors import LinearSegmentedColormap

    colormap = LinearSegmentedColormap.from_list(
        "shift", [DELTA_NEGATIVE, NEUTRAL_MID, DELTA_POSITIVE]
    )
    figure, axes = styled_axes(plt, 7, 1.5 + 0.125 * len(order))
    image = axes.imshow(
        data, aspect="auto", cmap=colormap, vmin=-peak, vmax=peak,
        interpolation="nearest",
    )
    axes.set_yticks(range(len(order)))
    axes.set_yticklabels(order, fontsize=4.5)
    axes.set_xticks(range(len(bands)))
    axes.set_xticklabels([BAND_LABELS[b] for b in bands], fontsize=7)
    axes.set_title(
        f"All emotions: activation shift vs band {BASELINE_BAND}, by severity",
        color=INK, loc="left", pad=12,
    )
    bar = figure.colorbar(image, ax=axes, shrink=0.3, pad=0.02)
    bar.ax.tick_params(labelsize=7, colors=MUTED)
    bar.set_label(f"shift vs band {BASELINE_BAND}", color=MUTED, fontsize=8)
    save_figure(plt, figure, overview / "heatmap_all_emotions.png")

    # Tall grouped bars: 10 risers + 10 fallers, raw mean per band.
    chosen = order[:TOP_N] + order[-TOP_N:]
    figure, axes = styled_axes(plt, 7.5, 13)
    bar_height = 0.8 / len(bands)
    positions = range(len(chosen))
    for j, band in enumerate(bands):
        if band == POOLED_BAND:
            color = SERIES_2  # the synthetic average stands apart from the ramp
        else:
            color = SEVERITY_RAMP[SEVERITY_BAND_ORDER.index(band)]
        axes.barh(
            [i + (j - len(bands) / 2 + 0.5) * bar_height for i in positions],
            [band_stats[band][n]["mean"] for n in chosen],
            height=bar_height * 0.92,
            color=color,
            label=BAND_LABELS[band],
        )
    axes.set_yticks(list(positions))
    axes.set_yticklabels(
        [f"{n} — {cluster_of.get(n, '?')}" for n in chosen], fontsize=8
    )
    axes.invert_yaxis()
    axes.axvline(0, color=MUTED, linewidth=1)
    axes.xaxis.grid(True, color=GRID, linewidth=0.8)
    axes.set_xlabel("mean cosine score", color=MUTED)
    axes.set_title(
        f"Top {TOP_N} risers and fallers: activation by severity band",
        color=INK, loc="left", pad=12,
    )
    axes.legend(frameon=False, labelcolor=INK, fontsize=8, loc="lower right")
    save_figure(plt, figure, overview / "top_movers_by_band.png")

    # ---- figure set 4: breakdowns (majority-vote groups) ---------------------
    breakdowns = figures / "breakdowns"
    hostility = clusters.get("Anger/Hostility")
    abusive = [e for e in examples if e["abusive"]]
    normal = [e for e in examples if not e["abusive"]]
    if not hostility or len(abusive) < MIN_GROUP_SIZE:
        print("note: breakdown figures skipped (not enough abusive examples).")
        return

    target_groups = [
        ("not abusive", normal),
        ("abusive,\nother target", [e for e in abusive if not e["system"]]),
        ("abusive,\nat the system", [e for e in abusive if e["system"]]),
    ]
    target_groups = [(l, g) for l, g in target_groups if len(g) >= MIN_GROUP_SIZE]
    group_bars(
        plt,
        [label for label, _ in target_groups],
        [
            describe([_cluster_score(e["scores"], hostility) for e in g])
            for _, g in target_groups
        ],
        "mean Anger/Hostility cluster score",
        "Hostility activation by abuse target",
        breakdowns / "target_comparison.png",
    )

    baseline_hostility = describe(
        [_cluster_score(e["scores"], hostility) for e in normal]
    )
    type_rows = []
    for flag, label in ABUSE_TYPES:
        group = [e for e in examples if e["types"][flag]]
        if len(group) < MIN_GROUP_SIZE:
            continue
        stats = describe([_cluster_score(e["scores"], hostility) for e in group])
        type_rows.append(
            {
                "label": f"{label} (n={int(stats['n'])})",
                "shift": stats["mean"] - baseline_hostility["mean"],
                "sem": sem_diff(stats, baseline_hostility),
            }
        )
    type_rows.sort(key=lambda r: r["shift"])
    if type_rows:
        diverging_barh(
            plt,
            [r["label"] for r in type_rows],
            [r["shift"] for r in type_rows],
            [r["sem"] for r in type_rows],
            "Anger/Hostility shift vs non-abusive baseline",
            "Hostility shift by abuse type",
            breakdowns / "type_breakdown.png",
            4,
        )

    direction_groups = [
        ("not abusive", normal),
        ("implicit abuse", [e for e in abusive if e["implicit"]]),
        ("explicit abuse", [e for e in abusive if e["explicit"]]),
    ]
    direction_groups = [
        (l, g) for l, g in direction_groups if len(g) >= MIN_GROUP_SIZE
    ]
    group_bars(
        plt,
        [label for label, _ in direction_groups],
        [
            describe([_cluster_score(e["scores"], hostility) for e in g])
            for _, g in direction_groups
        ],
        "mean Anger/Hostility cluster score",
        "Hostility activation by abuse directness",
        breakdowns / "direction_comparison.png",
    )
