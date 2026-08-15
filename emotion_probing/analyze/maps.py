"""2D emotion-vector maps: the cluster lens and the PC1/PC2 axes lens.

Both figures place all 171 emotions at their gemotions PCA coordinates
(PC1 = valence, PC2 = disposition) — the real geometry of the vectors, not a
decorative layout. The cluster map emphasizes the 15-cluster structure (hulls
and name labels); the PC1/PC2 map emphasizes the axes (quadrant guides, axis
interpretations) with only the highlighted emotions labeled.
"""

from __future__ import annotations

from pathlib import Path

from emotion_probing.analyze.common import (
    BASELINE,
    GRID,
    INK,
    MUTED,
    save_figure,
    styled_axes,
)


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain convex hull (stdlib-only)."""
    points = sorted(set(points))
    if len(points) <= 2:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


_LABEL_OFFSETS = (
    (6, 4, "left"),
    (6, -11, "left"),
    (-6, 4, "right"),
    (-6, -11, "right"),
)


def _draw_highlights(
    axes,
    coordinates: dict[str, tuple[float, float]],
    highlights: dict[str, str],
) -> None:
    """Draw highlighted emotions as labeled colored points.

    Neighbors in a dense knot get different label directions (a 4-way cycle by
    position rank) plus a translucent backing box, which keeps the labels
    legible without a layout solver.
    """
    ordered = sorted(highlights.items(), key=lambda kv: coordinates[kv[0]])
    for index, (name, color) in enumerate(ordered):
        x, y = coordinates[name]
        axes.scatter(
            [x], [y], s=46, color=color, edgecolors="white", linewidths=1.2,
            zorder=3,
        )
        dx, dy, align = _LABEL_OFFSETS[index % len(_LABEL_OFFSETS)]
        axes.annotate(
            name, (x, y), xytext=(dx, dy), textcoords="offset points",
            color=color, fontsize=8, ha=align, zorder=4,
            bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none",
                  "pad": 0.5},
        )


def _separate_labels(
    positions: dict[str, list[float]],
    min_gap: float = 0.55,
    x_window: float = 1.8,
) -> None:
    """Nudge vertically-colliding cluster labels apart (in place).

    A single bottom-up pass: labels that are horizontally close and less than
    min_gap apart vertically get pushed up. Crude but effective for 15 labels.
    """
    ordered = sorted(positions, key=lambda name: positions[name][1])
    for i, current in enumerate(ordered):
        for previous in ordered[:i]:
            close_x = abs(positions[current][0] - positions[previous][0]) < x_window
            gap = positions[current][1] - positions[previous][1]
            if close_x and gap < min_gap:
                positions[current][1] = positions[previous][1] + min_gap


def _base_scatter(plt, coordinates, highlights):
    figure, axes = styled_axes(plt, 10, 8)
    axes.tick_params(labelcolor=MUTED)
    axes.grid(True, color=GRID, linewidth=0.6)
    base = [name for name in coordinates if name not in highlights]
    axes.scatter(
        [coordinates[name][0] for name in base],
        [coordinates[name][1] for name in base],
        s=14, color=MUTED, alpha=0.45, linewidths=0,
    )
    return figure, axes


def cluster_map(
    plt,
    coordinates: dict[str, tuple[float, float]],
    clusters: dict[str, list[str]],
    highlights: dict[str, str],
    title: str,
    path: Path,
) -> None:
    """The structural lens: cluster hulls + names, highlighted emotions marked.

    `highlights` maps emotion name -> color. Hulls are drawn only for clusters
    containing at least one highlighted emotion, to keep the map readable.
    """
    figure, axes = _base_scatter(plt, coordinates, highlights)

    for cluster_name, members in clusters.items():
        points = [coordinates[m] for m in members if m in coordinates]
        if not points:
            continue
        if any(m in highlights for m in members):
            hull = _convex_hull(points)
            if len(hull) >= 3:
                axes.fill(
                    [p[0] for p in hull], [p[1] for p in hull],
                    color=MUTED, alpha=0.08, zorder=1,
                )
                axes.plot(
                    [p[0] for p in hull] + [hull[0][0]],
                    [p[1] for p in hull] + [hull[0][1]],
                    color=MUTED, alpha=0.35, linewidth=1, zorder=1,
                )
    label_positions = {
        name: [
            sum(coordinates[m][0] for m in members if m in coordinates)
            / max(1, sum(1 for m in members if m in coordinates)),
            sum(coordinates[m][1] for m in members if m in coordinates)
            / max(1, sum(1 for m in members if m in coordinates)),
        ]
        for name, members in clusters.items()
        if any(m in coordinates for m in members)
    }
    _separate_labels(label_positions)
    for cluster_name, (x, y) in label_positions.items():
        axes.annotate(
            cluster_name, (x, y), color=INK, fontsize=10,
            fontweight="bold", ha="center", alpha=0.7, zorder=2,
            bbox={"facecolor": "white", "alpha": 0.5, "edgecolor": "none",
                  "pad": 0.5},
        )

    _draw_highlights(axes, coordinates, highlights)
    axes.set_xlabel("PC1", color=MUTED)
    axes.set_ylabel("PC2", color=MUTED)
    axes.set_title(title, color=INK, loc="left", pad=12)
    save_figure(plt, figure, path)


def pca_map(
    plt,
    coordinates: dict[str, tuple[float, float]],
    highlights: dict[str, str],
    title: str,
    path: Path,
) -> None:
    """The axes lens: quadrant guides and axis meaning, highlights labeled."""
    figure, axes = _base_scatter(plt, coordinates, highlights)
    axes.axvline(0, color=BASELINE, linewidth=1)
    axes.axhline(0, color=BASELINE, linewidth=1)
    _draw_highlights(axes, coordinates, highlights)
    axes.set_xlabel("PC1 — valence (negative ← → positive)", color=MUTED)
    axes.set_ylabel("PC2 — disposition (tranquil ← → oppositional)", color=MUTED)
    axes.set_title(title, color=INK, loc="left", pad=12)
    save_figure(plt, figure, path)
