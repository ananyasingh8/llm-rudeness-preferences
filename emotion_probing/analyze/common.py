"""Shared helpers for run analysis: discovery, loading, stats, chart style."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PACKAGE_ROOT / "results"
GEMOTIONS_ANALYSIS_FILE = (
    PACKAGE_ROOT / "gemotions" / "results" / "gemma4-31b" / "analysis"
    / "analysis_results.json"
)

# Chart colors (project dataviz palette, light mode, CVD-validated).
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
DELTA_POSITIVE = "#e34948"  # rude/abusive scores higher
DELTA_NEGATIVE = "#2a78d6"  # rude/abusive scores lower
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
SERIES_3 = "#1b9e77"
NEUTRAL_MID = "#f0efec"  # diverging midpoint
# Sequential blue ramp (ordinal steps 250..650) for the 5 severity bands.
SEVERITY_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

SEVERITY_BAND_ORDER = (1, 0, -1, -2, -3)
SEVERITY_BAND_LABELS = {
    1: "1 (not abusive)",
    0: "0 (ambiguous)",
    -1: "-1 (mild)",
    -2: "-2 (strong)",
    -3: "-3 (very strong)",
}
MIN_GROUP_SIZE = 5

# Readable names for the gemotions clusters, identified by a distinctive
# member (numeric cluster ids in the analysis file are arbitrary).
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


def find_run_dir(explicit: str | None) -> Path:
    """Resolve the run folder to analyze (--run PATH or the latest run)."""
    if explicit is not None:
        run_dir = Path(explicit)
        if not (run_dir / "run_info.json").exists():
            raise SystemExit(f"error: {run_dir} is not a run folder.")
        return run_dir
    if RESULTS_DIR.exists():
        # Sort by the started stamp in run_info.json, not the folder name, so
        # legacy folder names (results/run1) order correctly among timestamps.
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


def load_clusters(run_dir: Path, emotions: list[str]) -> dict[str, list[str]]:
    """Load gemotions clusters as {readable name: member emotions}.

    Names are resolved from the FULL cluster membership (so "Positive/Joy" is
    recognized even when its identifying member "happy" is not among the run's
    scored emotions), then members are filtered to the scored emotions and
    clusters left empty by the filter are dropped — runs against a reduced
    vector set (e.g. the 20 base-model emotions) only analyze the clusters
    they can actually measure.
    """
    clusters_file = run_dir / "clusters.json"
    if not clusters_file.exists():
        raise SystemExit(
            f"error: {clusters_file} not found; it is written by the runner."
        )
    saved = json.loads(clusters_file.read_text(encoding="utf-8"))
    named: dict[str, list[str]] = {}
    for cluster_id, members in saved["clusters"].items():
        name = next(
            (
                CLUSTER_NAME_BY_MEMBER[m]
                for m in members
                if m in CLUSTER_NAME_BY_MEMBER
            ),
            f"cluster {cluster_id}",
        )
        present = [m for m in members if m in emotions]
        if present:
            named[name] = present
    return named


def load_pca_coordinates(probe_layer: int) -> dict[str, tuple[float, float]]:
    """Load {emotion: (pc1, pc2)} from the vendored gemotions analysis.

    The analysis file only covers the IT model's swept layers (5, 10, ..,
    55). Runs probed at other layers (e.g. the base model at 59) fall back to
    the layer-40 projection — the maps use these coordinates purely as a
    stable layout for the emotion names, not as measurements of the run.
    """
    if not GEMOTIONS_ANALYSIS_FILE.exists():
        raise SystemExit(
            f"error: {GEMOTIONS_ANALYSIS_FILE} not found (vendored file)."
        )
    analysis = json.loads(GEMOTIONS_ANALYSIS_FILE.read_text(encoding="utf-8"))
    layer_key = str(probe_layer) if str(probe_layer) in analysis else "40"
    layer = analysis[layer_key]["pca"]
    emotions = layer["emotions"]
    pc1, pc2 = layer["projections"]["pc1"], layer["projections"]["pc2"]
    return {name: (pc1[i], pc2[i]) for i, name in enumerate(emotions)}


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


def sem_diff(a: dict[str, float], b: dict[str, float]) -> float:
    """Standard error of the difference between two independent group means."""
    return math.sqrt(a["sem"] ** 2 + b["sem"] ** 2)


def get_matplotlib():
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


def styled_axes(plt, width: float, height: float):
    """A figure/axes pair with the project chart chrome applied."""
    figure, axes = plt.subplots(figsize=(width, height), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.tick_params(colors=MUTED, labelcolor=INK, length=0)
    axes.set_axisbelow(True)
    return figure, axes


def save_figure(plt, figure, path: Path) -> None:
    """Write a figure and report it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    print(f"Chart written to {path}")


def diverging_barh(
    plt,
    labels: list[str],
    means: list[float],
    sems: list[float],
    xlabel: str,
    title: str,
    path: Path,
    height: float,
) -> None:
    """Horizontal diverging bars around zero; position and color carry sign."""
    figure, axes = styled_axes(plt, 7.5, height)
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
    save_figure(plt, figure, path)


def group_bars(
    plt,
    labels: list[str],
    stats: list[dict[str, float]],
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    """Vertical bars of one measure across labeled groups (single series)."""
    figure, axes = styled_axes(plt, 6.5, 4.2)
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
    save_figure(plt, figure, path)
