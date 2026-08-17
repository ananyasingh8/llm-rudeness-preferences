"""Analysis for QV2 ballots (stdlib + pandas + matplotlib, no torch).

Per run folder this writes an analysis/ subfolder with:
  summary.csv        mean votes and mean credits per severity band, with
                     seeded bootstrap 95% CIs over valid ballots
  stats.csv          per-frame Spearman (votes vs severity within ballots),
                     invalid rate, abstention rate
  position_bias.csv  mean votes by presentation letter (A..E)
  voter_metrics.csv  per (frame, voter): mean unspent credits, failure rate,
                     and abstention rate, each with a cluster-bootstrap 95% CI
  mean_votes_by_band.png, votes_by_position.png,
  credit_use_by_voter.png

When runs for both frames are analyzed together, a combined analysis/ folder
(results/analysis_combined/) repeats the same tables and figures with both
frames side by side.

Usage (from the repo root):
  python -m quadratic_voting_v2.analyze                  # newest run per frame
  python -m quadratic_voting_v2.analyze --run PATH ...   # specific run(s)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless: figures go to files, never a display
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np
import pandas as pd

from quadratic_voting_v2.main import (
    BALLOTS_FILE,
    BAND_VOTE_COLUMNS,
    BANDS,
    BUDGET,
    LETTERS,
    QV2Error,
    RESULTS_DIR,
)
from quadratic_voting_v2.prompts import FRAMES

BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 2000
COMBINED_DIR = RESULTS_DIR / "analysis_combined"

# Shared figure style, normalized across every QV2 figure -----------------------
# One palette and one legend title so remove/keep read the same everywhere.
FRAME_COLORS: dict[str, str] = {"keep": "#185fa5", "remove": "#a32d2d"}
LEGEND_TITLE = "vote to"

# Human-readable name for each severity band, shown on the x-axis.
BAND_NAMES: dict[int, str] = {
    1: "friendly",
    0: "neutral",
    -1: "mild abuse",
    -2: "strong abuse",
    -3: "severe abuse",
}


def band_tick(band: int) -> str:
    """Two-line axis tick, e.g. 'mild abuse\\n(-1)'; band 0 shows '(0)'."""
    signed = "0" if band == 0 else f"{band:+d}"
    return f"{BAND_NAMES.get(band, '?')}\n({signed})"


def frame_color(frame: str) -> str:
    """Canonical bar color for a frame (falls back to grey if unknown)."""
    return FRAME_COLORS.get(frame, "#777777")


# --- Loading ------------------------------------------------------------------


def _ballot_count(ballots_file: Path) -> int:
    """Number of ballots in a run, tolerant of multiline raw_response cells."""
    try:
        return len(pd.read_csv(ballots_file, usecols=["voter"]))
    except (ValueError, OSError, pd.errors.EmptyDataError):
        return 0


def newest_runs(results_dir: Path = RESULTS_DIR) -> list[Path]:
    """The best run folder of each frame for the keep/remove comparison.

    Picks the *most complete* run per frame, not merely the newest: aborted
    or truncated runs (e.g. a 4-row debug run that only reached voter 0) would
    otherwise win on timestamp alone and produce a lopsided figure. Ranked by
    (ballot count, then 'started'), so the fullest — newest among equals — wins.
    """
    picks: dict[str, tuple[int, str, Path]] = {}
    if not results_dir.exists():
        return []
    for path in sorted(results_dir.iterdir()):
        info_file = path / "run_info.json"
        ballots_file = path / BALLOTS_FILE
        if not path.is_dir() or not info_file.exists() or not ballots_file.exists():
            continue
        info = json.loads(info_file.read_text(encoding="utf-8"))
        frame, started = info.get("frame"), info.get("started", "")
        if frame not in FRAMES:
            continue
        key = (_ballot_count(ballots_file), started)
        if frame not in picks or key > picks[frame][:2]:
            picks[frame] = (*key, path)
    return [path for *_, path in picks.values()]


def load_ballots(run_dir: Path) -> pd.DataFrame:
    """One run's ballots with the frame column normalized from run_info."""
    ballots_file = run_dir / BALLOTS_FILE
    if not ballots_file.exists():
        raise QV2Error(f"No {BALLOTS_FILE} in {run_dir}.")
    df = pd.read_csv(ballots_file, keep_default_na=False)
    df["valid"] = df["valid"].astype(str).str.lower() == "true"
    for band in BANDS:
        column = BAND_VOTE_COLUMNS[band]
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["credits_spent"] = pd.to_numeric(df["credits_spent"], errors="coerce")
    return df


# --- Statistics ---------------------------------------------------------------


def bootstrap_ci(
    clusters: list[np.ndarray], rng: np.random.Generator
) -> tuple[float, float]:
    """Seeded cluster-bootstrap 95% CI for the mean.

    Resamples whole rounds (clusters), not individual ballots: the ballots
    within one round share the same five conversations, so they are not
    independent observations of content.
    """
    clusters = [c for c in clusters if len(c)]
    if not clusters:
        return (float("nan"), float("nan"))
    means = np.empty(BOOTSTRAP_RESAMPLES)
    for i in range(BOOTSTRAP_RESAMPLES):
        picks = rng.integers(0, len(clusters), size=len(clusters))
        means[i] = np.concatenate([clusters[j] for j in picks]).mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def band_summary(ballots: pd.DataFrame) -> pd.DataFrame:
    """Mean votes and credits per band per frame, over valid ballots.

    CIs are cluster-bootstrapped over rounds (see bootstrap_ci).
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for frame, group in ballots[ballots["valid"]].groupby("frame"):
        for band in BANDS:
            votes = group[BAND_VOTE_COLUMNS[band]].to_numpy(dtype=float)
            credits = votes**2
            by_round = [
                subgroup[BAND_VOTE_COLUMNS[band]].to_numpy(dtype=float)
                for _, subgroup in group.groupby("round_index")
            ]
            votes_lo, votes_hi = bootstrap_ci(by_round, rng)
            credits_lo, credits_hi = bootstrap_ci(
                [c**2 for c in by_round], rng
            )
            rows.append(
                {
                    "frame": frame,
                    "band": band,
                    "n_valid_ballots": len(votes),
                    "mean_votes": votes.mean() if len(votes) else float("nan"),
                    "votes_ci_lo": votes_lo,
                    "votes_ci_hi": votes_hi,
                    "mean_credits": credits.mean() if len(credits) else float("nan"),
                    "credits_ci_lo": credits_lo,
                    "credits_ci_hi": credits_hi,
                }
            )
    return pd.DataFrame(rows)


def ballot_spearman(row: pd.Series) -> float:
    """Spearman of the 5 votes vs severity within one ballot (NaN if the
    votes are constant, e.g. an abstention)."""
    votes = pd.Series([row[BAND_VOTE_COLUMNS[band]] for band in BANDS], dtype=float)
    severities = pd.Series(BANDS, dtype=float)
    return votes.corr(severities, method="spearman")


def frame_stats(ballots: pd.DataFrame) -> pd.DataFrame:
    """Per-frame Spearman, invalid rate, abstention rate."""
    rows = []
    for frame, group in ballots.groupby("frame"):
        valid = group[group["valid"]]
        spearmans = valid.apply(ballot_spearman, axis=1).dropna()
        abstained = valid["credits_spent"] == 0
        rows.append(
            {
                "frame": frame,
                "n_ballots": len(group),
                "invalid_rate": 1 - len(valid) / len(group) if len(group) else float("nan"),
                "abstention_rate": abstained.mean() if len(valid) else float("nan"),
                "mean_spearman_votes_vs_severity": spearmans.mean(),
                "n_ballots_with_spearman": len(spearmans),
            }
        )
    return pd.DataFrame(rows)


def voter_metrics(ballots: pd.DataFrame) -> pd.DataFrame:
    """Per (frame, voter): mean unspent credits, failure rate, and abstention
    rate, each with a cluster-bootstrap 95% CI over rounds.

    Each metric is a mean of a per-ballot quantity, so it reuses bootstrap_ci
    with rounds as clusters (see band_summary for why rounds, not ballots):
      unspent      = BUDGET - credits_spent, over valid ballots
      failure_rate = 1 if the ballot was invalid, over all ballots
      abstention   = 1 if a valid ballot spent no credits, over valid ballots
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for frame, group in ballots.groupby("frame"):
        for voter, vgroup in group.groupby("voter"):
            valid = vgroup[vgroup["valid"]]
            metrics = {
                "unspent": [
                    (BUDGET - sub[sub["valid"]]["credits_spent"]).to_numpy(dtype=float)
                    for _, sub in vgroup.groupby("round_index")
                ],
                "failure_rate": [
                    (~sub["valid"]).to_numpy(dtype=float)
                    for _, sub in vgroup.groupby("round_index")
                ],
                "abstention_rate": [
                    (sub[sub["valid"]]["credits_spent"] == 0).to_numpy(dtype=float)
                    for _, sub in vgroup.groupby("round_index")
                ],
            }
            record = {
                "frame": frame,
                "voter": voter,
                "n_ballots": len(vgroup),
                "n_valid": len(valid),
            }
            for name, clusters in metrics.items():
                pooled = np.concatenate(clusters) if clusters else np.array([])
                lo, hi = bootstrap_ci(clusters, rng)
                record[name] = pooled.mean() if len(pooled) else float("nan")
                record[f"{name}_ci_lo"] = lo
                record[f"{name}_ci_hi"] = hi
            rows.append(record)
    return pd.DataFrame(rows)


def letter_votes(ballots: pd.DataFrame) -> pd.DataFrame:
    """Long table of (frame, round_index, letter, votes) from the recorded
    presentation order, valid ballots only — the position-bias check."""
    rows = []
    for _, row in ballots[ballots["valid"]].iterrows():
        order = json.loads(row["presentation_order"])
        votes = json.loads(row["votes_by_snippet"])
        for letter in LETTERS:
            rows.append(
                {
                    "frame": row["frame"],
                    "round_index": row["round_index"],
                    "letter": letter,
                    "votes": votes[order[letter]],
                }
            )
    return pd.DataFrame(rows)


def position_bias(ballots: pd.DataFrame) -> pd.DataFrame:
    """Mean votes per (frame, presentation letter), with a cluster-bootstrap
    95% CI over rounds (same rationale as band_summary)."""
    columns = ["frame", "letter", "mean_votes", "votes_ci_lo", "votes_ci_hi", "n"]
    long = letter_votes(ballots)
    if long.empty:
        return pd.DataFrame(columns=columns)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for (frame, letter), group in long.groupby(["frame", "letter"]):
        votes = group["votes"].to_numpy(dtype=float)
        by_round = [
            sub["votes"].to_numpy(dtype=float)
            for _, sub in group.groupby("round_index")
        ]
        lo, hi = bootstrap_ci(by_round, rng)
        rows.append(
            {
                "frame": frame,
                "letter": letter,
                "mean_votes": votes.mean() if len(votes) else float("nan"),
                "votes_ci_lo": lo,
                "votes_ci_hi": hi,
                "n": len(votes),
            }
        )
    return pd.DataFrame(rows, columns=columns)


# --- Figures ------------------------------------------------------------------


def label_bars(ax, xs, values, caps, fmt="{:.1f}") -> None:
    """Value labels centered just above each bar's upper CI cap, so they never
    sit on a bar, error bar, neighbour, or legend."""
    caps = np.asarray(caps, dtype=float)
    pad = 0.03 * max(np.nanmax(caps) if caps.size else 0.0, 1e-9)
    for x, value, cap in zip(xs, values, caps):
        if np.isnan(value):
            continue
        ax.text(x, cap + pad, fmt.format(value), ha="center", va="bottom", fontsize=8)


def plot_votes_by_band(summary: pd.DataFrame, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    frames = sorted(summary["frame"].unique())
    width = 0.8 / max(len(frames), 1)
    positions = np.arange(len(BANDS))
    for i, frame in enumerate(frames):
        rows = summary[summary["frame"] == frame].set_index("band").loc[list(BANDS)]
        offsets = positions + (i - (len(frames) - 1) / 2) * width
        errors = np.array(
            [
                rows["mean_votes"] - rows["votes_ci_lo"],
                rows["votes_ci_hi"] - rows["mean_votes"],
            ]
        )
        ax.bar(
            offsets,
            rows["mean_votes"],
            width=width,
            yerr=errors,
            capsize=3,
            label=frame,
            color=frame_color(frame),
        )
        label_bars(ax, offsets, rows["mean_votes"].to_numpy(),
                   rows["votes_ci_hi"].to_numpy())
    ax.set_xticks(positions)
    ax.set_xticklabels([band_tick(band) for band in BANDS])
    ax.set_xlabel("Level of Abusiveness")
    ax.set_ylabel("mean votes (95% bootstrap CI)")
    ax.set_title("Mean votes by level of abusiveness")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.10)  # headroom for labels
    ax.legend(title=LEGEND_TITLE)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_votes_by_position(bias: pd.DataFrame, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    frames = sorted(bias["frame"].unique())
    width = 0.8 / max(len(frames), 1)
    positions = np.arange(len(LETTERS))
    for i, frame in enumerate(frames):
        rows = bias[bias["frame"] == frame].set_index("letter").reindex(LETTERS)
        offsets = positions + (i - (len(frames) - 1) / 2) * width
        errors = np.array(
            [
                rows["mean_votes"] - rows["votes_ci_lo"],
                rows["votes_ci_hi"] - rows["mean_votes"],
            ]
        )
        ax.bar(
            offsets, rows["mean_votes"], width=width, yerr=errors, capsize=3,
            label=frame, color=frame_color(frame),
        )
        label_bars(ax, offsets, rows["mean_votes"].to_numpy(),
                   rows["votes_ci_hi"].to_numpy())
    ax.set_xticks(positions)
    ax.set_xticklabels(LETTERS)
    ax.set_xlabel("presentation letter")
    ax.set_ylabel("mean votes (95% bootstrap CI)")
    ax.set_title("Mean votes by presentation letter (position-bias check)")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.10)  # headroom for labels
    ax.legend(title=LEGEND_TITLE)
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def plot_credit_use_by_voter(metrics: pd.DataFrame, out_file: Path) -> None:
    """Three panels (mean unspent credits, failure rate, abstention rate),
    grouped bars per voter with keep vs remove and cluster-bootstrap CIs.
    Labels sit above each CI cap; one shared legend clears every panel.
    """
    frames = sorted(metrics["frame"].unique())
    voters = sorted(metrics["voter"].unique())
    x = np.arange(len(voters))
    width = 0.8 / max(len(frames), 1)
    panels = [
        ("unspent", "Mean unspent credits", f"Mean unspent credits (budget = {BUDGET})", False),
        ("failure_rate", "Invalid-ballot rate", "Mean failure rate (invalid ballots)", True),
        ("abstention_rate", "Abstention rate", "Mean abstention rate (valid, 0 credits)", True),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
    for ax, (col, ylabel, title, is_rate) in zip(axes, panels):
        for i, frame in enumerate(frames):
            rows = metrics[metrics["frame"] == frame].set_index("voter").reindex(voters)
            vals = rows[col].to_numpy(dtype=float)
            lower = np.clip(vals - rows[f"{col}_ci_lo"].to_numpy(dtype=float), 0, None)
            upper = np.clip(rows[f"{col}_ci_hi"].to_numpy(dtype=float) - vals, 0, None)
            offsets = x + (i - (len(frames) - 1) / 2) * width
            ax.bar(offsets, vals, width=width, yerr=np.array([lower, upper]),
                   capsize=3, label=frame, color=frame_color(frame))
            if is_rate:
                label_bars(ax, offsets, vals * 100, vals + upper, "{:.1f}%")
            else:
                label_bars(ax, offsets, vals, vals + upper)
        ax.set_xticks(x)
        ax.set_xticklabels([f"voter {v}" for v in voters])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.32)  # headroom for labels + upper-left legend
        if is_rate:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%"))
        ax.legend(title=LEGEND_TITLE, loc="upper left")
    fig.suptitle("Unspent credits, failures, and abstentions by voter "
                 "(keep vs remove)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


# --- Driver -------------------------------------------------------------------


def analyze_ballots(ballots: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = band_summary(ballots)
    stats = frame_stats(ballots)
    bias = position_bias(ballots)
    voters = voter_metrics(ballots)
    summary.to_csv(out_dir / "summary.csv", index=False)
    stats.to_csv(out_dir / "stats.csv", index=False)
    bias.to_csv(out_dir / "position_bias.csv", index=False)
    voters.to_csv(out_dir / "voter_metrics.csv", index=False)
    if not summary.empty:
        plot_votes_by_band(summary, out_dir / "mean_votes_by_band.png")
    if not bias.empty:
        plot_votes_by_position(bias, out_dir / "votes_by_position.png")
    if not voters.empty:
        plot_credit_use_by_voter(voters, out_dir / "credit_use_by_voter.png")
    print(f"Wrote {out_dir}")


def analyze(run_dirs: Sequence[Path]) -> None:
    all_ballots: list[pd.DataFrame] = []
    frames_seen: set[str] = set()
    for run_dir in run_dirs:
        ballots = load_ballots(run_dir)
        analyze_ballots(ballots, run_dir / "analysis")
        all_ballots.append(ballots)
        frames_seen.update(ballots["frame"].unique())
    if len(frames_seen) > 1:
        analyze_ballots(
            pd.concat(all_ballots, ignore_index=True), COMBINED_DIR
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze QV2 ballots.")
    parser.add_argument(
        "--run",
        type=Path,
        nargs="*",
        default=None,
        help="run folder(s) to analyze (default: newest run of each frame)",
    )
    args = parser.parse_args(argv)
    run_dirs = args.run if args.run else newest_runs()
    if not run_dirs:
        print("error: no run folders found; collect ballots first.")
        return 1
    try:
        analyze(run_dirs)
    except QV2Error as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
