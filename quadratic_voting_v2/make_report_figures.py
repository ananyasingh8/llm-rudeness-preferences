"""Report-candidate figures for QV v2 (both frames, full runs).

Run from the repo root:
  python3 quadratic_voting_v2/make_report_figures.py
Writes into quadratic_voting_v2/results/report_candidates/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNS = {
    "remove": REPO / "quadratic_voting_v2" / "results"
    / "2026-08-16_162510_881063_qv2-remove",
    "keep": REPO / "quadratic_voting_v2" / "results"
    / "2026-08-16_164946_110135_qv2-keep",
}
OUT = REPO / "quadratic_voting_v2" / "results" / "report_candidates"

BANDS = (1, 0, -1, -2, -3)
BAND_COLS = {1: "votes_band_p1", 0: "votes_band_0", -1: "votes_band_m1",
             -2: "votes_band_m2", -3: "votes_band_m3"}
BAND_LABELS = ["friendly\n(+1)", "neutral\n(0)", "mild abuse\n(-1)",
               "strong abuse\n(-2)", "severe abuse\n(-3)"]
C_REMOVE = "#A32D2D"
C_KEEP = "#185FA5"


def load(frame: str) -> pd.DataFrame:
    df = pd.read_csv(RUNS[frame] / "ballots.csv", keep_default_na=False)
    df = df[df["valid"].astype(str).str.lower() == "true"].copy()
    for col in BAND_COLS.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {frame: load(frame) for frame in RUNS}

    # ---- Fig 1: mean votes per band, both frames ----------------------------
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(BANDS))
    for i, (frame, color, label) in enumerate(
        [("remove", C_REMOVE, "votes to remove"),
         ("keep", C_KEEP, "votes to keep")]
    ):
        vals = [data[frame][BAND_COLS[b]].mean() for b in BANDS]
        bars = ax.bar(x + (i - 0.5) * 0.34, vals, width=0.32, color=color,
                      label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                    f"{v:.1f}", ha="center", fontsize=9, color="#2C2C2A")
    ax.set_xticks(x, BAND_LABELS, fontsize=9)
    ax.set_ylabel("mean votes per ballot")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "qv1_votes_by_band.png", dpi=200)
    plt.close(fig)

    # ---- Fig 2: share of all credits spent, per band, remove frame ----------
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for i, (frame, color) in enumerate([("remove", C_REMOVE), ("keep", C_KEEP)]):
        credits = np.array(
            [(data[frame][BAND_COLS[b]] ** 2).sum() for b in BANDS], dtype=float
        )
        share = credits / credits.sum() * 100
        bars = ax.bar(np.arange(len(BANDS)) + (i - 0.5) * 0.34, share,
                      width=0.32, color=color,
                      label=f"{frame} frame")
        for b, v in zip(bars, share):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    f"{v:.0f}%", ha="center", fontsize=9, color="#2C2C2A")
    ax.set_xticks(np.arange(len(BANDS)), BAND_LABELS, fontsize=9)
    ax.set_ylabel("share of all credits spent (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "qv2_credit_share.png", dpi=200)
    plt.close(fig)

    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
