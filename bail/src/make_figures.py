"""Report figures for the rudeness-bail experiment.

Outputs (outputs/figures/):
  fig1_primary.png   -- bail rate orig vs rude, per method cell, Gemma + Sonnet
  fig2_formulas.png  -- Gemma bail-rate lift by Culpeper rudeness formula
  fig3_categories.png-- Gemma bail-rate lift by BailBench category
Plus outputs/combined_results.csv with every cell (both models).

Run: python -m src.make_figures
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config
from src.analyze import load_cell, per_prompt  # Gemma loaders

BAIL = "\U0001f504"
GREEN = "\U0001f7e2"
OUT = os.path.join(config.ROOT, "outputs", "figures")
os.makedirs(OUT, exist_ok=True)

C_ORIG = "#2a78d6"   # blue  (categorical slot 1)
C_RUDE = "#eb6834"   # orange (slot 2)
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#d8d7d2", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "font.size": 10, "axes.titlesize": 11, "axes.spines.top": False,
    "axes.spines.right": False,
})


def sonnet_cell(method: str, condition: str) -> pd.Series:
    if method == "tool":
        df = pd.read_parquet(os.path.join(config.RESULTS_DIR,
                                          f"sonnet_results_tool_{condition}.parquet"))
        df = df[df["error"].isna()].copy()
        df["bail"] = df["tool_called"].astype(float)
    else:
        ordering = method.removeprefix("prompt_")
        df = pd.read_parquet(os.path.join(config.RESULTS_DIR,
                                          f"sonnet_results_prompt_{condition}.parquet"))
        df = df[(df["error"].isna()) & (df["ordering"] == ordering)].copy()
        df["bail"] = df["wellbeing"].map({BAIL: 1.0, GREEN: 0.0})
    return df.dropna(subset=["bail"]).groupby("bailbench_id")["bail"].mean()


METHODS = ["tool", "prompt_bail_first", "prompt_continue_first"]
LABELS = {"tool": "Bail tool", "prompt_bail_first": "Bail prompt\n(bail listed first)",
          "prompt_continue_first": "Bail prompt\n(continue listed first)"}


def combined_table() -> pd.DataFrame:
    rows = []
    for model, cellfn in [("Gemma 4 31B", lambda m, c: per_prompt(load_cell(m, c))),
                          ("Claude Sonnet 5", sonnet_cell)]:
        for m in METHODS:
            a, b = cellfn(m, "orig"), cellfn(m, "rude")
            ids = a.index.intersection(b.index)
            diff = (b[ids] - a[ids]).values
            nz = (diff != 0).sum()
            p = stats.wilcoxon(diff).pvalue if nz >= 5 else np.nan
            rng = np.random.default_rng(42)
            boots = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"model": model, "method": m, "n_prompts": len(ids),
                         "orig_rate": a[ids].mean(), "rude_rate": b[ids].mean(),
                         "diff": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                         "nonzero_pairs": int(nz), "wilcoxon_p": p})
    return pd.DataFrame(rows)


def fig1(tbl: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    for ax, model in zip(axes, ["Gemma 4 31B", "Claude Sonnet 5"]):
        sub = tbl[tbl["model"] == model].set_index("method").loc[METHODS]
        x = np.arange(len(METHODS))
        w = 0.36
        ax.bar(x - w / 2, sub["orig_rate"] * 100, w, color=C_ORIG, label="Original")
        ax.bar(x + w / 2, sub["rude_rate"] * 100, w, color=C_RUDE, label="Rude")
        for i, (_, r) in enumerate(sub.iterrows()):
            for dx, v in [(-w / 2, r["orig_rate"]), (w / 2, r["rude_rate"])]:
                ax.text(i + dx, v * 100 + 0.25, f"{v * 100:.1f}", ha="center",
                        fontsize=8.5, color=INK2)
        ax.set_xticks(x, [LABELS[m] for m in METHODS], fontsize=9)
        ax.set_title(model)
        ax.set_ylim(0, max(15, sub["rude_rate"].max() * 115))
    axes[0].set_ylabel("Bail rate (%)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Bail rate by elicitation method and condition", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_primary.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig2() -> None:
    aug = pd.read_csv(os.path.join(config.DATA_DIR, "bailbench_augmented.csv"))
    a = per_prompt(load_cell("prompt_continue_first", "orig")).rename("orig")
    b = per_prompt(load_cell("prompt_continue_first", "rude")).rename("rude")
    d = (pd.concat([a, b], axis=1).dropna().reset_index()
         .merge(aug[["bailbench_id", "rudeness_name"]], on="bailbench_id"))
    g = d.groupby("rudeness_name").apply(
        lambda x: (x["rude"] - x["orig"]).mean()).sort_values() * 100
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(g.index.str.capitalize(), g.values, color=C_RUDE, height=0.62)
    for i, v in enumerate(g.values):
        ax.text(v + 0.4, i, f"+{v:.1f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("Bail-rate increase, rude vs original (percentage points)")
    ax.set_title("Rudeness formulas ranked by bail-rate lift\n"
                 "(Gemma, bail prompt, continue-first ordering)", loc="left")
    ax.set_xlim(0, g.max() * 1.15)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_formulas.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig3() -> None:
    bench = pd.read_csv(os.path.join(config.DATA_DIR, "bailBench.csv"))
    bench["bailbench_id"] = range(len(bench))
    a = per_prompt(load_cell("prompt_continue_first", "orig")).rename("orig")
    b = per_prompt(load_cell("prompt_continue_first", "rude")).rename("rude")
    d = (pd.concat([a, b], axis=1).dropna().reset_index()
         .merge(bench[["bailbench_id", "category"]], on="bailbench_id"))
    g = d.groupby("category")[["orig", "rude"]].mean().sort_values("rude") * 100
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hlines(y, g["orig"], g["rude"], color="#d8d7d2", lw=2, zorder=1)
    ax.scatter(g["orig"], y, s=42, color=C_ORIG, zorder=2, label="Original")
    ax.scatter(g["rude"], y, s=42, color=C_RUDE, zorder=2, label="Rude")
    ax.set_yticks(y, g.index, fontsize=8.5)
    ax.set_xlabel("Bail rate (%)")
    ax.set_title("Bail rate by BailBench category\n"
                 "(Gemma, bail prompt, continue-first ordering)", loc="left")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_categories.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tbl = combined_table()
    tbl.to_csv(os.path.join(config.ROOT, "outputs", "combined_results.csv"), index=False)
    print(tbl.to_string(index=False))
    fig1(tbl)
    fig2()
    fig3()
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
