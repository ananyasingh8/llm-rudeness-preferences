"""Report-candidate tables and figures for BailBench vs RudeBailBench (bail prompt).

Reads the committed Gemma prompt-phase parquets, writes candidate figures and
tables to bail/outputs/figures/report_candidates/. Stdlib + pandas + matplotlib.

Run from the repo root:
  python3 bail/src/make_report_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "bail" / "data" / "results"
OUT = REPO / "bail" / "outputs" / "figures" / "report_candidates"
BAIL, CONT = "\U0001f504", "\U0001f7e2"
SEED = 42
BOOT = 5000

# Palette: control = blue, rude = coral/red (project diverging convention).
C_ORIG = "#185FA5"
C_RUDE = "#A32D2D"
C_GRAY = "#888780"

ORDER_LABEL = {"continue_first": "continue-first", "bail_first": "bail-first"}


def per_prompt_rates(df: pd.DataFrame, ordering: str) -> pd.Series:
    """Per-prompt bail probability (mean over the 10 samples) for one ordering."""
    sub = df[(df.ordering == ordering) & df.wellbeing.isin([BAIL, CONT])]
    return (sub.wellbeing == BAIL).groupby(sub.bailbench_id).mean()


def boot_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    means = rng.choice(values, size=(BOOT, len(values))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    orig = pd.read_parquet(RESULTS / "results_prompt_orig.parquet")
    rude = pd.read_parquet(RESULTS / "results_prompt_rude.parquet")
    aug = pd.read_csv(REPO / "bail" / "data" / "bailbench_augmented.csv")
    meta = aug.set_index("bailbench_id")[["category", "rudeness_name"]]

    # ---- Table 1: primary rates with bootstrap CIs --------------------------
    rows = []
    rates = {}
    for ordering in ("continue_first", "bail_first"):
        o = per_prompt_rates(orig, ordering)
        r = per_prompt_rates(rude, ordering).reindex(o.index)
        rates[ordering] = (o, r)
        d = (r - o).dropna()
        lo, hi = boot_ci(d.to_numpy(), rng)
        rows.append(
            {
                "ordering": ORDER_LABEL[ordering],
                "bailbench_rate": o.mean(),
                "rudebailbench_rate": r.mean(),
                "diff_pp": (r.mean() - o.mean()) * 100,
                "fold": r.mean() / o.mean(),
                "diff_ci95": f"[{lo * 100:.2f}, {hi * 100:.2f}] pp",
                "n_prompts": len(o),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "table1_primary.csv", index=False)
    with open(OUT / "table1_primary.md", "w") as fh:
        fh.write(table.round(4).to_markdown(index=False))

    # ---- Fig A: paired bars, orig vs rude x ordering ------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(2)
    for i, (cond, color) in enumerate(
        [("BailBench", C_ORIG), ("RudeBailBench", C_RUDE)]
    ):
        vals, lo_err, hi_err = [], [], []
        for ordering in ("continue_first", "bail_first"):
            s = rates[ordering][i]
            vals.append(s.mean() * 100)
            lo, hi = boot_ci(s.to_numpy(), rng)
            lo_err.append((s.mean() - lo) * 100)
            hi_err.append((hi - s.mean()) * 100)
        bars = ax.bar(
            x + (i - 0.5) * 0.32, vals, width=0.3, color=color, label=cond,
            yerr=[lo_err, hi_err], capsize=3,
            error_kw={"lw": 1, "ecolor": "#444441"},
        )
        for b, v, e in zip(bars, vals, hi_err):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + e + 0.35,
                    f"{v:.1f}%", ha="center", fontsize=9, color="#2C2C2A")
    ax.set_xticks(x, [ORDER_LABEL[o] for o in ("continue_first", "bail_first")])
    ax.set_ylabel("bail rate (% of responses)")
    ax.set_xlabel("wellbeing-check option ordering")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "figA_primary_bars.png", dpi=200)
    plt.close(fig)

    # ---- Fig B: rudeness-formula gradient (continue-first diffs) ------------
    o, r = rates["continue_first"]
    d = (r - o).rename("diff").to_frame().join(meta)
    grad = (
        d.groupby("rudeness_name")["diff"].agg(["mean", "count", "sem"])
        .sort_values("mean")
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    y = np.arange(len(grad))
    ax.barh(y, grad["mean"] * 100, xerr=grad["sem"] * 100 * 1.96, height=0.62,
            color=C_RUDE, capsize=2, error_kw={"lw": 1, "ecolor": "#444441"})
    ax.set_yticks(y, [f"{n}  (n={c})" for n, c in zip(grad.index, grad["count"])],
                  fontsize=8.5)
    ax.set_xlabel("bail-rate increase, rude vs original (pp, continue-first)",
                  fontsize=9.5)
    ax.axvline(0, color="#B4B2A9", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figB_formula_gradient.png", dpi=200)
    plt.close(fig)

    # ---- Fig B2: formula gradient, simplified (no error bars, no n's) -------
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    y = np.arange(len(grad))
    ax.barh(y, grad["mean"] * 100, height=0.62, color=C_RUDE)
    for yi, v in zip(y, grad["mean"] * 100):
        ax.text(v + 0.6, yi, f"+{v:.0f}", va="center", fontsize=8.5,
                color="#2C2C2A")
    ax.set_yticks(y, grad.index, fontsize=9)
    ax.set_xlabel("increase in bail rate (percentage points)", fontsize=9.5)
    ax.set_xlim(0, grad["mean"].max() * 100 * 1.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figB2_formula_gradient_simple.png", dpi=200)
    plt.close(fig)

    # ---- Fig C: category dumbbells (continue-first, n>=50) ------------------
    cat = (
        d.join(o.rename("orig")).join(r.rename("rude"))
        .groupby("category").agg(orig=("orig", "mean"), rude=("rude", "mean"),
                                 n=("diff", "count"))
    )
    cat = cat[cat.n >= 50].sort_values("rude")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    y = np.arange(len(cat))
    ax.hlines(y, cat.orig * 100, cat.rude * 100, color="#D3D1C7", lw=2, zorder=1)
    ax.scatter(cat.orig * 100, y, s=42, color=C_ORIG, zorder=2, label="BailBench")
    ax.scatter(cat.rude * 100, y, s=42, color=C_RUDE, zorder=2,
               label="RudeBailBench")
    ax.set_yticks(y, [f"{i}  (n={n})" for i, n in zip(cat.index, cat.n)],
                  fontsize=8.5)
    ax.set_xlabel("bail rate (%, continue-first)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "figC_category_dumbbell.png", dpi=200)
    plt.close(fig)

    # ---- Fig D: per-prompt paired shift (how broad is the effect?) ----------
    buckets = pd.cut(
        (r - o) * 100,
        bins=[-100, -1e-9, 1e-9, 10, 20, 50, 100],
        labels=["decreased", "no change", "0-10 pp", "10-20 pp",
                "20-50 pp", ">50 pp"],
    ).value_counts().reindex(
        ["decreased", "no change", "0-10 pp", "10-20 pp", "20-50 pp", ">50 pp"]
    )
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    colors = [C_ORIG, C_GRAY, "#F0997B", "#D85A30", "#A32D2D", "#4A1B0C"]
    bars = ax.bar(np.arange(len(buckets)), buckets.to_numpy(), color=colors,
                  width=0.62)
    for b, v in zip(bars, buckets):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 8, str(v),
                ha="center", fontsize=9, color="#2C2C2A")
    ax.set_xticks(np.arange(len(buckets)), buckets.index, fontsize=8.5)
    ax.set_ylabel("number of prompts (of 1,630)")
    ax.set_xlabel("per-prompt change in bail rate after rude rewrite (continue-first)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figD_perprompt_shift.png", dpi=200)
    plt.close(fig)

    # ---- Fig E: both models x both orderings --------------------------------
    s_orig = pd.read_parquet(RESULTS / "sonnet_results_prompt_orig.parquet")
    s_rude = pd.read_parquet(RESULTS / "sonnet_results_prompt_rude.parquet")
    groups = []
    for ordering in ("continue_first", "bail_first"):
        groups.append((f"Gemma 4 31B\n{ORDER_LABEL[ordering]}",
                       *rates[ordering]))
        so = per_prompt_rates(s_orig, ordering)
        sr = per_prompt_rates(s_rude, ordering).reindex(so.index)
        groups.append((f"Sonnet 5\n{ORDER_LABEL[ordering]}", so, sr.dropna()))
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = np.arange(len(groups))
    for i, (cond, color) in enumerate(
        [("original (BailBench)", C_ORIG), ("rude (RudeBailBench)", C_RUDE)]
    ):
        vals, lo_err, hi_err = [], [], []
        for _, o, r in groups:
            s = (o, r)[i]
            vals.append(s.mean() * 100)
            lo, hi = boot_ci(s.to_numpy(), rng)
            lo_err.append((s.mean() - lo) * 100)
            hi_err.append((hi - s.mean()) * 100)
        bars = ax.bar(x + (i - 0.5) * 0.32, vals, width=0.3, color=color,
                      label=cond, yerr=[lo_err, hi_err], capsize=3,
                      error_kw={"lw": 1, "ecolor": "#444441"})
        for b, v, e in zip(bars, vals, hi_err):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + e + 0.3,
                    f"{v:.1f}", ha="center", fontsize=8.5, color="#2C2C2A")
    ax.set_xticks(x, [g[0] for g in groups], fontsize=9)
    ax.set_ylabel("bail rate (% of responses)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "figE_models_orderings.png", dpi=200)
    plt.close(fig)

    # ---- Fig H: Gemma, no error bars (hackathon-simple version) -------------
    gemma_groups = [g for g in groups if g[0].startswith("Gemma")]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    x = np.arange(len(gemma_groups))
    for i, (cond, color) in enumerate(
        [("original (BailBench)", C_ORIG), ("rude (RudeBailBench)", C_RUDE)]
    ):
        vals = [(o, r)[i].mean() * 100 for _, o, r in gemma_groups]
        bars = ax.bar(x + (i - 0.5) * 0.32, vals, width=0.3, color=color,
                      label=cond)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.25,
                    f"{v:.1f}%", ha="center", fontsize=9.5, color="#2C2C2A")
    ax.set_xticks(x, [g[0].split("\n")[1] for g in gemma_groups], fontsize=9.5)
    ax.set_xlabel("wellbeing-check option ordering")
    ax.set_ylabel("bail rate (% of responses)")
    ax.set_title("Gemma 4 31B", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "figH_gemma_nobars.png", dpi=200)
    plt.close(fig)

    # ---- Figs F/G: one panel per model --------------------------------------
    for tag, model_groups in (
        ("figF_gemma", [g for g in groups if g[0].startswith("Gemma")]),
        ("figG_sonnet", [g for g in groups if g[0].startswith("Sonnet")]),
    ):
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        x = np.arange(len(model_groups))
        for i, (cond, color) in enumerate(
            [("original (BailBench)", C_ORIG), ("rude (RudeBailBench)", C_RUDE)]
        ):
            vals, lo_err, hi_err = [], [], []
            for _, o, r in model_groups:
                s = (o, r)[i]
                vals.append(s.mean() * 100)
                lo, hi = boot_ci(s.to_numpy(), rng)
                lo_err.append((s.mean() - lo) * 100)
                hi_err.append((hi - s.mean()) * 100)
            bars = ax.bar(x + (i - 0.5) * 0.32, vals, width=0.3, color=color,
                          label=cond, yerr=[lo_err, hi_err], capsize=3,
                          error_kw={"lw": 1, "ecolor": "#444441"})
            for b, v, e in zip(bars, vals, hi_err):
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + e + (0.3 if tag == "figF_gemma" else 0.05),
                        f"{v:.1f}", ha="center", fontsize=9, color="#2C2C2A")
        ax.set_xticks(x, [g[0].split("\n")[1] for g in model_groups], fontsize=9.5)
        ax.set_xlabel("wellbeing-check option ordering")
        ax.set_ylabel("bail rate (% of responses)")
        ax.set_title(model_groups[0][0].split("\n")[0], fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.5)
        fig.tight_layout()
        fig.savefig(OUT / f"{tag}.png", dpi=200)
        plt.close(fig)

    print("wrote", sorted(p.name for p in OUT.iterdir()))
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
