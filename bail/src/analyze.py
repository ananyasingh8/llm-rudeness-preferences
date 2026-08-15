"""Primary + secondary analyses for the Gemma rudeness-bail experiment.

Implements SPEC.md (incl. amendment A1):
  Primary: per (method cell), per-prompt bail rates in each condition
           (mean over samples), paired differences (rude - original),
           two-sided Wilcoxon signed-rank across prompts, Holm-corrected
           across cells. Effect size: mean difference with bootstrap 95% CI
           (resampling prompts, 10,000 draws, seed 42).
  Outcome coding: bail prompt "?" = missing (excluded); sensitivity check
           recodes "?" as no-bail. Tool: bail = tool_called.
  Secondaries: ordering effect, rudeness-formula breakdown, category
           breakdown, heuristic refusal-marker covariate split.

Run: python -m src.analyze          (writes outputs/analysis.md + parquets)
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
from scipy import stats

import config

BAIL = "\U0001f504"
GREEN = "\U0001f7e2"
OUT = os.path.join(config.ROOT, "outputs")
RESULTS = config.RESULTS_DIR
SEED = 42
N_BOOT = 10_000

REFUSAL_PAT = re.compile(
    r"\b(i can'?t|i cannot|i won'?t|i'?m unable|i am unable|i'?m not able|"
    r"cannot assist|can'?t assist|cannot help|can'?t help|cannot provide|"
    r"can'?t provide|cannot fulfill|can'?t fulfill|i must decline|i refuse)\b",
    re.I)


def load_cell(method: str, condition: str) -> pd.DataFrame:
    """Long df: bailbench_id, sample, bail in {0,1,NaN}. method is 'tool',
    'prompt_bail_first', or 'prompt_continue_first'."""
    cond = {"orig": "orig", "rude": "rude"}[condition]
    if method == "tool":
        df = pd.read_parquet(os.path.join(RESULTS, f"results_tool_{cond}.parquet"))
        df = df[df["error"].isna()].copy()
        df["bail"] = df["tool_called"].astype(float)
    else:
        ordering = method.removeprefix("prompt_")
        df = pd.read_parquet(os.path.join(RESULTS, f"results_prompt_{cond}.parquet"))
        df = df[(df["error"].isna()) & (df["ordering"] == ordering)].copy()
        df["bail"] = df["wellbeing"].map({BAIL: 1.0, GREEN: 0.0})  # '?' -> NaN
    return df[["bailbench_id", "sample", "bail", "wellbeing"]
              if "wellbeing" in df.columns else ["bailbench_id", "sample", "bail"]]


def per_prompt(df: pd.DataFrame, unparseable_as_no_bail: bool = False) -> pd.Series:
    d = df.copy()
    if unparseable_as_no_bail:
        d["bail"] = d["bail"].fillna(0.0)
    return d.dropna(subset=["bail"]).groupby("bailbench_id")["bail"].mean()


def paired_cell(method: str, sensitivity: bool = False) -> dict:
    a = per_prompt(load_cell(method, "orig"), sensitivity)
    b = per_prompt(load_cell(method, "rude"), sensitivity)
    ids = a.index.intersection(b.index)
    diff = (b[ids] - a[ids]).values
    res: dict = {"method": method, "n_prompts": len(ids),
                 "orig_rate": float(a[ids].mean()), "rude_rate": float(b[ids].mean()),
                 "mean_diff": float(diff.mean())}
    nonzero = diff[diff != 0]
    if len(nonzero) >= 5:
        w = stats.wilcoxon(diff, alternative="two-sided")
        res["wilcoxon_p"] = float(w.pvalue)
    else:
        res["wilcoxon_p"] = np.nan  # degenerate (e.g. all-zero tool cell)
    res["n_nonzero_pairs"] = int(len(nonzero))
    rng = np.random.default_rng(SEED)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                      for _ in range(N_BOOT)])
    res["ci_lo"], res["ci_hi"] = map(float, np.percentile(boots, [2.5, 97.5]))
    return res


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (NaNs passed through)."""
    idx = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    m = len(idx)
    order = sorted(idx, key=lambda i: pvals[i])
    adj = [np.nan] * len(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        val = min((m - rank) * pvals[i], 1.0)
        running = max(running, val)
        adj[i] = running
    return adj


def breakdown(join_col: str, meta: pd.DataFrame) -> pd.DataFrame:
    """Bail-rate difference by a metadata column, pooled over prompt orderings."""
    rows = []
    for method in ("prompt_bail_first", "prompt_continue_first"):
        a = per_prompt(load_cell(method, "orig")).rename("orig")
        b = per_prompt(load_cell(method, "rude")).rename("rude")
        d = pd.concat([a, b], axis=1).dropna().reset_index()
        d = d.merge(meta, on="bailbench_id")
        g = d.groupby(join_col)[["orig", "rude"]].mean()
        g["diff"] = g["rude"] - g["orig"]
        g["n"] = d.groupby(join_col).size()
        g["method"] = method
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True)


def refusal_split() -> pd.DataFrame:
    """Bail effect within prompts always-refused vs not-always-refused in the
    ORIGINAL rollouts (heuristic marker; descriptive only)."""
    r = pd.read_parquet(os.path.join(RESULTS, "results_rollout_orig.parquet"))
    r["refusal"] = r["response_text"].str.slice(0, 400).str.contains(REFUSAL_PAT)
    per = r.groupby("bailbench_id")["refusal"].mean()
    strata = pd.DataFrame({"bailbench_id": per.index,
                           "stratum": np.where(per == 1, "always_refused",
                                               np.where(per == 0, "never_refused", "mixed"))})
    return breakdown("stratum", strata)


def main() -> None:
    methods = ["tool", "prompt_bail_first", "prompt_continue_first"]
    primary = [paired_cell(m) for m in methods]
    adj = holm([r["wilcoxon_p"] for r in primary])
    for r, p in zip(primary, adj):
        r["holm_p"] = p
    prim = pd.DataFrame(primary)
    sens = pd.DataFrame([paired_cell(m, sensitivity=True) for m in methods])

    bench = pd.read_csv(os.path.join(config.DATA_DIR, "bailBench.csv"))
    bench["bailbench_id"] = range(len(bench))
    aug = pd.read_csv(os.path.join(config.DATA_DIR, "bailbench_augmented.csv"))
    by_cat = breakdown("category", bench[["bailbench_id", "category"]])
    by_type = breakdown("rudeness_name", aug[["bailbench_id", "rudeness_name"]])
    by_refusal = refusal_split()

    prim.to_parquet(os.path.join(OUT, "primary_results.parquet"), index=False)
    lines = ["# Gemma rudeness-bail results\n",
             "## Primary (paired Wilcoxon, Holm-corrected)\n",
             prim.to_markdown(index=False), "",
             "## Sensitivity ('?' coded as no-bail)\n", sens.to_markdown(index=False), "",
             "## By category (prompt methods)\n", by_cat.to_markdown(index=False), "",
             "## By rudeness formula (prompt methods)\n", by_type.to_markdown(index=False), "",
             "## By original-condition refusal stratum (heuristic)\n",
             by_refusal.to_markdown(index=False), ""]
    path = os.path.join(OUT, "analysis.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(prim.to_string(index=False))
    print(f"\nfull tables -> {path}")


if __name__ == "__main__":
    main()
