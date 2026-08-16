"""Pool per-repeat metrics across the repeat axis with t-based SEM error bars.

This module implements the across-repeat aggregation described in
``quadratic_voting/METHODS.md``. ``analysis.py`` computes statistics *within one
matched-set* (one repeat); this module reads the aggregated multi-repeat Parquet
export (which carries ``seed_repeat_index`` per row) and pools across the repeat
axis so ``plots.py`` can render pooled estimates with error bars.

Each repeat contributes ONE estimate per ``(severity_level, regime)`` cell (a
two-stage collapse that first reduces the non-independent voters within a repeat
to a single number); the independent replicate unit is therefore the repeat. For
a cell with ``N`` per-repeat estimates we report the mean and a two-sided 95%
Student's-t interval with ``df = N - 1`` (``scipy.stats``):

    SEM = s / sqrt(N),  ci = mean +/- t_{df, 0.975} * SEM

The pooling axis is a generic ``repeat_index`` (currently the seed-repeat index),
so the code aggregates over however many repeats are present without assuming a
fixed ``N``. No model or GPU is required.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]

from quadratic_voting.experiment.sampling import candidate_modal_severity

POOLED_ANALYSIS_VERSION = "qv-pooled-across-repeats/v1"
POOLED_ESTIMATOR = "t-sem"
POOLED_CI_LEVEL = 0.95

# Severity levels ordered least-abusive (1) to most-abusive (-3); this is the
# grouping/x-axis, not the sample. Its cardinality does not affect df.
SEVERITY_ORDER: tuple[int, ...] = (1, 0, -1, -2, -3)
REGIME_ORDER: tuple[str, ...] = ("support", "opposition")


class PooledMetric(StrEnum):
    """Metrics pooled across repeats and reported per (severity_level, regime)."""

    SURVIVAL_ROUNDS = "survival_rounds"
    NET_SIGNED_VOTES = "net_signed_votes"


def _rows(aggregate_dir: Path, name: str) -> list[dict[str, object]]:
    return pq.read_table(aggregate_dir / f"{name}.parquet").to_pylist()


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"pooled analysis expected an int value, got {value!r}")
    return int(value)


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"pooled analysis expected a numeric value, got {value!r}")
    return float(value)


def severity_level_by_candidate(
    source_annotation_rows: Iterable[Mapping[str, object]],
) -> dict[str, int]:
    """Map each candidate_id to its modal severity level from source annotations.

    Uses the same modal-severity rule as sampling (ties break toward the more
    severe/more negative level). Candidates whose annotations do not yield a
    single valid severity are omitted.
    """
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in source_annotation_rows:
        grouped[str(row["candidate_id"])].append(row)
    levels: dict[str, int] = {}
    for candidate_id, rows in grouped.items():
        try:
            level = candidate_modal_severity(rows)
        except ValueError:
            # Malformed/partial annotations (not a clean multiple of the 5-point
            # scale, e.g. in minimal fixtures) simply yield no severity mapping.
            continue
        if level is not None:
            levels[candidate_id] = level
    return levels


def t_sem_cell(values: Sequence[float]) -> dict[str, object]:
    """Mean and a two-sided 95% Student's-t interval over per-repeat estimates.

    ``df = N - 1``. With ``N < 2`` the SEM/interval are undefined (returned as
    ``None``); the mean is still reported.
    """
    array = np.asarray(values, dtype=float)
    n = int(array.size)
    mean = float(array.mean())
    if n < 2:
        return {
            "n_repeats": n,
            "mean": mean,
            "sem": None,
            "df": 0,
            "t_crit": None,
            "ci_lower": None,
            "ci_upper": None,
        }
    sem = float(stats.sem(array))  # sample std (ddof=1) / sqrt(n)
    df = n - 1
    t_crit = float(stats.t.ppf(1.0 - (1.0 - POOLED_CI_LEVEL) / 2.0, df))
    if sem == 0.0:
        # Identical repeats: a zero-width interval (scipy would return NaN for
        # scale=0). Avoids NaN leaking into the manifest/Parquet.
        lower, upper = mean, mean
    else:
        lower, upper = stats.t.interval(POOLED_CI_LEVEL, df=df, loc=mean, scale=sem)
    return {
        "n_repeats": n,
        "mean": mean,
        "sem": sem,
        "df": df,
        "t_crit": t_crit,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
    }


def _repeat(row: Mapping[str, object]) -> int:
    return _as_int(row.get("seed_repeat_index", 0))


def _survival_cells(
    aggregate_dir: Path, level_by_candidate: Mapping[str, int]
) -> dict[tuple[int, str], list[float]]:
    """One survival_round value per (level, regime) per repeat.

    Survival is a run-level outcome (no voters to collapse). Any multiple runs
    for the same (level, regime, repeat) are collapsed by mean before pooling.
    """
    per_repeat: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for row in _rows(aggregate_dir, "candidate_survival"):
        level = level_by_candidate.get(str(row["candidate_id"]))
        if level is None:
            continue
        per_repeat[(level, str(row["regime"]), _repeat(row))].append(
            _as_float(row["survival_round"])
        )
    cells: dict[tuple[int, str], list[float]] = defaultdict(list)
    for (level, regime, _r), run_values in per_repeat.items():
        cells[(level, regime)].append(sum(run_values) / len(run_values))
    return cells


def _net_votes_cells(
    aggregate_dir: Path, level_by_candidate: Mapping[str, int]
) -> dict[tuple[int, str], list[float]]:
    """One net-signed-votes value per (level, regime) per repeat.

    Two-stage collapse: within a repeat, sum each voter's regime-signed votes to
    the candidate across rounds, then average over voters (collapsing the
    non-independent voters). That per-repeat value feeds the across-repeat pool.
    """
    per_voter: dict[tuple[int, str, int, int], float] = defaultdict(float)
    for row in _rows(aggregate_dir, "candidate_analysis"):
        level = level_by_candidate.get(str(row["candidate_id"]))
        if level is None or row["signed_action"] is None:
            continue
        key = (level, str(row["regime"]), _repeat(row), _as_int(row["voter_index"]))
        per_voter[key] += _as_float(row["signed_action"])
    per_repeat: dict[tuple[int, str, int], list[float]] = defaultdict(list)
    for (level, regime, repeat, _voter), total in per_voter.items():
        per_repeat[(level, regime, repeat)].append(total)
    cells: dict[tuple[int, str], list[float]] = defaultdict(list)
    for (level, regime, _r), voter_totals in per_repeat.items():
        cells[(level, regime)].append(sum(voter_totals) / len(voter_totals))
    return cells


def _emit(
    metric: PooledMetric, cells: Mapping[tuple[int, str], Sequence[float]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for level in SEVERITY_ORDER:
        for regime in REGIME_ORDER:
            values = cells.get((level, regime))
            if not values:
                continue
            rows.append(
                {
                    "analysis_version": POOLED_ANALYSIS_VERSION,
                    "estimator": POOLED_ESTIMATOR,
                    "ci_level": POOLED_CI_LEVEL,
                    "metric": metric.value,
                    "severity_level": level,
                    "regime": regime,
                    **t_sem_cell(list(values)),
                }
            )
    return rows


def pooled_by_severity(aggregate_dir: Path) -> tuple[dict[str, object], ...]:
    """Pool survival and net-signed-votes across repeats, per (level, regime).

    Returns one row per (metric, severity_level, regime) cell with the mean and a
    two-sided 95% t-interval over the per-repeat estimates.
    """
    level_by_candidate = severity_level_by_candidate(
        _rows(aggregate_dir, "source_annotations")
    )
    return tuple(
        [
            *_emit(
                PooledMetric.SURVIVAL_ROUNDS,
                _survival_cells(aggregate_dir, level_by_candidate),
            ),
            *_emit(
                PooledMetric.NET_SIGNED_VOTES,
                _net_votes_cells(aggregate_dir, level_by_candidate),
            ),
        ]
    )


def _average_ranks(votes_by_level: Mapping[int, float]) -> dict[int, float]:
    """Rank levels by descending votes; ties receive the average of their ranks."""
    order = sorted(votes_by_level.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[int, float] = {}
    index = 0
    while index < len(order):
        tie_end = index
        while tie_end + 1 < len(order) and order[tie_end + 1][1] == order[index][1]:
            tie_end += 1
        average_rank = (index + tie_end) / 2.0 + 1.0  # positions are 0-based
        for position in range(index, tie_end + 1):
            ranks[order[position][0]] = average_rank
        index = tie_end + 1
    return ranks


def rank_trajectories_by_severity(
    aggregate_dir: Path,
) -> dict[str, dict[int, dict[str, list[float]]]]:
    """Median rank per (regime, severity_level) over rounds, across repeats.

    For each (regime, repeat, round) the active candidates are ranked by total
    raw votes that round (rank 1 = most votes). Ranks are then combined across
    repeats by the median at each round, over the repeats where the candidate is
    still active. Returns ``{regime: {level: {"rounds": [...], "median_rank":
    [...], "n_active": [...]}}}``.
    """
    level_by_candidate = severity_level_by_candidate(
        _rows(aggregate_dir, "source_annotations")
    )
    # (regime, repeat, round, level) -> total raw votes; presence tracks activity
    votes: dict[tuple[str, int, int, int], float] = defaultdict(float)
    present: set[tuple[str, int, int, int]] = set()
    for row in _rows(aggregate_dir, "candidate_analysis"):
        level = level_by_candidate.get(str(row["candidate_id"]))
        if level is None:
            continue
        key = (str(row["regime"]), _repeat(row), _as_int(row["round_index"]), level)
        present.add(key)
        raw = row["raw_votes"]
        if raw is not None:
            votes[key] += _as_float(raw)
    # rank within each (regime, repeat, round)
    per_group: dict[tuple[str, int, int], dict[int, float]] = defaultdict(dict)
    for regime, repeat, rnd, level in present:
        per_group[(regime, repeat, rnd)][level] = votes[(regime, repeat, rnd, level)]
    ranks: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for (regime, _repeat_idx, rnd), votes_by_level in per_group.items():
        for level, rank in _average_ranks(votes_by_level).items():
            ranks[(regime, rnd, level)].append(rank)
    result: dict[str, dict[int, dict[str, list[float]]]] = {
        regime: {} for regime in REGIME_ORDER
    }
    rounds_seen = sorted({key[1] for key in ranks})
    for regime in REGIME_ORDER:
        for level in SEVERITY_ORDER:
            xs: list[float] = []
            ys: list[float] = []
            counts: list[float] = []
            for rnd in rounds_seen:
                cell = ranks.get((regime, rnd, level))
                if not cell:
                    continue
                xs.append(float(rnd))
                ys.append(float(np.median(cell)))
                counts.append(float(len(cell)))
            if xs:
                result.setdefault(regime, {})[level] = {
                    "rounds": xs,
                    "median_rank": ys,
                    "n_active": counts,
                }
    return result


def rank_records_by_severity(aggregate_dir: Path) -> list[dict[str, object]]:
    """JSON-safe record form of :func:`rank_trajectories_by_severity`.

    Returns a flat list of ``{regime, severity_level, rounds, median_rank,
    n_active}`` records in deterministic (regime, severity) order — suitable for
    the plot manifest, which must round-trip through JSON (no int dict keys).
    """
    trajectories = rank_trajectories_by_severity(aggregate_dir)
    records: list[dict[str, object]] = []
    for regime in REGIME_ORDER:
        by_level = trajectories.get(regime, {})
        for level in SEVERITY_ORDER:
            cell = by_level.get(level)
            if not cell:
                continue
            records.append(
                {
                    "regime": regime,
                    "severity_level": level,
                    "rounds": cell["rounds"],
                    "median_rank": cell["median_rank"],
                    "n_active": cell["n_active"],
                }
            )
    return records
