"""Read-only, deterministic snapshot analytics over persisted Parquet exports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from quadratic_voting.experiment.analysis import AgreementNullReason, spearman_with_ties

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


DEFAULT_SNAPSHOT_COUNT = 5
Row: TypeAlias = dict[str, object]
LineStyle: TypeAlias = Literal["-", "--", ":"]

# These colours are Okabe-Ito colourblind-safe colours.  Rudeness is the one
# categorical quantity shared by the relevant figures, so it deliberately has
# one stable visual encoding everywhere it appears.
RUDENESS_COLORS = {
    "non_rude": "#0072B2",
    "rude": "#D55E00",
    "ambiguous_tie": "#CC79A7",
}
RUDENESS_ORDER = ("non_rude", "rude", "ambiguous_tie")
ARM_ORDER = ("action-only", "action-then-statement", "statement-then-action")
REGIME_ORDER = ("support", "opposition")
UNSPENT_COLOR = "#8A8A8A"
_CANDIDATE_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#59A14F",
    "#E15759",
    "#B07AA1",
    "#76B7B2",
    "#EDC948",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
)


def select_snapshot_rounds(
    round_indices: Sequence[int], count: int = DEFAULT_SNAPSHOT_COUNT
) -> tuple[int, ...]:
    """Select position-even observed rounds, always retaining both endpoints."""
    rounds = tuple(sorted(set(round_indices)))
    if count <= 0:
        raise ValueError("--snapshot-count must be positive")
    if len(rounds) <= 1:
        return rounds
    if count < 2:
        raise ValueError(
            "--snapshot-count must be at least 2 when a run has multiple observed "
            "rounds, so the first and last rounds can both be included"
        )
    if len(rounds) <= count:
        return rounds
    positions = tuple(
        round((len(rounds) - 1) * position / (count - 1)) for position in range(count)
    )
    return tuple(rounds[position] for position in dict.fromkeys(positions))


def _fields(*names: str) -> list[tuple[str, pa.DataType]]:
    return [
        ("matched_set_id", pa.string()),
        ("run_id", pa.string()),
        ("arm", pa.string()),
        ("regime", pa.string()),
        *[(name, pa.int64()) for name in names],
    ]


_DETAIL = pa.schema(
    _fields("snapshot_round", "voter_index")
    + [
        ("candidate_id", pa.string()),
        ("rudeness_label", pa.string()),
        ("rating_code", pa.int64()),
        ("statement_status", pa.string()),
        ("ballot_status", pa.string()),
        ("raw_votes", pa.int64()),
        ("signed_action", pa.int64()),
        ("current_credits", pa.int64()),
        ("signed_credit_spend", pa.int64()),
        ("cumulative_before_votes", pa.int64()),
        ("cumulative_through_votes", pa.int64()),
        ("cumulative_before_credits", pa.int64()),
        ("cumulative_through_credits", pa.int64()),
    ]
)
_VOTER = pa.schema(
    _fields("snapshot_round", "voter_index", "credit_budget")
    + [
        ("current_votes", pa.int64()),
        ("current_credits", pa.int64()),
        ("current_remaining_credit", pa.int64()),
        ("cumulative_before_votes", pa.int64()),
        ("cumulative_through_votes", pa.int64()),
        ("cumulative_before_credits", pa.int64()),
        ("cumulative_through_credits", pa.int64()),
    ]
)
_VOTER_RUDENESS = pa.schema(
    _fields("snapshot_round", "voter_index")
    + [
        ("rudeness_label", pa.string()),
        ("n_candidates", pa.int64()),
        ("has_null_action", pa.bool_()),
        ("current_votes", pa.int64()),
        ("current_credits", pa.int64()),
    ]
)
_SUMMARY_METRICS = (
    "current_votes",
    "current_credits",
    "cumulative_before_votes",
    "cumulative_through_votes",
    "cumulative_before_credits",
    "cumulative_through_credits",
)
_SUMMARY_COLUMNS = [
    column
    for metric in _SUMMARY_METRICS
    for column in (f"mean_{metric}", f"sum_{metric}")
]
_CANDIDATE = pa.schema(
    _fields("snapshot_round")
    + [
        ("candidate_id", pa.string()),
        ("rudeness_label", pa.string()),
        ("n_voters", pa.int64()),
    ]
    + [(column, pa.float64()) for column in _SUMMARY_COLUMNS]
)
_CANDIDATE_LABELS = pa.schema(
    [("candidate_id", pa.string()), ("candidate_label", pa.string())]
)
_BUDGET_DETAIL = pa.schema(
    _fields("snapshot_round", "voter_index", "credit_budget")
    + [
        ("candidate_id", pa.string()),
        ("candidate_label", pa.string()),
        ("rudeness_label", pa.string()),
        ("quadratic_credits", pa.int64()),
        ("ballot_status", pa.string()),
    ]
)
_BUDGET_UTILIZATION = pa.schema(
    _fields("snapshot_round", "voter_index", "credit_budget")
    + [
        ("current_spend", pa.int64()),
        ("unspent_credits", pa.int64()),
        ("utilization_fraction", pa.float64()),
        ("full_budget_used", pa.bool_()),
        ("ballot_status", pa.string()),
    ]
)
_RUDENESS = pa.schema(
    _fields("snapshot_round")
    + [("rudeness_label", pa.string()), ("n_voter_candidates", pa.int64())]
    + [(column, pa.float64()) for column in _SUMMARY_COLUMNS]
)
_SURVIVOR = pa.schema(
    _fields("snapshot_round")
    + [
        ("candidate_id", pa.string()),
        ("rudeness_label", pa.string()),
        ("first_turn_length", pa.int64()),
        ("second_turn_length", pa.int64()),
        ("total_two_turn_length", pa.int64()),
    ]
)
_AGREEMENT = pa.schema(
    _fields("snapshot_round", "voter_index")
    + [
        ("measure", pa.string()),
        ("spearman_rho", pa.float64()),
        ("null_reason", pa.string()),
        ("n_candidate_pairs", pa.int64()),
        ("estimand_language", pa.string()),
    ]
)
_AGREEMENT_RUDENESS = pa.schema(
    _fields("snapshot_round", "voter_index")
    + [
        ("rudeness_label", pa.string()),
        ("measure", pa.string()),
        ("spearman_rho", pa.float64()),
        ("null_reason", pa.string()),
        ("n_candidate_pairs", pa.int64()),
        ("estimand_language", pa.string()),
    ]
)


def _read(directory: Path, name: str) -> list[Row]:
    return [
        dict(row) for row in pq.read_table(directory / f"{name}.parquet").to_pylist()
    ]


def _integer(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(
            f"snapshot analytics expected an integer-compatible value, got {value!r}"
        )
    return int(value)


def _required_int(row: Mapping[str, object], field: str) -> int:
    value = _integer(row.get(field))
    if value is None:
        raise ValueError(
            f"snapshot analytics requires non-null {field!r} in exported data"
        )
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"snapshot analytics expected a numeric value, got {value!r}")
    return float(value)


def _write(
    directory: Path, name: str, rows: Iterable[Mapping[str, object]], schema: pa.Schema
) -> Path:
    ordered = sorted(
        rows,
        key=lambda row: tuple(
            "" if row.get(field.name) is None else str(row.get(field.name))
            for field in schema
        ),
    )
    table = pa.Table.from_pylist(
        [{field.name: row.get(field.name) for field in schema} for row in ordered],
        schema=schema,
    )
    path = directory / f"{name}.parquet"
    pq.write_table(table, path, compression="zstd", version="2.6")
    return path


def _summary(
    rows: Sequence[Row], identifiers: Mapping[str, object], count_name: str
) -> Row:
    output: Row = {**identifiers, count_name: len(rows)}
    sources = dict(
        zip(
            _SUMMARY_METRICS,
            (
                "raw_votes",
                "current_credits",
                "cumulative_before_votes",
                "cumulative_through_votes",
                "cumulative_before_credits",
                "cumulative_through_credits",
            ),
            strict=True,
        )
    )
    for metric, source in sources.items():
        numeric_values = [
            _number(value) for row in rows if (value := row[source]) is not None
        ]
        output[f"mean_{metric}"] = (
            None if not numeric_values else sum(numeric_values) / len(numeric_values)
        )
        output[f"sum_{metric}"] = None if not numeric_values else sum(numeric_values)
    return output


def build_snapshot_tables(
    export_dir: Path, out_dir: Path, *, snapshot_count: int = DEFAULT_SNAPSHOT_COUNT
) -> tuple[Path, ...]:
    """Write stable relations from an export into an already-owned output directory."""
    if not export_dir.is_dir():
        raise ValueError(f"export directory does not exist: {export_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates, runs = (
        _read(export_dir, "candidate_analysis"),
        _read(export_dir, "runs"),
    )
    run_by_id = {str(row["run_id"]): row for row in runs}
    active: dict[tuple[str, int], list[str]] = defaultdict(list)
    for row in _read(export_dir, "round_candidates"):
        active[(str(row["run_id"]), _required_int(row, "round_index"))].append(
            str(row["candidate_id"])
        )
    turns: dict[str, list[Row]] = defaultdict(list)
    for row in _read(export_dir, "candidate_source_turns"):
        turns[str(row["candidate_id"])].append(row)
    by_run: dict[str, list[Row]] = defaultdict(list)
    for row in candidates:
        by_run[str(row["run_id"])].append(row)

    details: list[Row] = []
    voters: list[Row] = []
    survivors: list[Row] = []
    for run_id, rows in sorted(by_run.items()):
        snapshots = set(
            select_snapshot_rounds(
                [index for candidate_run, index in active if candidate_run == run_id],
                snapshot_count,
            )
        )
        cumulative: dict[tuple[int, str], tuple[int, int]] = defaultdict(lambda: (0, 0))
        for round_index in sorted({_required_int(row, "round_index") for row in rows}):
            round_rows = [
                row for row in rows if _required_int(row, "round_index") == round_index
            ]
            selected_rows: list[Row] = []
            for row in round_rows:
                voter, candidate = (
                    _required_int(row, "voter_index"),
                    str(row["candidate_id"]),
                )
                raw = _integer(row.get("raw_votes"))
                credits = None if raw is None else raw**2
                before_votes, before_credits = cumulative[(voter, candidate)]
                if round_index in snapshots:
                    selected_rows.append(
                        {
                            "matched_set_id": row["matched_set_id"],
                            "run_id": run_id,
                            "arm": row["arm"],
                            "regime": row["regime"],
                            "snapshot_round": round_index,
                            "voter_index": voter,
                            "candidate_id": candidate,
                            "rudeness_label": row["rudeness_label"],
                            "rating_code": row.get("rating_code"),
                            "statement_status": row["statement_status"],
                            "ballot_status": row["ballot_status"],
                            "raw_votes": raw,
                            "signed_action": row.get("signed_action"),
                            "current_credits": credits,
                            "signed_credit_spend": None
                            if credits is None
                            else (1 if row["regime"] == "support" else -1) * credits,
                            "cumulative_before_votes": None
                            if raw is None
                            else before_votes,
                            "cumulative_through_votes": None
                            if raw is None
                            else before_votes + raw,
                            "cumulative_before_credits": None
                            if credits is None
                            else before_credits,
                            "cumulative_through_credits": None
                            if credits is None
                            else before_credits + credits,
                        }
                    )
                if raw is not None:
                    cumulative[(voter, candidate)] = (
                        before_votes + raw,
                        before_credits + raw**2,
                    )
            details.extend(selected_rows)
            if round_index not in snapshots:
                continue
            by_voter: dict[int, list[Row]] = defaultdict(list)
            for row in selected_rows:
                by_voter[_required_int(row, "voter_index")].append(row)
            for voter_index, values in by_voter.items():
                complete = all(row["raw_votes"] is not None for row in values)

                def total(field: str) -> int | None:
                    return (
                        sum(_required_int(row, field) for row in values)
                        if complete
                        else None
                    )

                run = run_by_id[run_id]
                current_credits = total("current_credits")
                voters.append(
                    {
                        "matched_set_id": run["matched_set_id"],
                        "run_id": run_id,
                        "arm": run["arm"],
                        "regime": run["regime"],
                        "snapshot_round": round_index,
                        "voter_index": voter_index,
                        "credit_budget": run["credit_budget"],
                        "current_votes": total("raw_votes"),
                        "current_credits": current_credits,
                        "current_remaining_credit": None
                        if current_credits is None
                        else _required_int(run, "credit_budget") - current_credits,
                        "cumulative_before_votes": total("cumulative_before_votes"),
                        "cumulative_through_votes": total("cumulative_through_votes"),
                        "cumulative_before_credits": total("cumulative_before_credits"),
                        "cumulative_through_credits": total(
                            "cumulative_through_credits"
                        ),
                    }
                )
            candidate_rows = {str(row["candidate_id"]): row for row in round_rows}
            for candidate_id in active[(run_id, round_index)]:
                source = candidate_rows.get(candidate_id)
                if source is None:
                    continue
                ordered_turns = sorted(
                    turns[candidate_id],
                    key=lambda row: _required_int(row, "turn_index"),
                )
                first = (
                    None if not ordered_turns else len(str(ordered_turns[0]["text"]))
                )
                second = (
                    None
                    if len(ordered_turns) < 2
                    else len(str(ordered_turns[1]["text"]))
                )
                survivors.append(
                    {
                        "matched_set_id": source["matched_set_id"],
                        "run_id": run_id,
                        "arm": source["arm"],
                        "regime": source["regime"],
                        "snapshot_round": round_index,
                        "candidate_id": candidate_id,
                        "rudeness_label": source["rudeness_label"],
                        "first_turn_length": first,
                        "second_turn_length": second,
                        "total_two_turn_length": None
                        if first is None or second is None
                        else first + second,
                    }
                )

    candidate_groups: dict[tuple[object, ...], list[Row]] = defaultdict(list)
    rudeness_groups: dict[tuple[object, ...], list[Row]] = defaultdict(list)
    voter_groups: dict[tuple[object, ...], list[Row]] = defaultdict(list)
    voter_rudeness_groups: dict[tuple[object, ...], list[Row]] = defaultdict(list)
    for row in details:
        base = tuple(
            row[field]
            for field in ("matched_set_id", "run_id", "arm", "regime", "snapshot_round")
        )
        candidate_groups[(*base, row["candidate_id"], row["rudeness_label"])].append(
            row
        )
        rudeness_groups[(*base, row["rudeness_label"])].append(row)
        voter_groups[(*base, row["voter_index"])].append(row)
        voter_rudeness_groups[
            (*base, row["voter_index"], row["rudeness_label"])
        ].append(row)
    candidate_summary = [
        _summary(
            rows,
            dict(
                zip(
                    (
                        "matched_set_id",
                        "run_id",
                        "arm",
                        "regime",
                        "snapshot_round",
                        "candidate_id",
                        "rudeness_label",
                    ),
                    key,
                    strict=True,
                )
            ),
            "n_voters",
        )
        for key, rows in candidate_groups.items()
    ]
    rudeness_summary = [
        _summary(
            rows,
            dict(
                zip(
                    (
                        "matched_set_id",
                        "run_id",
                        "arm",
                        "regime",
                        "snapshot_round",
                        "rudeness_label",
                    ),
                    key,
                    strict=True,
                )
            ),
            "n_voter_candidates",
        )
        for key, rows in rudeness_groups.items()
    ]
    voter_rudeness_summary: list[Row] = []
    for key, rows in voter_rudeness_groups.items():
        has_null_action = any(row["raw_votes"] is None for row in rows)
        voter_rudeness_summary.append(
            {
                **dict(
                    zip(
                        (
                            "matched_set_id",
                            "run_id",
                            "arm",
                            "regime",
                            "snapshot_round",
                            "voter_index",
                            "rudeness_label",
                        ),
                        key,
                        strict=True,
                    )
                ),
                "n_candidates": len(rows),
                "has_null_action": has_null_action,
                "current_votes": None
                if has_null_action
                else sum(_required_int(row, "raw_votes") for row in rows),
                "current_credits": None
                if has_null_action
                else sum(_required_int(row, "current_credits") for row in rows),
            }
        )

    def agreement_row(
        key: tuple[object, ...], rows: Sequence[Row], *, rudeness_label: str | None
    ) -> list[Row]:
        measure_rows: list[Row] = []
        for measure in ("signed_action", "signed_credit_spend"):
            pairs = [
                row
                for row in rows
                if row["rating_code"] is not None and row[measure] is not None
            ]
            ratings = [_required_int(row, "rating_code") for row in pairs]
            action_values = [_required_int(row, measure) for row in pairs]
            reason = (
                "NOT_APPLICABLE_ACTION_ONLY"
                if key[2] == "action-only"
                else (
                    AgreementNullReason.MISSING_STATEMENT.value
                    if any(row["statement_status"] != "accepted" for row in rows)
                    else AgreementNullReason.ABSTAINED_BALLOT.value
                    if any(row["ballot_status"] != "accepted" for row in rows)
                    else AgreementNullReason.N_LT_2.value
                    if len(pairs) < 2
                    else AgreementNullReason.CONSTANT_RATING.value
                    if len(set(ratings)) < 2
                    else AgreementNullReason.CONSTANT_ACTION.value
                    if len(set(action_values)) < 2
                    else None
                )
            )
            identifiers = dict(
                zip(
                    (
                        "matched_set_id",
                        "run_id",
                        "arm",
                        "regime",
                        "snapshot_round",
                        "voter_index",
                    ),
                    key,
                    strict=True,
                )
            )
            if rudeness_label is not None:
                identifiers["rudeness_label"] = rudeness_label
            measure_rows.append(
                {
                    **identifiers,
                    "measure": measure,
                    "spearman_rho": None
                    if reason
                    else spearman_with_ties(ratings, action_values),
                    "null_reason": reason,
                    "n_candidate_pairs": len(pairs),
                    "estimand_language": "descriptive stated-preference association; not causal",
                }
            )
        return measure_rows

    agreements: list[Row] = []
    for key, rows in voter_groups.items():
        agreements.extend(agreement_row(key, rows, rudeness_label=None))
    rudeness_agreements: list[Row] = []
    for key, rows in voter_rudeness_groups.items():
        rudeness_agreements.extend(
            agreement_row(key[:-1], rows, rudeness_label=str(key[-1]))
        )
    candidate_labels = [
        {"candidate_id": candidate_id, "candidate_label": f"C{index}"}
        for index, candidate_id in enumerate(
            sorted({str(row["candidate_id"]) for row in details}), start=1
        )
    ]
    labels_by_id = {
        str(row["candidate_id"]): str(row["candidate_label"])
        for row in candidate_labels
    }
    budget_detail = [
        {
            "matched_set_id": row["matched_set_id"],
            "run_id": row["run_id"],
            "arm": row["arm"],
            "regime": row["regime"],
            "snapshot_round": row["snapshot_round"],
            "voter_index": row["voter_index"],
            "credit_budget": run_by_id[str(row["run_id"])]["credit_budget"],
            "candidate_id": row["candidate_id"],
            "candidate_label": labels_by_id[str(row["candidate_id"])],
            "rudeness_label": row["rudeness_label"],
            "quadratic_credits": row["current_credits"],
            "ballot_status": row["ballot_status"],
        }
        for row in details
    ]
    status_by_voter = {
        (
            str(row["run_id"]),
            _required_int(row, "snapshot_round"),
            _required_int(row, "voter_index"),
        ): str(row["ballot_status"])
        for row in details
    }
    budget_utilization = [
        {
            **{
                key: row[key]
                for key in (
                    "matched_set_id",
                    "run_id",
                    "arm",
                    "regime",
                    "snapshot_round",
                    "voter_index",
                    "credit_budget",
                )
            },
            "current_spend": row["current_credits"],
            "unspent_credits": row["current_remaining_credit"],
            "utilization_fraction": None
            if row["current_credits"] is None
            else _required_int(row, "current_credits")
            / _required_int(row, "credit_budget"),
            "full_budget_used": None
            if row["current_credits"] is None
            else _required_int(row, "current_credits")
            == _required_int(row, "credit_budget"),
            "ballot_status": status_by_voter[
                (
                    str(row["run_id"]),
                    _required_int(row, "snapshot_round"),
                    _required_int(row, "voter_index"),
                )
            ],
        }
        for row in voters
    ]
    return (
        _write(out_dir, "snapshot_voter_candidate", details, _DETAIL),
        _write(out_dir, "snapshot_voter_summary", voters, _VOTER),
        _write(
            out_dir,
            "snapshot_voter_rudeness_summary",
            voter_rudeness_summary,
            _VOTER_RUDENESS,
        ),
        _write(out_dir, "snapshot_candidate_summary", candidate_summary, _CANDIDATE),
        _write(
            out_dir, "snapshot_candidate_labels", candidate_labels, _CANDIDATE_LABELS
        ),
        _write(
            out_dir, "snapshot_voter_budget_distribution", budget_detail, _BUDGET_DETAIL
        ),
        _write(
            out_dir,
            "snapshot_budget_utilization",
            budget_utilization,
            _BUDGET_UTILIZATION,
        ),
        _write(out_dir, "snapshot_rudeness_summary", rudeness_summary, _RUDENESS),
        _write(out_dir, "survivor_demographics", survivors, _SURVIVOR),
        _write(out_dir, "stated_preference_agreement", agreements, _AGREEMENT),
        _write(
            out_dir,
            "stated_preference_agreement_by_rudeness",
            rudeness_agreements,
            _AGREEMENT_RUDENESS,
        ),
    )


def _ordered_conditions(rows: Sequence[Row]) -> tuple[tuple[str, str], ...]:
    """Return observed arm/regime conditions in the stable experimental order."""
    present = {(str(row["arm"]), str(row["regime"])) for row in rows}
    known = tuple(
        (arm, regime)
        for regime in REGIME_ORDER
        for arm in ARM_ORDER
        if (arm, regime) in present
    )
    return known + tuple(sorted(present.difference(known)))


def _condition_title(arm: str, regime: str) -> str:
    return f"{_regime_title(regime)} — {arm.replace('-', ' ').title()}"


def _regime_title(regime: str) -> str:
    return {
        "support": "Most Votes Kept",
        "opposition": "Most Votes Kicked",
    }.get(regime, regime.replace("_", " ").title())


def _candidate_color(candidate_label: str) -> str:
    """Return the stable categorical colour for a compact candidate label."""
    try:
        return _CANDIDATE_COLORS[
            (int(candidate_label.removeprefix("C")) - 1) % len(_CANDIDATE_COLORS)
        ]
    except ValueError:
        return "#666666"


def _rudeness_title(label: str) -> str:
    return {
        "non_rude": "Non-rude",
        "rude": "Rude",
        "ambiguous_tie": "Ambiguous tie",
    }.get(label, label.replace("_", " ").title())


def _ordered_rudeness(rows: Sequence[Row]) -> tuple[str, ...]:
    present = {str(row["rudeness_label"]) for row in rows}
    known = tuple(label for label in RUDENESS_ORDER if label in present)
    return known + tuple(sorted(present.difference(known)))


def _mean_by_snapshot(rows: Sequence[Row], field: str) -> tuple[list[int], list[float]]:
    """Mean a table field at each observed snapshot, omitting null values."""
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            values[_required_int(row, "snapshot_round")].append(_number(value))
    rounds = sorted(values)
    return rounds, [
        sum(values[round_index]) / len(values[round_index]) for round_index in rounds
    ]


def _sum_by_snapshot(rows: Sequence[Row], field: str) -> tuple[list[int], list[float]]:
    """Sum a persisted total field at each observed snapshot, omitting nulls."""
    values: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            values[_required_int(row, "snapshot_round")].append(_number(value))
    rounds = sorted(values)
    return rounds, [sum(values[round_index]) for round_index in rounds]


def _survivor_counts(
    rows: Sequence[Row], condition: tuple[str, str], labels: Sequence[str]
) -> tuple[list[int], dict[str, list[int]]]:
    """Count active candidates by persisted rudeness at each snapshot."""
    counts: dict[str, dict[int, int]] = {label: defaultdict(int) for label in labels}
    rounds: set[int] = set()
    for row in rows:
        if (row["arm"], row["regime"]) != condition:
            continue
        snapshot_round = _required_int(row, "snapshot_round")
        rounds.add(snapshot_round)
        label = str(row["rudeness_label"])
        if label in counts:
            counts[label][snapshot_round] += 1
    ordered_rounds = sorted(rounds)
    return ordered_rounds, {
        label: [counts[label][snapshot_round] for snapshot_round in ordered_rounds]
        for label in labels
    }


def _condition_snapshot_rounds(
    rows: Sequence[Row], condition: tuple[str, str]
) -> list[int]:
    """Return only persisted snapshot rounds for one arm/regime panel."""
    return sorted(
        {
            _required_int(row, "snapshot_round")
            for row in rows
            if (row["arm"], row["regime"]) == condition
        }
    )


def _set_snapshot_ticks(axis: Axes, rounds: Sequence[int]) -> None:
    """Keep the x-axis faithful to observed snapshots, not locator interpolation."""
    axis.set_xticks(rounds)
    axis.tick_params(axis="x", labelbottom=True)


def aggregate_preference_agreement(rows: Sequence[Row]) -> tuple[Row, ...]:
    """Aggregate defined per-voter rho values before they are drawn as a series.

    The source table is at voter grain.  Returning one mean (and range) per
    run/condition/snapshot/measure prevents a plotted line from connecting
    different voters as though they were a longitudinal observation.
    """
    grouped: dict[tuple[str, str, str, int, str], list[float]] = defaultdict(list)
    for row in rows:
        rho = row.get("spearman_rho")
        if rho is None or str(row["arm"]) == "action-only":
            continue
        key = (
            str(row["run_id"]),
            str(row["arm"]),
            str(row["regime"]),
            _required_int(row, "snapshot_round"),
            str(row["measure"]),
        )
        grouped[key].append(_number(rho))
    return tuple(
        {
            "run_id": run_id,
            "arm": arm,
            "regime": regime,
            "snapshot_round": snapshot_round,
            "measure": measure,
            "mean_spearman_rho": sum(values) / len(values),
            "min_spearman_rho": min(values),
            "max_spearman_rho": max(values),
            "n_voters": len(values),
        }
        for (run_id, arm, regime, snapshot_round, measure), values in sorted(
            grouped.items()
        )
    )


def aggregate_preference_agreement_by_rudeness(rows: Sequence[Row]) -> tuple[Row, ...]:
    """Aggregate defined within-rudeness voter correlations for plotting."""
    grouped: dict[tuple[str, str, str, int, str, str], list[float]] = defaultdict(list)
    for row in rows:
        rho = row.get("spearman_rho")
        if rho is None or str(row["arm"]) == "action-only":
            continue
        key = (
            str(row["run_id"]),
            str(row["arm"]),
            str(row["regime"]),
            _required_int(row, "snapshot_round"),
            str(row["rudeness_label"]),
            str(row["measure"]),
        )
        grouped[key].append(_number(rho))
    return tuple(
        {
            "run_id": run_id,
            "arm": arm,
            "regime": regime,
            "snapshot_round": snapshot_round,
            "rudeness_label": rudeness_label,
            "measure": measure,
            "mean_spearman_rho": sum(values) / len(values),
            "min_spearman_rho": min(values),
            "max_spearman_rho": max(values),
            "n_voters": len(values),
        }
        for (
            run_id,
            arm,
            regime,
            snapshot_round,
            rudeness_label,
            measure,
        ), values in sorted(grouped.items())
    )


def build_snapshot_figures(out_dir: Path) -> tuple[tuple[str, Figure], ...]:
    """Build six compact descriptive figures from already-materialized tables.

    Kept separate from saving so artist-level tests can pin the scientific
    grouping without depending on pixels.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    (
        rudeness,
        survivor,
        agreement,
        voter_rudeness,
        candidate,
        candidate_labels,
        rudeness_agreement,
        budget_detail,
        budget_utilization,
    ) = (
        _read(out_dir, "snapshot_rudeness_summary"),
        _read(out_dir, "survivor_demographics"),
        _read(out_dir, "stated_preference_agreement"),
        _read(out_dir, "snapshot_voter_rudeness_summary"),
        _read(out_dir, "snapshot_candidate_summary"),
        _read(out_dir, "snapshot_candidate_labels"),
        _read(out_dir, "stated_preference_agreement_by_rudeness"),
        _read(out_dir, "snapshot_voter_budget_distribution"),
        _read(out_dir, "snapshot_budget_utilization"),
    )
    figures: list[tuple[str, Figure]] = []
    conditions = _ordered_conditions(rudeness or survivor or agreement)
    rudeness_labels = _ordered_rudeness(rudeness or survivor)
    rudeness_handles = [
        Line2D(
            [],
            [],
            color=RUDENESS_COLORS.get(label, "#666666"),
            marker="o",
            label=_rudeness_title(label),
        )
        for label in rudeness_labels
    ]
    survivor_rudeness_styles: dict[str, tuple[LineStyle, str]] = {
        "non_rude": ("-", "o"),
        "rude": ("--", "s"),
        "ambiguous_tie": (":", "^"),
    }

    figure, axes = plt.subplots(
        2, 3, figsize=(12, 6.8), sharex=True, constrained_layout=True
    )
    for axis, condition in zip(axes.flat, conditions, strict=False):
        arm, regime = condition
        axis.set_title(_condition_title(arm, regime), fontsize="small")
        for label in rudeness_labels:
            group = [
                row
                for row in rudeness
                if (row["arm"], row["regime"], row["rudeness_label"])
                == (*condition, label)
            ]
            for field, style, marker in (
                ("mean_current_votes", "-", "o"),
                ("mean_current_credits", "--", "s"),
            ):
                rounds, values = _mean_by_snapshot(group, field)
                axis.plot(
                    rounds,
                    values,
                    color=RUDENESS_COLORS.get(label, "#666666"),
                    linestyle=style,
                    marker=marker,
                )
        axis.set(xlabel="Snapshot round", ylabel="Mean allocation")
        _set_snapshot_ticks(axis, _condition_snapshot_rounds(rudeness, condition))
    figure.suptitle(
        "Current votes and quadratic credits by condition", fontsize="large"
    )
    figure.legend(
        handles=rudeness_handles
        + [
            Line2D([], [], color="#333333", marker="o", label="Votes"),
            Line2D(
                [], [], color="#333333", linestyle="--", marker="s", label="Credits"
            ),
        ],
        loc="outside lower center",
        ncols=5,
        fontsize="small",
    )
    figures.append(("average_current_votes_credits.png", figure))

    figure, axes = plt.subplots(
        4, 3, figsize=(12, 10), sharex=True, constrained_layout=True
    )
    for regime_index, regime in enumerate(REGIME_ORDER):
        for arm_index, arm in enumerate(ARM_ORDER):
            condition = (arm, regime)
            for metric_index, (metric, title) in enumerate(
                (("votes", "Votes"), ("credits", "Credits"))
            ):
                axis = axes[regime_index * 2 + metric_index, arm_index]
                axis.set_title(
                    f"{_condition_title(arm, regime)}\n{title}", fontsize="x-small"
                )
                for label in rudeness_labels:
                    group = [
                        row
                        for row in rudeness
                        if (row["arm"], row["regime"], row["rudeness_label"])
                        == (*condition, label)
                    ]
                    for status, style, marker in (
                        ("before", ":", "o"),
                        ("through", "-", "s"),
                    ):
                        rounds, values = _mean_by_snapshot(
                            group, f"mean_cumulative_{status}_{metric}"
                        )
                        axis.plot(
                            rounds,
                            values,
                            color=RUDENESS_COLORS.get(label, "#666666"),
                            linestyle=style,
                            marker=marker,
                        )
                axis.set(xlabel="Snapshot round", ylabel="Mean cumulative")
                _set_snapshot_ticks(
                    axis, _condition_snapshot_rounds(rudeness, condition)
                )
    figure.suptitle(
        "Cumulative allocations: before versus through snapshot", fontsize="large"
    )
    figure.legend(
        handles=rudeness_handles
        + [
            Line2D([], [], color="#333333", linestyle=":", marker="o", label="Before"),
            Line2D([], [], color="#333333", marker="s", label="Through"),
        ],
        loc="outside lower center",
        ncols=5,
        fontsize="small",
    )
    figures.append(("cumulative_votes_credits_before_through.png", figure))

    figure, axes = plt.subplots(
        2, 3, figsize=(12, 6.8), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, condition in zip(axes.flat, conditions, strict=False):
        axis.set_title(_condition_title(*condition), fontsize="small")
        for label in rudeness_labels:
            group = [
                row
                for row in survivor
                if (row["arm"], row["regime"], row["rudeness_label"])
                == (*condition, label)
            ]
            counts: dict[int, int] = defaultdict(int)
            for row in group:
                counts[_required_int(row, "snapshot_round")] += 1
            axis.plot(
                sorted(counts),
                [counts[round_index] for round_index in sorted(counts)],
                color=RUDENESS_COLORS.get(label, "#666666"),
                linestyle=survivor_rudeness_styles.get(label, ("-", "o"))[0],
                marker=survivor_rudeness_styles.get(label, ("-", "o"))[1],
            )
        axis.set(xlabel="Snapshot round", ylabel="Surviving candidates")
        _set_snapshot_ticks(axis, _condition_snapshot_rounds(survivor, condition))
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    figure.suptitle("Surviving candidates by rudeness", fontsize="large")
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=RUDENESS_COLORS.get(label, "#666666"),
                linestyle=survivor_rudeness_styles.get(label, ("-", "o"))[0],
                marker=survivor_rudeness_styles.get(label, ("-", "o"))[1],
                label=_rudeness_title(label),
            )
            for label in rudeness_labels
        ],
        loc="outside lower center",
        ncols=3,
        fontsize="small",
    )
    figures.append(("survivor_rudeness_distribution.png", figure))

    figure, axes = plt.subplots(
        2, 3, figsize=(12, 6.8), sharex=True, sharey=True, constrained_layout=True
    )
    for axis, condition in zip(axes.flat, conditions, strict=False):
        axis.set_title(_condition_title(*condition), fontsize="small")
        bar_rounds, category_counts = _survivor_counts(
            survivor, condition, rudeness_labels
        )
        min_spacing = min(
            (right - left for left, right in zip(bar_rounds, bar_rounds[1:])),
            default=1,
        )
        bar_width = 0.8 * min_spacing / max(len(rudeness_labels), 1)
        for label_index, label in enumerate(rudeness_labels):
            bar_values = category_counts[label]
            offset = (label_index - (len(rudeness_labels) - 1) / 2) * bar_width
            axis.bar(
                [round_index + offset for round_index in bar_rounds],
                bar_values,
                width=bar_width,
                color=RUDENESS_COLORS.get(label, "#666666"),
            )
        axis.set(xlabel="Snapshot round", ylabel="Surviving candidates")
        _set_snapshot_ticks(axis, bar_rounds)
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    figure.suptitle(
        "Candidate rudeness demographics at snapshot start", fontsize="large"
    )
    figure.legend(
        handles=[
            Patch(
                facecolor=RUDENESS_COLORS.get(label, "#666666"),
                label=_rudeness_title(label),
            )
            for label in rudeness_labels
        ],
        loc="outside lower center",
        ncols=3,
        fontsize="small",
    )
    figures.append(("candidate_rudeness_demographics.png", figure))

    figure, axes = plt.subplots(
        2, 3, figsize=(12, 6.8), sharex=True, sharey=True, constrained_layout=True
    )
    length_specs: tuple[tuple[str, str, LineStyle, str], ...] = (
        ("first_turn_length", "First turn", "-", "o"),
        ("second_turn_length", "Second turn", "--", "s"),
        ("total_two_turn_length", "First + second", ":", "^"),
    )
    for axis, condition in zip(axes.flat, conditions, strict=False):
        axis.set_title(_condition_title(*condition), fontsize="small")
        group = [row for row in survivor if (row["arm"], row["regime"]) == condition]
        for field, _label, style, marker in length_specs:
            rounds, values = _mean_by_snapshot(group, field)
            axis.plot(rounds, values, color="#333333", linestyle=style, marker=marker)
        axis.set(xlabel="Snapshot round", ylabel="Unicode characters")
        _set_snapshot_ticks(axis, _condition_snapshot_rounds(survivor, condition))
    figure.suptitle("Survivor source-message lengths", fontsize="large")
    figure.legend(
        handles=[
            Line2D([], [], color="#333333", linestyle=style, marker=marker, label=label)
            for _field, label, style, marker in length_specs
        ],
        loc="outside lower center",
        ncols=3,
        fontsize="small",
    )
    figures.append(("survivor_message_lengths.png", figure))

    preference_rows = aggregate_preference_agreement(agreement)
    preference_conditions = tuple(
        condition
        for condition in _ordered_conditions(preference_rows)
        if condition[0] != "action-only"
    )
    figure, axes = plt.subplots(
        2, 2, figsize=(10, 6.8), sharex=True, sharey=True, constrained_layout=True
    )
    measure_specs: tuple[tuple[str, str, str, LineStyle, str], ...] = (
        ("signed_action", "Signed action", "#555555", "-", "o"),
        ("signed_credit_spend", "Signed credit spend", "#000000", "--", "s"),
    )
    for axis, condition in zip(axes.flat, preference_conditions, strict=False):
        axis.set_title(_condition_title(*condition), fontsize="small")
        axis.axhline(0, color="#888888", linewidth=0.8, zorder=0)
        for measure, _label, color, style, marker in measure_specs:
            group = [
                row
                for row in preference_rows
                if (row["arm"], row["regime"], row["measure"]) == (*condition, measure)
            ]
            rounds = [_required_int(row, "snapshot_round") for row in group]
            means = [_number(row["mean_spearman_rho"]) for row in group]
            axis.plot(rounds, means, color=color, linestyle=style, marker=marker)
            for row in group:
                axis.vlines(
                    _required_int(row, "snapshot_round"),
                    _number(row["min_spearman_rho"]),
                    _number(row["max_spearman_rho"]),
                    color="#777777",
                    alpha=0.35,
                    linewidth=1,
                )
        axis.set(
            xlabel="Snapshot round", ylabel="Mean per-voter Spearman rho", ylim=(-1, 1)
        )
        _set_snapshot_ticks(
            axis, _condition_snapshot_rounds(preference_rows, condition)
        )
    figure.suptitle(
        "Stated-preference agreement (defined voter correlations)", fontsize="large"
    )
    figure.legend(
        handles=[
            Line2D([], [], color=color, linestyle=style, marker=marker, label=label)
            for _measure, label, color, style, marker in measure_specs
        ],
        loc="outside lower center",
        ncols=2,
        fontsize="small",
    )
    figures.append(("stated_preference_agreement.png", figure))

    # New figures deliberately use rudeness as a panel dimension rather than a
    # line colour: it remains legible if an export carries more than two labels.
    facet_conditions = _ordered_conditions(voter_rudeness or candidate or survivor)
    facet_labels = _ordered_rudeness(voter_rudeness or candidate or survivor)

    def facet_figure(
        title: str, *, height: float = 2.35
    ) -> tuple[Figure, list[list[Axes]]]:
        figure, axes = plt.subplots(
            len(facet_conditions),
            len(facet_labels),
            figsize=(3.45 * len(facet_labels), height * len(facet_conditions) + 0.8),
            squeeze=False,
            constrained_layout=True,
        )
        figure.suptitle(title, fontsize="large")
        return figure, cast("list[list[Axes]]", axes)

    for field, metric_title, filename in (
        ("current_votes", "current votes", "per_voter_current_votes_by_rudeness.png"),
        (
            "current_credits",
            "current quadratic-credit spend",
            "per_voter_current_credits_by_rudeness.png",
        ),
    ):
        figure, axes = facet_figure(f"Per-voter {metric_title} by persisted rudeness")
        values = [
            _number(row[field]) for row in voter_rudeness if row[field] is not None
        ]
        for condition_index, condition in enumerate(facet_conditions):
            for label_index, label in enumerate(facet_labels):
                axis = axes[condition_index][label_index]
                group = [
                    row
                    for row in voter_rudeness
                    if (row["arm"], row["regime"], row["rudeness_label"])
                    == (*condition, label)
                ]
                rounds = _condition_snapshot_rounds(voter_rudeness, condition)
                voters = sorted({_required_int(row, "voter_index") for row in group})
                matrix = [
                    [
                        next(
                            (
                                _number(row[field])
                                for row in group
                                if _required_int(row, "voter_index") == voter
                                and _required_int(row, "snapshot_round") == round_index
                                and row[field] is not None
                            ),
                            float("nan"),
                        )
                        for round_index in rounds
                    ]
                    for voter in voters
                ]
                if matrix:
                    axis.imshow(
                        matrix,
                        aspect="auto",
                        interpolation="nearest",
                        vmin=min(values, default=0),
                        vmax=max(values, default=1),
                        cmap=plt.get_cmap("viridis").with_extremes(bad="#eeeeee"),
                    )
                    for y, row_values in enumerate(matrix):
                        for x, value in enumerate(row_values):
                            axis.text(
                                x,
                                y,
                                "—" if value != value else f"{value:g}",
                                ha="center",
                                va="center",
                                fontsize="x-small",
                                color="white"
                                if value == value and value > max(values, default=1) / 2
                                else "black",
                            )
                else:
                    axis.text(
                        0.5,
                        0.5,
                        "No observed voters",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                    )
                axis.set_title(
                    f"{_condition_title(*condition)}\n{_rudeness_title(label)}",
                    fontsize="x-small",
                )
                axis.set(
                    xticks=range(len(rounds)),
                    xticklabels=rounds,
                    xlabel="Snapshot round",
                )
                axis.set_yticks(
                    range(len(voters)), [f"V{voter + 1}" for voter in voters]
                )
                if label_index == 0:
                    axis.set_ylabel("Voter")
        figures.append((filename, figure))

    label_by_candidate = {
        str(row["candidate_id"]): str(row["candidate_label"])
        for row in candidate_labels
    }

    # This intentionally remains an absolute-credit plot: a valid bar reaches
    # the replenished budget only through an explicit gray Unspent segment.
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    budget_conditions = _ordered_conditions(budget_detail)
    labels = sorted(
        {str(row["candidate_label"]) for row in budget_detail},
        key=lambda value: int(value[1:]),
    )
    for axis, condition in zip(axes.flat, budget_conditions, strict=False):
        group = [
            row for row in budget_detail if (row["arm"], row["regime"]) == condition
        ]
        utility = [
            row
            for row in budget_utilization
            if (row["arm"], row["regime"]) == condition
        ]
        keys = sorted(
            {
                (
                    _required_int(row, "snapshot_round"),
                    _required_int(row, "voter_index"),
                )
                for row in group
            }
        )
        # Boundaries are derived from observed row keys, so incomplete/uneven
        # voter populations remain legible without encoding a fixed voter count.
        for boundary, ((round_index, _voter), (next_round, _next_voter)) in enumerate(
            zip(keys, keys[1:], strict=False)
        ):
            axis.axhline(
                boundary + 0.5,
                color="#44515A",
                linestyle="-" if next_round != round_index else ":",
                linewidth=1.35 if next_round != round_index else 0.65,
                alpha=0.8 if next_round != round_index else 0.65,
                zorder=0,
            )
        for y, (round_index, voter_index) in enumerate(keys):
            util = next(
                row
                for row in utility
                if _required_int(row, "snapshot_round") == round_index
                and _required_int(row, "voter_index") == voter_index
            )
            row_group = [
                row
                for row in group
                if _required_int(row, "snapshot_round") == round_index
                and _required_int(row, "voter_index") == voter_index
            ]
            budget = _required_int(util, "credit_budget")
            if util["current_spend"] is None:
                axis.barh(y, budget, color="#F2F2F2", edgecolor="#555555", hatch="//")
                axis.text(
                    budget / 2,
                    y,
                    "Abstained / missing",
                    ha="center",
                    va="center",
                    fontsize="xx-small",
                )
                continue
            left = 0.0
            for label in labels:
                segment_credits = _integer(
                    next(
                        (
                            row["quadratic_credits"]
                            for row in row_group
                            if row["candidate_label"] == label
                        ),
                        0,
                    )
                )
                if segment_credits:
                    axis.barh(
                        y,
                        segment_credits,
                        left=left,
                        color=_candidate_color(label),
                        edgecolor="white",
                        linewidth=0.4,
                    )
                    left += segment_credits
            unspent = _required_int(util, "unspent_credits")
            if unspent:
                axis.barh(
                    y,
                    unspent,
                    left=left,
                    color=UNSPENT_COLOR,
                    edgecolor="white",
                    linewidth=0.4,
                )
        axis.set_title(_condition_title(*condition), fontsize="small")
        axis.set(
            yticks=range(len(keys)),
            yticklabels=[
                f"R{round_index} · V{voter_index + 1}"
                for round_index, voter_index in keys
            ],
            xlabel="Quadratic credits (absolute; budget per round)",
        )
        if utility:
            axis.set_xlim(
                0, max(_required_int(row, "credit_budget") for row in utility)
            )
        axis.invert_yaxis()
    figure.suptitle(
        "Per-voter replenished credit-budget distribution at observed snapshots",
        fontsize="large",
    )
    figure.legend(
        handles=[
            Patch(facecolor=_candidate_color(label), label=label) for label in labels
        ]
        + [
            Patch(facecolor=UNSPENT_COLOR, label="Unspent"),
            Patch(
                facecolor="#F2F2F2",
                edgecolor="#555555",
                hatch="//",
                label="Abstained / missing",
            ),
        ],
        loc="outside lower center",
        ncols=min(len(labels) + 2, 8),
        fontsize="x-small",
    )
    figures.append(("voter_credit_budget_distribution.png", figure))
    for field, metric_title, filename in (
        (
            "sum_current_votes",
            "current votes",
            "per_candidate_current_votes_by_rudeness.png",
        ),
        (
            "sum_current_credits",
            "current quadratic-credit spend",
            "per_candidate_current_credits_by_rudeness.png",
        ),
    ):
        figure, axes = facet_figure(
            f"Per-candidate {metric_title} by persisted rudeness"
        )
        for condition_index, condition in enumerate(facet_conditions):
            for label_index, label in enumerate(facet_labels):
                axis = axes[condition_index][label_index]
                group = [
                    row
                    for row in candidate
                    if (row["arm"], row["regime"], row["rudeness_label"])
                    == (*condition, label)
                ]
                rounds = _condition_snapshot_rounds(candidate, condition)
                candidate_ids = sorted({str(row["candidate_id"]) for row in group})
                for candidate_id in candidate_ids:
                    values_by_round = {
                        _required_int(row, "snapshot_round"): _number(row[field])
                        for row in group
                        if str(row["candidate_id"]) == candidate_id
                        and row[field] is not None
                    }
                    observed_rounds = [
                        round_index
                        for round_index in rounds
                        if round_index in values_by_round
                    ]
                    axis.plot(
                        observed_rounds,
                        [
                            values_by_round[round_index]
                            for round_index in observed_rounds
                        ],
                        marker="o",
                        label=label_by_candidate[candidate_id],
                    )
                axis.set_title(
                    f"{_condition_title(*condition)}\n{_rudeness_title(label)}",
                    fontsize="x-small",
                )
                axis.set(xlabel="Snapshot round", ylabel="Allocation (sum over voters)")
                _set_snapshot_ticks(axis, rounds)
                if candidate_ids:
                    axis.legend(fontsize="xx-small", ncols=2)
        figures.append((filename, figure))

    for metric, metric_title, filename in (
        (
            "votes",
            "vote totals (sums)",
            "cumulative_vote_totals_before_through_by_rudeness.png",
        ),
        (
            "credits",
            "quadratic-credit-spend totals (sums)",
            "cumulative_credit_totals_before_through_by_rudeness.png",
        ),
    ):
        figure, axes = facet_figure(
            f"Cumulative {metric_title}: before versus through snapshot"
        )
        for condition_index, condition in enumerate(facet_conditions):
            for label_index, label in enumerate(facet_labels):
                axis = axes[condition_index][label_index]
                group = [
                    row
                    for row in rudeness
                    if (row["arm"], row["regime"], row["rudeness_label"])
                    == (*condition, label)
                ]
                for status, style, marker in (
                    ("before", ":", "o"),
                    ("through", "-", "s"),
                ):
                    rounds, totals = _sum_by_snapshot(
                        group, f"sum_cumulative_{status}_{metric}"
                    )
                    axis.plot(
                        rounds,
                        totals,
                        linestyle=style,
                        marker=marker,
                        color="#333333",
                        label=status.title(),
                    )
                axis.set_title(
                    f"{_condition_title(*condition)}\n{_rudeness_title(label)}",
                    fontsize="x-small",
                )
                axis.set(xlabel="Snapshot round", ylabel="Total (sum)")
                _set_snapshot_ticks(
                    axis, _condition_snapshot_rounds(rudeness, condition)
                )
                if condition_index == 0 and label_index == 0:
                    axis.legend(fontsize="x-small")
        figures.append((filename, figure))

    for field, metric_title, filename in (
        (
            "first_turn_length",
            "first message",
            "survivor_first_message_length_distribution_by_rudeness.png",
        ),
        (
            "second_turn_length",
            "second message",
            "survivor_second_message_length_distribution_by_rudeness.png",
        ),
        (
            "total_two_turn_length",
            "first + second messages",
            "survivor_total_message_length_distribution_by_rudeness.png",
        ),
    ):
        figure, axes = facet_figure(f"Survivor {metric_title} length distributions")
        figure.suptitle(
            f"Survivor {metric_title} length distributions\n"
            "Persisted-rudeness facets; snapshot changes reflect survivor composition, not message change.",
            fontsize="medium",
        )
        for condition_index, condition in enumerate(facet_conditions):
            for label_index, label in enumerate(facet_labels):
                axis = axes[condition_index][label_index]
                group = [
                    row
                    for row in survivor
                    if (row["arm"], row["regime"], row["rudeness_label"])
                    == (*condition, label)
                ]
                rounds = _condition_snapshot_rounds(survivor, condition)
                distributions = [
                    [
                        _number(row[field])
                        for row in group
                        if _required_int(row, "snapshot_round") == round_index
                        and row[field] is not None
                    ]
                    for round_index in rounds
                ]
                populated = [
                    (index, values)
                    for index, values in enumerate(distributions, start=1)
                    if values
                ]
                if populated:
                    axis.boxplot(
                        [values for _index, values in populated],
                        positions=[index for index, _values in populated],
                        widths=0.55,
                        manage_ticks=False,
                    )
                else:
                    axis.text(
                        0.5,
                        0.5,
                        "No non-null lengths",
                        ha="center",
                        va="center",
                        transform=axis.transAxes,
                    )
                axis.set_title(
                    f"{_condition_title(*condition)}\n{_rudeness_title(label)}",
                    fontsize="x-small",
                )
                axis.set(
                    xlabel="Snapshot round",
                    ylabel="Unicode characters",
                    xticks=range(1, len(rounds) + 1),
                    xticklabels=rounds,
                )
        figures.append((filename, figure))

    rudeness_preference_rows = aggregate_preference_agreement_by_rudeness(
        rudeness_agreement
    )
    preference_conditions = tuple(
        condition
        for condition in _ordered_conditions(rudeness_agreement)
        if condition[0] != "action-only"
    )
    preference_labels = _ordered_rudeness(rudeness_agreement)
    if not preference_conditions or not preference_labels:
        # No non-action-only arms (e.g. an action-only pilot): stated-preference
        # agreement is undefined, so render an explicit placeholder rather than a
        # zero-row subplot grid (which matplotlib rejects).
        figure, axis = plt.subplots(figsize=(6.5, 3.0), constrained_layout=True)
        figure.suptitle(
            "Stated-preference agreement within persisted rudeness "
            "(descriptive, not causal)",
            fontsize="large",
        )
        axis.text(
            0.5,
            0.5,
            "Not applicable: this dataset has only action-only ballots,\n"
            "which carry no stated preference to correlate.",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        figures.append(("stated_preference_agreement_by_rudeness.png", figure))
        return tuple(figures)
    figure, axes = plt.subplots(
        len(preference_conditions),
        len(preference_labels),
        figsize=(
            3.45 * len(preference_labels),
            2.65 * len(preference_conditions) + 0.8,
        ),
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(
        "Stated-preference agreement within persisted rudeness (descriptive, not causal)",
        fontsize="large",
    )
    for condition_index, condition in enumerate(preference_conditions):
        for label_index, label in enumerate(preference_labels):
            axis = axes[condition_index][label_index]
            axis.axhline(0, color="#888888", linewidth=0.8, zorder=0)
            group = [
                row
                for row in rudeness_preference_rows
                if (row["arm"], row["regime"], row["rudeness_label"])
                == (*condition, label)
            ]
            for measure, color, style, marker in (
                ("signed_action", "#555555", "-", "o"),
                ("signed_credit_spend", "#000000", "--", "s"),
            ):
                measure_rows = [row for row in group if row["measure"] == measure]
                axis.plot(
                    [_required_int(row, "snapshot_round") for row in measure_rows],
                    [_number(row["mean_spearman_rho"]) for row in measure_rows],
                    color=color,
                    linestyle=style,
                    marker=marker,
                    label=measure.replace("_", " ").title(),
                )
                for row in measure_rows:
                    axis.vlines(
                        _required_int(row, "snapshot_round"),
                        _number(row["min_spearman_rho"]),
                        _number(row["max_spearman_rho"]),
                        color=color,
                        alpha=0.3,
                        linewidth=1,
                    )
            axis.set_title(
                f"{_condition_title(*condition)}\n{_rudeness_title(label)}",
                fontsize="x-small",
            )
            axis.set(
                xlabel="Snapshot round",
                ylabel="Mean per-voter Spearman rho",
                ylim=(-1, 1),
            )
            _set_snapshot_ticks(
                axis, _condition_snapshot_rounds(rudeness_agreement, condition)
            )
            if condition_index == 0 and label_index == 0:
                axis.legend(fontsize="xx-small")
    figures.append(("stated_preference_agreement_by_rudeness.png", figure))
    return tuple(figures)


def render_snapshot_figures(out_dir: Path) -> tuple[Path, ...]:
    """Render baseline and rudeness-faceted descriptive figures from snapshot tables."""
    import matplotlib.pyplot as plt

    paths = []
    for name, figure in build_snapshot_figures(out_dir):
        path = out_dir / name
        figure.savefig(path, dpi=180, metadata={"Software": "quadratic-voting"})
        plt.close(figure)
        paths.append(path)
    return tuple(paths)
