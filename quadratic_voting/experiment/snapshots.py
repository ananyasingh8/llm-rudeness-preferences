"""Read-only, deterministic snapshot analytics over persisted Parquet exports."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TypeAlias

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from quadratic_voting.experiment.analysis import AgreementNullReason, spearman_with_ties


DEFAULT_SNAPSHOT_COUNT = 5
Row: TypeAlias = dict[str, object]


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
    agreements: list[Row] = []
    for key, rows in voter_groups.items():
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
            agreements.append(
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
                            ),
                            key,
                            strict=True,
                        )
                    ),
                    "measure": measure,
                    "spearman_rho": None
                    if reason
                    else spearman_with_ties(ratings, action_values),
                    "null_reason": reason,
                    "n_candidate_pairs": len(pairs),
                    "estimand_language": "descriptive stated-preference association; not causal",
                }
            )
    return (
        _write(out_dir, "snapshot_voter_candidate", details, _DETAIL),
        _write(out_dir, "snapshot_voter_summary", voters, _VOTER),
        _write(out_dir, "snapshot_candidate_summary", candidate_summary, _CANDIDATE),
        _write(out_dir, "snapshot_rudeness_summary", rudeness_summary, _RUDENESS),
        _write(out_dir, "survivor_demographics", survivors, _SURVIVOR),
        _write(out_dir, "stated_preference_agreement", agreements, _AGREEMENT),
    )


def render_snapshot_figures(out_dir: Path) -> tuple[Path, ...]:
    """Render five grouped, labelled descriptive figures from snapshot tables."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    rudeness, survivor, agreement = (
        _read(out_dir, "snapshot_rudeness_summary"),
        _read(out_dir, "survivor_demographics"),
        _read(out_dir, "stated_preference_agreement"),
    )
    figures: list[tuple[str, Figure]] = []

    def grouped_lines(
        rows: Sequence[Row], fields: Sequence[str], title: str, y_label: str
    ) -> Figure:
        figure, axis = plt.subplots(figsize=(9, 4))
        for label in sorted(
            {f"{row['run_id']} / {row['rudeness_label']}" for row in rows}
        ):
            group = sorted(
                (
                    row
                    for row in rows
                    if f"{row['run_id']} / {row['rudeness_label']}" == label
                ),
                key=lambda row: _required_int(row, "snapshot_round"),
            )
            for field in fields:
                axis.plot(
                    [_required_int(row, "snapshot_round") for row in group],
                    [
                        _number(row[field]) if row[field] is not None else float("nan")
                        for row in group
                    ],
                    marker="o",
                    label=f"{label}: {field.removeprefix('mean_').replace('_', ' ')}",
                )
        axis.set(title=title, xlabel="Snapshot round", ylabel=y_label)
        if axis.lines:
            axis.legend(fontsize="x-small", ncol=2)
        else:
            axis.text(
                0.5,
                0.5,
                "No defined data",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        return figure

    figures.append(
        (
            "average_current_votes_credits.png",
            grouped_lines(
                rudeness,
                ("mean_current_votes", "mean_current_credits"),
                "Average current votes and credits by rudeness",
                "Average allocation",
            ),
        )
    )
    figures.append(
        (
            "cumulative_votes_credits_before_through.png",
            grouped_lines(
                rudeness,
                (
                    "mean_cumulative_before_votes",
                    "mean_cumulative_through_votes",
                    "mean_cumulative_before_credits",
                    "mean_cumulative_through_credits",
                ),
                "Cumulative votes and credits before versus through snapshot",
                "Average cumulative allocation",
            ),
        )
    )
    figure, axis = plt.subplots(figsize=(9, 4))
    for label in sorted(
        {f"{row['run_id']} / {row['rudeness_label']}" for row in survivor}
    ):
        group = sorted(
            (
                row
                for row in survivor
                if f"{row['run_id']} / {row['rudeness_label']}" == label
            ),
            key=lambda row: _required_int(row, "snapshot_round"),
        )
        axis.plot(
            [_required_int(row, "snapshot_round") for row in group],
            [
                len(
                    [
                        item
                        for item in group
                        if _required_int(item, "snapshot_round") == round_index
                    ]
                )
                for round_index in [
                    _required_int(item, "snapshot_round") for item in group
                ]
            ],
            marker="o",
            label=label,
        )
    axis.set(
        title="Survivor rudeness distribution by run and snapshot",
        xlabel="Snapshot round",
        ylabel="Surviving candidates",
    )
    if axis.lines:
        axis.legend(fontsize="x-small")
    figures.append(("survivor_rudeness_distribution.png", figure))
    figure, axis = plt.subplots(figsize=(9, 4))
    for run_id in sorted({str(row["run_id"]) for row in survivor}):
        group = sorted(
            (row for row in survivor if row["run_id"] == run_id),
            key=lambda row: _required_int(row, "snapshot_round"),
        )
        rounds = sorted({_required_int(row, "snapshot_round") for row in group})
        lengths_by_snapshot = {
            snapshot: [
                _number(row["total_two_turn_length"])
                for row in group
                if _required_int(row, "snapshot_round") == snapshot
                and row["total_two_turn_length"] is not None
            ]
            for snapshot in rounds
        }
        axis.plot(
            rounds,
            [
                sum(lengths_by_snapshot[snapshot]) / len(lengths_by_snapshot[snapshot])
                if lengths_by_snapshot[snapshot]
                else float("nan")
                for snapshot in rounds
            ],
            marker="o",
            label=run_id,
        )
    axis.set(
        title="Mean survivor first-two-turn length by run and snapshot",
        xlabel="Snapshot round",
        ylabel="Unicode characters",
    )
    if axis.lines:
        axis.legend(fontsize="x-small")
    figures.append(("survivor_message_lengths.png", figure))
    figure, axis = plt.subplots(figsize=(9, 4))
    for measure in ("signed_action", "signed_credit_spend"):
        for run_id in sorted({str(row["run_id"]) for row in agreement}):
            group = sorted(
                (
                    row
                    for row in agreement
                    if row["run_id"] == run_id
                    and row["measure"] == measure
                    and row["spearman_rho"] is not None
                ),
                key=lambda row: _required_int(row, "snapshot_round"),
            )
            if group:
                axis.plot(
                    [_required_int(row, "snapshot_round") for row in group],
                    [_number(row["spearman_rho"]) for row in group],
                    marker="o",
                    label=f"{run_id}: {measure.replace('_', ' ')}",
                )
    axis.set(
        title="Stated-preference agreement by measure and snapshot",
        xlabel="Snapshot round",
        ylabel="Spearman rho",
        ylim=(-1, 1),
    )
    if axis.lines:
        axis.legend(fontsize="x-small")
    else:
        axis.text(
            0.5,
            0.5,
            "No defined agreement",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    figures.append(("stated_preference_agreement.png", figure))
    paths = []
    for name, figure in figures:
        path = out_dir / name
        figure.tight_layout()
        figure.savefig(path, dpi=120, metadata={"Software": "quadratic-voting"})
        plt.close(figure)
        paths.append(path)
    return tuple(paths)
