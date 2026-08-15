"""Versioned, deterministic analysis of quadratic-voting export relations.

The only causal contrasts emitted here are round-one elicitation-order
contrasts.  Later-round candidate intersections are explicitly post-treatment
and descriptive.  Rudeness-label estimates are associations.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import struct
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from quadratic_voting.experiment.types import ElicitationArm, LikertRating, VotingRegime


ANALYSIS_VERSION = "qv-analysis/v1"
ANALYSIS_SEED_VERSION = "qv-seed/v1"
BOOTSTRAP_VERSION = "qv-paired-voter-bootstrap/v1"
ANALYSIS_MASTER_SEED = 0
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_CI_LEVEL = 0.95


class AgreementScope(StrEnum):
    OVERALL = "overall"
    RUDENESS_LABEL = "rudeness-label"


class AgreementNullReason(StrEnum):
    MISSING_STATEMENT = "MISSING_STATEMENT"
    ABSTAINED_BALLOT = "ABSTAINED_BALLOT"
    N_LT_2 = "N_LT_2"
    CONSTANT_RATING = "CONSTANT_RATING"
    CONSTANT_ACTION = "CONSTANT_ACTION"


class ClaimKind(StrEnum):
    ROUND_1_CAUSAL_ORDER = "round-1-causal-order"
    DESCRIPTIVE_POST_TREATMENT = "descriptive-post-treatment"
    RUDENESS_ASSOCIATION = "rudeness-association"


class ContrastMetric(StrEnum):
    MEAN_SIGNED_ACTION = "mean-signed-action"
    AGREEMENT_RHO = "agreement-rho"
    RUDENESS_MEAN_SIGNED_ACTION = "rudeness-mean-signed-action"
    RUDENESS_AGREEMENT_RHO = "rudeness-agreement-rho"


@dataclass(frozen=True, slots=True)
class AnalysisInputs:
    """Normalized relations required by the analysis boundary.

    Candidate metadata is selected by each run's immutable label-policy and
    presentation-template IDs; missing lineage is an error, never a fallback.
    ``round_candidate_rows`` is authoritative for active pools, including
    action-only and terminal-missing voter rounds.
    """

    runs: tuple[Mapping[str, object], ...]
    voters: tuple[Mapping[str, object], ...]
    rounds: tuple[Mapping[str, object], ...]
    round_candidate_rows: tuple[Mapping[str, object], ...]
    candidate_rows: tuple[Mapping[str, object], ...]
    source_annotation_rows: tuple[Mapping[str, object], ...]
    run_definition_rows: tuple[Mapping[str, object], ...]
    turns: tuple[Mapping[str, object], ...]
    calls: tuple[Mapping[str, object], ...]
    validation_failures: tuple[Mapping[str, object], ...]
    runtime_failures: tuple[Mapping[str, object], ...]
    ballots: tuple[Mapping[str, object], ...]
    allocations: tuple[Mapping[str, object], ...]
    statements: tuple[Mapping[str, object], ...]
    statement_items: tuple[Mapping[str, object], ...]
    outcomes: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class AnalysisOutputs:
    candidate_rows: tuple[dict[str, object], ...]
    agreement_cells: tuple[dict[str, object], ...]
    agreement_summaries: tuple[dict[str, object], ...]
    contrasts: tuple[dict[str, object], ...]
    bootstrap_replicates: tuple[dict[str, object], ...]
    candidate_survival: tuple[dict[str, object], ...]
    run_quality: tuple[dict[str, object], ...]
    round_trajectories: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _AnalysisIndex:
    runs: dict[str, Mapping[str, object]]
    rounds: dict[str, Mapping[str, object]]
    voters: dict[str, Mapping[str, object]]
    turns: dict[str, Mapping[str, object]]
    ballots: dict[str, Mapping[str, object]]
    statements: dict[str, Mapping[str, object]]
    candidates: dict[tuple[str, str, str], Mapping[str, object]]
    run_definitions: dict[str, Mapping[str, object]]
    source_annotations_json: dict[str, str]
    pools: dict[str, list[Mapping[str, object]]]
    turn_by_key: dict[tuple[str, str, str], Mapping[str, object]]
    allocations: dict[str, dict[str, int]]
    items: dict[str, dict[str, Mapping[str, object]]]
    calls_by_turn: dict[str, list[Mapping[str, object]]]
    validation_by_turn: Counter[str]
    runtime_by_turn: Counter[str]


_RATING_CODES = {rating.value: index - 2 for index, rating in enumerate(LikertRating)}
_NULL_REASONS = tuple(reason.value for reason in AgreementNullReason)


def midranks(values: Sequence[int | float]) -> tuple[float, ...]:
    """Return one-based average ranks; equal values receive equal midranks."""
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = rank
        start = end
    return tuple(ranks)


def spearman_with_ties(
    left: Sequence[int | float], right: Sequence[int | float]
) -> float | None:
    """Pearson correlation of midranks, or ``None`` when undefined."""
    if len(left) != len(right):
        raise ValueError(
            "cannot compute Spearman correlation in analysis.spearman_with_ties: "
            f"vector lengths differ ({len(left)} != {len(right)}); the caller must "
            "construct candidate-matched vectors"
        )
    if len(left) < 2 or len(set(left)) < 2 or len(set(right)) < 2:
        return None
    ranked_left, ranked_right = midranks(left), midranks(right)
    mean_left = sum(ranked_left) / len(ranked_left)
    mean_right = sum(ranked_right) / len(ranked_right)
    numerator = sum(
        (a - mean_left) * (b - mean_right)
        for a, b in zip(ranked_left, ranked_right, strict=True)
    )
    denominator = math.sqrt(
        sum((value - mean_left) ** 2 for value in ranked_left)
        * sum((value - mean_right) ** 2 for value in ranked_right)
    )
    return None if denominator == 0 else numerator / denominator


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(f"analysis expected an integer-compatible value, got {value!r}")
    return int(value)


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(f"analysis expected a numeric value, got {value!r}")
    return float(value)


def _lp(value: bytes) -> bytes:
    return struct.pack(">I", len(value)) + value


def analysis_seed(contrast_id: str) -> int:
    """Derive the fixed qv-seed/v1 analysis stream for one contrast."""
    payload = b"".join(
        (
            _lp(ANALYSIS_SEED_VERSION.encode()),
            _lp(b"analysis-bootstrap"),
            _lp(struct.pack(">Q", ANALYSIS_MASTER_SEED)),
            _lp(ANALYSIS_VERSION.encode()),
            _lp(contrast_id.encode()),
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("analysis percentile requires at least one finite replicate")
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def paired_cluster_bootstrap(
    values_by_voter: Mapping[int, float],
    *,
    contrast_id: str,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    """Bootstrap stable matched-set voter indices and preserve paired rows."""
    if replicates <= 0:
        raise ValueError(
            "analysis paired bootstrap requires a positive replicate count; "
            f"received {replicates} for {contrast_id}"
        )
    population = tuple(sorted(values_by_voter))
    seed = analysis_seed(contrast_id)
    if not population:
        summary: dict[str, object] = {
            "estimate": None,
            "ci_lower": None,
            "ci_upper": None,
            "n_clusters": 0,
            "bootstrap_replicates": replicates,
            "analysis_version": ANALYSIS_VERSION,
            "seed_version": ANALYSIS_SEED_VERSION,
            "bootstrap_version": BOOTSTRAP_VERSION,
            "analysis_seed": seed,
            "voter_population": [],
            "resample_sha256": hashlib.sha256(b"").hexdigest(),
            "ci_method": "percentile",
            "ci_level": DEFAULT_CI_LEVEL,
        }
        return summary, tuple()
    rng = random.Random(seed)
    digest = hashlib.sha256()
    replicate_rows: list[dict[str, object]] = []
    estimates: list[float] = []
    for replicate_index in range(replicates):
        sampled = tuple(population[rng.randrange(len(population))] for _ in population)
        for voter_index in sampled:
            digest.update(struct.pack(">Q", voter_index))
        estimate = sum(values_by_voter[index] for index in sampled) / len(sampled)
        estimates.append(estimate)
        replicate_rows.append(
            {
                "contrast_id": contrast_id,
                "replicate_index": replicate_index,
                "estimate": estimate,
                "sampled_voter_indices": list(sampled),
            }
        )
    estimates.sort()
    bootstrap_summary: dict[str, object] = {
        "estimate": sum(values_by_voter.values()) / len(values_by_voter),
        "ci_lower": _percentile(estimates, 0.025),
        "ci_upper": _percentile(estimates, 0.975),
        "n_clusters": len(population),
        "bootstrap_replicates": replicates,
        "analysis_version": ANALYSIS_VERSION,
        "seed_version": ANALYSIS_SEED_VERSION,
        "bootstrap_version": BOOTSTRAP_VERSION,
        "analysis_seed": seed,
        "voter_population": list(population),
        "resample_sha256": digest.hexdigest(),
        "ci_method": "percentile",
        "ci_level": DEFAULT_CI_LEVEL,
    }
    return bootstrap_summary, tuple(replicate_rows)


def _index(inputs: AnalysisInputs) -> _AnalysisIndex:
    runs = {str(row["run_id"]): row for row in inputs.runs}
    rounds = {str(row["round_id"]): row for row in inputs.rounds}
    voters = {str(row["voter_id"]): row for row in inputs.voters}
    turns = {str(row["turn_id"]): row for row in inputs.turns}
    ballots = {str(row["turn_id"]): row for row in inputs.ballots}
    statements = {str(row["turn_id"]): row for row in inputs.statements}
    candidates = {
        (
            str(row["candidate_id"]),
            str(row["label_policy_id"]),
            str(row["template_id"]),
        ): row
        for row in inputs.candidate_rows
    }
    run_definitions = {str(row["run_id"]): row for row in inputs.run_definition_rows}
    annotations: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in inputs.source_annotation_rows:
        annotations[str(row["candidate_id"])].append(row)
    source_annotations_json = {
        candidate_id: json.dumps(
            sorted(
                rows,
                key=lambda row: (
                    _as_int(row["annotation_index"]),
                    str(row["annotator_hash"]),
                ),
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        for candidate_id, rows in annotations.items()
    }
    pools: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in inputs.round_candidate_rows:
        pools[str(row["round_id"])].append(row)
    for rows in pools.values():
        rows.sort(
            key=lambda row: (_as_int(row["sample_position"]), str(row["candidate_id"]))
        )
    turn_by_key = {
        (str(row["round_id"]), str(row["voter_id"]), str(row["kind"])): row
        for row in inputs.turns
    }
    allocations: dict[str, dict[str, int]] = defaultdict(dict)
    for row in inputs.allocations:
        allocations[str(row["ballot_id"])][str(row["candidate_id"])] = _as_int(
            row["votes"]
        )
    items: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in inputs.statement_items:
        items[str(row["statement_id"])][str(row["candidate_id"])] = row
    calls_by_turn: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    call_turn: dict[str, str] = {}
    for row in inputs.calls:
        turn_id = str(row["turn_id"])
        calls_by_turn[turn_id].append(row)
        call_turn[str(row["call_id"])] = turn_id
    validation_by_turn: Counter[str] = Counter()
    for row in inputs.validation_failures:
        validation_by_turn[call_turn[str(row["call_id"])]] += 1
    runtime_by_turn: Counter[str] = Counter()
    for row in inputs.runtime_failures:
        runtime_by_turn[call_turn[str(row["call_id"])]] += 1
    return _AnalysisIndex(
        runs,
        rounds,
        voters,
        turns,
        ballots,
        statements,
        candidates,
        run_definitions,
        source_annotations_json,
        dict(pools),
        turn_by_key,
        dict(allocations),
        dict(items),
        dict(calls_by_turn),
        validation_by_turn,
        runtime_by_turn,
    )


def _intersection_by_round(inputs: AnalysisInputs) -> dict[str, set[str]]:
    runs_by_matched: dict[str, set[str]] = defaultdict(set)
    for run in inputs.runs:
        runs_by_matched[str(run["matched_set_id"])].add(str(run["run_id"]))
    pool_by_coordinate: dict[tuple[str, int], set[str]] = defaultdict(set)
    round_coordinates = {
        str(row["round_id"]): (str(row["run_id"]), _as_int(row["round_index"]))
        for row in inputs.rounds
    }
    for row in inputs.round_candidate_rows:
        run_id, round_index = round_coordinates[str(row["round_id"])]
        pool_by_coordinate[(run_id, round_index)].add(str(row["candidate_id"]))
    result: dict[str, set[str]] = {}
    runs = {str(row["run_id"]): row for row in inputs.runs}
    for round_id, (run_id, round_index) in round_coordinates.items():
        matched_id = str(runs[run_id]["matched_set_id"])
        available = [
            pool_by_coordinate[(other, round_index)]
            for other in sorted(runs_by_matched[matched_id])
            if (other, round_index) in pool_by_coordinate
        ]
        result[round_id] = set.intersection(*available) if available else set()
    return result


def _candidate_rows(
    inputs: AnalysisInputs, indexed: _AnalysisIndex
) -> list[dict[str, object]]:
    runs, pools, candidates = indexed.runs, indexed.pools, indexed.candidates
    turn_by_key, ballots, statements = (
        indexed.turn_by_key,
        indexed.ballots,
        indexed.statements,
    )
    allocations, items = indexed.allocations, indexed.items
    calls_by_turn = indexed.calls_by_turn
    validation_by_turn, runtime_by_turn = (
        indexed.validation_by_turn,
        indexed.runtime_by_turn,
    )
    intersections = _intersection_by_round(inputs)
    voters_by_run: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for voter in inputs.voters:
        voters_by_run[str(voter["run_id"])].append(voter)
    result: list[dict[str, object]] = []
    for round_row in sorted(
        inputs.rounds, key=lambda row: (str(row["run_id"]), _as_int(row["round_index"]))
    ):
        round_id, run_id = str(round_row["round_id"]), str(round_row["run_id"])
        run = runs[run_id]
        definition = indexed.run_definitions.get(run_id)
        if definition is None:
            raise RuntimeError(
                f"analysis candidate construction failed because run {run_id} has no exported "
                "immutable model definition. Lineage selection stopped in analysis._candidate_rows "
                "before producing candidate values, so labels or presentations were not guessed. "
                "Expose this run through ExportStore.run_definition_rows() and retry export."
            )
        pool = pools.get(round_id, [])
        for voter in sorted(
            voters_by_run[run_id], key=lambda row: _as_int(row["voter_index"])
        ):
            voter_id = str(voter["voter_id"])
            ballot_turn = turn_by_key.get((round_id, voter_id, "ballot"))
            statement_turn = turn_by_key.get((round_id, voter_id, "statement"))
            ballot = (
                None
                if ballot_turn is None
                else ballots.get(str(ballot_turn["turn_id"]))
            )
            statement = (
                None
                if statement_turn is None
                else statements.get(str(statement_turn["turn_id"]))
            )
            # Pending rounds do not create terminal observations.
            if ballot is None:
                continue
            assert ballot_turn is not None
            action_only = str(run["arm"]) == ElicitationArm.ACTION_ONLY.value
            if not action_only and statement is None:
                continue
            ballot_status = str(ballot["status"])
            statement_status = (
                "not-applicable"
                if action_only
                else str(statement["status"] if statement is not None else "")
            )
            missing_reason: str | None = None
            if not action_only and statement_status != "accepted":
                missing_reason = AgreementNullReason.MISSING_STATEMENT.value
            elif ballot_status != "accepted":
                missing_reason = AgreementNullReason.ABSTAINED_BALLOT.value
            ballot_id = str(ballot["ballot_id"])
            statement_id = None if statement is None else str(statement["statement_id"])
            sign = 1 if str(run["regime"]) == VotingRegime.SUPPORT.value else -1
            ballot_turn_id = str(ballot_turn["turn_id"])
            statement_turn_id = (
                None if statement_turn is None else str(statement_turn["turn_id"])
            )
            candidate_votes = allocations.get(ballot_id, {})
            statement_values = (
                {} if statement_id is None else items.get(statement_id, {})
            )
            for pool_row in pool:
                candidate_id = str(pool_row["candidate_id"])
                metadata_key = (
                    candidate_id,
                    str(definition["label_policy_id"]),
                    str(definition["presentation_template_id"]),
                )
                metadata = candidates.get(metadata_key)
                if metadata is None:
                    raise RuntimeError(
                        "analysis candidate construction failed because candidate "
                        f"{candidate_id} in run {run_id} has no metadata row for label policy "
                        f"{metadata_key[1]} and presentation template {metadata_key[2]}. "
                        "The mismatch was detected in analysis._candidate_rows before output, "
                        "so no unversioned label or presentation fallback was emitted. Ensure "
                        "candidate_rows() exports the run-selected normalized lineage and retry."
                    )
                item = statement_values.get(candidate_id)
                rating = None if item is None else str(item["rating"])
                raw_votes = _as_int(candidate_votes.get(candidate_id, 0))
                values_available = missing_reason is None and not action_only
                action_available = ballot_status == "accepted" and (
                    action_only or missing_reason is None
                )
                result.append(
                    {
                        "matched_set_id": run["matched_set_id"],
                        "run_id": run_id,
                        "regime": run["regime"],
                        "arm": run["arm"],
                        "voter_index": _as_int(voter["voter_index"]),
                        "round_index": _as_int(round_row["round_index"]),
                        "candidate_id": candidate_id,
                        "rating_code": _RATING_CODES[rating]
                        if values_available and rating is not None
                        else None,
                        "statement_text": item.get("text")
                        if values_available and item is not None
                        else None,
                        "raw_votes": raw_votes if action_available else None,
                        "signed_action": sign * raw_votes if action_available else None,
                        "rudeness_label": metadata["rudeness_label"],
                        "label_policy_id": metadata["label_policy_id"],
                        "label_policy_name": metadata["label_policy_name"],
                        "label_policy_version": metadata["label_policy_version"],
                        "label_policy_sha256": metadata["label_policy_sha256"],
                        "source_annotations_json": indexed.source_annotations_json.get(
                            candidate_id, "[]"
                        ),
                        "presentation_id": metadata["presentation_id"],
                        "presentation_template_id": metadata["template_id"],
                        "presentation_template_name": metadata[
                            "presentation_template_name"
                        ],
                        "presentation_template_version": metadata[
                            "presentation_template_version"
                        ],
                        "presentation_template_sha256": metadata[
                            "presentation_template_sha256"
                        ],
                        "presentation_sha256": metadata["presentation_sha256"],
                        "statement_status": statement_status,
                        "ballot_status": ballot_status,
                        "missing_reason": "NOT_APPLICABLE_ACTION_ONLY"
                        if action_only
                        else missing_reason,
                        "statement_retry_count": 0
                        if statement_turn_id is None
                        else sum(
                            _as_int(call["attempt_index"]) > 0
                            for call in calls_by_turn.get(statement_turn_id, [])
                        ),
                        "ballot_retry_count": sum(
                            _as_int(call["attempt_index"]) > 0
                            for call in calls_by_turn.get(ballot_turn_id, [])
                        ),
                        "statement_validation_failure_count": 0
                        if statement_turn_id is None
                        else validation_by_turn[statement_turn_id],
                        "ballot_validation_failure_count": validation_by_turn[
                            ballot_turn_id
                        ],
                        "statement_runtime_failure_count": 0
                        if statement_turn_id is None
                        else runtime_by_turn[statement_turn_id],
                        "ballot_runtime_failure_count": runtime_by_turn[ballot_turn_id],
                        "sample_position": _as_int(pool_row["sample_position"]),
                        "active_pool_size": len(pool),
                        "intersection_pool_size": len(intersections[round_id]),
                        "in_all_run_intersection": candidate_id
                        in intersections[round_id],
                        "post_treatment_intersection": _as_int(round_row["round_index"])
                        > 1,
                    }
                )
    return result


def _agreement_cells(
    candidate_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in candidate_rows:
        if row["arm"] == ElicitationArm.ACTION_ONLY.value:
            continue
        base = (
            row["matched_set_id"],
            row["run_id"],
            row["arm"],
            row["regime"],
            row["round_index"],
            row["voter_index"],
        )
        groups[(*base, AgreementScope.OVERALL.value, None)].append(row)
        groups[
            (*base, AgreementScope.RUDENESS_LABEL.value, row["rudeness_label"])
        ].append(row)
    result: list[dict[str, object]] = []
    for group_key, rows in sorted(
        groups.items(),
        key=lambda item: tuple(
            "" if value is None else str(value) for value in item[0]
        ),
    ):
        missing_statement = any(row["statement_status"] != "accepted" for row in rows)
        abstained = any(row["ballot_status"] != "accepted" for row in rows)
        pairs = [
            row
            for row in rows
            if row["rating_code"] is not None and row["signed_action"] is not None
        ]
        ratings = [_as_int(row["rating_code"]) for row in pairs]
        actions = [_as_int(row["signed_action"]) for row in pairs]
        reason: AgreementNullReason | None = None
        if missing_statement:
            reason = AgreementNullReason.MISSING_STATEMENT
        elif abstained:
            reason = AgreementNullReason.ABSTAINED_BALLOT
        elif len(pairs) < 2:
            reason = AgreementNullReason.N_LT_2
        elif len(set(ratings)) < 2:
            reason = AgreementNullReason.CONSTANT_RATING
        elif len(set(actions)) < 2:
            reason = AgreementNullReason.CONSTANT_ACTION
        rho = None if reason is not None else spearman_with_ties(ratings, actions)
        result.append(
            {
                "matched_set_id": group_key[0],
                "run_id": group_key[1],
                "arm": group_key[2],
                "regime": group_key[3],
                "round_index": group_key[4],
                "voter_index": group_key[5],
                "scope": group_key[6],
                "rudeness_label": group_key[7],
                "spearman_rho": rho,
                "null_reason": None if reason is None else reason.value,
                "n_candidate_pairs": len(pairs),
                "n_eligible_candidates": len(rows),
                "active_pool_size": max(
                    _as_int(row["active_pool_size"]) for row in rows
                ),
                "intersection_pool_size": max(
                    _as_int(row["intersection_pool_size"]) for row in rows
                ),
                "label_policy_version": rows[0]["label_policy_version"],
                "label_policy_id": rows[0]["label_policy_id"],
                "label_policy_sha256": rows[0]["label_policy_sha256"],
            }
        )
    return result


def _agreement_summaries(
    cells: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in cells:
        key = (
            row["matched_set_id"],
            row["arm"],
            row["regime"],
            row["round_index"],
            row["scope"],
            row["rudeness_label"],
        )
        groups[key].append(row)
    result: list[dict[str, object]] = []
    for group_key, rows in sorted(
        groups.items(),
        key=lambda item: tuple(
            "" if value is None else str(value) for value in item[0]
        ),
    ):
        defined = sorted(
            _as_float(row["spearman_rho"])
            for row in rows
            if row["spearman_rho"] is not None
        )
        counts = Counter(
            str(row["null_reason"]) for row in rows if row["null_reason"] is not None
        )
        middle = len(defined) // 2
        median = (
            None
            if not defined
            else (
                defined[middle]
                if len(defined) % 2
                else (defined[middle - 1] + defined[middle]) / 2
            )
        )
        output = {
            "matched_set_id": group_key[0],
            "arm": group_key[1],
            "regime": group_key[2],
            "round_index": group_key[3],
            "scope": group_key[4],
            "rudeness_label": group_key[5],
            "mean_spearman_rho": None if not defined else sum(defined) / len(defined),
            "median_spearman_rho": median,
            "n_defined_cells": len(defined),
            "n_total_eligible_cells": len(rows),
            "n_candidate_pairs": sum(_as_int(row["n_candidate_pairs"]) for row in rows),
            "active_pool_size": max(_as_int(row["active_pool_size"]) for row in rows),
            "intersection_pool_size": max(
                _as_int(row["intersection_pool_size"]) for row in rows
            ),
            "label_policy_version": rows[0]["label_policy_version"],
            "label_policy_id": rows[0]["label_policy_id"],
            "label_policy_sha256": rows[0]["label_policy_sha256"],
            "estimand_language": "rudeness association"
            if group_key[4] == AgreementScope.RUDENESS_LABEL.value
            else "preference-action association",
        }
        for reason in _NULL_REASONS:
            output[f"n_null_{reason.lower()}"] = counts[reason]
        result.append(output)
    return result


def _contrasts(
    candidate_rows: Sequence[Mapping[str, object]],
    cells: Sequence[Mapping[str, object]],
    *,
    bootstrap_replicates: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    del cells  # Contrasts must be recomputed on their exact compared candidate sets.
    grouped: dict[tuple[str, str, int, str, int], list[Mapping[str, object]]] = (
        defaultdict(list)
    )
    for row in candidate_rows:
        grouped[
            (
                str(row["matched_set_id"]),
                str(row["regime"]),
                _as_int(row["round_index"]),
                str(row["arm"]),
                _as_int(row["voter_index"]),
            )
        ].append(row)

    def complete_mean(
        rows: Sequence[Mapping[str, object]],
        candidates: set[str],
        label: str | None = None,
    ) -> float | None:
        selected = [
            row
            for row in rows
            if str(row["candidate_id"]) in candidates
            and (label is None or row["rudeness_label"] == label)
        ]
        expected = {
            candidate
            for candidate in candidates
            if label is None
            or any(
                str(row["candidate_id"]) == candidate and row["rudeness_label"] == label
                for row in rows
            )
        }
        if (
            not expected
            or len(selected) != len(expected)
            or any(row["signed_action"] is None for row in selected)
        ):
            return None
        return sum(_as_float(row["signed_action"]) for row in selected) / len(selected)

    def complete_rho(
        rows: Sequence[Mapping[str, object]],
        candidates: set[str],
        label: str | None = None,
    ) -> float | None:
        selected = [
            row
            for row in rows
            if str(row["candidate_id"]) in candidates
            and (label is None or row["rudeness_label"] == label)
        ]
        if len(selected) < 2 or any(
            row["rating_code"] is None or row["signed_action"] is None
            for row in selected
        ):
            return None
        return spearman_with_ties(
            [_as_int(row["rating_code"]) for row in selected],
            [_as_int(row["signed_action"]) for row in selected],
        )

    specs: list[dict[str, object]] = []
    arms = tuple(arm.value for arm in ElicitationArm)
    coordinates = sorted({key[:3] for key in grouped})
    for matched, regime, round_index in coordinates:
        pools = {
            arm: {
                str(row["candidate_id"])
                for key, rows in grouped.items()
                if key[:4] == (matched, regime, round_index, arm)
                for row in rows
            }
            for arm in arms
        }
        for left_index, left_arm in enumerate(arms):
            for right_arm in arms[left_index + 1 :]:
                intersection = pools[left_arm] & pools[right_arm]
                left = {
                    key[4]: value
                    for key, rows in grouped.items()
                    if key[:4] == (matched, regime, round_index, left_arm)
                    and (value := complete_mean(rows, intersection)) is not None
                }
                right = {
                    key[4]: value
                    for key, rows in grouped.items()
                    if key[:4] == (matched, regime, round_index, right_arm)
                    and (value := complete_mean(rows, intersection)) is not None
                }
                values = {
                    voter: left[voter] - right[voter]
                    for voter in sorted(left.keys() & right.keys())
                }
                if values:
                    specs.append(
                        {
                            "matched": matched,
                            "regime": regime,
                            "round": round_index,
                            "left": left_arm,
                            "right": right_arm,
                            "metric": ContrastMetric.MEAN_SIGNED_ACTION,
                            "values": values,
                            "original": max(
                                len(pools[left_arm]), len(pools[right_arm])
                            ),
                            "intersection": len(intersection),
                            "kind": "elicitation-order",
                            "language": "round-1 causal elicitation-order contrast"
                            if round_index == 1
                            else "descriptive post-treatment candidate-intersection contrast",
                        }
                    )
        joint = (
            ElicitationArm.STATEMENT_THEN_ACTION.value,
            ElicitationArm.ACTION_THEN_STATEMENT.value,
        )
        intersection = pools[joint[0]] & pools[joint[1]]
        left = {
            key[4]: value
            for key, rows in grouped.items()
            if key[:4] == (matched, regime, round_index, joint[0])
            and (value := complete_rho(rows, intersection)) is not None
        }
        right = {
            key[4]: value
            for key, rows in grouped.items()
            if key[:4] == (matched, regime, round_index, joint[1])
            and (value := complete_rho(rows, intersection)) is not None
        }
        values = {
            voter: left[voter] - right[voter]
            for voter in sorted(left.keys() & right.keys())
        }
        if values:
            specs.append(
                {
                    "matched": matched,
                    "regime": regime,
                    "round": round_index,
                    "left": joint[0],
                    "right": joint[1],
                    "metric": ContrastMetric.AGREEMENT_RHO,
                    "values": values,
                    "original": max(len(pools[joint[0]]), len(pools[joint[1]])),
                    "intersection": len(intersection),
                    "kind": "elicitation-order",
                    "language": "round-1 causal elicitation-order agreement contrast"
                    if round_index == 1
                    else "descriptive post-treatment candidate-intersection agreement contrast",
                }
            )

        for arm in arms:
            arm_rows = [
                (key, rows)
                for key, rows in grouped.items()
                if key[:4] == (matched, regime, round_index, arm)
            ]
            rude = {
                key[4]: value
                for key, rows in arm_rows
                if (value := complete_mean(rows, pools[arm], "rude")) is not None
            }
            non_rude = {
                key[4]: value
                for key, rows in arm_rows
                if (value := complete_mean(rows, pools[arm], "non_rude")) is not None
            }
            label_values = {
                voter: rude[voter] - non_rude[voter]
                for voter in sorted(rude.keys() & non_rude.keys())
            }
            if label_values:
                specs.append(
                    {
                        "matched": matched,
                        "regime": regime,
                        "round": round_index,
                        "left": "rude",
                        "right": "non_rude",
                        "metric": ContrastMetric.RUDENESS_MEAN_SIGNED_ACTION,
                        "values": label_values,
                        "original": len(pools[arm]),
                        "intersection": len(pools[arm]),
                        "kind": f"rudeness-label:{arm}",
                        "language": "associational rudeness-label action contrast; labels are not randomized",
                    }
                )
            rude_rho = {
                key[4]: value
                for key, rows in arm_rows
                if (value := complete_rho(rows, pools[arm], "rude")) is not None
            }
            non_rude_rho = {
                key[4]: value
                for key, rows in arm_rows
                if (value := complete_rho(rows, pools[arm], "non_rude")) is not None
            }
            rho_values = {
                voter: rude_rho[voter] - non_rude_rho[voter]
                for voter in sorted(rude_rho.keys() & non_rude_rho.keys())
            }
            if rho_values:
                specs.append(
                    {
                        "matched": matched,
                        "regime": regime,
                        "round": round_index,
                        "left": "rude",
                        "right": "non_rude",
                        "metric": ContrastMetric.RUDENESS_AGREEMENT_RHO,
                        "values": rho_values,
                        "original": len(pools[arm]),
                        "intersection": len(pools[arm]),
                        "kind": f"rudeness-label:{arm}",
                        "language": "associational rudeness-label agreement contrast; labels are not randomized",
                    }
                )
    contrasts: list[dict[str, object]] = []
    replicate_rows: list[dict[str, object]] = []
    for spec in specs:
        matched, regime, round_index = (
            str(spec["matched"]),
            str(spec["regime"]),
            _as_int(spec["round"]),
        )
        left_arm, right_arm, metric = (
            str(spec["left"]),
            str(spec["right"]),
            spec["metric"],
        )
        contrast_values = cast(dict[int, float], spec["values"])
        assert isinstance(metric, ContrastMetric)
        contrast_id = f"{matched}|{regime}|r{round_index}|{metric.value}|{left_arm}-minus-{right_arm}"
        bootstrap, replicates = paired_cluster_bootstrap(
            contrast_values, contrast_id=contrast_id, replicates=bootstrap_replicates
        )
        order_contrast = spec["kind"] == "elicitation-order"
        claim = (
            ClaimKind.RUDENESS_ASSOCIATION
            if not order_contrast
            else (
                ClaimKind.ROUND_1_CAUSAL_ORDER
                if round_index == 1
                else ClaimKind.DESCRIPTIVE_POST_TREATMENT
            )
        )
        contrasts.append(
            {
                "contrast_id": contrast_id,
                "matched_set_id": matched,
                "regime": regime,
                "round_index": round_index,
                "metric": metric.value,
                "left_arm": left_arm,
                "right_arm": right_arm,
                "claim_kind": claim.value,
                "causal": order_contrast and round_index == 1,
                "post_treatment": order_contrast and round_index > 1,
                "original_pool_size": spec["original"],
                "intersection_pool_size": spec["intersection"],
                "contrast_kind": spec["kind"],
                "estimand_language": spec["language"],
                **bootstrap,
            }
        )
        replicate_rows.extend(replicates)
    return contrasts, replicate_rows


def _survival_and_trajectories(
    inputs: AnalysisInputs, indexed: _AnalysisIndex
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    runs, rounds, turns, allocations = (
        indexed.runs,
        indexed.rounds,
        indexed.turns,
        indexed.allocations,
    )
    totals: dict[tuple[str, int], Counter[str]] = defaultdict(Counter)
    for ballot in inputs.ballots:
        turn = turns[str(ballot["turn_id"])]
        round_row = rounds[str(turn["round_id"])]
        key = (str(round_row["run_id"]), _as_int(round_row["round_index"]))
        for candidate, votes in allocations.get(str(ballot["ballot_id"]), {}).items():
            totals[key][candidate] += votes
    outcomes = {
        (
            str(rounds[str(row["round_id"])]["run_id"]),
            _as_int(rounds[str(row["round_id"])]["round_index"]),
        ): row
        for row in inputs.outcomes
    }
    pools: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in inputs.round_candidate_rows:
        pools[(str(row["run_id"]), _as_int(row["round_index"]))].append(row)
    survival: list[dict[str, object]] = []
    trajectories: list[dict[str, object]] = []
    for run_id, run in sorted(runs.items()):
        indices = sorted(index for rid, index in pools if rid == run_id)
        candidates = sorted(
            {
                str(row["candidate_id"])
                for key, rows in pools.items()
                if key[0] == run_id
                for row in rows
            }
        )
        removed = {
            str(row["removed_candidate_id"]): index
            for (rid, index), row in outcomes.items()
            if rid == run_id
        }
        for index in indices:
            values = totals[(run_id, index)]
            trajectories.append(
                {
                    "run_id": run_id,
                    "arm": run["arm"],
                    "regime": run["regime"],
                    "round_index": index,
                    "active_pool_size": len(pools[(run_id, index)]),
                    "total_votes": sum(values.values()),
                    "max_votes": max(values.values(), default=0),
                    "distinct_supported_candidates": sum(
                        value > 0 for value in values.values()
                    ),
                }
            )
        final_round = max(indices, default=0)
        for candidate in candidates:
            removed_round = removed.get(candidate)
            active_indices = [
                index
                for index in indices
                if any(
                    str(row["candidate_id"]) == candidate
                    for row in pools[(run_id, index)]
                )
            ]
            survival.append(
                {
                    "run_id": run_id,
                    "arm": run["arm"],
                    "regime": run["regime"],
                    "candidate_id": candidate,
                    "round_indices": active_indices,
                    "votes_by_round": [
                        totals[(run_id, index)][candidate] for index in active_indices
                    ],
                    "protected_rounds": [
                        index
                        for (rid, index), row in outcomes.items()
                        if rid == run_id and row["protected_candidate_id"] == candidate
                    ],
                    "removed_round": removed_round,
                    "survival_round": final_round
                    if removed_round is None
                    else removed_round,
                    "winner": bool(
                        run["status"] == "complete" and removed_round is None
                    ),
                }
            )
    return survival, trajectories


def _quality(
    inputs: AnalysisInputs, indexed: _AnalysisIndex
) -> list[dict[str, object]]:
    runs, rounds, turns = indexed.runs, indexed.rounds, indexed.turns
    call_run: dict[str, str] = {}
    calls: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for call in inputs.calls:
        run_id = str(rounds[str(turns[str(call["turn_id"])]["round_id"])]["run_id"])
        call_run[str(call["call_id"])] = run_id
        calls[run_id].append(call)
    errors: dict[str, Counter[str]] = defaultdict(Counter)
    for row in inputs.validation_failures:
        errors[call_run[str(row["call_id"])]][str(row["error_code"])] += 1
    runtime = Counter(call_run[str(row["call_id"])] for row in inputs.runtime_failures)
    ballot_run = {
        str(row["turn_id"]): str(
            rounds[str(turns[str(row["turn_id"])]["round_id"])]["run_id"]
        )
        for row in inputs.ballots
    }
    statement_run = {
        str(row["turn_id"]): str(
            rounds[str(turns[str(row["turn_id"])]["round_id"])]["run_id"]
        )
        for row in inputs.statements
    }
    result: list[dict[str, object]] = []
    for run_id, run in sorted(runs.items()):
        run_calls = calls[run_id]
        invalid_call_ids = {str(row["call_id"]) for row in inputs.validation_failures}
        result.append(
            {
                "run_id": run_id,
                "arm": run["arm"],
                "regime": run["regime"],
                "invalid_attempts": sum(
                    str(call["call_id"]) in invalid_call_ids for call in run_calls
                ),
                "invalid_attempts_by_error_code": sorted(errors[run_id].items()),
                "correction_attempts": sum(
                    _as_int(call["attempt_index"]) > 0 for call in run_calls
                ),
                "retry_invocations": sum(
                    _as_int(call["invocation_index"]) > 0 for call in run_calls
                ),
                "abstentions": sum(
                    row["status"] == "abstained"
                    and ballot_run[str(row["turn_id"])] == run_id
                    for row in inputs.ballots
                ),
                "invalid_missing_statements": sum(
                    row["status"] == "invalid-missing"
                    and statement_run[str(row["turn_id"])] == run_id
                    for row in inputs.statements
                ),
                "runtime_failures": runtime[run_id],
                "interruptions": sum(
                    call["status"] == "interrupted" for call in run_calls
                ),
                "prompt_tokens": sum(
                    _as_int(call.get("prompt_token_count") or 0) for call in run_calls
                ),
                "completion_tokens": sum(
                    _as_int(call.get("completion_token_count") or 0)
                    for call in run_calls
                ),
                "total_duration_ms": sum(
                    _as_int(call.get("duration_ms") or 0) for call in run_calls
                ),
            }
        )
    return result


def analyze(
    inputs: AnalysisInputs, *, bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES
) -> AnalysisOutputs:
    """Execute the complete deterministic analysis contract."""
    indexed = _index(inputs)
    candidates = _candidate_rows(inputs, indexed)
    cells = _agreement_cells(candidates)
    summaries = _agreement_summaries(cells)
    contrasts, replicates = _contrasts(
        candidates, cells, bootstrap_replicates=bootstrap_replicates
    )
    survival, trajectories = _survival_and_trajectories(inputs, indexed)
    return AnalysisOutputs(
        tuple(candidates),
        tuple(cells),
        tuple(summaries),
        tuple(contrasts),
        tuple(replicates),
        tuple(survival),
        tuple(_quality(inputs, indexed)),
        tuple(trajectories),
    )
