"""Pure round aggregation for support and opposition voting regimes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from quadratic_voting.experiment.seeds import DrawSelection, SeededDraw
from quadratic_voting.experiment.types import CandidateId, RoundResult, VotingRegime


@dataclass(frozen=True, slots=True)
class AggregationOutcome:
    result: RoundResult
    tie_draw: DrawSelection | None
    removal_draw: DrawSelection | None


def aggregate_round(
    regime: VotingRegime,
    active: Sequence[CandidateId],
    accepted_allocations: Sequence[Mapping[CandidateId, int]],
    tie_draw: SeededDraw,
    removal_draw: SeededDraw,
) -> AggregationOutcome:
    """Aggregate accepted allocations using only injected named draws."""
    ordered_active = tuple(active)
    if not ordered_active:
        raise ValueError(
            "Round aggregation failed because the active snapshot is empty. Validation "
            "failed in quadratic_voting.experiment.engine.aggregate_round before totals "
            "were computed, so no candidate can be protected or removed. Supply the "
            "persisted non-empty round snapshot and retry."
        )
    if len(set(ordered_active)) != len(ordered_active):
        raise ValueError(
            "Round aggregation failed because the active snapshot contains duplicate "
            "candidate IDs. Validation failed in "
            "quadratic_voting.experiment.engine.aggregate_round before drawing an "
            "outcome, so persisted draw populations would be ambiguous. Supply each "
            "active candidate exactly once in frozen sample order and retry."
        )
    active_set = set(ordered_active)
    totals = {candidate: 0 for candidate in ordered_active}
    for ballot_index, ballot in enumerate(accepted_allocations):
        for candidate, votes in ballot.items():
            if candidate not in active_set:
                raise ValueError(
                    "Round aggregation failed because accepted ballot "
                    f"{ballot_index} contains non-active candidate {candidate!s}. "
                    "Validation failed in quadratic_voting.experiment.engine.aggregate_round "
                    "while computing totals, so the outcome cannot be trusted. Validate "
                    "ballots against the frozen active snapshot before aggregation and retry."
                )
            if type(votes) is not int or votes < 0:
                raise ValueError(
                    "Round aggregation failed because accepted ballot "
                    f"{ballot_index} has invalid votes={votes!r} for candidate {candidate!s}. "
                    "Validation failed in quadratic_voting.experiment.engine.aggregate_round "
                    "while computing totals, so the outcome cannot be trusted. Supply a "
                    "validated non-negative JSON integer allocation and retry."
                )
            totals[candidate] += votes

    maximum = max(totals.values())
    maxima = tuple(
        candidate for candidate in ordered_active if totals[candidate] == maximum
    )
    performed_tie_draw = tie_draw.choose(maxima) if len(maxima) > 1 else None
    selected = performed_tie_draw.selected if performed_tie_draw else maxima[0]
    tie_among = frozenset(maxima) if len(maxima) > 1 else frozenset()

    performed_removal_draw: DrawSelection | None = None
    if regime is VotingRegime.SUPPORT:
        removal_population = tuple(
            candidate for candidate in ordered_active if candidate != selected
        )
        if not removal_population:
            raise ValueError(
                "Support round aggregation failed because protecting the sole active "
                f"candidate {selected!s} leaves no removal population. Validation failed "
                "in quadratic_voting.experiment.engine.aggregate_round during support "
                "removal, so the round cannot eliminate a candidate. Aggregate support "
                "only while at least two candidates are active; otherwise seal the prior "
                "round as the terminal result."
            )
        performed_removal_draw = removal_draw.choose(removal_population)
        protected: CandidateId | None = selected
        removed = performed_removal_draw.selected
    elif regime is VotingRegime.OPPOSITION:
        protected = None
        removed = selected
    else:
        raise AssertionError(f"unhandled closed voting regime: {regime!r}")

    result = RoundResult(
        protected=protected,
        removed=removed,
        totals=MappingProxyType(totals),
        tie_among=tie_among,
    )
    return AggregationOutcome(result, performed_tie_draw, performed_removal_draw)
