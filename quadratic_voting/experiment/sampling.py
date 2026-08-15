"""Deterministic balanced sampling over immutable catalog candidates."""

from __future__ import annotations

import random
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, cast

from quadratic_voting.experiment.seeds import balanced_extra_stratum_draw
from quadratic_voting.experiment.store import ExperimentStore, SampleExtraDraw
from quadratic_voting.experiment.types import (
    CandidateId,
    ReleaseId,
    RudenessLabel,
    SampleId,
    SamplerPolicy,
    TemplateId,
)


class _SqliteStoreView(Protocol):
    connection: sqlite3.Connection


CandidateRow = Mapping[str, object]


def candidates_by_label(
    store_or_rows: ExperimentStore | Iterable[CandidateRow],
    release_id: ReleaseId | None = None,
) -> Mapping[RudenessLabel, tuple[CandidateId, ...]]:
    """Group candidate IDs by label in stable order.

    SQLite stores use the normalized candidate table directly because the final
    Store API exposes no catalog export dataset. Iterable rows may instead
    provide ``candidate_id``, ``rudeness_label``, and optional ``release_id``.
    """
    if hasattr(store_or_rows, "connection"):
        connection = cast(_SqliteStoreView, store_or_rows).connection
        if release_id is None:
            rows: Iterable[CandidateRow] = (
                dict(row)
                for row in connection.execute(
                    "SELECT c.candidate_id,l.rudeness_label,c.release_id FROM candidate c "
                    "JOIN candidate_label l ON l.candidate_id=c.candidate_id "
                    "ORDER BY c.candidate_id"
                )
            )
        else:
            rows = (
                dict(row)
                for row in connection.execute(
                    "SELECT c.candidate_id,l.rudeness_label,c.release_id FROM candidate c "
                    "JOIN candidate_label l ON l.candidate_id=c.candidate_id "
                    "WHERE c.release_id=? ORDER BY c.candidate_id",
                    (release_id,),
                )
            )
    else:
        rows = cast(Iterable[CandidateRow], store_or_rows)
    grouped: dict[RudenessLabel, list[CandidateId]] = {
        RudenessLabel.RUDE: [],
        RudenessLabel.NON_RUDE: [],
    }
    for row in rows:
        if release_id is not None and row.get("release_id") not in (None, release_id):
            continue
        try:
            label = RudenessLabel(str(row["rudeness_label"]))
            candidate_id = CandidateId(str(row["candidate_id"]))
        except (KeyError, ValueError) as error:
            raise ValueError(
                "Candidate grouping failed because an input row lacks a valid candidate_id or "
                "closed rudeness_label. Validation failed in "
                "experiment.sampling.candidates_by_label before sampling, so no sample was "
                "persisted. Supply normalized candidate rows with rude/non_rude labels."
            ) from error
        grouped[label].append(candidate_id)
    return {label: tuple(sorted(ids)) for label, ids in grouped.items()}


def create_balanced_sample(
    store: ExperimentStore,
    release_id: ReleaseId,
    template_id: TemplateId,
    *,
    size: int,
    seed: int,
    candidates_by_label: Mapping[RudenessLabel, Sequence[CandidateId]],
) -> SampleId:
    """Persist a DRAFT balanced-matched sample selected by a local RNG."""
    if type(size) is not int or size < 2:
        raise ValueError(
            f"Balanced sample creation failed because size={size!r} is not an integer of at "
            "least two. Validation failed before RNG selection, so no DRAFT sample was "
            "persisted. Set size to an integer >= 2 and retry."
        )
    if type(seed) is not int or not 0 <= seed <= (1 << 64) - 1:
        raise ValueError(
            f"Balanced sample creation failed because seed={seed} is negative. Validation "
            "failed before local RNG construction, so no sample was persisted. Use a "
            "non-negative integer seed and retry."
        )
    per_label = size // 2
    extra_draw_record: SampleExtraDraw | None = None
    extra_label: RudenessLabel | None = None
    if size % 2:
        population = (RudenessLabel.NON_RUDE, RudenessLabel.RUDE)
        draw = balanced_extra_stratum_draw(seed, size).choose(
            tuple(CandidateId(label.value) for label in population)
        )
        extra_label = RudenessLabel(str(draw.selected))
        extra_draw_record = SampleExtraDraw(
            population=population,
            selected_index=draw.selected_index,
            seed=draw.seed,
            coordinates=(seed, size),
        )
    ordered = {
        label: tuple(sorted(candidates_by_label.get(label, ())))
        for label in RudenessLabel
    }
    shortages = {
        label: len(ids)
        for label, ids in ordered.items()
        if len(ids) < per_label + int(label is extra_label)
    }
    if shortages:
        counts = ", ".join(
            f"{label.value}={len(ordered[label])}" for label in RudenessLabel
        )
        raise ValueError(
            f"Balanced sample creation failed because size={size} requires {per_label} "
            f"candidates per label but available counts are {counts}. Stratum validation "
            "failed in experiment.sampling.create_balanced_sample before persistence. "
            "Ingest more candidates for the undersized stratum or reduce --size to at most "
            f"{2 * min(len(ids) for ids in ordered.values())}."
        )
    rng = random.Random(seed)
    selected = {
        label: tuple(
            rng.sample(list(ordered[label]), per_label + int(label is extra_label))
        )
        for label in RudenessLabel
    }
    members_list = [
        candidate_id
        for index in range(per_label)
        for candidate_id in (
            selected[RudenessLabel.RUDE][index],
            selected[RudenessLabel.NON_RUDE][index],
        )
    ]
    if extra_label is not None:
        members_list.append(selected[extra_label][-1])
    members = tuple(members_list)
    return store.create_sample(
        release_id,
        template_id,
        SamplerPolicy.BALANCED_MATCHED,
        seed,
        members,
        extra_draw_record,
    )
