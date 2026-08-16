"""Deterministic balanced sampling over immutable catalog candidates."""

from __future__ import annotations

import random
import sqlite3
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol, cast

from quadratic_voting.experiment.seeds import (
    SeededDraw,
    balanced_extra_stratum_draw,
    derive_seed,
)
from quadratic_voting.experiment.store import ExperimentStore, SampleExtraDraw
from quadratic_voting.experiment.types import (
    CandidateId,
    ReleaseId,
    RudenessLabel,
    SampleId,
    SamplerPolicy,
    SeedDomain,
    TemplateId,
)


class _SqliteStoreView(Protocol):
    connection: sqlite3.Connection


CandidateRow = Mapping[str, object]
AnnotationRow = Mapping[str, object]
_BALANCED_LABELS = (RudenessLabel.RUDE, RudenessLabel.NON_RUDE)

# ConvAbuse per-annotation severity is a five-point one-hot scale. Each annotator
# row is stored as exactly these five ``source_annotation`` rows (see
# catalog.load_convabuse and catalog._SEVERITY_COLUMNS), so a candidate's stored
# annotations are always a multiple of five in this stable label order.
_SEVERITY_LABEL_TO_LEVEL: Mapping[str, int] = {
    "is_abuse.1": 1,
    "is_abuse.0": 0,
    "is_abuse.-1": -1,
    "is_abuse.-2": -2,
    "is_abuse.-3": -3,
}
_LEVELS_PER_ANNOTATOR = len(_SEVERITY_LABEL_TO_LEVEL)
# Candidate set draws exactly one candidate per level, most-severe (most
# negative) last, matching the ConvAbuse severity ordering.
LEVEL_STRATIFIED_LEVELS: tuple[int, ...] = (1, 0, -1, -2, -3)


def candidate_modal_severity(rows: Iterable[AnnotationRow]) -> int | None:
    """Return one candidate's modal per-annotator severity level, or None.

    ``rows`` are the candidate's persisted ``source_annotation`` rows, each with
    ``annotation_index``, ``source_label`` (e.g. ``is_abuse.-1``) and
    ``source_value`` (``"1"`` for the annotator's chosen level). Annotations are
    grouped into their originating annotator by ``annotation_index`` (five stored
    labels per annotator). An annotator contributes a severity only when exactly
    one of its five labels is active; malformed annotators are ignored, matching
    catalog._annotation_is_abusive. The mode across contributing annotators is the
    candidate level; ties break toward the MORE severe (more negative) level. None
    means no annotator supplied a valid single-severity vote.
    """
    ordered = sorted(rows, key=lambda row: int(cast(int, row["annotation_index"])))
    if len(ordered) % _LEVELS_PER_ANNOTATOR != 0:
        raise ValueError(
            "Modal-severity derivation failed because a candidate has "
            f"{len(ordered)} source_annotation rows, which is not a multiple of the "
            f"{_LEVELS_PER_ANNOTATOR}-point ConvAbuse severity scale. Validation failed "
            "in experiment.sampling.candidate_modal_severity before level stratification, "
            "so no sample was persisted. Re-ingest the release through catalog.ingest_"
            "convabuse so each annotator stores its full five one-hot severity labels."
        )
    per_annotator: dict[int, list[int]] = {}
    for row in ordered:
        annotator = int(cast(int, row["annotation_index"])) // _LEVELS_PER_ANNOTATOR
        label = str(row["source_label"])
        if label not in _SEVERITY_LABEL_TO_LEVEL:
            continue
        if str(row["source_value"]).strip() == "1":
            per_annotator.setdefault(annotator, []).append(
                _SEVERITY_LABEL_TO_LEVEL[label]
            )
    votes = [levels[0] for levels in per_annotator.values() if len(levels) == 1]
    if not votes:
        return None
    counts = Counter(votes)
    top = max(counts.values())
    # Tie-break toward the most severe (most negative) level.
    return min(level for level, count in counts.items() if count == top)


def candidates_by_severity_level(
    store_or_rows: ExperimentStore | Iterable[AnnotationRow],
    release_id: ReleaseId | None = None,
) -> Mapping[int, tuple[CandidateId, ...]]:
    """Group candidate IDs by modal severity level in stable order.

    SQLite stores read ``source_annotation`` joined to ``candidate`` directly.
    Iterable rows must provide ``candidate_id``, ``annotation_index``,
    ``source_label``, ``source_value`` and optional ``release_id``.
    """
    if hasattr(store_or_rows, "connection"):
        connection = cast(_SqliteStoreView, store_or_rows).connection
        if release_id is None:
            annotation_rows: Iterable[AnnotationRow] = (
                dict(row)
                for row in connection.execute(
                    "SELECT sa.candidate_id,c.release_id,sa.annotation_index,"
                    "sa.source_label,sa.source_value FROM source_annotation sa "
                    "JOIN candidate c ON c.candidate_id=sa.candidate_id "
                    "ORDER BY sa.candidate_id,sa.annotation_index"
                )
            )
        else:
            annotation_rows = (
                dict(row)
                for row in connection.execute(
                    "SELECT sa.candidate_id,c.release_id,sa.annotation_index,"
                    "sa.source_label,sa.source_value FROM source_annotation sa "
                    "JOIN candidate c ON c.candidate_id=sa.candidate_id "
                    "WHERE c.release_id=? ORDER BY sa.candidate_id,sa.annotation_index",
                    (release_id,),
                )
            )
    else:
        annotation_rows = cast(Iterable[AnnotationRow], store_or_rows)
    grouped_rows: dict[CandidateId, list[AnnotationRow]] = {}
    for row in annotation_rows:
        if release_id is not None and row.get("release_id") not in (None, release_id):
            continue
        try:
            candidate_id = CandidateId(str(row["candidate_id"]))
        except KeyError as error:
            raise ValueError(
                "Candidate severity grouping failed because a source_annotation row lacks "
                "a candidate_id. Validation failed in "
                "experiment.sampling.candidates_by_severity_level before sampling, so no "
                "sample was persisted. Supply normalized source_annotation rows and retry."
            ) from error
        grouped_rows.setdefault(candidate_id, []).append(row)
    by_level: dict[int, list[CandidateId]] = {
        level: [] for level in LEVEL_STRATIFIED_LEVELS
    }
    for candidate_id, rows in grouped_rows.items():
        level = candidate_modal_severity(rows)
        if level is None:
            continue
        by_level.setdefault(level, []).append(candidate_id)
    return {level: tuple(sorted(ids)) for level, ids in by_level.items()}


def create_level_stratified_sample(
    store: ExperimentStore,
    release_id: ReleaseId,
    template_id: TemplateId,
    *,
    seed: int,
    candidates_by_level: Mapping[int, Sequence[CandidateId]],
) -> SampleId:
    """Persist a DRAFT level-stratified sample of one candidate per severity level.

    Exactly one candidate is drawn per ConvAbuse severity level in
    ``LEVEL_STRATIFIED_LEVELS`` using a per-level ``qv-seed/v1`` draw, giving a
    deterministic five-candidate set ordered from least to most severe. Draws are
    reproducible from ``seed`` and the sorted candidate population; the
    lightweight level policy records no extra RNG rows.
    """
    if type(seed) is not int or not 0 <= seed <= (1 << 64) - 1:
        raise ValueError(
            f"Level-stratified sample creation failed because seed={seed!r} is not a "
            "uint64. Validation failed in experiment.sampling.create_level_stratified_"
            "sample before RNG selection, so no DRAFT sample was persisted. Use an integer "
            "from zero through 2**64 - 1 and retry."
        )
    shortages = {
        level: len(tuple(candidates_by_level.get(level, ())))
        for level in LEVEL_STRATIFIED_LEVELS
        if len(tuple(candidates_by_level.get(level, ()))) < 1
    }
    if shortages:
        counts = ", ".join(
            f"level {level}={len(tuple(candidates_by_level.get(level, ())))}"
            for level in LEVEL_STRATIFIED_LEVELS
        )
        raise ValueError(
            "Level-stratified sample creation failed because these severity levels have no "
            f"candidates: {sorted(shortages)}. Available counts are {counts}. Stratum "
            "validation failed in experiment.sampling.create_level_stratified_sample "
            f"before persistence. Ingest at least one candidate whose modal severity is "
            "each of the five levels (1, 0, -1, -2, -3), or select a release that spans "
            "all five levels, and retry."
        )
    members: list[CandidateId] = []
    for level in LEVEL_STRATIFIED_LEVELS:
        population = tuple(sorted(candidates_by_level[level]))
        draw = SeededDraw(
            stream_name=f"level-stratified/{seed}/{level}",
            # Levels are signed (down to -3); derive_seed encodes integer
            # coordinates as unsigned, so pass the level as a string coordinate.
            seed=derive_seed(seed, SeedDomain.LEVEL_STRATIFIED_DRAW, str(level)),
            domain=SeedDomain.LEVEL_STRATIFIED_DRAW.value,
            coordinates=(seed, str(level)),
        ).choose(population)
        members.append(draw.selected)
    return store.create_sample(
        release_id,
        template_id,
        SamplerPolicy.LEVEL_STRATIFIED,
        seed,
        tuple(members),
    )


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
        label: [] for label in RudenessLabel
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
    # Ambiguous ties remain available to future experiment-specific mapping
    # policies, but balanced-matched/v1 samples only strict-majority strata.
    ordered = {
        label: tuple(sorted(candidates_by_label.get(label, ())))
        for label in _BALANCED_LABELS
    }
    shortages = {
        label: len(ids)
        for label, ids in ordered.items()
        if len(ids) < per_label + int(label is extra_label)
    }
    if shortages:
        counts = ", ".join(
            f"{label.value}={len(ordered[label])}" for label in _BALANCED_LABELS
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
        for label in _BALANCED_LABELS
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
