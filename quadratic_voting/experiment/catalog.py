"""ConvAbuse catalog loading, immutable ingestion, and candidate-card rendering."""

from __future__ import annotations

import csv
import hashlib
import sqlite3
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol, cast

from quadratic_voting.experiment.store import (
    CandidateRecord,
    ExperimentStore,
    SourceAnnotation,
)
from quadratic_voting.experiment.types import (
    CandidateId,
    ReleaseId,
    RudenessLabel,
    TemplateId,
)


class RudenessDerivationRule(StrEnum):
    """Versioned association-definition choices, not established ground truth."""

    MAJORITY_SEVERITY_NEGATIVE_COMPLETE_CONTEXT = (
        "majority-severity-negative-complete-context/v3"
    )


DEFAULT_PRESENTATION_TEMPLATE_BODY: Final[str] = (
    "User {candidate_id}\n"
    "User: {prev_user}\n"
    "Assistant: {prev_agent}\n"
    "User: {user}\n"
    "Assistant: {agent}"
)
DEFAULT_PRESENTATION_TEMPLATE_NAME: Final[str] = "candidate-card"
DEFAULT_PRESENTATION_TEMPLATE_VERSION: Final[str] = "v3"
_COMPLETE_CONTEXT_ROLES: Final[tuple[str, ...]] = (
    "user",
    "assistant",
    "user",
    "assistant",
)

_SEVERITY_COLUMNS: Final[tuple[str, ...]] = (
    "is_abuse.1",
    "is_abuse.0",
    "is_abuse.-1",
    "is_abuse.-2",
    "is_abuse.-3",
)
_REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "conv_id",
        "prev_agent",
        "prev_user",
        "agent",
        "user",
        *_SEVERITY_COLUMNS,
    }
)


class _SqliteStoreView(Protocol):
    connection: sqlite3.Connection


def _normalized_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _annotation_is_abusive(row: dict[str, str]) -> bool | None:
    selected = [column for column in _SEVERITY_COLUMNS if row[column].strip() == "1"]
    if len(selected) != 1:
        return None
    return selected[0] in {"is_abuse.-1", "is_abuse.-2", "is_abuse.-3"}


def _content_digest(turns: tuple[tuple[str, str], ...]) -> str:
    canonical = "".join(
        f"{len(role.encode('utf-8'))}:{role}{len(text.encode('utf-8'))}:{text}"
        for role, text in turns
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _group_digest(group_key: tuple[str, ...]) -> str:
    canonical = "".join(f"{len(value.encode('utf-8'))}:{value}" for value in group_key)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_convabuse(
    csv_path: Path, *, rule: RudenessDerivationRule
) -> tuple[CandidateRecord, ...]:
    """Collapse annotator rows into deterministic, normalized candidate records.

    Under ``majority-severity-negative-complete-context/v3``, negative severity annotations are
    abusive. A strict negative majority is RUDE, a strict non-negative majority is
    NON_RUDE, and an exact tie is AMBIGUOUS_TIE.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"ConvAbuse loading failed because dataset file {csv_path} does not exist or "
            "is not a regular file. The failure occurred in experiment.catalog.load_convabuse "
            "before CSV parsing, so no candidates were produced. Supply --dataset-path pointing "
            "to ConvAbuseEMNLPfull.csv and retry."
        )
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    try:
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = sorted(_REQUIRED_COLUMNS - columns)
            if missing:
                raise ValueError(
                    f"ConvAbuse loading failed because {csv_path} is missing required CSV "
                    f"columns {missing}. Schema validation failed in "
                    "experiment.catalog.load_convabuse before grouping, so candidate labels "
                    "cannot be derived. Use the full ConvAbuse EMNLP CSV with the listed "
                    "conversation and one-hot severity columns."
                )
            for row in reader:
                normalized = tuple(
                    _normalized_text(row[column])
                    for column in (
                        "conv_id",
                        "prev_agent",
                        "prev_user",
                        "agent",
                        "user",
                    )
                )
                conversation_fields = (
                    "prev_agent",
                    "prev_user",
                    "agent",
                    "user",
                )
                blank_fields = tuple(
                    column
                    for column, value in zip(
                        conversation_fields, normalized[1:], strict=True
                    )
                    if not value
                )
                if blank_fields:
                    raise ValueError(
                        f"ConvAbuse loading failed because CSV record {reader.line_num} has "
                        f"blank required conversation fields {list(blank_fields)}. Validation "
                        "failed in experiment.catalog.load_convabuse before grouping, so no "
                        "malformed four-message candidate card can be persisted. Supply all "
                        "prev_user, prev_agent, user, and agent message text, then retry."
                    )
                groups[normalized].append(row)
    except UnicodeDecodeError as error:
        raise ValueError(
            f"ConvAbuse loading failed because {csv_path} is not valid UTF-8 CSV data. "
            "Decoding failed in experiment.catalog.load_convabuse before ingestion, so no "
            "release was changed. Obtain an unmodified UTF-8 ConvAbuse CSV and retry."
        ) from error

    records: list[CandidateRecord] = []
    for key, annotation_rows in groups.items():
        if (
            rule
            is not RudenessDerivationRule.MAJORITY_SEVERITY_NEGATIVE_COMPLETE_CONTEXT
        ):
            raise AssertionError(f"unhandled closed rudeness derivation rule: {rule!r}")
        votes = tuple(
            vote
            for vote in (_annotation_is_abusive(row) for row in annotation_rows)
            if vote is not None
        )
        if not votes:
            continue
        turns = (
            ("user", key[2]),
            ("assistant", key[1]),
            ("user", key[4]),
            ("assistant", key[3]),
        )
        digest = _content_digest(turns)
        source_row_id = f"{key[0]}:{_group_digest(key)[:16]}"
        abusive_votes = sum(votes)
        if abusive_votes > len(votes) / 2:
            rudeness_label = RudenessLabel.RUDE
        elif abusive_votes * 2 == len(votes):
            rudeness_label = RudenessLabel.AMBIGUOUS_TIE
        else:
            rudeness_label = RudenessLabel.NON_RUDE
        records.append(
            CandidateRecord(
                source_row_id=source_row_id,
                rudeness_label=rudeness_label,
                turns=turns,
                content_sha256=digest,
                annotations=tuple(
                    SourceAnnotation(
                        annotator_hash=hashlib.sha256(
                            row.get("annotator_id", "").encode("utf-8")
                        ).hexdigest(),
                        source_label=column,
                        source_value=row[column],
                    )
                    for row in annotation_rows
                    for column in _SEVERITY_COLUMNS
                ),
            )
        )
    records.sort(key=lambda record: record.source_row_id)
    if not records:
        raise ValueError(
            f"ConvAbuse loading failed because {csv_path} produced zero candidates after "
            "invalid-annotation filtering. Validation failed in "
            "experiment.catalog.load_convabuse, so an empty release must not be ingested. "
            "Check that the CSV has all four nonblank conversation turns and exactly one active severity "
            "one-hot value per annotation, then retry."
        )
    return tuple(records)


def render_candidate_card(candidate_id: str, record: CandidateRecord, body: str) -> str:
    """Render a neutral, chronologically complete card without derived labels."""
    roles = tuple(role for role, _text in record.turns)
    if roles != _COMPLETE_CONTEXT_ROLES:
        raise ValueError(
            f"Candidate-card rendering failed because candidate {candidate_id!r} has "
            f"roles {roles!r}, not required chronological roles {_COMPLETE_CONTEXT_ROLES!r}. "
            "Validation failed in "
            "experiment.catalog.render_candidate_card before model-visible text was frozen. "
            "Re-ingest a normalized ConvAbuse candidate as user(prev_user), "
            "assistant(prev_agent), user(user), assistant(agent)."
        )
    if any(not text.strip() for _role, text in record.turns):
        raise ValueError(
            f"Candidate-card rendering failed because candidate {candidate_id!r} has a blank "
            "required conversation message. Validation failed in "
            "experiment.catalog.render_candidate_card before model-visible text was frozen. "
            "Re-ingest all four nonblank ConvAbuse conversation fields."
        )
    (
        (_prev_user_role, prev_user),
        (_prev_agent_role, prev_agent),
        (_user_role, user),
        (
            _agent_role,
            agent,
        ),
    ) = record.turns
    try:
        return body.format(
            candidate_id=candidate_id,
            prev_user=prev_user,
            prev_agent=prev_agent,
            user=user,
            agent=agent,
        )
    except KeyError as error:
        missing = str(error.args[0])
        raise ValueError(
            f"Candidate-card rendering failed because template body requires unsupported "
            f"placeholder {missing!r}. Rendering failed in "
            "experiment.catalog.render_candidate_card before presentation persistence. Use "
            "only {candidate_id}, {prev_user}, {prev_agent}, {user}, and {agent} "
            "placeholders, then retry."
        ) from error


def _existing_presentation_template(store: ExperimentStore) -> TemplateId:
    if not hasattr(store, "connection"):
        raise RuntimeError(
            "Candidate-card template reuse failed because this ExperimentStore does not expose "
            "the SQLite catalog query used by Stage 2. The failure occurred after release "
            f"ingestion while resolving {DEFAULT_PRESENTATION_TEMPLATE_NAME}/"
            f"{DEFAULT_PRESENTATION_TEMPLATE_VERSION}. Use open_sqlite_store for catalog "
            "ingestion or extend the store protocol with a template lookup operation."
        )
    connection = cast(_SqliteStoreView, store).connection
    row = connection.execute(
        "SELECT template_id FROM presentation_template WHERE name=? AND version=?",
        (DEFAULT_PRESENTATION_TEMPLATE_NAME, DEFAULT_PRESENTATION_TEMPLATE_VERSION),
    ).fetchone()
    if row is None:
        raise RuntimeError(
            "Candidate-card template reuse failed because the current candidate-card version triggered a "
            "uniqueness conflict but no matching row was found. The catalog is inconsistent; "
            "inspect presentation_template and restore the database before retrying ingestion."
        )
    return TemplateId(str(row[0]))


def _candidate_ids_by_source(
    store: ExperimentStore, release_id: ReleaseId
) -> dict[str, CandidateId]:
    if not hasattr(store, "connection"):
        # The final store renderer callback omits CandidateId. Non-SQLite adapters can
        # still render a stable card, but should add a catalog lookup API before use.
        return {}
    connection = cast(_SqliteStoreView, store).connection
    return {
        str(row[0]): CandidateId(str(row[1]))
        for row in connection.execute(
            "SELECT source_row_id,candidate_id FROM candidate WHERE release_id=?",
            (release_id,),
        )
    }


def render_release_candidate_cards(
    store: ExperimentStore,
    release_id: ReleaseId,
    template_id: TemplateId,
    body: str = DEFAULT_PRESENTATION_TEMPLATE_BODY,
) -> None:
    """Render the complete release with stable persisted candidate IDs."""
    candidate_ids = _candidate_ids_by_source(store, release_id)
    store.render_presentations(
        release_id,
        template_id,
        lambda record: render_candidate_card(
            str(candidate_ids.get(record.source_row_id, record.source_row_id)),
            record,
            body,
        ),
    )


def ingest_convabuse(
    store: ExperimentStore,
    csv_path: Path,
    dataset_version: str,
    rule: RudenessDerivationRule,
) -> ReleaseId:
    """Ingest one immutable release and freeze its neutral presentations.

    The source metadata records ``#rudeness-rule=<value>`` because the current
    normalized schema has no separate association-definition column.
    """
    candidates = load_convabuse(csv_path, rule=rule)
    file_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    source_metadata = f"{csv_path.resolve()}#rudeness-rule={rule.value}"
    try:
        release_id = store.ingest_release(
            "ConvAbuse",
            dataset_version,
            source_metadata,
            file_sha256,
            candidates,
            label_policy_name="convabuse-rudeness",
            label_policy_version=rule.value,
            label_policy_rule=(
                "Negative severity annotations (-1,-2,-3) are abusive; RUDE requires a "
                "strict majority of valid one-hot annotations; NON_RUDE requires a strict "
                "non-negative majority; exact ties are AMBIGUOUS_TIE. Candidate conversation "
                "context is normalized in chronological order as user(prev_user), "
                "assistant(prev_agent), user(user), assistant(agent)."
            ),
        )
    except sqlite3.IntegrityError as error:
        raise ValueError(
            f"ConvAbuse ingestion failed because dataset release ConvAbuse/{dataset_version} "
            "already exists or conflicts with immutable catalog identity. The UNIQUE check "
            "failed in experiment.catalog.ingest_convabuse before a duplicate release could "
            "commit. Choose a new --dataset-version for changed source bytes, or reuse the "
            "existing release ID instead of re-ingesting it."
        ) from error
    try:
        template_id = store.register_template(
            DEFAULT_PRESENTATION_TEMPLATE_NAME,
            DEFAULT_PRESENTATION_TEMPLATE_VERSION,
            DEFAULT_PRESENTATION_TEMPLATE_BODY,
        )
    except sqlite3.IntegrityError:
        template_id = _existing_presentation_template(store)
    render_release_candidate_cards(store, release_id, template_id)
    return release_id
