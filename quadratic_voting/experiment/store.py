"""Durable SQLite storage and the single-writer boundary for experiments.

The optional ``commit_hook`` is a deterministic crash-injection seam.  It is
called immediately before each store-owned transaction commit with a
monotonically increasing ordinal; production callers leave it unset.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from string import Formatter
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, cast

from quadratic_voting.experiment.artifacts import (
    FrozenCandidateSample,
    SampleSidecar,
    canonical_sample_bytes,
    read_frozen_sample,
    write_sidecar,
)
from quadratic_voting.experiment.errors import (
    DefinitionDriftError,
    DirtyPrimaryTreeError,
    FreezeMismatchError,
)
from quadratic_voting.experiment.sample_file import write_fsynced_temp
from quadratic_voting.experiment.seeds import (
    DrawSelection,
    support_removal_draw,
    tie_break_draw,
    voter_permutation_draw,
    seed_from_blob,
    seed_to_blob,
)
from quadratic_voting.experiment.lock import (
    WriterLock,
    acquire_writer_lock,
    lock_matches_database,
)
from quadratic_voting.experiment.types import (
    BarrierReady,
    CallId,
    CandidateId,
    ElicitationArm,
    ExecutionEnvironment,
    ExecutionClass,
    ExecutionId,
    ExportDataset,
    FinalResultEvent,
    FreezePoint,
    GenerationResult,
    LikertRating,
    MatchedSetConfig,
    MatchedSetCreation,
    MatchedSetId,
    NextUnit,
    PendingTurn,
    PresentationId,
    PriorTurnEvent,
    ReleaseId,
    RoundId,
    RoundOutcomeEvent,
    RunComplete,
    RunForkReason,
    RunId,
    RunStatus,
    RuntimeFailureKind,
    RudenessLabel,
    SampleId,
    SamplerPolicy,
    SamplingProfile,
    SetupContext,
    TemplateId,
    TemplateKind,
    TurnId,
    TurnKind,
    VoterId,
    VoterRoundView,
    VotingRegime,
    WorkUnit,
    arm_turn_order,
)
from quadratic_voting.experiment.config import MatchedSetConfigV1

if TYPE_CHECKING:
    from quadratic_voting.experiment.ballots import ValidationFailure

KNOWN_SCHEMA_VERSION = 1
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_DIAGNOSTIC_ALLOWLIST = frozenset(
    {
        "backend",
        "error_type",
        "code",
        "operation",
        "device_index",
        "memory_total_bytes",
        "memory_free_bytes",
        "provider_request_id",
        "status",
        "native_finish_reason",
    }
)


def _ulid() -> str:
    """Return a dependency-free, lexically time-ordered text ULID."""
    value = ((int(time.time() * 1000) & ((1 << 48) - 1)) << 80) | int.from_bytes(
        os.urandom(10), "big"
    )
    chars = ["0"] * 26
    for index in range(25, -1, -1):
        chars[index] = _CROCKFORD[value & 31]
        value >>= 5
    return "".join(chars)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"cannot encode {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        default=_json_default,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _validated_diagnostics(values: Mapping[str, object]) -> dict[str, object]:
    invalid_keys = sorted(set(values) - _DIAGNOSTIC_ALLOWLIST)
    invalid_values = {
        key: type(value).__name__
        for key, value in values.items()
        if not isinstance(value, (str, int, float, bool, type(None)))
        or (isinstance(value, float) and not math.isfinite(value))
    }
    if invalid_keys or invalid_values:
        raise ValueError(
            "Diagnostics persistence refused non-allowlisted keys or non-scalar values: "
            f"keys={invalid_keys}, values={invalid_values}. Validation failed before durable "
            "write/export, so credentials or arbitrary exception data were not exposed. "
            "Map diagnostics to the documented sanitized scalar allowlist and retry."
        )
    return dict(values)


@dataclass(frozen=True, slots=True)
class SourceAnnotation:
    annotator_hash: str
    source_label: str
    source_value: str


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    source_row_id: str
    rudeness_label: RudenessLabel
    turns: tuple[tuple[str, str], ...]
    content_sha256: str
    annotations: tuple[SourceAnnotation, ...] = ()


@dataclass(frozen=True, slots=True)
class SampleExtraDraw:
    population: tuple[RudenessLabel, ...]
    selected_index: int
    seed: int
    coordinates: tuple[int, int]


@dataclass(frozen=True, slots=True)
class RunDefinition:
    model_id: str
    provider_id: str
    quantization_id: str
    artifact_repository: str
    artifact_revision: str
    presentation_template_id: TemplateId
    presentation_template_hash: str
    instruction_templates: Mapping[TemplateKind, tuple[TemplateId, str]]
    dataset_release_hash: str
    sample_artifact_hash: str
    runtime_id: str | None = None
    tokenizer_repository: str | None = None
    tokenizer_revision: str | None = None
    dtype: str | None = None
    route_registry_hash: str | None = None
    sampling_profile_hash: str | None = None
    instruction_profile_hash: str | None = None
    canonical_json_version: str = "qv-canonical-json/v1"
    prompt_encoding_version: str = "qv-prompt/v1"
    seed_version: str = "qv-seed/v1"
    source_release_id: str | None = None
    label_policy_id: str | None = None
    label_policy_version: str | None = None
    label_policy_hash: str | None = None
    sample_id: str | None = None
    prompt_reviewed: bool = False
    prompt_review_version: str | None = None
    prompt_review_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class RunInfo:
    run_id: RunId
    matched_set_id: MatchedSetId
    arm: ElicitationArm
    regime: VotingRegime
    status: RunStatus
    master_seed: int
    voter_count: int
    credit_budget: int
    max_correction_attempts: int
    max_consecutive_runtime_failures: int
    sampling: SamplingProfile


@dataclass(frozen=True, slots=True)
class AcceptedBallot:
    rationale: str
    allocations: Mapping[CandidateId, int]
    engine_cost: int


@dataclass(frozen=True, slots=True)
class AcceptedStatement:
    items: Mapping[CandidateId, tuple[LikertRating, str]]


@dataclass(frozen=True, slots=True)
class BallotAbstention:
    pass


@dataclass(frozen=True, slots=True)
class StatementInvalidMissing:
    pass


TerminalWrite: TypeAlias = (
    AcceptedBallot | AcceptedStatement | BallotAbstention | StatementInvalidMissing
)


@dataclass(frozen=True, slots=True)
class RoundOutcome:
    round_index: int
    protected: CandidateId | None
    removed: CandidateId
    tie: bool


class ExperimentStore(Protocol):
    def run_info(self, run_id: RunId) -> RunInfo: ...

    def voters(self, run_id: RunId) -> tuple[tuple[int, VoterId], ...]: ...

    def ingest_release(
        self,
        dataset_name: str,
        version: str,
        source_path: str,
        file_sha256: str,
        candidates: Sequence[CandidateRecord],
        *,
        label_policy_name: str = "embedded-rudeness-label",
        label_policy_version: str = "v1",
        label_policy_rule: str = "caller-supplied immutable candidate labels",
        label_policy_reviewed: bool = False,
        label_policy_review_version: str | None = None,
        label_policy_review_sha256: str | None = None,
    ) -> ReleaseId: ...

    def register_template(
        self, kind_or_name: TemplateKind | str, version: str, body: str
    ) -> TemplateId: ...

    def render_presentations(
        self,
        release_id: ReleaseId,
        template_id: TemplateId,
        renderer: Callable[[CandidateRecord], str],
    ) -> tuple[PresentationId, ...]: ...

    def create_sample(
        self,
        release_id: ReleaseId,
        template_id: TemplateId,
        policy: SamplerPolicy,
        seed: int,
        members: Sequence[CandidateId],
        extra_draw: SampleExtraDraw | None = None,
    ) -> SampleId: ...

    def freeze_sample(
        self, sample_id: SampleId, path: Path
    ) -> FrozenCandidateSample: ...

    def reconcile_sample(
        self, sample_id: SampleId, path: Path | None = None
    ) -> FrozenCandidateSample: ...

    def validate_sample(self, path: Path) -> SampleId: ...

    def create_matched_set(
        self,
        sample: FrozenCandidateSample,
        sample_id: SampleId,
        config: MatchedSetConfig,
        definition: RunDefinition,
        artifact_path: Path | None = None,
    ) -> MatchedSetCreation: ...

    def create_matched_set_v1(
        self, config: MatchedSetConfigV1
    ) -> MatchedSetCreation: ...

    def register_static_route(self, definition: RunDefinition) -> str: ...

    def record_run_fork(
        self,
        parent: MatchedSetId,
        child: MatchedSetId,
        reason: RunForkReason,
    ) -> None: ...

    def begin_execution(
        self, run_id: RunId, env: ExecutionEnvironment, drift_override: bool = False
    ) -> ExecutionId: ...

    def preflight_execution(
        self,
        run_id: RunId,
        env: ExecutionEnvironment,
        current_definition: RunDefinition | None = None,
    ) -> None: ...

    def verify_run_definition(self, run_id: RunId, current: RunDefinition) -> None: ...

    def end_execution(self, execution_id: ExecutionId, exit_reason: str) -> None: ...

    def next_incomplete_unit(self, run_id: RunId) -> NextUnit: ...

    def mark_interrupted(self, call_id: CallId) -> None: ...

    def resolve_turn_id(self, unit: WorkUnit) -> TurnId: ...

    def begin_call(
        self,
        turn_id: TurnId,
        attempt_index: int,
        prompt_messages_json: str,
        prompt_sha256: str,
        seed: int,
    ) -> CallId: ...

    def commit_call(
        self,
        call_id: CallId,
        result: GenerationResult,
        failures: Sequence[ValidationFailure],
        terminal: TerminalWrite | None,
    ) -> None: ...

    def record_runtime_failure(
        self, call_id: CallId, kind: RuntimeFailureKind, diagnostics: Mapping[str, str]
    ) -> None: ...

    def interrupt_call_with_failure(
        self, call_id: CallId, kind: RuntimeFailureKind, diagnostics: Mapping[str, str]
    ) -> None: ...

    def pause_run(self, run_id: RunId, reason: str) -> None: ...

    def set_run_in_progress(self, run_id: RunId) -> None: ...

    def aggregate_and_seal_round(self, run_id: RunId) -> RoundOutcome | RunComplete: ...

    def voter_round_view(self, run_id: RunId, voter_id: VoterId) -> VoterRoundView: ...

    def export_rows(self, dataset: ExportDataset) -> tuple[dict[str, object], ...]: ...

    def candidate_rows(self) -> tuple[dict[str, object], ...]: ...

    def round_candidate_rows(self) -> tuple[dict[str, object], ...]: ...

    def source_annotation_rows(self) -> tuple[dict[str, object], ...]: ...

    def candidate_presentation_rows(self) -> tuple[dict[str, object], ...]: ...

    def candidate_turn_rows(self) -> tuple[dict[str, object], ...]: ...

    def voter_permutation_rows(self) -> tuple[dict[str, object], ...]: ...

    def experiment_config_rows(self) -> tuple[dict[str, object], ...]: ...

    def run_definition_rows(self) -> tuple[dict[str, object], ...]: ...


class SqliteExperimentStore:
    def __init__(
        self,
        connection: sqlite3.Connection,
        path: Path,
        commit_hook: Callable[[int], None] | None = None,
        freeze_hook: Callable[[FreezePoint], None] | None = None,
        writer_lock: WriterLock | None = None,
        owns_writer_lock: bool = False,
    ) -> None:
        self.connection = connection
        self.path = path
        self._commit_hook = commit_hook or (lambda _ordinal: None)
        self._commit_ordinal = 0
        self._freeze_hook = freeze_hook or (lambda _point: None)
        self._writer_lock = writer_lock
        self._owns_writer_lock = owns_writer_lock
        self.last_environment_drift: Mapping[str, tuple[object, object]] = {}

    def close(self) -> None:
        self.connection.close()
        if self._owns_writer_lock and self._writer_lock is not None:
            self._writer_lock.release()
            self._owns_writer_lock = False

    def __enter__(self) -> SqliteExperimentStore:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def run_info(self, run_id: RunId) -> RunInfo:
        row = self.connection.execute(
            "SELECT r.run_id,r.matched_set_id,r.arm,r.regime,r.status,c.master_seed,"
            "c.voter_count,c.credit_budget,c.ballot_max_corrections+1 AS attempt_limit,"
            "c.runtime_max_failures AS max_consecutive_runtime_failures,"
            "c.temperature,c.top_p,c.top_k,c.max_new_tokens "
            "FROM experiment_run r JOIN matched_set m ON m.matched_set_id=r.matched_set_id "
            "JOIN experiment_config c ON c.config_id=m.config_id "
            "WHERE r.run_id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Run-info lookup failed because run {run_id} does not exist. Lookup failed in "
                "experiment.store.SqliteExperimentStore.run_info before runner configuration, "
                "so execution cannot safely infer seeds, limits, or sampling settings. Supply "
                "a run ID returned by create_matched_set and retry."
            )
        return RunInfo(
            RunId(row["run_id"]),
            MatchedSetId(row["matched_set_id"]),
            ElicitationArm(row["arm"]),
            VotingRegime(row["regime"]),
            RunStatus(row["status"]),
            seed_from_blob(row["master_seed"]),
            row["voter_count"],
            row["credit_budget"],
            row["attempt_limit"] - 1,
            row["max_consecutive_runtime_failures"],
            SamplingProfile(
                row["temperature"],
                row["top_p"],
                row["top_k"],
                row["max_new_tokens"],
            ),
        )

    def voters(self, run_id: RunId) -> tuple[tuple[int, VoterId], ...]:
        if (
            self.connection.execute(
                "SELECT 1 FROM experiment_run WHERE run_id=?", (run_id,)
            ).fetchone()
            is None
        ):
            raise ValueError(
                f"Voter lookup failed because run {run_id} does not exist. Lookup failed in "
                "experiment.store.SqliteExperimentStore.voters before deterministic traversal, "
                "so returning an empty population would hide an invalid run ID. Supply a run ID "
                "returned by create_matched_set and retry."
            )
        return tuple(
            (row["voter_index"], VoterId(row["voter_id"]))
            for row in self.connection.execute(
                "SELECT voter_index,voter_id FROM voter WHERE run_id=? ORDER BY voter_index",
                (run_id,),
            )
        )

    @contextmanager
    def _transaction(self) -> Any:
        if self.connection.in_transaction:
            yield
            return
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._commit_ordinal += 1
            self._commit_hook(self._commit_ordinal)
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise

    def ingest_release(
        self,
        dataset_name: str,
        version: str,
        source_path: str,
        file_sha256: str,
        candidates: Sequence[CandidateRecord],
        *,
        label_policy_name: str = "embedded-rudeness-label",
        label_policy_version: str = "v1",
        label_policy_rule: str = "caller-supplied immutable candidate labels",
        label_policy_reviewed: bool = False,
        label_policy_review_version: str | None = None,
        label_policy_review_sha256: str | None = None,
    ) -> ReleaseId:
        release_id = ReleaseId(_ulid())
        with self._transaction():
            policy = self.connection.execute(
                "SELECT label_policy_id,rule_text,reviewed,review_version,review_sha256 "
                "FROM label_policy WHERE name=? AND version=?",
                (label_policy_name, label_policy_version),
            ).fetchone()
            if policy is None:
                label_policy_id = _ulid()
                self.connection.execute(
                    "INSERT INTO label_policy VALUES (?,?,?,?,?,?,?,?)",
                    (
                        label_policy_id,
                        label_policy_name,
                        label_policy_version,
                        label_policy_rule,
                        hashlib.sha256(label_policy_rule.encode()).hexdigest(),
                        int(label_policy_reviewed),
                        label_policy_review_version,
                        label_policy_review_sha256,
                    ),
                )
            elif (
                policy["rule_text"],
                bool(policy["reviewed"]),
                policy["review_version"],
                policy["review_sha256"],
            ) != (
                label_policy_rule,
                label_policy_reviewed,
                label_policy_review_version,
                label_policy_review_sha256,
            ):
                raise ValueError(
                    f"Release ingestion failed because label policy {label_policy_name}/"
                    f"{label_policy_version} already exists with different immutable rule text. "
                    "Validation failed before release commit. Use a new policy version."
                )
            else:
                label_policy_id = policy["label_policy_id"]
            self.connection.execute(
                "INSERT INTO dataset_release VALUES (?,?,?,?,?,?)",
                (release_id, dataset_name, version, source_path, file_sha256, _now()),
            )
            for candidate in candidates:
                candidate_id = CandidateId(_ulid())
                self.connection.execute(
                    "INSERT INTO candidate VALUES (?,?,?,?)",
                    (
                        candidate_id,
                        release_id,
                        candidate.source_row_id,
                        candidate.content_sha256,
                    ),
                )
                self.connection.execute(
                    "INSERT INTO candidate_label VALUES (?,?,?)",
                    (candidate_id, label_policy_id, candidate.rudeness_label.value),
                )
                for index, (role, text) in enumerate(candidate.turns):
                    normalized_role = "assistant" if role == "agent" else role
                    self.connection.execute(
                        "INSERT INTO candidate_turn VALUES (?,?,?,?)",
                        (candidate_id, index, normalized_role, text),
                    )
                for index, annotation in enumerate(candidate.annotations):
                    self.connection.execute(
                        "INSERT INTO source_annotation VALUES (?,?,?,?,?)",
                        (
                            candidate_id,
                            index,
                            annotation.annotator_hash,
                            annotation.source_label,
                            annotation.source_value,
                        ),
                    )
        return release_id

    def register_template(
        self, kind_or_name: TemplateKind | str, version: str, body: str
    ) -> TemplateId:
        template_id = TemplateId(_ulid())
        digest = hashlib.sha256(body.encode()).hexdigest()
        if isinstance(kind_or_name, TemplateKind):
            table = "instruction_template"
            name = kind_or_name.value
        else:
            table = "presentation_template"
            name = kind_or_name
        with self._transaction():
            self.connection.execute(
                f"INSERT INTO {table} VALUES (?,?,?,?,?)",
                (template_id, name, version, body, digest),
            )
        return template_id

    def _candidate_record(self, row: sqlite3.Row) -> CandidateRecord:
        turns = self.connection.execute(
            "SELECT role, text FROM candidate_turn WHERE candidate_id=? ORDER BY turn_index",
            (row["candidate_id"],),
        ).fetchall()
        annotations = self.connection.execute(
            "SELECT annotator_hash,source_label,source_value FROM source_annotation "
            "WHERE candidate_id=? ORDER BY annotation_index",
            (row["candidate_id"],),
        ).fetchall()
        return CandidateRecord(
            source_row_id=row["source_row_id"],
            rudeness_label=RudenessLabel(row["rudeness_label"]),
            turns=tuple((turn["role"], turn["text"]) for turn in turns),
            content_sha256=row["content_sha256"],
            annotations=tuple(
                SourceAnnotation(
                    annotation["annotator_hash"],
                    annotation["source_label"],
                    annotation["source_value"],
                )
                for annotation in annotations
            ),
        )

    def render_presentations(
        self,
        release_id: ReleaseId,
        template_id: TemplateId,
        renderer: Callable[[CandidateRecord], str],
    ) -> tuple[PresentationId, ...]:
        rows = self.connection.execute(
            "SELECT c.*,l.rudeness_label FROM candidate c JOIN candidate_label l "
            "ON l.candidate_id=c.candidate_id WHERE c.release_id=? ORDER BY c.candidate_id",
            (release_id,),
        ).fetchall()
        presentations: list[PresentationId] = []
        with self._transaction():
            for row in rows:
                rendered = renderer(self._candidate_record(row))
                presentation_id = PresentationId(_ulid())
                self.connection.execute(
                    "INSERT INTO candidate_presentation VALUES (?,?,?,?,?)",
                    (
                        presentation_id,
                        row["candidate_id"],
                        template_id,
                        rendered,
                        hashlib.sha256(rendered.encode()).hexdigest(),
                    ),
                )
                presentations.append(presentation_id)
        return tuple(presentations)

    def create_sample(
        self,
        release_id: ReleaseId,
        template_id: TemplateId,
        policy: SamplerPolicy,
        seed: int,
        members: Sequence[CandidateId],
        extra_draw: SampleExtraDraw | None = None,
    ) -> SampleId:
        if len(members) < 2:
            raise ValueError(
                "Sample creation failed because the member list has fewer than two candidates. Validation failed "
                "in experiment.store.create_sample before persistence, so no eliminations "
                "could run. Supply at least two distinct candidates from the release and retry."
            )
        sample_id = SampleId(_ulid())
        policy_rows = self.connection.execute(
            "SELECT DISTINCT l.label_policy_id FROM candidate_label l "
            f"WHERE l.candidate_id IN ({','.join('?' for _ in members)})",
            tuple(members),
        ).fetchall()
        if len(policy_rows) != 1:
            raise ValueError(
                "Sample creation failed because members do not resolve to exactly one versioned "
                "label policy. Validation failed before persistence. Select candidates labeled "
                "under the same policy and retry."
            )
        label_policy_id = policy_rows[0][0]
        with self._transaction():
            self.connection.execute(
                "INSERT INTO candidate_sample VALUES (?,?,?,?,?,?,?,?,NULL,NULL,NULL,NULL)",
                (
                    sample_id,
                    release_id,
                    label_policy_id,
                    template_id,
                    policy.value,
                    seed_to_blob(seed),
                    len(members),
                    "draft",
                ),
            )
            for position, candidate_id in enumerate(members):
                candidate_release = self.connection.execute(
                    "SELECT release_id FROM candidate WHERE candidate_id=?",
                    (candidate_id,),
                ).fetchone()
                if candidate_release is None or candidate_release[0] != release_id:
                    raise ValueError(
                        f"Sample creation failed because candidate {candidate_id} is absent "
                        f"from release {release_id}. Membership validation failed in "
                        "experiment.store.create_sample before commit, so the sample was not "
                        "created. Use candidate IDs from the selected release and retry."
                    )
                self.connection.execute(
                    "INSERT INTO candidate_sample_member VALUES (?,?,?)",
                    (sample_id, position, candidate_id),
                )
            if extra_draw is not None:
                if len(members) % 2 != 1:
                    raise ValueError(
                        "Sample draw persistence failed because an extra-stratum draw was "
                        "provided for an even sample. Validation failed in store.create_sample "
                        "before commit. Omit the draw for even sizes."
                    )
                draw_id = _ulid()
                selected = extra_draw.population[extra_draw.selected_index]
                self.connection.execute(
                    "INSERT INTO sample_rng_draw VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        draw_id,
                        sample_id,
                        "balanced-extra-stratum",
                        seed_to_blob(extra_draw.seed),
                        "qv-seed/v1",
                        _canonical_json(extra_draw.coordinates),
                        "pyrandom-randrange/v1",
                        extra_draw.selected_index,
                        selected.value,
                    ),
                )
                for draw_position, label in enumerate(extra_draw.population):
                    self.connection.execute(
                        "INSERT INTO sample_rng_draw_population VALUES (?,?,?)",
                        (draw_id, draw_position, label.value),
                    )
        return sample_id

    def freeze_sample(self, sample_id: SampleId, path: Path) -> FrozenCandidateSample:
        row = self.connection.execute(
            "SELECT * FROM candidate_sample WHERE sample_id=?", (sample_id,)
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Sample freeze failed because sample {sample_id} does not exist. Lookup failed "
                "in experiment.store.freeze_sample before F1, so no artifact changed. Create a "
                "DRAFT sample and retry under the writer lock."
            )
        requested = path.resolve()
        if row["status"] == "frozen":
            if Path(row["artifact_path"]) != requested:
                raise FreezeMismatchError(
                    f"Sample freeze refused {sample_id} because it is already FROZEN at "
                    f"{row['artifact_path']}, not {requested}. The mismatch was detected before "
                    "filesystem writes. Use the recorded artifact or create a new sample."
                )
            return self.reconcile_sample(sample_id, requested)
        if row["status"] == "freeze_pending":
            return self.reconcile_sample(sample_id, requested)

        ids = self._sample_member_ids(sample_id)
        sample = FrozenCandidateSample(ids)
        content = canonical_sample_bytes(sample)
        digest = hashlib.sha256(content).hexdigest()
        requested.parent.mkdir(parents=True, exist_ok=True)
        for stale in requested.parent.glob(f".{requested.name}.*.freeze.tmp"):
            stale.unlink(missing_ok=True)
        temp = write_fsynced_temp(requested, content)
        self._freeze_hook(FreezePoint.TEMP_FSYNC)
        with self._transaction():
            changed = self.connection.execute(
                "UPDATE candidate_sample SET status='freeze_pending',artifact_path=?,"
                "artifact_sha256=?,artifact_bytes=?,temp_basename=? "
                "WHERE sample_id=? AND status='draft'",
                (str(requested), digest, len(content), temp.name, sample_id),
            ).rowcount
            if changed != 1:
                temp.unlink(missing_ok=True)
                raise ValueError(
                    f"Sample freeze F1 failed because sample {sample_id} no longer has DRAFT "
                    "status. The intent transaction rolled back and its temp was removed. "
                    "Reconcile the current sample state under the writer lock."
                )
        self._freeze_hook(FreezePoint.F1_COMMIT)
        return self.reconcile_sample(sample_id, requested)

    def _sample_member_ids(self, sample_id: SampleId) -> tuple[str, ...]:
        return tuple(
            item[0]
            for item in self.connection.execute(
                "SELECT candidate_id FROM candidate_sample_member WHERE sample_id=? "
                "ORDER BY position",
                (sample_id,),
            )
        )

    def reconcile_sample(
        self, sample_id: SampleId, path: Path | None = None
    ) -> FrozenCandidateSample:
        """Idempotently finish F1→rename/fsync→F2 without overwriting conflicts."""
        row = self.connection.execute(
            "SELECT * FROM candidate_sample WHERE sample_id=?", (sample_id,)
        ).fetchone()
        if row is None or row["status"] == "draft":
            status = "missing" if row is None else "draft"
            raise ValueError(
                f"Sample reconciliation failed because sample {sample_id} has status {status}, "
                "not FREEZE_PENDING/FROZEN. Validation failed before filesystem access. Start "
                "freeze with an artifact path under the common writer lock."
            )
        final = Path(row["artifact_path"])
        if path is not None and path.resolve() != final:
            raise FreezeMismatchError(
                f"Sample reconciliation refused {sample_id} because requested path "
                f"{path.resolve()} differs from durable F1 path {final}. No file was replaced. "
                "Use the recorded path or inspect and recreate the sample."
            )
        sample = FrozenCandidateSample(self._sample_member_ids(sample_id))
        content = canonical_sample_bytes(sample)
        digest = hashlib.sha256(content).hexdigest()
        if digest != row["artifact_sha256"] or len(content) != row["artifact_bytes"]:
            raise FreezeMismatchError(
                f"Sample reconciliation refused {sample_id} because normalized DB members no "
                "longer match the F1 hash/length. Integrity validation failed before rename; "
                "FREEZE_PENDING remains for inspection. Restore the immutable member rows."
            )
        if final.exists():
            actual = final.read_bytes()
            if actual != content:
                raise FreezeMismatchError(
                    f"Sample reconciliation refused {sample_id} because existing final artifact "
                    f"{final} has SHA-256 {hashlib.sha256(actual).hexdigest()}, not F1 hash "
                    f"{digest}. It was not overwritten and FREEZE_PENDING remains. Restore the "
                    "expected bytes or move the conflicting file after forensic inspection."
                )
            if row["status"] == "freeze_pending":
                directory_fd = os.open(
                    final.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                self._freeze_hook(FreezePoint.DIRECTORY_FSYNC)
        elif row["status"] == "freeze_pending":
            temp = final.parent / row["temp_basename"]
            if not temp.exists() or temp.read_bytes() != content:
                temp.unlink(missing_ok=True)
                temp = write_fsynced_temp(final, content)
                with self._transaction():
                    self.connection.execute(
                        "UPDATE candidate_sample SET temp_basename=? WHERE sample_id=? "
                        "AND status='freeze_pending'",
                        (temp.name, sample_id),
                    )
            os.replace(temp, final)
            self._freeze_hook(FreezePoint.RENAME)
            directory_fd = os.open(
                final.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            self._freeze_hook(FreezePoint.DIRECTORY_FSYNC)
        parsed, actual_hash = read_frozen_sample(final)
        if parsed != sample or actual_hash != digest:
            raise FreezeMismatchError(
                f"Sample reconciliation F2 refused {sample_id} because strict parse/hash/order "
                f"of {final} differs from F1. FREEZE_PENDING remains. Preserve the file and "
                "inspect storage corruption before retrying."
            )
        if row["status"] == "freeze_pending":
            with self._transaction():
                changed = self.connection.execute(
                    "UPDATE candidate_sample SET status='frozen',temp_basename=NULL "
                    "WHERE sample_id=? AND status='freeze_pending'",
                    (sample_id,),
                ).rowcount
                if changed != 1:
                    raise RuntimeError(
                        f"Sample reconciliation F2 failed because sample {sample_id} did not "
                        "transition FREEZE_PENDING→FROZEN. The transaction rolled back. Reopen "
                        "under the writer lock and reconcile again."
                    )
            self._freeze_hook(FreezePoint.F2_COMMIT)
        refreshed = self.connection.execute(
            "SELECT * FROM candidate_sample WHERE sample_id=?", (sample_id,)
        ).fetchone()
        assert refreshed is not None
        write_sidecar(
            SampleSidecar(
                sample_id=sample_id,
                dataset_release_id=refreshed["release_id"],
                presentation_template_id=refreshed["template_id"],
                sampler_policy=SamplerPolicy(refreshed["sampler_policy"]),
                sampler_seed=seed_from_blob(refreshed["sampler_seed"]),
                artifact_sha256=digest,
            ),
            final,
        )
        return sample

    def validate_sample(self, path: Path) -> SampleId:
        sample, digest = read_frozen_sample(path)
        rows = self.connection.execute(
            "SELECT sample_id FROM candidate_sample WHERE status='frozen' AND artifact_sha256=?",
            (digest,),
        ).fetchall()
        for row in rows:
            expected = tuple(
                item[0]
                for item in self.connection.execute(
                    "SELECT candidate_id FROM candidate_sample_member WHERE sample_id=? "
                    "ORDER BY position",
                    (row[0],),
                )
            )
            if expected == sample.root:
                return SampleId(row[0])
        raise ValueError(
            f"Frozen sample validation failed because {path} hash {digest} and ordered "
            "membership do not match any FROZEN candidate_sample row. Validation failed in "
            "experiment.store.validate_sample before use, so matched runs must not trust "
            "this file. Restore the originally frozen bytes or run qv sample freeze for a "
            "new DRAFT sample."
        )

    def _insert_round(
        self,
        run_id: RunId,
        round_index: int,
        candidates: Sequence[CandidateId],
        arm: ElicitationArm,
    ) -> RoundId:
        round_id = RoundId(_ulid())
        self.connection.execute(
            'INSERT INTO "round" VALUES (?,?,?,?)',
            (round_id, run_id, round_index, "eliciting"),
        )
        for position, candidate_id in enumerate(candidates):
            self.connection.execute(
                "INSERT INTO round_candidate VALUES (?,?,?)",
                (round_id, candidate_id, position),
            )
        voters = self.connection.execute(
            "SELECT voter_id FROM voter WHERE run_id=? ORDER BY voter_index", (run_id,)
        ).fetchall()
        for voter in voters:
            for kind in arm_turn_order(arm):
                self.connection.execute(
                    "INSERT INTO turn VALUES (?,?,?,?,?)",
                    (_ulid(), round_id, voter[0], kind.value, "pending"),
                )
        return round_id

    def create_matched_set(
        self,
        sample: FrozenCandidateSample,
        sample_id: SampleId,
        config: MatchedSetConfig,
        definition: RunDefinition,
        artifact_path: Path | None = None,
        *,
        canonical_config: Mapping[str, object] | None = None,
    ) -> MatchedSetCreation:
        route_missing = {
            field
            for field in (
                "runtime_id",
                "tokenizer_repository",
                "tokenizer_revision",
                "dtype",
            )
            if getattr(definition, field) is None
        }
        if route_missing and config.execution_class is not ExecutionClass.FIXTURE:
            raise ValueError(
                f"Matched-set creation refused pilot/primary definition because route fields "
                f"are missing: {sorted(route_missing)}. Resolve the exact enabled static route "
                "before persistence; fixture-only defaults are not permitted."
            )
        definition = replace(
            definition,
            runtime_id=definition.runtime_id or "fixture-runtime",
            tokenizer_repository=definition.tokenizer_repository
            or definition.artifact_repository,
            tokenizer_revision=definition.tokenizer_revision
            or definition.artifact_revision,
            dtype=definition.dtype or definition.quantization_id,
        )
        sample_row = self.connection.execute(
            "SELECT * FROM candidate_sample WHERE sample_id=?", (sample_id,)
        ).fetchone()
        if sample_row is None or sample_row["status"] != "frozen":
            status = "missing" if sample_row is None else sample_row["status"]
            raise ValueError(
                f"Matched-set creation refused sample {sample_id} because its status is "
                f"{status}, not frozen. Validation failed in "
                "experiment.store.create_matched_set before run creation, so no run was "
                "created. Run `qv sample freeze --sample-id {sample_id}` and retry."
            )
        recorded_path = Path(sample_row["artifact_path"])
        supplied_path = (
            recorded_path if artifact_path is None else artifact_path.resolve()
        )
        if supplied_path != recorded_path:
            raise ValueError(
                f"Matched-set creation refused sample {sample_id} because supplied artifact "
                f"path {supplied_path} differs from frozen path {recorded_path}. Binding "
                "validation failed before config insertion, so no runs were created. Use the "
                "sample_id and exact recorded path from the strict config."
            )
        file_sample, file_hash = read_frozen_sample(supplied_path)
        expected = tuple(
            row[0]
            for row in self.connection.execute(
                "SELECT candidate_id FROM candidate_sample_member WHERE sample_id=? "
                "ORDER BY position",
                (sample_id,),
            )
        )
        artifact_json = json.dumps(
            sample.model_dump(mode="json"), separators=(",", ":")
        )
        actual_hash = hashlib.sha256(artifact_json.encode()).hexdigest()
        if (
            sample.root != expected
            or file_sample != sample
            or actual_hash != sample_row["artifact_sha256"]
            or file_hash != actual_hash
        ):
            raise ValueError(
                f"Matched-set creation refused sample {sample_id} because the supplied "
                f"artifact hash {actual_hash} or membership differs from database hash "
                f"{sample_row['artifact_sha256']}. Validation failed before six-run creation, "
                "so no trajectories were created. Validate the frozen file and pass its "
                "unchanged FrozenCandidateSample value."
            )
        release_hash = self.connection.execute(
            "SELECT file_sha256 FROM dataset_release WHERE release_id=?",
            (sample_row["release_id"],),
        ).fetchone()[0]
        presentation = self.connection.execute(
            "SELECT body_sha256 FROM presentation_template WHERE template_id=?",
            (sample_row["template_id"],),
        ).fetchone()
        if (
            definition.sample_artifact_hash != actual_hash
            or definition.dataset_release_hash != release_hash
            or definition.presentation_template_id != sample_row["template_id"]
            or presentation is None
            or definition.presentation_template_hash != presentation[0]
        ):
            raise ValueError(
                f"Matched-set creation refused run definition for sample {sample_id} because "
                "its sample, release, or presentation-template identity/hash does not match "
                "the frozen catalog state. Provenance validation failed in "
                "experiment.store.create_matched_set before any run was inserted, so replay "
                "would not be auditable. Rebuild RunDefinition from the selected sample's "
                "recorded artifact, release, and template hashes, then retry."
            )
        if set(definition.instruction_templates) != set(TemplateKind):
            raise ValueError(
                f"Matched-set creation refused run definition for sample {sample_id} because "
                "instruction template references do not contain exactly setup, statement, "
                "ballot, correction, result, and final-result. Validation failed before run "
                "creation, so transcript reconstruction remains complete. Register every "
                "TemplateKind and pass each immutable ID/hash pair."
            )
        for kind, (
            template_id,
            expected_hash,
        ) in definition.instruction_templates.items():
            template = self.connection.execute(
                "SELECT name,body_sha256 FROM instruction_template WHERE template_id=?",
                (template_id,),
            ).fetchone()
            if (
                template is None
                or template["name"] != kind.value
                or template["body_sha256"] != expected_hash
            ):
                raise ValueError(
                    f"Matched-set creation refused instruction template {template_id} for "
                    f"{kind.value} because its name/hash differs from the registered immutable "
                    "row. Provenance validation failed before any run was inserted. Restore or "
                    "register the intended template and pass its exact ID/hash."
                )
        label_policy = self.connection.execute(
            "SELECT name,version,rule_sha256 FROM label_policy WHERE label_policy_id=?",
            (sample_row["label_policy_id"],),
        ).fetchone()
        assert label_policy is not None
        instruction_json = _canonical_json(
            {
                kind.value: [str(value[0]), value[1]]
                for kind, value in definition.instruction_templates.items()
            }
        )
        derived = {
            "route_registry_hash": hashlib.sha256(
                _canonical_json(
                    {
                        "model_id": definition.model_id,
                        "provider_id": definition.provider_id,
                        "quantization_id": definition.quantization_id,
                        "runtime_id": definition.runtime_id,
                        "artifact_repository": definition.artifact_repository,
                        "artifact_revision": definition.artifact_revision,
                        "tokenizer_repository": definition.tokenizer_repository,
                        "tokenizer_revision": definition.tokenizer_revision,
                        "dtype": definition.dtype,
                    }
                ).encode()
            ).hexdigest(),
            "sampling_profile_hash": hashlib.sha256(
                _canonical_json(asdict(config.sampling)).encode()
            ).hexdigest(),
            "instruction_profile_hash": hashlib.sha256(
                instruction_json.encode()
            ).hexdigest(),
            "source_release_id": str(sample_row["release_id"]),
            "label_policy_id": str(sample_row["label_policy_id"]),
            "label_policy_version": str(label_policy["version"]),
            "label_policy_hash": str(label_policy["rule_sha256"]),
            "sample_id": str(sample_id),
        }
        mismatches = {
            field: (getattr(definition, field), expected_value)
            for field, expected_value in derived.items()
            if getattr(definition, field) not in (None, expected_value)
        }
        if mismatches:
            raise ValueError(
                f"Matched-set creation refused definition bindings for sample {sample_id} "
                f"because immutable release/label/sample/profile values differ: {mismatches}. "
                "Validation failed before config insertion. Rebuild the definition from the "
                "strict config and normalized catalog rows."
            )
        definition = replace(
            definition,
            route_registry_hash=derived["route_registry_hash"],
            sampling_profile_hash=derived["sampling_profile_hash"],
            instruction_profile_hash=derived["instruction_profile_hash"],
            source_release_id=derived["source_release_id"],
            label_policy_id=derived["label_policy_id"],
            label_policy_version=derived["label_policy_version"],
            label_policy_hash=derived["label_policy_hash"],
            sample_id=derived["sample_id"],
        )
        definition_json = _canonical_json(asdict(definition))
        definition_hash = hashlib.sha256(definition_json.encode()).hexdigest()
        config_json = _canonical_json(
            canonical_config
            if canonical_config is not None
            else {"config": asdict(config), "definition_hash": definition_hash}
        )
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()
        matched_set_id = MatchedSetId(_ulid())
        run_ids: dict[tuple[ElicitationArm, VotingRegime], RunId] = {}
        with self._transaction():
            artifact = self.connection.execute(
                "SELECT artifact_id FROM model_artifact WHERE repository=? AND revision=?",
                (definition.artifact_repository, definition.artifact_revision),
            ).fetchone()
            artifact_id = artifact[0] if artifact is not None else _ulid()
            if artifact is None:
                self.connection.execute(
                    "INSERT INTO model_artifact VALUES (?,?,?)",
                    (
                        artifact_id,
                        definition.artifact_repository,
                        definition.artifact_revision,
                    ),
                )
            tokenizer = self.connection.execute(
                "SELECT tokenizer_id FROM tokenizer_artifact WHERE repository=? AND revision=?",
                (definition.tokenizer_repository, definition.tokenizer_revision),
            ).fetchone()
            tokenizer_id = tokenizer[0] if tokenizer is not None else _ulid()
            if tokenizer is None:
                self.connection.execute(
                    "INSERT INTO tokenizer_artifact VALUES (?,?,?)",
                    (
                        tokenizer_id,
                        definition.tokenizer_repository,
                        definition.tokenizer_revision,
                    ),
                )
            route = self.connection.execute(
                "SELECT route_id,registry_hash,enabled FROM model_route WHERE model_id=? "
                "AND provider_id=? AND quantization_id=? AND runtime_id=?",
                (
                    definition.model_id,
                    definition.provider_id,
                    definition.quantization_id,
                    definition.runtime_id,
                ),
            ).fetchone()
            if route is None:
                if config.execution_class is not ExecutionClass.FIXTURE:
                    raise ValueError(
                        "Matched-set creation refused an unregistered model route for pilot/"
                        "primary execution. Route resolution failed before config insertion. "
                        "Register the exact enabled static route and retry."
                    )
                route_id = _ulid()
                self.connection.execute(
                    "INSERT INTO model_route VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        route_id,
                        definition.model_id,
                        definition.provider_id,
                        definition.quantization_id,
                        definition.runtime_id,
                        artifact_id,
                        tokenizer_id,
                        definition.dtype,
                        definition.route_registry_hash,
                        1,
                    ),
                )
            else:
                route_id = route["route_id"]
                if (
                    route["registry_hash"] != definition.route_registry_hash
                    or not route["enabled"]
                ):
                    raise ValueError(
                        "Matched-set creation refused the model route because its registry hash "
                        "or enabled state differs from the immutable definition. Create/use the "
                        "exact enabled static route; no override exists."
                    )
            sampling_row = self.connection.execute(
                "SELECT sampling_profile_id FROM sampling_profile WHERE profile_hash=?",
                (definition.sampling_profile_hash,),
            ).fetchone()
            sampling_profile_id = (
                sampling_row[0] if sampling_row is not None else _ulid()
            )
            if sampling_row is None:
                self.connection.execute(
                    "INSERT INTO sampling_profile VALUES (?,?,?,?,?,?)",
                    (
                        sampling_profile_id,
                        definition.sampling_profile_hash,
                        float(config.sampling.temperature),
                        float(config.sampling.top_p),
                        config.sampling.top_k,
                        config.sampling.max_new_tokens,
                    ),
                )
            profile_row = self.connection.execute(
                "SELECT profile_id FROM instruction_profile WHERE profile_hash=?",
                (definition.instruction_profile_hash,),
            ).fetchone()
            instruction_profile_id = (
                profile_row[0] if profile_row is not None else _ulid()
            )
            if profile_row is None:
                self.connection.execute(
                    "INSERT INTO instruction_profile VALUES (?,?,?,?,?)",
                    (
                        instruction_profile_id,
                        definition.instruction_profile_hash,
                        int(definition.prompt_reviewed),
                        definition.prompt_review_version,
                        definition.prompt_review_sha256,
                    ),
                )
                for kind, (
                    template_id,
                    _digest,
                ) in definition.instruction_templates.items():
                    self.connection.execute(
                        "INSERT INTO instruction_profile_member VALUES (?,?,?)",
                        (instruction_profile_id, kind.value, template_id),
                    )
            retry_row = self.connection.execute(
                "SELECT turn_retry_policy_id FROM turn_retry_policy WHERE max_corrections=?",
                (config.retry_policy.max_correction_attempts,),
            ).fetchone()
            turn_retry_policy_id = retry_row[0] if retry_row is not None else _ulid()
            if retry_row is None:
                self.connection.execute(
                    "INSERT INTO turn_retry_policy VALUES (?,?)",
                    (turn_retry_policy_id, config.retry_policy.max_correction_attempts),
                )
            runtime_row = self.connection.execute(
                "SELECT runtime_retry_policy_id FROM runtime_retry_policy WHERE "
                "max_failures_per_execution=3 AND initial_backoff_ms=1000 "
                "AND multiplier=2.0 AND max_backoff_ms=2000"
            ).fetchone()
            runtime_retry_policy_id = (
                runtime_row[0] if runtime_row is not None else _ulid()
            )
            if runtime_row is None:
                self.connection.execute(
                    "INSERT INTO runtime_retry_policy VALUES (?,3,1000,2.0,2000)",
                    (runtime_retry_policy_id,),
                )
            definition_row = self.connection.execute(
                "SELECT definition_id FROM experiment_definition WHERE definition_hash=?",
                (definition_hash,),
            ).fetchone()
            definition_id = definition_row[0] if definition_row is not None else _ulid()
            if definition_row is None:
                self.connection.execute(
                    "INSERT INTO experiment_definition VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        definition_id,
                        definition_hash,
                        route_id,
                        sampling_profile_id,
                        instruction_profile_id,
                        definition.presentation_template_id,
                        definition.source_release_id,
                        definition.label_policy_id,
                        definition.sample_id,
                        definition.canonical_json_version,
                        definition.prompt_encoding_version,
                        definition.seed_version,
                    ),
                )
            config_id = _ulid()
            self.connection.execute(
                "INSERT INTO experiment_config_record VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    config_id,
                    config_hash,
                    definition_id,
                    turn_retry_policy_id,
                    turn_retry_policy_id,
                    runtime_retry_policy_id,
                    seed_to_blob(config.master_seed),
                    config.credit_budget,
                    config.voter_count,
                    config.tie_policy.value,
                    config.presentation_policy.value,
                    config.action_format.value,
                    "qv-run-config/v1",
                    "balanced-matched/v1",
                    config.execution_class.value,
                ),
            )
            self.connection.execute(
                "INSERT INTO matched_set VALUES (?,?,?)",
                (matched_set_id, config_id, _now()),
            )
            permutations = {
                voter_index: voter_permutation_draw(
                    config.master_seed, voter_index
                ).permutation(tuple(CandidateId(value) for value in sample.root))
                for voter_index in range(config.voter_count)
            }
            for arm in ElicitationArm:
                for regime in VotingRegime:
                    run_id = RunId(_ulid())
                    run_ids[(arm, regime)] = run_id
                    self.connection.execute(
                        "INSERT INTO experiment_run VALUES (?,?,?,?,?,NULL)",
                        (run_id, matched_set_id, arm.value, regime.value, "created"),
                    )
                    for voter_index, draw in permutations.items():
                        voter_id = VoterId(_ulid())
                        self.connection.execute(
                            "INSERT INTO voter VALUES (?,?,?,?,?,?)",
                            (
                                voter_id,
                                run_id,
                                voter_index,
                                seed_to_blob(draw.seed),
                                draw.algorithm.value,
                                _canonical_json(draw.coordinates),
                            ),
                        )
                        for position, candidate_id in enumerate(draw.permutation):
                            self.connection.execute(
                                "INSERT INTO voter_permutation VALUES (?,?,?)",
                                (voter_id, position, candidate_id),
                            )
                    self._insert_round(
                        run_id,
                        1,
                        tuple(CandidateId(value) for value in sample.root),
                        arm,
                    )
        return MatchedSetCreation(matched_set_id, run_ids)

    def register_static_route(self, definition: RunDefinition) -> str:
        """Register one immutable enabled route before pilot/primary creation."""
        if self._writer_lock is None or not lock_matches_database(
            self._writer_lock, self.path
        ):
            raise RuntimeError(
                "Static-route registration requires the common writer lock before database "
                "open. No route was changed. Reopen with writer_lock and retry."
            )
        required = {
            field: getattr(definition, field)
            for field in (
                "runtime_id",
                "tokenizer_repository",
                "tokenizer_revision",
                "dtype",
                "route_registry_hash",
            )
            if getattr(definition, field) is None
        }
        if required:
            raise ValueError(
                f"Static-route registration refused incomplete fields {sorted(required)}. "
                "No route changed. Resolve the complete pinned route and retry."
            )
        with self._transaction():
            artifact = self.connection.execute(
                "SELECT artifact_id FROM model_artifact WHERE repository=? AND revision=?",
                (definition.artifact_repository, definition.artifact_revision),
            ).fetchone()
            artifact_id = artifact[0] if artifact is not None else _ulid()
            if artifact is None:
                self.connection.execute(
                    "INSERT INTO model_artifact VALUES (?,?,?)",
                    (
                        artifact_id,
                        definition.artifact_repository,
                        definition.artifact_revision,
                    ),
                )
            tokenizer = self.connection.execute(
                "SELECT tokenizer_id FROM tokenizer_artifact WHERE repository=? AND revision=?",
                (definition.tokenizer_repository, definition.tokenizer_revision),
            ).fetchone()
            tokenizer_id = tokenizer[0] if tokenizer is not None else _ulid()
            if tokenizer is None:
                self.connection.execute(
                    "INSERT INTO tokenizer_artifact VALUES (?,?,?)",
                    (
                        tokenizer_id,
                        definition.tokenizer_repository,
                        definition.tokenizer_revision,
                    ),
                )
            existing = self.connection.execute(
                "SELECT * FROM model_route WHERE model_id=? AND provider_id=? "
                "AND quantization_id=? AND runtime_id=?",
                (
                    definition.model_id,
                    definition.provider_id,
                    definition.quantization_id,
                    definition.runtime_id,
                ),
            ).fetchone()
            if existing is not None:
                expected = (
                    artifact_id,
                    tokenizer_id,
                    definition.dtype,
                    definition.route_registry_hash,
                    1,
                )
                actual = (
                    existing["artifact_id"],
                    existing["tokenizer_id"],
                    existing["dtype"],
                    existing["registry_hash"],
                    existing["enabled"],
                )
                if actual != expected:
                    raise DefinitionDriftError(
                        f"Static route registration refused immutable drift: persisted={actual}, "
                        f"requested={expected}. No route changed. Register a new versioned route."
                    )
                return str(existing["route_id"])
            route_id = _ulid()
            self.connection.execute(
                "INSERT INTO model_route VALUES (?,?,?,?,?,?,?,?,?,1)",
                (
                    route_id,
                    definition.model_id,
                    definition.provider_id,
                    definition.quantization_id,
                    definition.runtime_id,
                    artifact_id,
                    tokenizer_id,
                    definition.dtype,
                    definition.route_registry_hash,
                ),
            )
            return route_id

    def record_run_fork(
        self,
        parent: MatchedSetId,
        child: MatchedSetId,
        reason: RunForkReason,
    ) -> None:
        """Persist an immutable typed provenance edge between matched sets."""
        if not isinstance(reason, RunForkReason):
            raise ValueError(
                "Run-fork recording refused an untyped reason before persistence. Construct a "
                "RunForkReason and retry; arbitrary strings are not durable provenance."
            )
        with self._transaction():
            self.connection.execute(
                "INSERT INTO run_fork VALUES (?,?,?,?)",
                (child, parent, reason.value, _now()),
            )

    def create_matched_set_v1(self, config: MatchedSetConfigV1) -> MatchedSetCreation:
        """Verify the complete strict config graph and atomically create its six runs."""
        if self._writer_lock is None or not lock_matches_database(
            self._writer_lock, self.path
        ):
            raise RuntimeError(
                f"Strict matched-set creation refused {self.path} because the store was not "
                "opened with its live common WriterLock. Lock validation failed before graph "
                "reads or insertion. Acquire the lock before open, pass writer_lock=lock with "
                "require_writer_lock=True, and hold it through store close."
            )
        sample_id = SampleId(config.sample.sample_id)
        row = self.connection.execute(
            "SELECT s.*,r.dataset_name,r.version AS release_version,r.file_sha256,"
            "lp.name AS label_policy_name,lp.version AS label_policy_version,lp.rule_sha256,"
            "lp.reviewed AS label_policy_reviewed,lp.review_version AS label_policy_review_version,"
            "lp.review_sha256 AS label_policy_review_sha256,"
            "pt.name AS presentation_template_name,pt.version AS presentation_template_version,"
            "pt.body_sha256 FROM candidate_sample s JOIN dataset_release r "
            "ON r.release_id=s.release_id JOIN label_policy lp "
            "ON lp.label_policy_id=s.label_policy_id JOIN presentation_template pt "
            "ON pt.template_id=s.template_id WHERE s.sample_id=?",
            (sample_id,),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Strict matched-set creation failed because sample {sample_id} does not exist. "
                "Graph verification failed in store.create_matched_set_v1 before insertion. "
                "Create and freeze the selected sample, then retry with its exact identifiers."
            )
        expected = {
            "status": (row["status"], "frozen"),
            "artifact_path": (
                str(Path(row["artifact_path"])),
                str(config.sample.artifact_path.resolve()),
            ),
            "artifact_sha256": (row["artifact_sha256"], config.sample.expected_sha256),
            "release_id": (row["release_id"], config.sample.release.release_id),
            "dataset_name": (row["dataset_name"], config.sample.release.dataset_name),
            "release_version": (row["release_version"], config.sample.release.version),
            "release_hash": (row["file_sha256"], config.sample.release.expected_sha256),
            "label_policy_id": (
                row["label_policy_id"],
                config.sample.label_policy.label_policy_id,
            ),
            "label_policy_name": (
                row["label_policy_name"],
                config.sample.label_policy.name,
            ),
            "label_policy_version": (
                row["label_policy_version"],
                config.sample.label_policy.version,
            ),
            "label_policy_hash": (
                row["rule_sha256"],
                config.sample.label_policy.expected_sha256,
            ),
            "label_policy_reviewed": (
                bool(row["label_policy_reviewed"]),
                config.sample.label_policy.reviewed,
            ),
            "label_policy_review_version": (
                row["label_policy_review_version"],
                config.sample.label_policy.review_version,
            ),
            "label_policy_review_sha256": (
                row["label_policy_review_sha256"],
                config.sample.label_policy.review_sha256,
            ),
            "presentation_template_id": (
                row["template_id"],
                config.sample.presentation_template.template_id,
            ),
            "presentation_template_name": (
                row["presentation_template_name"],
                config.sample.presentation_template.name,
            ),
            "presentation_template_version": (
                row["presentation_template_version"],
                config.sample.presentation_template.version,
            ),
            "presentation_template_hash": (
                row["body_sha256"],
                config.sample.presentation_template.expected_sha256,
            ),
        }
        mismatches = {
            field: values
            for field, values in expected.items()
            if values[0] != values[1]
        }
        if mismatches:
            raise ValueError(
                f"Strict matched-set creation refused sample {sample_id} because config graph "
                f"bindings differ from normalized rows: {mismatches}. Verification failed "
                "before config insertion. Regenerate the config from the frozen catalog state."
            )
        artifact_sample, artifact_hash = read_frozen_sample(
            config.sample.artifact_path.resolve()
        )
        members = self._sample_member_ids(sample_id)
        if (
            artifact_sample.root != members
            or artifact_hash != config.sample.expected_sha256
        ):
            raise ValueError(
                f"Strict matched-set creation refused sample {sample_id} because artifact hash/"
                "order differs from normalized sample members. Verification failed before "
                "insertion. Restore the frozen artifact bytes and retry."
            )
        presentations = self.connection.execute(
            "SELECT sm.candidate_id,cp.rendered_text,cp.rendered_sha256,cl.candidate_id AS labeled "
            "FROM candidate_sample_member sm LEFT JOIN candidate_label cl "
            "ON cl.candidate_id=sm.candidate_id AND cl.label_policy_id=? "
            "LEFT JOIN candidate_presentation cp ON cp.candidate_id=sm.candidate_id "
            "AND cp.template_id=? WHERE sm.sample_id=? ORDER BY sm.position",
            (row["label_policy_id"], row["template_id"], sample_id),
        ).fetchall()
        invalid_presentations = [
            item["candidate_id"]
            for item in presentations
            if item["labeled"] is None
            or item["rendered_text"] is None
            or hashlib.sha256(item["rendered_text"].encode()).hexdigest()
            != item["rendered_sha256"]
        ]
        if invalid_presentations:
            raise ValueError(
                f"Strict matched-set creation refused sample {sample_id} because candidate label/"
                f"presentation lineage is incomplete or hash-invalid: {invalid_presentations}. "
                "Verification failed before insertion. Re-render and re-freeze a new sample."
            )
        prompt_selectors = {
            TemplateKind.SETUP: config.prompts.setup,
            TemplateKind.STATEMENT: config.prompts.statement,
            TemplateKind.BALLOT: config.prompts.ballot,
            TemplateKind.CORRECTION: config.prompts.correction,
            TemplateKind.RESULT: config.prompts.result,
            TemplateKind.FINAL_RESULT: config.prompts.final_result,
        }
        instructions: dict[TemplateKind, tuple[TemplateId, str]] = {}
        for kind, selector in prompt_selectors.items():
            template = self.connection.execute(
                "SELECT name,version,body_sha256 FROM instruction_template WHERE template_id=?",
                (selector.template_id,),
            ).fetchone()
            if (
                selector.name != kind.value
                or template is None
                or (template["name"], template["version"], template["body_sha256"])
                != (kind.value, selector.version, selector.expected_sha256)
            ):
                raise ValueError(
                    f"Strict matched-set creation refused {kind.value} template "
                    f"{selector.template_id} because ID/name/version/hash does not resolve "
                    "exactly once. Verification failed before insertion. Register the exact "
                    "reviewed template and retry."
                )
            instructions[kind] = (
                TemplateId(selector.template_id),
                selector.expected_sha256,
            )
        legacy_config = MatchedSetConfig(
            voter_count=config.voter_count,
            master_seed=config.master_seed,
            sampling=SamplingProfile(
                temperature=config.sampling.temperature,
                top_p=config.sampling.top_p,
                top_k=config.sampling.top_k,
                max_new_tokens=config.sampling.max_new_tokens,
            ),
            credit_budget=config.credit_budget,
            max_consecutive_runtime_failures=config.runtime_retry.max_failures_per_execution,
            execution_class=ExecutionClass(config.execution_class),
        )
        instruction_json = _canonical_json(
            {
                kind.value: [str(value[0]), value[1]]
                for kind, value in instructions.items()
            }
        )
        route_json = _canonical_json(config.route.model_dump(mode="json"))
        definition = RunDefinition(
            model_id=config.route.model_id,
            provider_id=config.route.provider_id,
            quantization_id=config.route.quantization_id,
            artifact_repository=config.route.artifact_repository,
            artifact_revision=config.route.artifact_revision,
            presentation_template_id=TemplateId(row["template_id"]),
            presentation_template_hash=row["body_sha256"],
            instruction_templates=instructions,
            dataset_release_hash=row["file_sha256"],
            sample_artifact_hash=artifact_hash,
            runtime_id=config.route.runtime_id,
            tokenizer_repository=config.route.tokenizer_repository,
            tokenizer_revision=config.route.tokenizer_revision,
            dtype=config.route.dtype,
            route_registry_hash=hashlib.sha256(route_json.encode()).hexdigest(),
            sampling_profile_hash=hashlib.sha256(
                _canonical_json(config.sampling.model_dump(mode="json")).encode()
            ).hexdigest(),
            instruction_profile_hash=hashlib.sha256(
                instruction_json.encode()
            ).hexdigest(),
            canonical_json_version=config.canonical_json_version,
            prompt_encoding_version=config.prompt_encoding_version,
            seed_version=config.seed_version,
            source_release_id=row["release_id"],
            label_policy_id=row["label_policy_id"],
            label_policy_version=row["label_policy_version"],
            label_policy_hash=row["rule_sha256"],
            sample_id=str(sample_id),
            prompt_reviewed=config.prompts.reviewed,
            prompt_review_version=config.prompts.review_version,
            prompt_review_sha256=config.prompts.review_sha256,
        )
        return self.create_matched_set(
            artifact_sample,
            sample_id,
            legacy_config,
            definition,
            config.sample.artifact_path,
            canonical_config=config.model_dump(mode="json"),
        )

    def begin_execution(
        self, run_id: RunId, env: ExecutionEnvironment, drift_override: bool = False
    ) -> ExecutionId:
        if drift_override:
            raise DefinitionDriftError(
                f"Execution start refused run {run_id} because drift_override=True was supplied. "
                "Immutable model-definition drift has no override in qv-run-config/v1. The "
                "failure occurred in store.begin_execution before an execution row was created. "
                "Create a linked fork/new matched set for a changed treatment."
            )
        self.preflight_execution(run_id, env)
        previous = self.connection.execute(
            "SELECT * FROM run_execution WHERE run_id=? ORDER BY started_at DESC LIMIT 1",
            (run_id,),
        ).fetchone()
        env_values = asdict(env)
        drift: dict[str, tuple[object, object]] = {}
        if previous is not None:
            boolean_fields = {
                "git_dirty",
                "deterministic_algorithms",
                "tf32_enabled",
                "cudnn_benchmark",
            }
            for field, current in env_values.items():
                old: object = (
                    bool(previous[field])
                    if field in boolean_fields
                    else previous[field]
                )
                if old != current:
                    drift[field] = (old, current)
        self.last_environment_drift = drift
        execution_id = ExecutionId(_ulid())
        with self._transaction():
            self.connection.execute(
                "INSERT INTO run_execution (execution_id,run_id,python_version,torch_version,"
                "transformers_version,uv_lock_hash,device,dtype,hostname,git_commit,git_dirty,"
                "started_at,ended_at,exit_reason,drift_override,environment_drift_json,"
                "cuda_runtime_version,nvidia_driver_version,cudnn_version,gpu_model,gpu_count,"
                "gpu_compute_capability,gpu_uuid_hash,os_name,os_version,kernel_version,"
                "cpu_architecture,deterministic_algorithms,tf32_enabled,cudnn_benchmark,"
                "tracked_tree_hash,binary_diff_sha256,untracked_manifest_hash,"
                "untracked_tree_hash,hostname_hash) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
                "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    execution_id,
                    run_id,
                    env.python_version,
                    env.torch_version,
                    env.transformers_version,
                    env.uv_lock_hash,
                    env.device,
                    env.dtype,
                    env.hostname,
                    env.git_commit,
                    int(env.git_dirty),
                    _now(),
                    None,
                    None,
                    0,
                    _canonical_json(drift),
                    env.cuda_runtime_version,
                    env.nvidia_driver_version,
                    env.cudnn_version,
                    env.gpu_model,
                    env.gpu_count,
                    env.gpu_compute_capability,
                    env.gpu_uuid_hash,
                    env.os_name,
                    env.os_version,
                    env.kernel_version,
                    env.cpu_architecture,
                    int(env.deterministic_algorithms),
                    int(env.tf32_enabled),
                    int(env.cudnn_benchmark),
                    env.tracked_tree_hash,
                    env.binary_diff_sha256,
                    env.untracked_manifest_hash,
                    env.untracked_tree_hash,
                    env.hostname_hash,
                ),
            )
        return execution_id

    def preflight_execution(
        self,
        run_id: RunId,
        env: ExecutionEnvironment,
        current_definition: RunDefinition | None = None,
    ) -> None:
        """Validate immutable definition and primary source state without writing."""
        definition = self.connection.execute(
            "SELECT d.*,c.execution_class FROM run_definition d JOIN experiment_run r "
            "ON r.run_id=d.run_id JOIN matched_set m ON m.matched_set_id=r.matched_set_id "
            "JOIN experiment_config c ON c.config_id=m.config_id WHERE d.run_id=?",
            (run_id,),
        ).fetchone()
        if definition is None:
            raise ValueError(
                f"Execution start failed because run {run_id} has no immutable run_definition. "
                "Lookup failed in experiment.store.begin_execution before an execution row "
                "was appended, so provenance cannot be audited. Recreate the matched set from "
                "a frozen sample instead of running this incomplete database record."
            )
        if current_definition is not None:
            self.verify_run_definition(run_id, current_definition)
        expected_dtype = definition["dtype"]
        if env.dtype.casefold() != str(expected_dtype).casefold():
            raise DefinitionDriftError(
                f"Execution start refused run {run_id} because environment dtype {env.dtype!r} "
                f"conflicts with immutable dtype {expected_dtype!r} "
                f"for {definition['artifact_repository']}@{definition['artifact_revision']}. "
                "The conflict was detected in experiment.store.begin_execution before model "
                "loading or execution-row creation, so continuing could change results. Select "
                "the frozen route or create a linked fork; no override exists."
            )
        if (
            definition["execution_class"] == ExecutionClass.PRIMARY.value
            and env.git_dirty
        ):
            raise DirtyPrimaryTreeError(
                f"Execution start refused primary run {run_id} because git_dirty=True. The "
                "dirty-tree preflight failed in store.begin_execution before execution-row "
                "creation, so no primary observation was started. Commit/review the exact tree "
                "or create a non-primary pilot definition and retry."
            )

    def verify_run_definition(self, run_id: RunId, current: RunDefinition) -> None:
        """Fail on every immutable model/template/sample definition mismatch."""
        row = self.connection.execute(
            "SELECT * FROM run_definition WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise DefinitionDriftError(
                f"Definition verification failed because run {run_id} has no run_definition. "
                "The preflight stopped before execution. Recreate the matched set."
            )
        expected_templates = _canonical_json(
            {
                kind.value: [str(value[0]), value[1]]
                for kind, value in current.instruction_templates.items()
            }
        )
        values: dict[str, object] = {
            "model_id": current.model_id,
            "provider_id": current.provider_id,
            "quantization_id": current.quantization_id,
            "artifact_repository": current.artifact_repository,
            "artifact_revision": current.artifact_revision,
            "presentation_template_id": current.presentation_template_id,
            "presentation_template_hash": current.presentation_template_hash,
            "instruction_templates_json": expected_templates,
            "dataset_release_hash": current.dataset_release_hash,
            "sample_artifact_hash": current.sample_artifact_hash,
            "runtime_id": current.runtime_id,
            "tokenizer_repository": current.tokenizer_repository,
            "tokenizer_revision": current.tokenizer_revision,
            "dtype": current.dtype,
            "route_registry_hash": current.route_registry_hash,
            "sampling_profile_hash": current.sampling_profile_hash,
            "instruction_profile_hash": current.instruction_profile_hash,
            "canonical_json_version": current.canonical_json_version,
            "prompt_encoding_version": current.prompt_encoding_version,
            "seed_version": current.seed_version,
            "source_release_id": current.source_release_id,
            "label_policy_id": current.label_policy_id,
            "label_policy_version": current.label_policy_version,
            "label_policy_hash": current.label_policy_hash,
            "sample_id": current.sample_id,
        }
        mismatches = {
            field: (row[field], value)
            for field, value in values.items()
            if row[field] != value
        }
        if mismatches:
            raise DefinitionDriftError(
                f"Definition verification refused run {run_id} because immutable fields differ: "
                f"{mismatches}. The mismatch was detected in store.verify_run_definition before "
                "execution. Create a linked fork/new matched set; no override exists."
            )

    def end_execution(self, execution_id: ExecutionId, exit_reason: str) -> None:
        if exit_reason not in {"completed", "paused", "interrupted", "error"}:
            raise ValueError(
                f"Execution end failed because exit_reason={exit_reason!r} is not a closed "
                "execution reason. Validation failed in experiment.store.end_execution, so "
                "the append-only execution remains open. Use completed, paused, interrupted, "
                "or error and retry."
            )
        with self._transaction():
            changed = self.connection.execute(
                "UPDATE run_execution SET ended_at=?, exit_reason=? "
                "WHERE execution_id=? AND ended_at IS NULL",
                (_now(), exit_reason, execution_id),
            ).rowcount
            if changed != 1:
                raise ValueError(
                    f"Execution end failed because execution {execution_id} is missing or "
                    "already ended. The transition check failed in "
                    "experiment.store.end_execution, so no provenance was overwritten. End "
                    "each execution exactly once using its returned execution ID."
                )

    def next_incomplete_unit(self, run_id: RunId) -> NextUnit:
        run = self.connection.execute(
            "SELECT status, arm FROM experiment_run WHERE run_id=?", (run_id,)
        ).fetchone()
        if run is None:
            raise ValueError(
                f"Resume lookup failed because run {run_id} does not exist. Lookup failed in "
                "experiment.store.next_incomplete_unit before state traversal, so there is "
                "nothing safe to execute. Supply a run ID created by create_matched_set."
            )
        if run["status"] == "complete":
            return RunComplete(run_id)
        round_row = self.connection.execute(
            "SELECT * FROM \"round\" WHERE run_id=? AND phase='eliciting' "
            "ORDER BY round_index LIMIT 1",
            (run_id,),
        ).fetchone()
        if round_row is None:
            raise RuntimeError(
                f"Resume lookup failed because incomplete run {run_id} has no eliciting round. "
                "State validation failed in experiment.store.next_incomplete_unit, so execution "
                "cannot advance without risking a duplicate result. Inspect or restore the "
                "database from the last durable commit."
            )
        attempt_limit = self.connection.execute(
            "SELECT c.ballot_max_corrections+1 FROM matched_set m "
            "JOIN experiment_config c ON c.config_id=m.config_id JOIN experiment_run r "
            "ON r.matched_set_id=m.matched_set_id WHERE r.run_id=?",
            (run_id,),
        ).fetchone()[0]
        for voter in self.connection.execute(
            "SELECT voter_id, voter_index FROM voter WHERE run_id=? ORDER BY voter_index",
            (run_id,),
        ):
            for kind in arm_turn_order(ElicitationArm(run["arm"])):
                turn = self.connection.execute(
                    "SELECT turn_id, status FROM turn WHERE round_id=? AND voter_id=? AND kind=?",
                    (round_row["round_id"], voter["voter_id"], kind.value),
                ).fetchone()
                if turn is None:
                    raise RuntimeError(
                        f"Resume lookup failed because voter {voter['voter_id']} round "
                        f"{round_row['round_index']} lacks required {kind.value} turn. State "
                        "validation failed in experiment.store.next_incomplete_unit, so the "
                        "barrier cannot be trusted. Restore the database or recreate the run."
                    )
                if turn["status"] == "committed":
                    continue
                committed_attempts = self.connection.execute(
                    "SELECT COUNT(*) FROM model_call WHERE turn_id=? AND status='committed'",
                    (turn["turn_id"],),
                ).fetchone()[0]
                if committed_attempts >= attempt_limit:
                    raise RuntimeError(
                        f"Resume lookup found pending turn {turn['turn_id']} with "
                        f"{committed_attempts} committed attempts at limit {attempt_limit}. The "
                        "terminal write is missing after commit_call, so continuing could create "
                        "a fifth response. Inspect crash consistency and restore the transaction."
                    )
                return WorkUnit(
                    run_id,
                    VoterId(voter["voter_id"]),
                    voter["voter_index"],
                    RoundId(round_row["round_id"]),
                    round_row["round_index"],
                    kind,
                    committed_attempts,
                )
        return BarrierReady(
            run_id, RoundId(round_row["round_id"]), round_row["round_index"]
        )

    def mark_interrupted(self, call_id: CallId) -> None:
        with self._transaction():
            changed = self.connection.execute(
                "UPDATE model_call SET status='interrupted' WHERE call_id=? AND status='started'",
                (call_id,),
            ).rowcount
            if changed != 1:
                raise ValueError(
                    f"Call interruption failed because call {call_id} is missing or is not "
                    "started. The transition check failed in experiment.store.mark_interrupted, "
                    "so committed history was left untouched. Interrupt only a currently "
                    "started invocation and create a new invocation for retry."
                )

    def resolve_turn_id(self, unit: WorkUnit) -> TurnId:
        """Resolve the normalized turn key omitted from the final WorkUnit contract."""
        row = self.connection.execute(
            "SELECT turn_id FROM turn WHERE round_id=? AND voter_id=? AND kind=?",
            (unit.round_id, unit.voter_id, unit.kind.value),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"Turn resolution failed because WorkUnit for run {unit.run_id}, voter "
                f"{unit.voter_id}, round {unit.round_index}, kind {unit.kind.value} has no "
                "normalized turn row. Lookup failed in experiment.store.resolve_turn_id before "
                "T1, so no call was started. Ask next_incomplete_unit for a current WorkUnit "
                "from this store and retry."
            )
        return TurnId(row[0])

    def begin_call(
        self,
        turn_id: TurnId,
        attempt_index: int,
        prompt_messages_json: str,
        prompt_sha256: str,
        seed: int,
    ) -> CallId:
        computed_hash = hashlib.sha256(prompt_messages_json.encode()).hexdigest()
        if computed_hash != prompt_sha256:
            raise ValueError(
                f"Call start failed because prompt hash {prompt_sha256} does not match "
                f"computed hash {computed_hash}. Validation failed in "
                "experiment.store.begin_call before T1, so an unreplayable prompt was not "
                "persisted. Serialize the exact messages, recompute SHA-256, and retry."
            )
        call_id = CallId(_ulid())
        with self._transaction():
            turn = self.connection.execute(
                "SELECT status FROM turn WHERE turn_id=?", (turn_id,)
            ).fetchone()
            if turn is None or turn[0] != "pending":
                status = "missing" if turn is None else turn[0]
                raise ValueError(
                    f"Call start failed because turn {turn_id} has status {status}, not pending. "
                    "Validation failed in experiment.store.begin_call before T1, so no duplicate "
                    "invocation was inserted. Resume from next_incomplete_unit."
                )
            committed = self.connection.execute(
                "SELECT 1 FROM model_call WHERE turn_id=? AND attempt_index=? "
                "AND status='committed'",
                (turn_id, attempt_index),
            ).fetchone()
            if committed is not None:
                raise ValueError(
                    f"Call start failed because turn {turn_id} attempt {attempt_index} already "
                    "has a committed invocation. Uniqueness validation failed before T1, so no "
                    "second response was created. Ask next_incomplete_unit for the next unit."
                )
            # A crash-left STARTED invocation is invisible and terminally interrupted as part
            # of the same T1 transaction that appends its replacement.
            self.connection.execute(
                "UPDATE model_call SET status='interrupted' WHERE turn_id=? AND attempt_index=? "
                "AND status='started'",
                (turn_id, attempt_index),
            )
            invocation_index = self.connection.execute(
                "SELECT COALESCE(MAX(invocation_index),-1)+1 FROM model_call "
                "WHERE turn_id=? AND attempt_index=?",
                (turn_id, attempt_index),
            ).fetchone()[0]
            self.connection.execute(
                "INSERT INTO model_call (call_id,turn_id,attempt_index,invocation_index,status,"
                "prompt_messages_json,prompt_sha256,seed,started_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    call_id,
                    turn_id,
                    attempt_index,
                    invocation_index,
                    "started",
                    prompt_messages_json,
                    prompt_sha256,
                    seed_to_blob(seed),
                    _now(),
                ),
            )
        return call_id

    def commit_call(
        self,
        call_id: CallId,
        result: GenerationResult,
        failures: Sequence[ValidationFailure],
        terminal: TerminalWrite | None,
    ) -> None:
        diagnostics = _validated_diagnostics(result.diagnostics)
        with self._transaction():
            call = self.connection.execute(
                "SELECT c.*, t.kind, t.status AS turn_status FROM model_call c "
                "JOIN turn t ON t.turn_id=c.turn_id WHERE c.call_id=?",
                (call_id,),
            ).fetchone()
            if (
                call is None
                or call["status"] != "started"
                or call["turn_status"] != "pending"
            ):
                status = "missing" if call is None else call["status"]
                raise ValueError(
                    f"Call commit failed because call {call_id} has status {status} or its turn "
                    "is no longer pending. T3 validation failed in experiment.store.commit_call, "
                    "so no response or terminal result was duplicated. Commit only the active "
                    "started invocation returned by begin_call."
                )
            self.connection.execute(
                "UPDATE model_call SET status='committed', raw_text=?, prompt_token_count=?, "
                "completion_token_count=?, completion_token_ids_json=?, stop_reason=?, "
                "duration_ms=?, diagnostics_json=?, committed_at=? WHERE call_id=?",
                (
                    result.text,
                    result.prompt_token_count,
                    result.completion_token_count,
                    None
                    if result.completion_token_ids is None
                    else _canonical_json(result.completion_token_ids),
                    result.stop_reason.value,
                    result.duration_ms,
                    _canonical_json(diagnostics),
                    _now(),
                    call_id,
                ),
            )
            for failure in sorted(failures, key=lambda item: item.ordinal):
                self.connection.execute(
                    "INSERT INTO validation_failure VALUES (?,?,?,?,?)",
                    (
                        _ulid(),
                        call_id,
                        failure.code.value,
                        failure.ordinal,
                        failure.message,
                    ),
                )
            if terminal is None:
                return
            turn_id = TurnId(call["turn_id"])
            if isinstance(terminal, AcceptedBallot):
                if call["kind"] != TurnKind.BALLOT.value or failures:
                    raise ValueError(
                        f"Terminal ballot write failed for call {call_id} because its turn kind "
                        "or validation failures conflict with acceptance. T3 was rolled back in "
                        "experiment.store.commit_call. Pass AcceptedBallot only for a valid "
                        "ballot call."
                    )
                ballot_id = _ulid()
                computed_cost = sum(
                    votes * votes for votes in terminal.allocations.values()
                )
                if computed_cost != terminal.engine_cost:
                    raise ValueError(
                        f"Terminal ballot write failed because engine_cost={terminal.engine_cost} "
                        f"but allocations cost {computed_cost}. T3 was rolled back before the "
                        "turn committed. Pass the domain engine's exact quadratic cost."
                    )
                self.connection.execute(
                    "INSERT INTO ballot VALUES (?,?,?,?,?,?)",
                    (
                        ballot_id,
                        turn_id,
                        "accepted",
                        call_id,
                        terminal.rationale,
                        computed_cost,
                    ),
                )
                for candidate_id, votes in terminal.allocations.items():
                    if votes < 1:
                        raise ValueError(
                            f"Terminal ballot write failed because candidate {candidate_id} has "
                            f"votes={votes}, below one. Canonicalization validation failed in "
                            "experiment.store.commit_call, so T3 was rolled back. Remove explicit "
                            "zero allocations in the domain layer and retry."
                        )
                    self.connection.execute(
                        "INSERT INTO allocation VALUES (?,?,?)",
                        (ballot_id, candidate_id, votes),
                    )
            elif isinstance(terminal, AcceptedStatement):
                if call["kind"] != TurnKind.STATEMENT.value or failures:
                    raise ValueError(
                        f"Terminal statement write failed for call {call_id} because its turn "
                        "kind or validation failures conflict with acceptance. T3 was rolled back. "
                        "Pass AcceptedStatement only for a valid statement call."
                    )
                statement_id = _ulid()
                self.connection.execute(
                    "INSERT INTO statement VALUES (?,?,?,?)",
                    (statement_id, turn_id, "accepted", call_id),
                )
                for candidate_id, (rating, text) in terminal.items.items():
                    self.connection.execute(
                        "INSERT INTO statement_item VALUES (?,?,?,?)",
                        (statement_id, candidate_id, rating.value, text),
                    )
            elif isinstance(terminal, BallotAbstention):
                if call["kind"] != TurnKind.BALLOT.value or not failures:
                    raise ValueError(
                        f"Ballot abstention failed for call {call_id} because exhaustion requires "
                        "an invalid ballot response. T3 was rolled back. Supply persisted "
                        "validation failures only on the configured final attempt."
                    )
                attempt_limit = self.connection.execute(
                    "SELECT c.ballot_max_corrections+1 FROM turn t JOIN voter v ON v.voter_id=t.voter_id "
                    "JOIN experiment_run r ON r.run_id=v.run_id JOIN matched_set m "
                    "ON m.matched_set_id=r.matched_set_id JOIN experiment_config c "
                    "ON c.config_id=m.config_id WHERE t.turn_id=?",
                    (turn_id,),
                ).fetchone()[0]
                if call["attempt_index"] != attempt_limit - 1:
                    raise ValueError(
                        f"Ballot abstention failed because call {call_id} is attempt "
                        f"{call['attempt_index']} but final attempt is {attempt_limit - 1}. T3 "
                        "was rolled back before committing the turn. Persist the invalid response "
                        "without a terminal write, issue the next correction, and abstain only "
                        "when the configured attempt limit is exhausted."
                    )
                self.connection.execute(
                    "INSERT INTO ballot VALUES (?,?,?,?,?,?)",
                    (_ulid(), turn_id, "abstained", None, None, 0),
                )
            elif isinstance(terminal, StatementInvalidMissing):
                if call["kind"] != TurnKind.STATEMENT.value or not failures:
                    raise ValueError(
                        f"Invalid-missing statement failed for call {call_id} because exhaustion "
                        "requires an invalid statement response. T3 was rolled back. Supply "
                        "persisted validation failures only on the configured final attempt."
                    )
                attempt_limit = self.connection.execute(
                    "SELECT c.statement_max_corrections+1 FROM turn t JOIN voter v ON v.voter_id=t.voter_id "
                    "JOIN experiment_run r ON r.run_id=v.run_id JOIN matched_set m "
                    "ON m.matched_set_id=r.matched_set_id JOIN experiment_config c "
                    "ON c.config_id=m.config_id WHERE t.turn_id=?",
                    (turn_id,),
                ).fetchone()[0]
                if call["attempt_index"] != attempt_limit - 1:
                    raise ValueError(
                        f"Invalid-missing statement failed because call {call_id} is attempt "
                        f"{call['attempt_index']} but final attempt is {attempt_limit - 1}. T3 "
                        "was rolled back before committing the turn. Persist the invalid response "
                        "without a terminal write, issue the next correction, and terminate only "
                        "when the configured attempt limit is exhausted."
                    )
                self.connection.execute(
                    "INSERT INTO statement VALUES (?,?,?,?)",
                    (_ulid(), turn_id, "invalid-missing", None),
                )
            else:  # pragma: no cover - closed union defense
                raise AssertionError(f"unhandled terminal write {terminal!r}")
            self.connection.execute(
                "UPDATE turn SET status='committed' WHERE turn_id=?", (turn_id,)
            )

    def record_runtime_failure(
        self, call_id: CallId, kind: RuntimeFailureKind, diagnostics: Mapping[str, str]
    ) -> None:
        self.interrupt_call_with_failure(call_id, kind, diagnostics)

    def interrupt_call_with_failure(
        self, call_id: CallId, kind: RuntimeFailureKind, diagnostics: Mapping[str, str]
    ) -> None:
        """Atomically insert the failure and transition STARTED to INTERRUPTED."""
        sanitized = _validated_diagnostics(diagnostics)
        with self._transaction():
            changed = self.connection.execute(
                "UPDATE model_call SET status='interrupted' WHERE call_id=? AND status='started'",
                (call_id,),
            ).rowcount
            if changed != 1:
                raise ValueError(
                    f"Runtime-failure recording failed because call {call_id} is missing or not "
                    "started. Validation failed in experiment.store.record_runtime_failure, so "
                    "no attempt state was changed. Attach machinery failures only to the active "
                    "invocation."
                )
            self.connection.execute(
                "INSERT INTO runtime_failure VALUES (?,?,?,?,?)",
                (_ulid(), call_id, kind.value, _canonical_json(sanitized), _now()),
            )

    def pause_run(self, run_id: RunId, reason: str) -> None:
        with self._transaction():
            changed = self.connection.execute(
                "UPDATE experiment_run SET status='paused', pause_reason=? "
                "WHERE run_id=? AND status='in_progress'",
                (reason, run_id),
            ).rowcount
            if changed != 1:
                raise ValueError(
                    f"Run pause failed because run {run_id} is not in_progress. The transition "
                    "check failed in experiment.store.pause_run, so its status was unchanged. "
                    "Pause only a running experiment after a bounded runtime-failure budget."
                )

    def set_run_in_progress(self, run_id: RunId) -> None:
        with self._transaction():
            changed = self.connection.execute(
                "UPDATE experiment_run SET status='in_progress', pause_reason=NULL "
                "WHERE run_id=? AND status IN ('created','paused')",
                (run_id,),
            ).rowcount
            if changed != 1:
                row = self.connection.execute(
                    "SELECT status FROM experiment_run WHERE run_id=?", (run_id,)
                ).fetchone()
                if row is not None and row[0] == "in_progress":
                    return
                status = "missing" if row is None else row[0]
                raise ValueError(
                    f"Run start failed because run {run_id} has status {status}, which cannot "
                    "transition to in_progress. Validation failed in "
                    "experiment.store.set_run_in_progress. Resume only created or paused runs; "
                    "complete runs are side-effect-free."
                )

    def _persist_draw(
        self,
        run_id: RunId,
        round_index: int,
        domain: str,
        selection: DrawSelection,
    ) -> None:
        existing = self.connection.execute(
            "SELECT * FROM rng_draw WHERE run_id=? AND stream_domain=? AND round_index=?",
            (run_id, domain, round_index),
        ).fetchone()
        if existing is not None:
            population = tuple(
                row[0]
                for row in self.connection.execute(
                    "SELECT candidate_id FROM rng_draw_population WHERE draw_id=? ORDER BY position",
                    (existing["draw_id"],),
                )
            )
            if (
                seed_from_blob(existing["derived_seed"]) != selection.seed
                or existing["draw_index"] != selection.selected_index
                or existing["chosen_candidate_id"] != selection.selected
                or population != selection.eligible
            ):
                raise RuntimeError(
                    f"RNG replay verification failed for run {run_id}, stream {domain}, round "
                    f"{round_index}: persisted draw differs from the deterministic selection. "
                    "Verification failed in experiment.store._persist_draw while sealing, so "
                    "the transaction was rolled back and the stream was not redrawn. Restore "
                    "the unmodified database or investigate seed/config corruption."
                )
            return
        draw_id = _ulid()
        self.connection.execute(
            "INSERT INTO rng_draw VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                draw_id,
                run_id,
                domain,
                round_index,
                selection.stream_name,
                seed_to_blob(selection.seed),
                "qv-seed/v1",
                _canonical_json(selection.coordinates),
                selection.algorithm.value,
                selection.selected_index,
                selection.selected,
            ),
        )
        for position, candidate_id in enumerate(selection.eligible):
            self.connection.execute(
                "INSERT INTO rng_draw_population VALUES (?,?,?)",
                (draw_id, position, candidate_id),
            )

    def aggregate_and_seal_round(self, run_id: RunId) -> RoundOutcome | RunComplete:
        """Read, compute, persist, and seal through COMPLETE in one transaction."""
        with self._transaction():
            return self._aggregate_and_seal_round_locked(run_id)

    def _aggregate_and_seal_round_locked(
        self, run_id: RunId
    ) -> RoundOutcome | RunComplete:
        run = self.connection.execute(
            "SELECT r.*, c.master_seed FROM experiment_run r JOIN matched_set m "
            "ON m.matched_set_id=r.matched_set_id JOIN experiment_config c "
            "ON c.config_id=m.config_id WHERE r.run_id=?",
            (run_id,),
        ).fetchone()
        if run is None:
            raise ValueError(
                f"Round aggregation failed because run {run_id} does not exist. Lookup failed "
                "in experiment.store.aggregate_and_seal_round before the barrier transaction, "
                "so nothing was sealed. Supply a run ID created by create_matched_set."
            )
        if run["status"] == "complete":
            return RunComplete(run_id)
        round_row = self.connection.execute(
            "SELECT * FROM \"round\" WHERE run_id=? AND phase='eliciting' "
            "ORDER BY round_index LIMIT 1",
            (run_id,),
        ).fetchone()
        if round_row is None:
            raise RuntimeError(
                f"Round aggregation failed because incomplete run {run_id} has no eliciting "
                "round. State validation failed before aggregation, so no caller-supplied "
                "outcome was accepted. Restore the database from its last durable commit."
            )
        # Sealing and successor creation are atomic. A caller replaying the just-finished
        # aggregation therefore sees the untouched successor round; recognize that exact
        # state (no calls yet) and observe the durable prior outcome without redrawing.
        current_call_count = self.connection.execute(
            "SELECT COUNT(*) FROM model_call c JOIN turn t ON t.turn_id=c.turn_id "
            "WHERE t.round_id=?",
            (round_row["round_id"],),
        ).fetchone()[0]
        if round_row["round_index"] > 1 and current_call_count == 0:
            prior = self.connection.execute(
                'SELECT r.round_index,o.* FROM "round" r JOIN round_outcome o '
                "ON o.round_id=r.round_id WHERE r.run_id=? AND r.round_index=?",
                (run_id, round_row["round_index"] - 1),
            ).fetchone()
            if prior is not None:
                return RoundOutcome(
                    prior["round_index"],
                    None
                    if prior["protected_candidate_id"] is None
                    else CandidateId(prior["protected_candidate_id"]),
                    CandidateId(prior["removed_candidate_id"]),
                    bool(prior["tie_flag"]),
                )
        incomplete = self.connection.execute(
            "SELECT v.voter_id, v.voter_index, t.turn_id, t.kind FROM turn t "
            "JOIN voter v ON v.voter_id=t.voter_id WHERE t.round_id=? AND t.status='pending' "
            "ORDER BY v.voter_index, CASE t.kind WHEN 'statement' THEN 0 ELSE 1 END LIMIT 1",
            (round_row["round_id"],),
        ).fetchone()
        if incomplete is not None:
            raise RuntimeError(
                f"Round barrier refused run {run_id} round {round_row['round_index']} because "
                f"voter {incomplete['voter_id']} (index {incomplete['voter_index']}) has "
                f"incomplete {incomplete['kind']} turn {incomplete['turn_id']}. Barrier "
                "validation failed in experiment.store.aggregate_and_seal_round, so the round "
                "remains eliciting and no peer activity is revealed. Resume that exact turn, "
                "commit it, then retry aggregation."
            )
        active = tuple(
            CandidateId(row[0])
            for row in self.connection.execute(
                "SELECT candidate_id FROM round_candidate WHERE round_id=? ORDER BY sample_position",
                (round_row["round_id"],),
            )
        )
        allocation_rows = self.connection.execute(
            "SELECT b.ballot_id, a.candidate_id, a.votes FROM turn t "
            "JOIN ballot b ON b.turn_id=t.turn_id LEFT JOIN allocation a ON a.ballot_id=b.ballot_id "
            "WHERE t.round_id=? ORDER BY b.ballot_id, a.candidate_id",
            (round_row["round_id"],),
        ).fetchall()
        allocations_by_ballot: dict[str, dict[CandidateId, int]] = {}
        for row in allocation_rows:
            ballot = allocations_by_ballot.setdefault(row["ballot_id"], {})
            if row["candidate_id"] is not None:
                ballot[CandidateId(row["candidate_id"])] = row["votes"]
        arm = ElicitationArm(run["arm"])
        regime = VotingRegime(run["regime"])
        tie_seeded = tie_break_draw(
            seed_from_blob(run["master_seed"]), arm, regime, round_row["round_index"]
        )
        removal_seeded = support_removal_draw(
            seed_from_blob(run["master_seed"]), arm, regime, round_row["round_index"]
        )
        # Imported only here so catalog/sample operations remain usable while the pure
        # engine slice is developed independently.
        from quadratic_voting.experiment.engine import aggregate_round

        aggregation = aggregate_round(
            regime,
            active,
            tuple(allocations_by_ballot.values()),
            tie_seeded,
            removal_seeded,
        )
        outcome = RoundOutcome(
            round_row["round_index"],
            aggregation.result.protected,
            aggregation.result.removed,
            len(aggregation.result.tie_among) > 1,
        )
        with self._transaction():
            existing = self.connection.execute(
                "SELECT * FROM round_outcome WHERE round_id=?", (round_row["round_id"],)
            ).fetchone()
            if existing is not None:
                persisted = RoundOutcome(
                    round_row["round_index"],
                    None
                    if existing["protected_candidate_id"] is None
                    else CandidateId(existing["protected_candidate_id"]),
                    CandidateId(existing["removed_candidate_id"]),
                    bool(existing["tie_flag"]),
                )
                if persisted != outcome:
                    raise RuntimeError(
                        f"Round replay failed because recomputed outcome {outcome} differs from "
                        f"persisted outcome {persisted} for run {run_id}. Verification failed "
                        "before sealing replay, so no rows changed. Inspect persisted ballots, "
                        "draws, and immutable config."
                    )
                return persisted
            if aggregation.tie_draw is not None:
                self._persist_draw(
                    run_id, round_row["round_index"], "tie-break", aggregation.tie_draw
                )
            if aggregation.removal_draw is not None:
                self._persist_draw(
                    run_id,
                    round_row["round_index"],
                    "support-removal",
                    aggregation.removal_draw,
                )
            self.connection.execute(
                "INSERT INTO round_outcome VALUES (?,?,?,?,?)",
                (
                    round_row["round_id"],
                    outcome.protected,
                    outcome.removed,
                    int(outcome.tie),
                    _now(),
                ),
            )
            self.connection.execute(
                "UPDATE \"round\" SET phase='sealed' WHERE round_id=? AND phase='eliciting'",
                (round_row["round_id"],),
            )
            remaining = tuple(
                candidate for candidate in active if candidate != outcome.removed
            )
            if len(remaining) == 1:
                self.connection.execute(
                    "INSERT INTO final_result VALUES (?,?,?)",
                    (run_id, remaining[0], _now()),
                )
                self.connection.execute(
                    "UPDATE experiment_run SET status='complete', pause_reason=NULL WHERE run_id=?",
                    (run_id,),
                )
            else:
                self._insert_round(run_id, round_row["round_index"] + 1, remaining, arm)
        return RunComplete(run_id) if len(active) == 2 else outcome

    @staticmethod
    def _final_user_text(prompt_messages_json: str) -> str:
        try:
            messages = json.loads(prompt_messages_json)
            for message in reversed(messages):
                if message.get("role") == "user":
                    return str(message.get("content", ""))
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        raise RuntimeError(
            "Transcript projection failed because a committed call's persisted prompt is not "
            "a JSON message list containing a user message. Reconstruction failed in "
            "experiment.store._final_user_text, so returning a guessed transcript would break "
            "auditability. Inspect and repair the corrupted prompt row from its source artifact."
        )

    def voter_round_view(self, run_id: RunId, voter_id: VoterId) -> VoterRoundView:
        run = self.connection.execute(
            "SELECT r.arm,r.regime,c.credit_budget,c.sample_id,d.instruction_templates_json "
            "FROM experiment_run r JOIN matched_set m ON m.matched_set_id=r.matched_set_id "
            "JOIN experiment_config c ON c.config_id=m.config_id "
            "JOIN run_definition d ON d.run_id=r.run_id WHERE r.run_id=?",
            (run_id,),
        ).fetchone()
        voter = self.connection.execute(
            "SELECT voter_index FROM voter WHERE voter_id=? AND run_id=?",
            (voter_id, run_id),
        ).fetchone()
        if run is None or voter is None:
            raise ValueError(
                f"Transcript projection failed because run {run_id} or voter {voter_id} does "
                "not exist in that run. Lookup failed in experiment.store.voter_round_view, so "
                "no potentially cross-voter state was exposed. Supply a voter ID from the run."
            )
        template_map = json.loads(run["instruction_templates_json"])
        setup_reference = template_map.get(TemplateKind.SETUP.value)
        if setup_reference is None:
            raise RuntimeError(
                f"Transcript projection failed because run {run_id} has no setup template in "
                "its immutable definition. Reconstruction stopped before producing model-visible "
                "text. Recreate the matched set with all instruction template references."
            )
        instruction_row = self.connection.execute(
            "SELECT body,body_sha256 FROM instruction_template WHERE template_id=?",
            (setup_reference[0],),
        ).fetchone()
        if (
            instruction_row is None
            or instruction_row["body_sha256"] != setup_reference[1]
        ):
            raise RuntimeError(
                f"Transcript projection failed because setup template {setup_reference[0]} is "
                "missing or its hash drifted. Integrity validation failed before prompt assembly, "
                "so the model must not receive changed instructions. Restore the registered "
                "template row matching the immutable run definition."
            )
        setup_body = cast(str, instruction_row["body"])
        placeholders = {
            field_name
            for _literal, field_name, _format_spec, _conversion in Formatter().parse(
                setup_body
            )
            if field_name is not None
        }
        transcript_composed_fields = {
            "regime_rules",
            "arm_instructions",
            "response_formats",
            "candidate_cards",
        }
        legacy_fields = {"regime", "arm", "budget", "credit_budget"}
        if placeholders & transcript_composed_fields:
            from quadratic_voting.experiment.transcript import TEMPLATE_BODIES

            canonical_body = TEMPLATE_BODIES[TemplateKind.SETUP]
            canonical_hash = hashlib.sha256(canonical_body.encode()).hexdigest()
            if instruction_row["body_sha256"] != canonical_hash:
                raise RuntimeError(
                    f"Transcript setup drift detected for run {run_id} because stored setup "
                    f"template {setup_reference[0]} uses renderer-composed placeholders but hash "
                    f"{instruction_row['body_sha256']} differs from code template hash "
                    f"{canonical_hash}. Validation failed in "
                    "experiment.store.voter_round_view while reconstructing the setup prompt, "
                    "so the model and inspect command must not receive silently changed or "
                    "double-composed instructions. Re-register the current transcript templates "
                    "and create a new matched set, or update the executing code to the exact "
                    "template version pinned by the run."
                )
            instructions = ""
        elif placeholders <= legacy_fields:
            instructions = setup_body.format(
                regime=run["regime"],
                arm=run["arm"],
                budget=run["credit_budget"],
                credit_budget=run["credit_budget"],
            )
        else:
            unsupported = ", ".join(sorted(placeholders - legacy_fields))
            raise RuntimeError(
                f"Transcript setup rendering failed for run {run_id} because stored setup "
                f"template {setup_reference[0]} requires unsupported placeholders: "
                f"{unsupported}. Placeholder validation failed in "
                "experiment.store.voter_round_view before model-visible setup composition, so "
                "the prompt cannot be reconstructed safely. Re-register either the canonical "
                "transcript setup template or a legacy template using only regime, arm, budget, "
                "and credit_budget, then create a new matched set."
            )
        cards = tuple(
            (CandidateId(row["candidate_id"]), row["rendered_text"])
            for row in self.connection.execute(
                "SELECT vp.candidate_id,cp.rendered_text FROM voter_permutation vp "
                "JOIN candidate_sample cs ON cs.sample_id=? "
                "JOIN candidate_presentation cp ON cp.candidate_id=vp.candidate_id "
                "AND cp.template_id=cs.template_id WHERE vp.voter_id=? ORDER BY vp.position",
                (run["sample_id"], voter_id),
            )
        )
        setup = SetupContext(
            run_id,
            ElicitationArm(run["arm"]),
            VotingRegime(run["regime"]),
            run["credit_budget"],
            instructions,
            cards,
        )
        current = self.next_incomplete_unit(run_id)
        complete = isinstance(current, RunComplete)
        if not complete and (
            not isinstance(current, WorkUnit) or current.voter_id != voter_id
        ):
            raise ValueError(
                f"Transcript projection refused voter {voter_id} because the deterministic next "
                f"unit is {current!r}, not that voter's pending turn. Validation failed in "
                "experiment.store.voter_round_view, so current-round peer ordering was not "
                "exposed. Request the view only for the WorkUnit returned by "
                "next_incomplete_unit."
            )
        if complete:
            latest_round = self.connection.execute(
                'SELECT round_index,round_id FROM "round" WHERE run_id=? '
                "ORDER BY round_index DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            if latest_round is None:
                raise RuntimeError(
                    f"Transcript projection failed because complete run {run_id} has no sealed "
                    "round from which to derive its winner. State validation failed in "
                    "experiment.store.voter_round_view, so inspect cannot present a guessed "
                    "final result. Restore the run's durable round and outcome rows."
                )
            current_round_index = latest_round["round_index"]
        else:
            assert isinstance(current, WorkUnit)
            current_round_index = current.round_index
        history: list[PriorTurnEvent | RoundOutcomeEvent | FinalResultEvent] = []
        round_rows = self.connection.execute(
            'SELECT * FROM "round" WHERE run_id=? AND round_index<=? ORDER BY round_index',
            (run_id, current_round_index),
        ).fetchall()
        turn_order = arm_turn_order(ElicitationArm(run["arm"]))
        for round_row in round_rows:
            for kind in turn_order:
                turn = self.connection.execute(
                    "SELECT turn_id,status FROM turn WHERE round_id=? AND voter_id=? AND kind=?",
                    (round_row["round_id"], voter_id, kind.value),
                ).fetchone()
                if turn is None or turn["status"] != "committed":
                    continue
                calls = self.connection.execute(
                    "SELECT prompt_messages_json,raw_text FROM model_call WHERE turn_id=? "
                    "AND status='committed' ORDER BY attempt_index",
                    (turn["turn_id"],),
                ).fetchall()
                history.append(
                    PriorTurnEvent(
                        round_row["round_index"],
                        kind,
                        tuple(
                            (self._final_user_text(call[0]), cast(str, call[1]))
                            for call in calls
                        ),
                    )
                )
            if round_row["phase"] == "sealed":
                result = self.connection.execute(
                    "SELECT * FROM round_outcome WHERE round_id=?",
                    (round_row["round_id"],),
                ).fetchone()
                history.append(
                    RoundOutcomeEvent(
                        round_row["round_index"],
                        None
                        if result["protected_candidate_id"] is None
                        else CandidateId(result["protected_candidate_id"]),
                        CandidateId(result["removed_candidate_id"]),
                    )
                )
        if complete:
            removed = {
                row[0]
                for row in self.connection.execute(
                    'SELECT o.removed_candidate_id FROM "round" r JOIN round_outcome o '
                    "ON o.round_id=r.round_id WHERE r.run_id=?",
                    (run_id,),
                )
            }
            winners = tuple(
                candidate_id
                for candidate_id, _text in cards
                if candidate_id not in removed
            )
            if len(winners) != 1:
                raise RuntimeError(
                    f"Transcript projection failed because complete run {run_id} has "
                    f"{len(winners)} surviving candidates instead of one. Winner derivation "
                    "failed in experiment.store.voter_round_view after reading sealed outcomes, "
                    "so inspect cannot emit an auditable final result. Verify every terminal "
                    "round outcome and candidate snapshot."
                )
            history.append(FinalResultEvent(winners[0]))
            return VoterRoundView(
                setup,
                tuple(history),
                PendingTurn(
                    current_round_index,
                    TurnKind.BALLOT,
                    winners,
                    0,
                    (),
                ),
            )
        assert isinstance(current, WorkUnit)
        active_set = {
            row[0]
            for row in self.connection.execute(
                "SELECT candidate_id FROM round_candidate WHERE round_id=?",
                (current.round_id,),
            )
        }
        active = tuple(
            candidate_id for candidate_id, _text in cards if candidate_id in active_set
        )
        current_turn = self.connection.execute(
            "SELECT turn_id FROM turn WHERE round_id=? AND voter_id=? AND kind=?",
            (current.round_id, voter_id, current.kind.value),
        ).fetchone()[0]
        correction_errors = tuple(
            row[0]
            for row in self.connection.execute(
                "SELECT vf.message FROM validation_failure vf JOIN model_call c "
                "ON c.call_id=vf.call_id WHERE c.turn_id=? AND c.status='committed' "
                "ORDER BY c.attempt_index,vf.ordinal",
                (current_turn,),
            )
        )
        return VoterRoundView(
            setup,
            tuple(history),
            PendingTurn(
                current.round_index,
                current.kind,
                active,
                current.attempt_index,
                correction_errors,
            ),
        )

    def candidate_rows(self) -> tuple[dict[str, object], ...]:
        """Return catalog identity and label rows for sampling and normalized exports."""
        return tuple(
            dict(row)
            for row in self.connection.execute(
                "SELECT c.candidate_id,c.release_id,r.dataset_name,r.version AS release_version,"
                "r.file_sha256 AS release_sha256,c.source_row_id,c.content_sha256,"
                "l.rudeness_label,l.label_policy_id,lp.name AS label_policy_name,"
                "lp.version AS label_policy_version,lp.rule_sha256 AS label_policy_sha256,"
                "cp.presentation_id,cp.template_id,pt.name AS presentation_template_name,"
                "pt.version AS presentation_template_version,"
                "pt.body_sha256 AS presentation_template_sha256,"
                "cp.rendered_sha256 AS presentation_sha256 FROM candidate c "
                "JOIN dataset_release r ON r.release_id=c.release_id "
                "JOIN candidate_label l ON l.candidate_id=c.candidate_id "
                "JOIN label_policy lp ON lp.label_policy_id=l.label_policy_id "
                "LEFT JOIN candidate_presentation cp ON cp.candidate_id=c.candidate_id "
                "LEFT JOIN presentation_template pt ON pt.template_id=cp.template_id "
                "ORDER BY c.release_id,c.candidate_id,l.label_policy_id,cp.template_id"
            )
        )

    def source_annotation_rows(self) -> tuple[dict[str, object], ...]:
        """Return exact source annotation values with release and policy lineage."""
        return tuple(
            dict(row)
            for row in self.connection.execute(
                "SELECT sa.candidate_id,c.release_id,r.dataset_name,r.version AS release_version,"
                "r.file_sha256 AS release_sha256,cl.label_policy_id,"
                "lp.name AS label_policy_name,lp.version AS label_policy_version,"
                "lp.rule_sha256 AS label_policy_sha256,sa.annotation_index,sa.annotator_hash,"
                "sa.source_label,sa.source_value FROM source_annotation sa "
                "JOIN candidate c ON c.candidate_id=sa.candidate_id "
                "JOIN dataset_release r ON r.release_id=c.release_id "
                "JOIN candidate_label cl ON cl.candidate_id=c.candidate_id "
                "JOIN label_policy lp ON lp.label_policy_id=cl.label_policy_id "
                "ORDER BY c.release_id,sa.candidate_id,cl.label_policy_id,sa.annotation_index"
            )
        )

    def candidate_presentation_rows(self) -> tuple[dict[str, object], ...]:
        """Return exact rendered candidate cards and immutable template lineage."""
        return tuple(
            dict(row)
            for row in self.connection.execute(
                "SELECT cp.presentation_id,cp.candidate_id,c.release_id,cp.template_id,"
                "pt.name AS template_name,pt.version AS template_version,pt.body_sha256,"
                "cp.rendered_text,cp.rendered_sha256 FROM candidate_presentation cp "
                "JOIN candidate c ON c.candidate_id=cp.candidate_id "
                "JOIN presentation_template pt ON pt.template_id=cp.template_id "
                "ORDER BY c.release_id,c.source_row_id,cp.template_id"
            )
        )

    def candidate_turn_rows(self) -> tuple[dict[str, object], ...]:
        """Return ordered source conversation turns with release provenance."""
        return tuple(
            dict(row)
            for row in self.connection.execute(
                "SELECT ct.candidate_id,c.release_id,r.dataset_name,"
                "r.version AS release_version,r.file_sha256 AS release_sha256,"
                "c.content_sha256,ct.turn_index,ct.role,ct.text "
                "FROM candidate_turn ct JOIN candidate c ON c.candidate_id=ct.candidate_id "
                "JOIN dataset_release r ON r.release_id=c.release_id "
                "ORDER BY c.release_id,c.source_row_id,ct.turn_index"
            )
        )

    def voter_permutation_rows(self) -> tuple[dict[str, object], ...]:
        """Return normalized persisted voter orders with complete seed coordinates."""
        rows = [
            dict(row)
            for row in self.connection.execute(
                "SELECT v.run_id,v.voter_id,v.voter_index,v.permutation_seed,"
                "v.permutation_algorithm,v.permutation_coordinates_json,vp.position,"
                "vp.candidate_id FROM voter v JOIN voter_permutation vp "
                "ON vp.voter_id=v.voter_id ORDER BY v.run_id,v.voter_index,vp.position"
            )
        ]
        for row in rows:
            row["permutation_seed"] = seed_from_blob(row["permutation_seed"])
        return tuple(rows)

    def experiment_config_rows(self) -> tuple[dict[str, object], ...]:
        """Return the normalized immutable config graph without a duplicate JSON body."""
        rows = [
            dict(row)
            for row in self.connection.execute(
                "SELECT ec.*,m.matched_set_id,m.created_at,d.model_id,d.provider_id,"
                "d.quantization_id,d.runtime_id,d.artifact_repository,d.artifact_revision,"
                "d.tokenizer_repository,d.tokenizer_revision,d.dtype,d.route_registry_hash,"
                "d.sampling_profile_hash,d.instruction_profile_hash,"
                "d.instruction_templates_json,d.dataset_release_hash,d.source_release_id,"
                "d.label_policy_id,d.label_policy_version,d.label_policy_hash,"
                "d.sample_artifact_hash,d.presentation_template_id,"
                "d.presentation_template_hash FROM experiment_config ec "
                "JOIN matched_set m ON m.config_id=ec.config_id JOIN experiment_run r "
                "ON r.matched_set_id=m.matched_set_id JOIN run_definition d ON d.run_id=r.run_id "
                "WHERE r.run_id=(SELECT MIN(r2.run_id) FROM experiment_run r2 "
                "WHERE r2.matched_set_id=m.matched_set_id) ORDER BY ec.config_id"
            )
        ]
        for row in rows:
            row["master_seed"] = seed_from_blob(row["master_seed"])
        return tuple(rows)

    def run_definition_rows(self) -> tuple[dict[str, object], ...]:
        """Return every immutable run definition in deterministic run-ID order."""
        return tuple(
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM run_definition ORDER BY run_id"
            )
        )

    def round_candidate_rows(self) -> tuple[dict[str, object], ...]:
        """Return every frozen active-pool snapshot with its run and round coordinates."""
        return tuple(
            dict(row)
            for row in self.connection.execute(
                "SELECT rc.round_id,r.run_id,r.round_index,rc.candidate_id,rc.sample_position "
                'FROM round_candidate rc JOIN "round" r ON r.round_id=rc.round_id '
                "ORDER BY r.run_id,r.round_index,rc.sample_position"
            )
        )

    def export_rows(self, dataset: ExportDataset) -> tuple[dict[str, object], ...]:
        """Return normalized rows; RNG draw rows include ordered ``population`` lists."""
        tables = {
            ExportDataset.RUN_EXECUTIONS: "run_execution",
            ExportDataset.VOTERS: "voter",
            ExportDataset.ROUNDS: '"round"',
            ExportDataset.TURNS: "turn",
            ExportDataset.CALLS: "model_call",
            ExportDataset.VALIDATION_FAILURES: "validation_failure",
            ExportDataset.RUNTIME_FAILURES: "runtime_failure",
            ExportDataset.BALLOTS: "ballot",
            ExportDataset.ALLOCATIONS: "allocation",
            ExportDataset.STATEMENTS: "statement",
            ExportDataset.STATEMENT_ITEMS: "statement_item",
            ExportDataset.OUTCOMES: "round_outcome",
            ExportDataset.RNG_DRAWS: "rng_draw",
        }
        if dataset is ExportDataset.RUNS:
            query = (
                "SELECT r.*,c.sample_id,c.master_seed,c.temperature,c.top_p,c.top_k,"
                "c.max_new_tokens,c.credit_budget,c.ballot_max_corrections+1 AS attempt_limit,"
                "c.voter_count,c.runtime_max_failures AS max_consecutive_runtime_failures,"
                "c.tie_policy,c.presentation_policy,c.action_format,c.config_hash "
                "FROM experiment_run r JOIN matched_set m "
                "ON m.matched_set_id=r.matched_set_id JOIN experiment_config c "
                "ON c.config_id=m.config_id ORDER BY r.run_id"
            )
        else:
            query = f"SELECT * FROM {tables[dataset]} ORDER BY rowid"
        rows = [dict(row) for row in self.connection.execute(query)]
        if dataset is ExportDataset.RNG_DRAWS:
            for row in rows:
                row["population"] = [
                    item[0]
                    for item in self.connection.execute(
                        "SELECT candidate_id FROM rng_draw_population WHERE draw_id=? "
                        "ORDER BY position",
                        (row["draw_id"],),
                    )
                ]
        return tuple(rows)


def open_sqlite_store(
    path: Path,
    *,
    commit_hook: Callable[[int], None] | None = None,
    freeze_hook: Callable[[FreezePoint], None] | None = None,
    writer_lock: WriterLock | None = None,
    require_writer_lock: bool = True,
) -> SqliteExperimentStore:
    """Open a durable store, transactionally applying known forward migrations."""
    owns_writer_lock = False
    if require_writer_lock and writer_lock is None:
        writer_lock = acquire_writer_lock(path, command="open-sqlite-store")
        owns_writer_lock = True
    if require_writer_lock and (
        writer_lock is None or not lock_matches_database(writer_lock, path)
    ):
        raise RuntimeError(
            f"SQLite writer open refused {path} because no matching live WriterLock was supplied. "
            "The preflight failed in store.open_sqlite_store before directory creation, database "
            "open, or migration. Acquire acquire_writer_lock(path), pass writer_lock=lock and "
            "require_writer_lock=True, and hold it through store close/artifact fsync."
        )

    def release_owned_lock() -> None:
        if owns_writer_lock and writer_lock is not None:
            writer_lock.release()

    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA journal_mode=WAL")
    table = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'"
    ).fetchone()
    if table is None:
        schema_path = Path(__file__).with_name("schema.sql")
        schema = schema_path.read_text(encoding="utf-8")
        try:
            connection.executescript(
                "BEGIN IMMEDIATE;\n"
                + schema
                + f"\nINSERT INTO schema_version(version) VALUES ({KNOWN_SCHEMA_VERSION});\n"
                + "COMMIT;"
            )
        except BaseException:
            if connection.in_transaction:
                connection.rollback()
            connection.close()
            release_owned_lock()
            raise
    row = connection.execute("SELECT MAX(version) FROM schema_version").fetchone()
    version = 0 if row is None or row[0] is None else int(row[0])
    if version > KNOWN_SCHEMA_VERSION:
        connection.close()
        release_owned_lock()
        raise RuntimeError(
            f"SQLite store open refused {path} because database schema version {version} is "
            f"newer than this code's known version {KNOWN_SCHEMA_VERSION}. Compatibility "
            "validation failed in experiment.store.open_sqlite_store before normal reads or "
            "writes, so using this binary could corrupt newer data. Upgrade the application to "
            f"code supporting schema version {version}; do not downgrade the database."
        )
    if version < KNOWN_SCHEMA_VERSION:
        connection.close()
        release_owned_lock()
        raise RuntimeError(
            f"SQLite store open cannot migrate {path} from schema version {version} because "
            f"this build only contains the fresh v{KNOWN_SCHEMA_VERSION} migration. Migration "
            "stopped transactionally before writes. Upgrade through the missing intermediate "
            "application version, then reopen with this build."
        )
    # Additive identity storage also upgrades databases initialized by the
    # previous v1 schema without changing experiment data.
    connection.execute(
        "CREATE TABLE IF NOT EXISTS pipeline_database_identity "
        "(database_id TEXT PRIMARY KEY) STRICT"
    )
    return SqliteExperimentStore(
        connection, path, commit_hook, freeze_hook, writer_lock, owns_writer_lock
    )


@contextmanager
def open_locked_sqlite_store(
    path: Path,
    *,
    command: str,
    commit_hook: Callable[[int], None] | None = None,
    freeze_hook: Callable[[FreezePoint], None] | None = None,
) -> Any:
    """Acquire the common writer lock before open/migration and release after close."""
    with acquire_writer_lock(path, command=command) as lock:
        store = open_sqlite_store(
            path,
            commit_hook=commit_hook,
            freeze_hook=freeze_hook,
            writer_lock=lock,
            require_writer_lock=True,
        )
        try:
            yield store
        finally:
            store.close()


def open_readonly_sqlite_store(path: Path) -> SqliteExperimentStore:
    """Open an existing compatible database read-only without migration."""
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        row = connection.execute("SELECT MAX(version) FROM schema_version").fetchone()
    except sqlite3.Error as error:
        connection.close()
        raise RuntimeError(
            f"Read-only SQLite open refused {path} because schema_version is unavailable. "
            "Compatibility checking failed in store.open_readonly_sqlite_store without "
            "migration. Initialize/migrate under the writer lock, then inspect again."
        ) from error
    version = 0 if row is None or row[0] is None else int(row[0])
    if version != KNOWN_SCHEMA_VERSION:
        connection.close()
        raise RuntimeError(
            f"Read-only SQLite open refused {path} because schema version {version} differs "
            f"from supported {KNOWN_SCHEMA_VERSION}. No migration was attempted. Use matching "
            "code or migrate under the common writer lock."
        )
    return SqliteExperimentStore(connection, path)
