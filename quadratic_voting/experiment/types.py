"""Shared typed contracts for the resumable voting experiment."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import NewType, Protocol, TypeAlias

from llm_runtime.types import ChatMessage


class ElicitationArm(StrEnum):
    ACTION_ONLY = "action-only"
    STATEMENT_THEN_ACTION = "statement-then-action"
    ACTION_THEN_STATEMENT = "action-then-statement"


class VotingRegime(StrEnum):
    SUPPORT = "support"
    OPPOSITION = "opposition"


class LikertRating(StrEnum):
    STRONGLY_PREFER_NOT_TO_CONTINUE = "strongly prefer not to continue"
    PREFER_NOT_TO_CONTINUE = "prefer not to continue"
    NEUTRAL = "neutral"
    PREFER_TO_CONTINUE = "prefer to continue"
    STRONGLY_PREFER_TO_CONTINUE = "strongly prefer to continue"


class TurnKind(StrEnum):
    STATEMENT = "statement"
    BALLOT = "ballot"


class CallStatus(StrEnum):
    STARTED = "started"
    COMMITTED = "committed"
    INTERRUPTED = "interrupted"


class TurnStatus(StrEnum):
    PENDING = "pending"
    COMMITTED = "committed"


class BallotStatus(StrEnum):
    ACCEPTED = "accepted"
    ABSTAINED = "abstained"


class StatementStatus(StrEnum):
    ACCEPTED = "accepted"
    INVALID_MISSING = "invalid-missing"


class RoundPhase(StrEnum):
    ELICITING = "eliciting"
    SEALED = "sealed"


class RunStatus(StrEnum):
    CREATED = "created"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETE = "complete"


class ExecutionClass(StrEnum):
    FIXTURE = "fixture"
    PILOT = "pilot"
    PRIMARY = "primary"


class RunForkReason(StrEnum):
    MODEL_DEFINITION_CHANGE = "model-definition-change"
    PROMPT_PROFILE_CHANGE = "prompt-profile-change"
    SAMPLING_PROFILE_CHANGE = "sampling-profile-change"
    LABEL_POLICY_CHANGE = "label-policy-change"
    OPERATOR_REQUEST = "operator-request"


class SampleStatus(StrEnum):
    DRAFT = "draft"
    FREEZE_PENDING = "freeze_pending"
    FROZEN = "frozen"


class FreezePoint(StrEnum):
    TEMP_FSYNC = "after-temp-fsync"
    F1_COMMIT = "after-f1-intent-commit"
    RENAME = "after-rename"
    DIRECTORY_FSYNC = "after-directory-fsync"
    F2_COMMIT = "after-f2-frozen-commit"


class StopReason(StrEnum):
    EOS = "eos"
    MAX_TOKENS = "max-tokens"
    STOP_SEQUENCE = "stop-sequence"


class RuntimeFailureKind(StrEnum):
    OOM = "oom"
    DRIVER = "driver"
    TIMEOUT = "timeout"
    TOKENIZER = "tokenizer"
    PROVIDER_REJECTED = "provider-rejected"
    UNKNOWN = "unknown"


class RudenessLabel(StrEnum):
    RUDE = "rude"
    NON_RUDE = "non_rude"


class SamplerPolicy(StrEnum):
    BALANCED_MATCHED = "balanced-matched"


class PresentationPolicy(StrEnum):
    SETUP_ONCE_IDS_LATER = "setup-once-ids-later"


class TiePolicy(StrEnum):
    UNIFORM_SEEDED_DRAW = "uniform-seeded-draw"


class ActionFormat(StrEnum):
    JSON_WITH_RATIONALE = "json-with-rationale"


class RngAlgorithm(StrEnum):
    FISHER_YATES_PYRANDOM_V1 = "fisher-yates-pyrandom/v1"
    PYRANDOM_RANDRANGE_V1 = "pyrandom-randrange/v1"


class SeedDomain(StrEnum):
    GENERATION = "generation"
    VOTER_PERMUTATION = "voter-permutation"
    TIE_BREAK = "tie-break"
    SUPPORT_REMOVAL = "support-removal"
    BALANCED_EXTRA_STRATUM = "balanced-extra-stratum"
    ANALYSIS_BOOTSTRAP = "analysis-bootstrap"


class ValidationErrorCode(StrEnum):
    MALFORMED_JSON = "malformed-json"
    MISSING_FIELD = "missing-field"
    EXTRA_FIELD = "extra-field"
    INVALID_TYPE = "invalid-type"
    UNKNOWN_CANDIDATE = "unknown-candidate"
    INACTIVE_CANDIDATE = "inactive-candidate"
    DUPLICATE_CANDIDATE = "duplicate-candidate"
    MISSING_CANDIDATE = "missing-candidate"
    NON_INTEGER_VOTES = "non-integer-votes"
    NEGATIVE_VOTES = "negative-votes"
    BUDGET_EXCEEDED = "budget-exceeded"
    UNKNOWN_RATING = "unknown-rating"
    EMPTY_STATEMENT = "empty-statement"
    EMPTY_RATIONALE = "empty-rationale"


class TemplateKind(StrEnum):
    SETUP = "setup"
    STATEMENT = "statement"
    BALLOT = "ballot"
    CORRECTION = "correction"
    RESULT = "result"
    FINAL_RESULT = "final-result"


class ExportDataset(StrEnum):
    RUNS = "runs"
    RUN_EXECUTIONS = "run_executions"
    VOTERS = "voters"
    ROUNDS = "rounds"
    TURNS = "turns"
    CALLS = "calls"
    VALIDATION_FAILURES = "validation_failures"
    RUNTIME_FAILURES = "runtime_failures"
    BALLOTS = "ballots"
    ALLOCATIONS = "allocations"
    STATEMENTS = "statements"
    STATEMENT_ITEMS = "statement_items"
    OUTCOMES = "outcomes"
    RNG_DRAWS = "rng_draws"


RunId = NewType("RunId", str)
MatchedSetId = NewType("MatchedSetId", str)
SampleId = NewType("SampleId", str)
VoterId = NewType("VoterId", str)
ReleaseId = NewType("ReleaseId", str)
TemplateId = NewType("TemplateId", str)
LabelPolicyId = NewType("LabelPolicyId", str)
CandidateId = NewType("CandidateId", str)
PresentationId = NewType("PresentationId", str)
TurnId = NewType("TurnId", str)
CallId = NewType("CallId", str)
RoundId = NewType("RoundId", str)
ExecutionId = NewType("ExecutionId", str)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_correction_attempts: int = 3

    def __post_init__(self) -> None:
        if (
            type(self.max_correction_attempts) is not int
            or self.max_correction_attempts < 0
        ):
            raise ValueError(
                "Retry policy construction failed because "
                f"max_correction_attempts={self.max_correction_attempts} is negative. "
                "Validation failed in quadratic_voting.experiment.types.RetryPolicy "
                "while configuring a run, so the attempt limit is unusable and no "
                "experiment can safely start. Set max_correction_attempts to zero or "
                "a positive integer and retry."
            )


@dataclass(frozen=True, slots=True)
class SamplingProfile:
    temperature: float
    top_p: float
    top_k: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        if (
            type(self.temperature) is not float
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError(
                "Sampling profile construction failed because "
                f"temperature={self.temperature!r} is not finite and non-negative. "
                "Validation failed in quadratic_voting.experiment.types.SamplingProfile "
                "while configuring generation, so no model call can safely start. Set "
                "temperature to a finite value greater than or equal to zero and retry."
            )
        if (
            type(self.top_p) is not float
            or not math.isfinite(self.top_p)
            or not 0 < self.top_p <= 1
        ):
            raise ValueError(
                "Sampling profile construction failed because "
                f"top_p={self.top_p!r} is not finite and within (0, 1]. Validation "
                "failed in quadratic_voting.experiment.types.SamplingProfile while "
                "configuring generation, so no model call can safely start. Set top_p "
                "to a finite value greater than zero and at most one, then retry."
            )
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError(
                "Sampling profile construction failed because "
                f"top_k={self.top_k} is not positive. Validation failed in "
                "quadratic_voting.experiment.types.SamplingProfile while configuring "
                "generation, so no model call can safely start. Set top_k to a positive "
                "integer and retry."
            )
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError(
                "Sampling profile construction failed because "
                f"max_new_tokens={self.max_new_tokens} is not positive. Validation "
                "failed in quadratic_voting.experiment.types.SamplingProfile while "
                "configuring generation, so no model call can safely start. Set "
                "max_new_tokens to a positive integer and retry."
            )


@dataclass(frozen=True, slots=True)
class RunConfig:
    arm: ElicitationArm
    regime: VotingRegime
    voter_count: int
    sampling: SamplingProfile
    credit_budget: int = 100
    retry_policy: RetryPolicy = RetryPolicy()
    tie_policy: TiePolicy = TiePolicy.UNIFORM_SEEDED_DRAW
    presentation_policy: PresentationPolicy = PresentationPolicy.SETUP_ONCE_IDS_LATER
    action_format: ActionFormat = ActionFormat.JSON_WITH_RATIONALE

    def __post_init__(self) -> None:
        if not isinstance(self.arm, ElicitationArm) or not isinstance(
            self.regime, VotingRegime
        ):
            raise ValueError(
                "Run configuration construction failed because arm/regime are not closed "
                "ElicitationArm/VotingRegime values. Validation failed before run creation. "
                "Parse dynamic strings into the typed enums and retry."
            )
        if type(self.voter_count) is not int or self.voter_count <= 0:
            raise ValueError(
                "Run configuration construction failed because "
                f"voter_count={self.voter_count} is not positive. Validation failed in "
                "quadratic_voting.experiment.types.RunConfig before run creation, so "
                "the round barrier would have no valid voter population. Set voter_count "
                "to a positive integer and retry."
            )
        if type(self.credit_budget) is not int or self.credit_budget <= 0:
            raise ValueError(
                "Run configuration construction failed because "
                f"credit_budget={self.credit_budget} is not positive. Validation failed "
                "in quadratic_voting.experiment.types.RunConfig before run creation, so "
                "ballots cannot use a valid quadratic-credit budget. Set credit_budget "
                "to a positive integer and retry."
            )


def arm_turn_order(arm: ElicitationArm) -> tuple[TurnKind, ...]:
    if arm is ElicitationArm.ACTION_ONLY:
        return (TurnKind.BALLOT,)
    if arm is ElicitationArm.STATEMENT_THEN_ACTION:
        return (TurnKind.STATEMENT, TurnKind.BALLOT)
    if arm is ElicitationArm.ACTION_THEN_STATEMENT:
        return (TurnKind.BALLOT, TurnKind.STATEMENT)
    raise AssertionError(f"unhandled closed elicitation arm: {arm!r}")


@dataclass(frozen=True, slots=True)
class RoundResult:
    protected: CandidateId | None
    removed: CandidateId
    totals: Mapping[CandidateId, int]
    tie_among: frozenset[CandidateId]


@dataclass(frozen=True, slots=True)
class SetupContext:
    run_id: RunId
    arm: ElicitationArm
    regime: VotingRegime
    credit_budget: int
    instructions: str
    candidate_cards: tuple[tuple[CandidateId, str], ...]


@dataclass(frozen=True, slots=True)
class PriorTurnEvent:
    round_index: int
    kind: TurnKind
    exchanges: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class RoundOutcomeEvent:
    round_index: int
    protected: CandidateId | None
    removed: CandidateId


@dataclass(frozen=True, slots=True)
class FinalResultEvent:
    winner: CandidateId


TranscriptEvent: TypeAlias = PriorTurnEvent | RoundOutcomeEvent | FinalResultEvent


@dataclass(frozen=True, slots=True)
class PendingTurn:
    round_index: int
    kind: TurnKind
    active: tuple[CandidateId, ...]
    attempt_index: int
    correction_errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VoterRoundView:
    setup: SetupContext
    history: tuple[TranscriptEvent, ...]
    pending: PendingTurn


@dataclass(frozen=True, slots=True)
class WorkUnit:
    run_id: RunId
    voter_id: VoterId
    voter_index: int
    round_id: RoundId
    round_index: int
    kind: TurnKind
    attempt_index: int


@dataclass(frozen=True, slots=True)
class RunComplete:
    run_id: RunId


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    prompt_token_count: int
    completion_token_count: int
    completion_token_ids: tuple[int, ...] | None
    stop_reason: StopReason
    duration_ms: int
    diagnostics: Mapping[str, str]

    def __post_init__(self) -> None:
        numeric = {
            "prompt_token_count": self.prompt_token_count,
            "completion_token_count": self.completion_token_count,
            "duration_ms": self.duration_ms,
        }
        invalid = {
            name: value
            for name, value in numeric.items()
            if type(value) is not int or value < 0
        }
        if invalid:
            raise ValueError(
                f"Generation result construction failed because nonnegative strict integer "
                f"metadata is invalid: {invalid}. Validation failed in experiment.types before "
                "T3 persistence. Return exact integer counts/duration (excluding bool) and retry."
            )
        if self.completion_token_ids is not None and any(
            type(token) is not int or token < 0 for token in self.completion_token_ids
        ):
            raise ValueError(
                "Generation result construction failed because completion_token_ids contains "
                "a non-integer or negative token ID. Validation failed before T3 persistence. "
                "Return tokenizer IDs as nonnegative strict integers or None."
            )


@dataclass(frozen=True, slots=True)
class ExecutionEnvironment:
    python_version: str
    torch_version: str
    transformers_version: str
    uv_lock_hash: str
    device: str
    dtype: str
    hostname: str
    git_commit: str
    git_dirty: bool
    cuda_runtime_version: str = "unknown"
    nvidia_driver_version: str = "unknown"
    cudnn_version: str = "unknown"
    gpu_model: str = "unknown"
    gpu_count: int = 0
    gpu_compute_capability: str = "unknown"
    gpu_uuid_hash: str = "unknown"
    os_name: str = "unknown"
    os_version: str = "unknown"
    kernel_version: str = "unknown"
    cpu_architecture: str = "unknown"
    deterministic_algorithms: bool = False
    tf32_enabled: bool = False
    cudnn_benchmark: bool = False
    tracked_tree_hash: str = "unknown"
    binary_diff_sha256: str = "unknown"
    untracked_manifest_hash: str = "unknown"
    untracked_tree_hash: str = "unknown"
    hostname_hash: str = "unknown"

    def __post_init__(self) -> None:
        if (
            type(self.git_dirty) is not bool
            or type(self.gpu_count) is not int
            or self.gpu_count < 0
        ):
            raise ValueError(
                "Execution environment construction failed because git_dirty must be bool and "
                "gpu_count must be a nonnegative strict integer. Validation failed before "
                "execution creation. Collect typed provenance and retry."
            )
        for name in ("deterministic_algorithms", "tf32_enabled", "cudnn_benchmark"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(
                    f"Execution environment construction failed because {name} is not bool. "
                    "Validation failed before execution creation. Collect typed flags and retry."
                )


@dataclass(frozen=True, slots=True)
class BarrierReady:
    run_id: RunId
    round_id: RoundId
    round_index: int


NextUnit: TypeAlias = WorkUnit | BarrierReady | RunComplete


class VoterGenerator(Protocol):
    def generate(
        self,
        messages: Sequence[ChatMessage],
        profile: SamplingProfile,
        seed: int,
    ) -> GenerationResult: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


@dataclass(frozen=True, slots=True)
class MatchedSetConfig:
    voter_count: int
    master_seed: int
    sampling: SamplingProfile
    credit_budget: int = 100
    retry_policy: RetryPolicy = RetryPolicy()
    tie_policy: TiePolicy = TiePolicy.UNIFORM_SEEDED_DRAW
    presentation_policy: PresentationPolicy = PresentationPolicy.SETUP_ONCE_IDS_LATER
    action_format: ActionFormat = ActionFormat.JSON_WITH_RATIONALE
    max_consecutive_runtime_failures: int = 3
    execution_class: ExecutionClass = ExecutionClass.FIXTURE

    def __post_init__(self) -> None:
        if not isinstance(self.sampling, SamplingProfile) or not isinstance(
            self.retry_policy, RetryPolicy
        ):
            raise ValueError(
                "Matched-set configuration construction failed because sampling/retry values "
                "are not typed SamplingProfile/RetryPolicy instances. Validation failed before "
                "persistence. Construct the strict nested types and retry."
            )
        if self.retry_policy.max_correction_attempts != 3:
            raise ValueError(
                "Matched-set configuration construction failed because qv-run-config/v1 "
                "requires exactly three corrections (four responses) for ballot and statement "
                "turns. Create a new versioned treatment rather than overriding retries."
            )
        if type(self.voter_count) is not int or self.voter_count <= 0:
            raise ValueError(
                "Matched-set configuration construction failed because "
                f"voter_count={self.voter_count} is not positive. Validation failed in "
                "quadratic_voting.experiment.types.MatchedSetConfig before matched-set "
                "creation, so the six runs cannot have a valid voter population. Set "
                "voter_count to a positive integer and retry."
            )
        if type(self.credit_budget) is not int or self.credit_budget <= 0:
            raise ValueError(
                "Matched-set configuration construction failed because "
                f"credit_budget={self.credit_budget} is not positive. Validation failed "
                "in quadratic_voting.experiment.types.MatchedSetConfig before matched-set "
                "creation, so the six runs cannot use a valid quadratic-credit budget. "
                "Set credit_budget to a positive integer and retry."
            )
        if (
            type(self.max_consecutive_runtime_failures) is not int
            or self.max_consecutive_runtime_failures <= 0
        ):
            raise ValueError(
                "Matched-set configuration construction failed because "
                "max_consecutive_runtime_failures="
                f"{self.max_consecutive_runtime_failures} is not positive. Validation "
                "failed in quadratic_voting.experiment.types.MatchedSetConfig before "
                "matched-set creation, so runtime faults could pause immediately or retry "
                "without a valid bound. Set max_consecutive_runtime_failures to a "
                "positive integer and retry."
            )
        if (
            type(self.master_seed) is not int
            or not 0 <= self.master_seed <= (1 << 64) - 1
        ):
            raise ValueError(
                "Matched-set configuration construction failed because master_seed="
                f"{self.master_seed!r} is not an unsigned 64-bit integer. Validation failed "
                "in quadratic_voting.experiment.types.MatchedSetConfig before persistence, "
                "so deterministic streams cannot be represented exactly. Supply an integer "
                "from zero through 2**64 - 1 (excluding bool) and retry."
            )


@dataclass(frozen=True, slots=True)
class MatchedSetCreation:
    matched_set_id: MatchedSetId
    run_ids: Mapping[tuple[ElicitationArm, VotingRegime], RunId]
