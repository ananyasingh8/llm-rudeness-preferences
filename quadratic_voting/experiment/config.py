"""Strict, versioned, complete matched-set configuration file models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    StrictBool,
    ValidationError,
    field_validator,
    model_validator,
)

from quadratic_voting.experiment.types import ElicitationArm, VotingRegime

NonEmptyStr = Annotated[str, StringConstraints(strict=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
UInt64 = Annotated[int, Field(strict=True, ge=0, le=(1 << 64) - 1)]
PositiveStrictInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeStrictInt = Annotated[int, Field(strict=True, ge=0)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class ReleaseSelector(_StrictModel):
    release_id: NonEmptyStr
    dataset_name: NonEmptyStr
    version: NonEmptyStr
    expected_sha256: Sha256


class LabelPolicySelector(_StrictModel):
    label_policy_id: NonEmptyStr
    name: NonEmptyStr
    version: NonEmptyStr
    expected_sha256: Sha256
    reviewed: StrictBool = False
    review_version: NonEmptyStr | None = None
    review_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def review_binding(self) -> LabelPolicySelector:
        if self.reviewed != (
            self.review_version is not None and self.review_sha256 is not None
        ):
            raise ValueError(
                "reviewed label policy requires review_version and review_sha256"
            )
        return self


class TemplateSelector(_StrictModel):
    template_id: NonEmptyStr
    name: NonEmptyStr
    version: NonEmptyStr
    expected_sha256: Sha256


class SampleSelector(_StrictModel):
    sample_id: NonEmptyStr
    artifact_path: Path
    expected_sha256: Sha256
    release: ReleaseSelector
    label_policy: LabelPolicySelector
    presentation_template: TemplateSelector


class RouteSelector(_StrictModel):
    model_id: NonEmptyStr
    provider_id: NonEmptyStr
    quantization_id: NonEmptyStr
    runtime_id: NonEmptyStr
    artifact_repository: NonEmptyStr
    artifact_revision: NonEmptyStr
    tokenizer_repository: NonEmptyStr
    tokenizer_revision: NonEmptyStr
    dtype: NonEmptyStr


class TemplateProfileSelector(_StrictModel):
    setup: TemplateSelector
    statement: TemplateSelector
    ballot: TemplateSelector
    correction: TemplateSelector
    result: TemplateSelector
    final_result: TemplateSelector
    reviewed: StrictBool = False
    review_version: NonEmptyStr | None = None
    review_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def unique_template_ids(self) -> TemplateProfileSelector:
        values = (
            self.setup,
            self.statement,
            self.ballot,
            self.correction,
            self.result,
            self.final_result,
        )
        ids = [value.template_id for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError(
                "prompt template binding failed because template IDs are not unique across "
                "the six required kinds; use one versioned template row per kind"
            )
        if self.reviewed != (
            self.review_version is not None and self.review_sha256 is not None
        ):
            raise ValueError(
                "reviewed prompt profile requires review_version and review_sha256"
            )
        return self


class SamplingProfileV1(_StrictModel):
    temperature: float
    top_p: float
    top_k: PositiveStrictInt
    max_new_tokens: PositiveStrictInt = 2048

    @field_validator("temperature", "top_p")
    @classmethod
    def finite_float(cls, value: float) -> float:
        if isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(
                "sampling values must be finite JSON numbers, excluding bool"
            )
        return value

    @model_validator(mode="after")
    def bounds(self) -> SamplingProfileV1:
        if self.temperature < 0:
            raise ValueError("temperature must be greater than or equal to zero")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be greater than zero and at most one")
        return self


class TurnRetryPolicy(_StrictModel):
    max_corrections: NonNegativeStrictInt = 3

    @model_validator(mode="after")
    def settled_v1(self) -> TurnRetryPolicy:
        if self.max_corrections != 3:
            raise ValueError(
                "qv-run-config/v1 requires exactly three corrections (four total responses)"
            )
        return self


class RuntimeRetryPolicy(_StrictModel):
    max_failures_per_execution: PositiveStrictInt = 3
    initial_backoff_ms: PositiveStrictInt = 1000
    multiplier: float = 2.0
    max_backoff_ms: PositiveStrictInt = 2000

    @field_validator("multiplier")
    @classmethod
    def multiplier_is_finite(cls, value: float) -> float:
        if isinstance(value, bool) or not math.isfinite(value) or value < 1:
            raise ValueError("runtime retry multiplier must be finite and at least one")
        return value

    @model_validator(mode="after")
    def settled_v1(self) -> RuntimeRetryPolicy:
        expected = (3, 1000, 2.0, 2000)
        actual = (
            self.max_failures_per_execution,
            self.initial_backoff_ms,
            self.multiplier,
            self.max_backoff_ms,
        )
        if actual != expected:
            raise ValueError(
                "qv-run-config/v1 runtime retry policy must be 3 failures with 1000ms, "
                "2x, and 2000ms cap"
            )
        return self


class MatchedSetConfigV1(_StrictModel):
    schema_version: Literal["qv-run-config/v1"]
    canonical_json_version: Literal["qv-canonical-json/v1"]
    prompt_encoding_version: Literal["qv-prompt/v1"]
    seed_version: Literal["qv-seed/v1"]
    sample: SampleSelector
    route: RouteSelector
    prompts: TemplateProfileSelector
    sampling: SamplingProfileV1
    ballot_retry: TurnRetryPolicy
    statement_retry: TurnRetryPolicy
    runtime_retry: RuntimeRetryPolicy
    master_seed: UInt64
    voter_count: PositiveStrictInt
    credit_budget: PositiveStrictInt = 100
    sampler_policy: Literal["balanced-matched/v1"]
    presentation_policy: Literal["setup-once-ids-later/v1"]
    tie_policy: Literal["uniform-seeded/v1"]
    action_format: Literal["json-with-rationale/v1"]
    execution_class: Literal["fixture", "pilot", "primary"]

    @model_validator(mode="after")
    def sampled_execution_temperature(self) -> MatchedSetConfigV1:
        if (
            self.execution_class in ("pilot", "primary")
            and self.sampling.temperature <= 0
        ):
            raise ValueError(
                "pilot and primary qv-run-config/v1 executions require temperature > 0; "
                "use fixture for deterministic zero-temperature tests"
            )
        if self.execution_class in ("pilot", "primary") and (
            not self.sample.label_policy.reviewed or not self.prompts.reviewed
        ):
            raise ValueError(
                "pilot/primary execution requires explicitly reviewed label policy and prompt "
                "profile version/hash approval; fixture-only defaults are not permitted"
            )
        return self

    @classmethod
    def from_json_file(cls, path: Path) -> MatchedSetConfigV1:
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"Run-config loading failed because {path} could not be read in "
                "experiment.config.MatchedSetConfigV1.from_json_file before database open. "
                "No experiment state was changed. Supply a readable qv-run-config/v1 file."
            ) from error
        try:
            return cls.model_validate_json(content)
        except ValidationError as error:
            details = error.errors(include_input=False, include_url=False)
            raise ValueError(
                f"Run-config validation failed for {path} in experiment.config before database "
                f"open: {details}. No matched set was created. Correct the reported field and "
                "retry without coercing its JSON type."
            ) from error


PRIMARY_RUN_MATRIX: tuple[tuple[ElicitationArm, VotingRegime], ...] = tuple(
    (arm, regime) for arm in ElicitationArm for regime in VotingRegime
)
