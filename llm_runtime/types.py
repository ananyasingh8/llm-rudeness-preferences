"""Provider-independent model identities and generation values."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

MAX_RETRY_DELAY_SECONDS = 30.0


class ModelId(StrEnum):
    GEMMA_4_E2B_IT = "gemma-4-e2b-it"
    GEMMA_4_31B_IT = "gemma-4-31b-it"
    GEMMA_4_31B = "gemma-4-31b"
    GEMMA_2_2B_IT = "gemma-2-2b-it"
    DOLPHIN_MISTRAL_24B_VENICE = "dolphin-mistral-24b-venice"


class ProviderId(StrEnum):
    LOCAL = "local"
    OPENROUTER = "openrouter"


class QuantizationId(StrEnum):
    BF16 = "bf16"
    W4A16_COMPRESSED_TENSORS = "w4a16-compressed-tensors"
    BITSANDBYTES_FP4 = "bitsandbytes-fp4"


class RuntimeId(StrEnum):
    TRANSFORMERS = "transformers"
    OPENAI_COMPATIBLE_HTTP = "openai-compatible-http"


class Capability(StrEnum):
    TEXT_GENERATION = "text-generation"
    LOCAL_ACTIVATIONS = "local-activations"


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class FinishReason(StrEnum):
    EOS = "eos"
    LENGTH = "length"
    STOP_SEQUENCE = "stop-sequence"
    CONTENT_FILTER = "content-filter"
    PROVIDER_OTHER = "provider-other"


DiagnosticValue = str | int | float | bool | None
SANITIZED_DIAGNOSTIC_KEYS = frozenset(
    {
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


def sanitized_diagnostics(
    diagnostics: Mapping[str, DiagnosticValue],
) -> Mapping[str, DiagnosticValue]:
    """Validate and freeze provider-safe scalar diagnostics at a public boundary."""
    if not isinstance(diagnostics, Mapping):
        raise ValueError(
            "Runtime diagnostics must be a mapping of allowlisted scalar metadata; "
            "remove raw bodies, prompts, headers, credentials, and exceptions."
        )
    clean: dict[str, DiagnosticValue] = {}
    for key, value in diagnostics.items():
        if (
            not isinstance(key, str)
            or not key
            or not (value is None or isinstance(value, (str, int, float, bool)))
        ):
            raise ValueError(
                "Runtime diagnostics must use non-empty string keys and scalar values."
            )
        if key not in SANITIZED_DIAGNOSTIC_KEYS:
            raise ValueError(
                f"Runtime diagnostics key {key!r} is not in the sanitized allowlist; "
                "remove credentials, headers, prompts, raw bodies, and arbitrary "
                "exception data before crossing the runtime boundary."
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Runtime diagnostics cannot contain non-finite floats.")
        clean[key] = value
    return MappingProxyType(clean)


@dataclass(frozen=True, slots=True)
class GenerationResult:
    raw_text: str
    prompt_token_count: int
    completion_token_count: int
    completion_token_ids: tuple[int, ...] | None
    finish_reason: FinishReason
    duration_ms: int
    diagnostics: Mapping[str, DiagnosticValue]

    def __post_init__(self) -> None:
        if not isinstance(self.raw_text, str):
            raise ValueError(
                "GenerationResult.raw_text must be a strict string; the provider "
                "returned an incompatible response at the shared runtime boundary, "
                "so the caller must reject the call rather than persist coerced text."
            )
        if not isinstance(self.finish_reason, FinishReason):
            raise ValueError(
                "GenerationResult.finish_reason must be a FinishReason enum value; "
                "map the provider-native reason in its runtime adapter before returning."
            )
        for name in ("prompt_token_count", "completion_token_count", "duration_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"GenerationResult.{name} must be a non-negative strict integer; "
                    f"got {value!r} at the provider result boundary. Reject or repair "
                    "the provider parser before persisting this response."
                )
        if self.completion_token_ids is not None:
            if not isinstance(self.completion_token_ids, tuple) or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0
                for v in self.completion_token_ids
            ):
                raise ValueError(
                    "GenerationResult completion_token_ids must be a tuple containing "
                    "only non-negative strict integers; fix the local runtime adapter."
                )
            if len(self.completion_token_ids) != self.completion_token_count:
                raise ValueError(
                    "GenerationResult completion_token_ids length must equal "
                    "completion_token_count; the local runtime returned inconsistent "
                    "token provenance and the caller must not persist it."
                )
        object.__setattr__(self, "diagnostics", sanitized_diagnostics(self.diagnostics))


class GenerationFailureKind(StrEnum):
    CONFIGURATION = "configuration"
    TRANSIENT_TRANSPORT = "transient-transport"
    RETRYABLE_PROVIDER_STATUS = "retryable-provider-status"
    PERMANENT_PROVIDER_STATUS = "permanent-provider-status"
    MALFORMED_PROVIDER_RESPONSE = "malformed-provider-response"


class GenerationError(RuntimeError):
    """Provider failure with a typed retryability decision."""

    def __init__(
        self,
        message: str,
        *,
        kind: GenerationFailureKind,
        retry_after_seconds: float | None = None,
        diagnostics: Mapping[str, DiagnosticValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retry_after_seconds = retry_after_seconds
        self.diagnostics = sanitized_diagnostics(diagnostics or {})

    @property
    def retryable(self) -> bool:
        return self.kind in {
            GenerationFailureKind.TRANSIENT_TRANSPORT,
            GenerationFailureKind.RETRYABLE_PROVIDER_STATUS,
        }


class UnsupportedSettingError(GenerationError):
    """A provider/runtime rejected one generation setting before its request."""

    def __init__(
        self,
        *,
        setting: str,
        value: object,
        provider_id: ProviderId,
        runtime_id: RuntimeId,
        location: str,
        alternative_provider_id: ProviderId,
    ) -> None:
        super().__init__(
            f"Generation setting {setting}={value!r} was rejected by provider "
            f"{provider_id.value} using runtime {runtime_id.value} because that "
            "provider/runtime contract does not support the setting. Validation "
            f"failed in {location} before any provider request was made, so "
            "generation did not start and the caller has no generated text. Set "
            f"{setting}=None or use provider {alternative_provider_id.value}, then "
            "retry.",
            kind=GenerationFailureKind.CONFIGURATION,
        )
        self.setting = setting
        self.value = value
        self.provider_id = provider_id
        self.runtime_id = runtime_id
        self.location = location
        self.alternative_provider_id = alternative_provider_id


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Provider-independent generation controls.

    Local Transformers honors ``seed`` with a scoped generator and applies
    ``top_p``/``top_k`` when ``temperature > 0``; sampling filters are intentionally
    not forwarded for greedy decoding. OpenRouter forwards ``top_p`` regardless of
    temperature and rejects ``top_k`` and ``seed`` with UnsupportedSettingError.
    """

    max_new_tokens: int
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens <= 0
        ):
            raise ValueError(
                "Generation settings construction failed because "
                f"max_new_tokens={self.max_new_tokens} is not positive. "
                "Validation failed in llm_runtime.types.GenerationSettings before "
                "generation, so no provider request was made. Set max_new_tokens "
                "to a positive integer and retry."
            )
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise ValueError(
                "Generation settings construction failed because "
                f"temperature={self.temperature!r} is not finite and non-negative. "
                "Validation failed in llm_runtime.types.GenerationSettings before "
                "generation, so no provider request was made. Set temperature to "
                "a finite value greater than or equal to zero and retry."
            )
        if self.top_p is not None and (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not math.isfinite(self.top_p)
            or not 0 < self.top_p <= 1
        ):
            raise ValueError(
                "Generation settings construction failed because "
                f"top_p={self.top_p!r} is not finite and in the interval (0, 1]. "
                "Validation failed in llm_runtime.types.GenerationSettings before "
                "generation, so no provider request was made. Set top_p to a finite "
                "value greater than zero and less than or equal to one, or set it "
                "to None, then retry."
            )
        if self.top_k is not None and (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k <= 0
        ):
            raise ValueError(
                "Generation settings construction failed because "
                f"top_k={self.top_k!r} is not a positive integer. Validation "
                "failed in llm_runtime.types.GenerationSettings before generation, "
                "so no provider request was made. Set top_k to an integer greater "
                "than zero, or set it to None, then retry."
            )
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or self.seed > 2**64 - 1
        ):
            raise ValueError(
                "Generation settings construction failed because "
                f"seed={self.seed!r} is not a non-negative integer. Validation "
                "failed in llm_runtime.types.GenerationSettings before generation, "
                "so no provider request was made. Set seed to an integer greater "
                "than or equal to zero, or set it to None, then retry."
            )


class TextGenerator(Protocol):
    @property
    def model_id(self) -> ModelId: ...

    @property
    def provider_id(self) -> ProviderId: ...

    @property
    def quantization_id(self) -> QuantizationId | None: ...

    def generate(
        self,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
    ) -> GenerationResult: ...
