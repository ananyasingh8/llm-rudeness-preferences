"""Provider-independent model identities and generation values."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

MAX_RETRY_DELAY_SECONDS = 30.0


class ModelId(StrEnum):
    GEMMA_4_E2B_IT = "gemma-4-e2b-it"
    GEMMA_2_2B_IT = "gemma-2-2b-it"
    DOLPHIN_MISTRAL_24B_VENICE = "dolphin-mistral-24b-venice"


class ProviderId(StrEnum):
    LOCAL = "local"
    OPENROUTER = "openrouter"


class QuantizationId(StrEnum):
    BF16 = "bf16"
    W4A16_COMPRESSED_TENSORS = "w4a16-compressed-tensors"


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
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retry_after_seconds = retry_after_seconds

    @property
    def retryable(self) -> bool:
        return self.kind in {
            GenerationFailureKind.TRANSIENT_TRANSPORT,
            GenerationFailureKind.RETRYABLE_PROVIDER_STATUS,
        }


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    max_new_tokens: int
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(
                "Generation settings construction failed because "
                f"max_new_tokens={self.max_new_tokens} is not positive. "
                "Validation failed in llm_runtime.types.GenerationSettings before "
                "generation, so no provider request was made. Set max_new_tokens "
                "to a positive integer and retry."
            )
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError(
                "Generation settings construction failed because "
                f"temperature={self.temperature!r} is not finite and non-negative. "
                "Validation failed in llm_runtime.types.GenerationSettings before "
                "generation, so no provider request was made. Set temperature to "
                "a finite value greater than or equal to zero and retry."
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
    ) -> str: ...
