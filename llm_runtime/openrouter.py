"""Minimal OpenRouter text-generation adapter using httpx directly."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import TracebackType

import httpx

from llm_runtime.registry import OpenRouterRoute, require_registered_route
from llm_runtime.types import (
    ChatMessage,
    GenerationError,
    GenerationFailureKind,
    GenerationSettings,
    MAX_RETRY_DELAY_SECONDS,
    ModelId,
    ProviderId,
    QuantizationId,
)

OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})


class OpenRouterRuntimeError(GenerationError):
    """An actionable OpenRouter construction or response failure."""


@dataclass(frozen=True, slots=True, repr=False)
class OpenRouterCredentials:
    api_key: str = field(repr=False)

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise OpenRouterRuntimeError(
                "OpenRouter credential construction failed because the API key "
                "was blank. Validation failed in "
                "llm_runtime.openrouter.OpenRouterCredentials before HTTP work, "
                "so no request was made. Export a non-empty OPENROUTER_API_KEY "
                "and retry.",
                kind=GenerationFailureKind.CONFIGURATION,
            )


def openrouter_credentials_from_env(
    environ: Mapping[str, str] | None = None,
) -> OpenRouterCredentials:
    source = os.environ if environ is None else environ
    api_key = source.get("OPENROUTER_API_KEY", "")
    if not api_key.strip():
        raise OpenRouterRuntimeError(
            "OpenRouter runtime construction failed because OPENROUTER_API_KEY "
            "is missing or blank. Environment validation failed in "
            "llm_runtime.openrouter.openrouter_credentials_from_env before HTTP "
            "work, so the caller cannot generate remotely. Export "
            "OPENROUTER_API_KEY in the process environment and retry.",
            kind=GenerationFailureKind.CONFIGURATION,
        )
    return OpenRouterCredentials(api_key=api_key)


class OpenRouterGenerator:
    def __init__(
        self,
        route: OpenRouterRoute,
        credentials: OpenRouterCredentials,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._route = require_registered_route(route)
        self._credentials = credentials
        self._owns_client = http_client is None
        self._client = (
            http_client if http_client is not None else httpx.Client(timeout=60.0)
        )
        self._closed = False

    def close(self) -> None:
        """Close only a client created by this generator; safe to call repeatedly."""
        if self._owns_client and not self._closed:
            self._client.close()
            self._closed = True

    def __enter__(self) -> OpenRouterGenerator:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def model_id(self) -> ModelId:
        return self._route.model_id

    @property
    def provider_id(self) -> ProviderId:
        return self._route.provider_id

    @property
    def quantization_id(self) -> QuantizationId | None:
        return self._route.quantization_id

    def generate(
        self,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
    ) -> str:
        request = {
            "model": self._route.model_slug,
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in messages
            ],
            "max_tokens": settings.max_new_tokens,
            "temperature": settings.temperature,
        }
        try:
            response = self._client.post(
                OPENROUTER_CHAT_URL,
                headers={"Authorization": f"Bearer {self._credentials.api_key}"},
                json=request,
            )
        except httpx.TransportError as error:
            raise OpenRouterRuntimeError(
                "OpenRouter generation failed during HTTP transport in "
                "llm_runtime.openrouter.OpenRouterGenerator.generate. The request "
                "did not produce a response, so the caller has no generated text. "
                "Check network access and OpenRouter availability, then retry. "
                f"Transport type: {type(error).__name__}.",
                kind=GenerationFailureKind.TRANSIENT_TRANSPORT,
            ) from None
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            retryable = status_code in RETRYABLE_HTTP_STATUSES
            retry_after = _retry_after_seconds(error.response) if retryable else None
            remediation = (
                "Honor Retry-After when present and retry within the caller's "
                "bounded policy."
                if retryable
                else "Do not retry this unchanged request; verify credentials, "
                "account access, the registered model, and request validation."
            )
            raise OpenRouterRuntimeError(
                "OpenRouter generation failed during status validation in "
                "llm_runtime.openrouter.OpenRouterGenerator.generate because the "
                f"service returned HTTP {status_code}. No generated text is "
                f"available. {remediation}",
                kind=(
                    GenerationFailureKind.RETRYABLE_PROVIDER_STATUS
                    if retryable
                    else GenerationFailureKind.PERMANENT_PROVIDER_STATUS
                ),
                retry_after_seconds=retry_after,
            ) from None
        try:
            payload: object = response.json()
        except ValueError:
            raise OpenRouterRuntimeError(
                "OpenRouter generation failed during response parsing in "
                "llm_runtime.openrouter.OpenRouterGenerator.generate because the "
                "successful response was not JSON. The caller has no trustworthy "
                "generated text. Retry the request and check provider status if it "
                "persists. Do not retry this unchanged response automatically.",
                kind=GenerationFailureKind.MALFORMED_PROVIDER_RESPONSE,
            ) from None
        content = _response_content(payload)
        if content is None or not content.strip():
            raise OpenRouterRuntimeError(
                "OpenRouter generation failed during response validation in "
                "llm_runtime.openrouter.OpenRouterGenerator.generate because the "
                "successful response had no non-empty first assistant message. "
                "The caller has no generated text. Retry, or inspect OpenRouter "
                "service status and the registered model route. Do not retry this "
                "unchanged response automatically.",
                kind=GenerationFailureKind.MALFORMED_PROVIDER_RESPONSE,
            )
        return content


def _retry_after_seconds(response: httpx.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return min(seconds, MAX_RETRY_DELAY_SECONDS)


def _response_content(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    message = choice.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def create_openrouter_generator(
    route: OpenRouterRoute,
    *,
    credentials: OpenRouterCredentials,
    http_client: httpx.Client | None = None,
) -> OpenRouterGenerator:
    return OpenRouterGenerator(route, credentials, http_client)
