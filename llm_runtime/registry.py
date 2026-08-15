"""Static reviewed model/provider/quantization routes."""

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, TypeAlias, overload

from llm_runtime.types import (
    Capability,
    ModelId,
    ProviderId,
    QuantizationId,
    RuntimeId,
)

TransformersModelId: TypeAlias = Literal[
    ModelId.GEMMA_4_E2B_IT,
    ModelId.GEMMA_4_31B_IT,
    ModelId.GEMMA_2_2B_IT,
]
OpenRouterModelId: TypeAlias = Literal[ModelId.DOLPHIN_MISTRAL_24B_VENICE]
TransformersQuantizationId: TypeAlias = Literal[
    QuantizationId.BF16,
    QuantizationId.W4A16_COMPRESSED_TENSORS,
    QuantizationId.BITSANDBYTES_FP4,
]
RouteKey: TypeAlias = tuple[ModelId, ProviderId, QuantizationId | None]


class RouteAvailability(StrEnum):
    ENABLED = "enabled"
    UNAVAILABLE = "unavailable"


class LocalLoaderKind(StrEnum):
    BF16 = "bf16"
    COMPRESSED_TENSORS_W4A16 = "compressed-tensors-w4a16"
    BITSANDBYTES_4BIT = "bitsandbytes-4bit"


class FourBitQuantType(StrEnum):
    FP4 = "fp4"


class TorchDTypeId(StrEnum):
    BFLOAT16 = "bfloat16"
    UINT8 = "uint8"


@dataclass(frozen=True, slots=True)
class BitsAndBytes4BitSettings:
    load_in_4bit: Literal[True]
    quant_type: Literal[FourBitQuantType.FP4]
    compute_dtype: Literal[TorchDTypeId.BFLOAT16]
    quant_storage: Literal[TorchDTypeId.UINT8]
    use_double_quant: Literal[False]


@dataclass(frozen=True, slots=True)
class HuggingFaceArtifact:
    repository: str
    revision: str


@dataclass(frozen=True, slots=True)
class LocalTransformersRoute:
    model_id: TransformersModelId
    quantization_id: TransformersQuantizationId
    artifact: HuggingFaceArtifact
    loader: LocalLoaderKind
    context_window: int
    bitsandbytes: BitsAndBytes4BitSettings | None = None
    availability: RouteAvailability = RouteAvailability.ENABLED
    unavailable_reason: str | None = None
    provider_id: Literal[ProviderId.LOCAL] = ProviderId.LOCAL
    runtime_id: Literal[RuntimeId.TRANSFORMERS] = RuntimeId.TRANSFORMERS
    capabilities: frozenset[Capability] = frozenset(
        {Capability.TEXT_GENERATION, Capability.LOCAL_ACTIVATIONS}
    )


@dataclass(frozen=True, slots=True)
class OpenRouterRoute:
    model_id: OpenRouterModelId
    model_slug: str
    quantization_id: None = None
    availability: RouteAvailability = RouteAvailability.ENABLED
    unavailable_reason: str | None = None
    provider_id: Literal[ProviderId.OPENROUTER] = ProviderId.OPENROUTER
    runtime_id: Literal[RuntimeId.OPENAI_COMPATIBLE_HTTP] = (
        RuntimeId.OPENAI_COMPATIBLE_HTTP
    )
    capabilities: frozenset[Capability] = frozenset({Capability.TEXT_GENERATION})


ModelRoute: TypeAlias = LocalTransformersRoute | OpenRouterRoute


_ROUTES: Mapping[RouteKey, ModelRoute] = MappingProxyType(
    {
        (
            ModelId.GEMMA_4_E2B_IT,
            ProviderId.LOCAL,
            QuantizationId.BF16,
        ): LocalTransformersRoute(
            model_id=ModelId.GEMMA_4_E2B_IT,
            quantization_id=QuantizationId.BF16,
            artifact=HuggingFaceArtifact(
                repository="google/gemma-4-E2B-it-qat-q4_0-unquantized",
                revision="6befbaca7398925921802abd1f277b495b78b738",
            ),
            loader=LocalLoaderKind.BF16,
            context_window=131_072,
        ),
        (
            ModelId.GEMMA_4_E2B_IT,
            ProviderId.LOCAL,
            QuantizationId.W4A16_COMPRESSED_TENSORS,
        ): LocalTransformersRoute(
            model_id=ModelId.GEMMA_4_E2B_IT,
            quantization_id=QuantizationId.W4A16_COMPRESSED_TENSORS,
            artifact=HuggingFaceArtifact(
                repository="google/gemma-4-E2B-it-qat-w4a16-ct",
                revision="971342c08f607aa7779983f6b5289778b5d271a7",
            ),
            loader=LocalLoaderKind.COMPRESSED_TENSORS_W4A16,
            context_window=131_072,
            availability=RouteAvailability.UNAVAILABLE,
            unavailable_reason=(
                "the exact pinned W4A16 artifact has passed metadata parsing only, "
                "not exact pinned-revision weight loading, text generation, and "
                "verification that the runtime exposes the real model and tokenizer "
                "for activation access. Metadata validation cannot prove executable "
                "or activation compatibility. Keep this candidate unavailable with "
                "no capabilities until all three checks pass in review"
            ),
            capabilities=frozenset(),
        ),
        (
            ModelId.GEMMA_4_31B_IT,
            ProviderId.LOCAL,
            QuantizationId.BITSANDBYTES_FP4,
        ): LocalTransformersRoute(
            model_id=ModelId.GEMMA_4_31B_IT,
            quantization_id=QuantizationId.BITSANDBYTES_FP4,
            artifact=HuggingFaceArtifact(
                repository="google/gemma-4-31B-it",
                revision="842da3794eaa0b77d5f08bae87a17459d91ff475",
            ),
            loader=LocalLoaderKind.BITSANDBYTES_4BIT,
            context_window=131_072,
            bitsandbytes=BitsAndBytes4BitSettings(
                load_in_4bit=True,
                quant_type=FourBitQuantType.FP4,
                compute_dtype=TorchDTypeId.BFLOAT16,
                quant_storage=TorchDTypeId.UINT8,
                use_double_quant=False,
            ),
        ),
        (
            ModelId.GEMMA_4_31B_IT,
            ProviderId.LOCAL,
            QuantizationId.W4A16_COMPRESSED_TENSORS,
        ): LocalTransformersRoute(
            model_id=ModelId.GEMMA_4_31B_IT,
            quantization_id=QuantizationId.W4A16_COMPRESSED_TENSORS,
            artifact=HuggingFaceArtifact(
                repository="google/gemma-4-31B-it-qat-w4a16-ct",
                revision="52f3f65bc7a02d555763bc923bd1d9094898219d",
            ),
            loader=LocalLoaderKind.COMPRESSED_TENSORS_W4A16,
            context_window=131_072,
        ),
        (
            ModelId.GEMMA_2_2B_IT,
            ProviderId.LOCAL,
            QuantizationId.BF16,
        ): LocalTransformersRoute(
            model_id=ModelId.GEMMA_2_2B_IT,
            quantization_id=QuantizationId.BF16,
            artifact=HuggingFaceArtifact(
                repository="google/gemma-2-2b-it",
                revision="299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8",
            ),
            loader=LocalLoaderKind.BF16,
            context_window=8_192,
        ),
        (
            ModelId.DOLPHIN_MISTRAL_24B_VENICE,
            ProviderId.OPENROUTER,
            None,
        ): OpenRouterRoute(
            model_id=ModelId.DOLPHIN_MISTRAL_24B_VENICE,
            model_slug=("cognitivecomputations/dolphin-mistral-24b-venice-edition"),
        ),
    }
)


class ModelRouteError(ValueError):
    """A route failed closed-registry validation before provider construction."""


def registered_routes() -> tuple[ModelRoute, ...]:
    """Return the immutable reviewed route set in declaration order."""
    return tuple(_ROUTES.values())


@overload
def require_registered_route(
    route: LocalTransformersRoute,
) -> LocalTransformersRoute: ...


@overload
def require_registered_route(route: OpenRouterRoute) -> OpenRouterRoute: ...


def require_registered_route(route: ModelRoute) -> ModelRoute:
    """Reject forged route dataclasses at every provider construction boundary."""
    key = (route.model_id, route.provider_id, route.quantization_id)
    canonical = _ROUTES.get(key)
    if canonical != route:
        raise ModelRouteError(
            f"Provider construction rejected {_route_label(key)} because the "
            "supplied route does not exactly match the static registry entry. "
            "Validation failed in llm_runtime.registry.require_registered_route "
            "at the provider boundary, so no artifact download, model load, or "
            "HTTP request can run. Obtain the route from resolve_route() instead "
            "of constructing a route or supplying an arbitrary artifact or slug."
        )
    if canonical.availability is RouteAvailability.UNAVAILABLE:
        raise ModelRouteError(
            f"Provider construction rejected {_route_label(key)} because its "
            f"canonical route is unavailable: {canonical.unavailable_reason}. "
            "Validation failed in llm_runtime.registry.require_registered_route "
            "before provider work, so no fallback will run. Select an enabled "
            "route or resolve and review the compatibility blocker."
        )
    return canonical


def _route_label(key: RouteKey) -> str:
    model_id, provider_id, quantization_id = key
    quantization = quantization_id.value if quantization_id is not None else "none"
    return f"{model_id.value}/{provider_id.value}/{quantization}"


def _alternatives() -> str:
    return ", ".join(
        _route_label(key)
        for key, route in _ROUTES.items()
        if route.availability is RouteAvailability.ENABLED
    )


def resolve_route(
    model_id: ModelId,
    provider_id: ProviderId,
    quantization_id: QuantizationId | None,
    *,
    required: Set[Capability] = frozenset(),
) -> ModelRoute:
    """Validate a dynamic route once before constructing a provider runtime."""
    key = (model_id, provider_id, quantization_id)
    route = _ROUTES.get(key)
    if route is None:
        raise ModelRouteError(
            f"Model route resolution failed for {_route_label(key)} because that "
            "model/provider/quantization triple is not registered. Validation "
            "failed in llm_runtime.registry.resolve_route during runtime "
            "construction, so the caller cannot start generation. Supported "
            f"routes are: {_alternatives()}. Select one of those exact triples or "
            "add and review a pinned route in llm_runtime.registry."
        )
    if route.availability is RouteAvailability.UNAVAILABLE:
        raise ModelRouteError(
            f"Model route resolution failed for {_route_label(key)} because the "
            f"route is explicitly unavailable: {route.unavailable_reason}. "
            "Validation failed in llm_runtime.registry.resolve_route during "
            "runtime construction, so no fallback model will run. Select an "
            f"enabled route ({_alternatives()}) or resolve and review the stated "
            "compatibility blocker."
        )
    missing = required - route.capabilities
    if missing:
        missing_text = ", ".join(sorted(capability.value for capability in missing))
        available_text = ", ".join(
            sorted(capability.value for capability in route.capabilities)
        )
        raise ModelRouteError(
            f"Capability validation failed for {_route_label(key)} because it "
            f"does not provide: {missing_text}. The route provides: "
            f"{available_text}. Validation failed in "
            "llm_runtime.registry.resolve_route before runtime construction, so "
            "the requested experiment cannot safely start. Choose a registered "
            "local route for activation access or request only capabilities the "
            "selected route declares."
        )
    return route
