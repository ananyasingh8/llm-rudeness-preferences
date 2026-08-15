"""Intentional mypy failure proving remote generators lack activation access."""

from llm_runtime.openrouter import OpenRouterGenerator
from llm_runtime.transformers import LocalActivationRuntime, TransformersRuntime


def prove_activation_boundary(
    local: TransformersRuntime,
    remote: OpenRouterGenerator,
) -> None:
    accepted: LocalActivationRuntime = local
    rejected: LocalActivationRuntime = remote  # type: ignore[assignment]
    _ = (accepted, rejected)
