"""Gemma adapter from the local Transformers runtime to experiment results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from llm_runtime.transformers import TransformersRuntime
from llm_runtime.types import (
    ChatMessage,
    DiagnosticValue,
    FinishReason,
    GenerationSettings,
    sanitized_diagnostics,
)
from quadratic_voting.experiment.types import (
    GenerationResult,
    SamplingProfile,
    StopReason,
)

# Gemma model-card recommended stochastic settings. Matched-set creation freezes
# this profile; operators may tune it only after evaluating the real-model pilot.
GEMMA_MODEL_CARD_PROFILE: Final[SamplingProfile] = SamplingProfile(
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    max_new_tokens=8192,
)


class GemmaVoterGeneratorError(RuntimeError):
    """Actionable failure while adapting local generation into experiment data."""

    def __init__(
        self,
        message: str,
        *,
        finish_reason: FinishReason,
        diagnostics: Mapping[str, DiagnosticValue],
    ) -> None:
        super().__init__(message)
        self.finish_reason = finish_reason
        self.diagnostics = sanitized_diagnostics(diagnostics)


class GemmaVoterGenerator:
    """Generate one seeded response and derive auditable local token metadata."""

    def __init__(
        self,
        runtime: TransformersRuntime,
    ) -> None:
        self._runtime = runtime

    def generate(
        self,
        messages: Sequence[ChatMessage],
        profile: SamplingProfile,
        seed: int,
    ) -> GenerationResult:
        settings = GenerationSettings(
            max_new_tokens=profile.max_new_tokens,
            temperature=profile.temperature,
            top_p=profile.top_p,
            top_k=profile.top_k,
            seed=seed,
        )
        result = self._runtime.generate(messages, settings)
        if result.finish_reason is FinishReason.LENGTH:
            stop_reason = StopReason.MAX_TOKENS
        else:
            try:
                stop_reason = StopReason(result.finish_reason.value)
            except ValueError:
                raise GemmaVoterGeneratorError(
                    "Gemma result adaptation failed in quadratic_voting.experiment."
                    "gemma.GemmaVoterGenerator.generate after local generation because "
                    f"finish reason {result.finish_reason.value!r} is not representable "
                    "by the experiment result enum. The caller must not persist this "
                    "response under a false EOS reason. Expand the core-owned StopReason "
                    "enum and retry the interrupted call; sanitized runtime diagnostics "
                    "remain attached to this error.",
                    finish_reason=result.finish_reason,
                    diagnostics=result.diagnostics,
                ) from None
        return GenerationResult(
            text=result.raw_text,
            prompt_token_count=result.prompt_token_count,
            completion_token_count=result.completion_token_count,
            completion_token_ids=result.completion_token_ids,
            stop_reason=stop_reason,
            duration_ms=result.duration_ms,
            diagnostics={key: str(value) for key, value in result.diagnostics.items()},
        )
