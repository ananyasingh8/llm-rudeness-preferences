import unittest
from typing import cast
from unittest.mock import MagicMock

from llm_runtime.transformers import TransformersRuntime
from llm_runtime.types import (
    ChatMessage,
    FinishReason,
    GenerationResult as RuntimeGenerationResult,
    GenerationSettings,
    MessageRole,
)
from quadratic_voting.experiment.gemma import (
    GEMMA_MODEL_CARD_PROFILE,
    GemmaVoterGenerator,
    GemmaVoterGeneratorError,
)
from quadratic_voting.experiment.types import SamplingProfile, StopReason


class GemmaVoterGeneratorTests(unittest.TestCase):
    messages = (ChatMessage(MessageRole.USER, "Choose carefully."),)

    def _runtime(
        self,
        completion_ids: list[int],
        text: str = "model response",
        finish_reason: FinishReason = FinishReason.EOS,
    ) -> tuple[TransformersRuntime, MagicMock]:
        runtime = MagicMock(spec=TransformersRuntime)
        runtime.generate.return_value = RuntimeGenerationResult(
            raw_text=text,
            prompt_token_count=4,
            completion_token_count=len(completion_ids),
            completion_token_ids=tuple(completion_ids),
            finish_reason=finish_reason,
            duration_ms=125,
            diagnostics={"device_index": 0},
        )
        return cast(TransformersRuntime, runtime), runtime

    def test_maps_settings_counts_tokens_and_records_safe_diagnostics(self) -> None:
        runtime, runtime_mock = self._runtime([101, 102, 103])
        profile = SamplingProfile(
            temperature=0.8,
            top_p=0.9,
            top_k=32,
            max_new_tokens=8,
        )

        result = GemmaVoterGenerator(runtime).generate(self.messages, profile, seed=987)

        runtime_mock.generate.assert_called_once_with(
            self.messages,
            GenerationSettings(
                max_new_tokens=8,
                temperature=0.8,
                top_p=0.9,
                top_k=32,
                seed=987,
            ),
        )
        self.assertEqual(result.text, "model response")
        self.assertEqual(result.prompt_token_count, 4)
        self.assertEqual(result.completion_token_count, 3)
        self.assertEqual(result.completion_token_ids, (101, 102, 103))
        self.assertIs(result.stop_reason, StopReason.EOS)
        self.assertEqual(result.duration_ms, 125)
        self.assertEqual(
            result.diagnostics,
            {"device_index": "0"},
        )
        self.assertEqual(
            set(result.diagnostics),
            {"device_index"},
        )

    def test_stop_reason_inference_covers_token_limit_and_eos(self) -> None:
        for finish_reason, expected in (
            (FinishReason.LENGTH, StopReason.MAX_TOKENS),
            (FinishReason.EOS, StopReason.EOS),
            (FinishReason.STOP_SEQUENCE, StopReason.STOP_SEQUENCE),
        ):
            with self.subTest(finish_reason=finish_reason):
                runtime, _ = self._runtime([101, 102], finish_reason=finish_reason)
                result = GemmaVoterGenerator(runtime).generate(
                    self.messages,
                    SamplingProfile(
                        temperature=1.0,
                        top_p=0.95,
                        top_k=64,
                        max_new_tokens=3,
                    ),
                    seed=1,
                )
                self.assertIs(result.stop_reason, expected)

    def test_filter_and_provider_other_are_preserved_or_rejected_never_eos(
        self,
    ) -> None:
        for finish_reason in (
            FinishReason.CONTENT_FILTER,
            FinishReason.PROVIDER_OTHER,
        ):
            with self.subTest(finish_reason=finish_reason):
                runtime, _ = self._runtime([101], finish_reason=finish_reason)
                try:
                    expected = StopReason(finish_reason.value)
                except ValueError:
                    with self.assertRaisesRegex(
                        GemmaVoterGeneratorError,
                        "not representable.*must not persist.*false EOS",
                    ) as rejected:
                        GemmaVoterGenerator(runtime).generate(
                            self.messages,
                            GEMMA_MODEL_CARD_PROFILE,
                            seed=1,
                        )
                    self.assertIs(rejected.exception.finish_reason, finish_reason)
                    self.assertEqual(
                        rejected.exception.diagnostics,
                        {"device_index": 0},
                    )
                else:
                    result = GemmaVoterGenerator(runtime).generate(
                        self.messages,
                        GEMMA_MODEL_CARD_PROFILE,
                        seed=1,
                    )
                    self.assertIs(result.stop_reason, expected)
                    self.assertIsNot(result.stop_reason, StopReason.EOS)

    def test_model_card_profile_is_the_frozen_recommended_starting_point(self) -> None:
        self.assertEqual(
            GEMMA_MODEL_CARD_PROFILE,
            SamplingProfile(
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                max_new_tokens=8192,
            ),
        )


if __name__ == "__main__":
    unittest.main()
