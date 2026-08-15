import json
import unittest
from collections.abc import Callable
from typing import cast
from unittest.mock import MagicMock

import httpx
import torch

from llm_runtime import (
    ChatMessage,
    GenerationSettings,
    LocalTransformersRoute,
    MessageRole,
    ModelId,
    OpenRouterRoute,
    ProviderId,
    QuantizationId,
    RuntimeId,
    resolve_route,
)
from llm_runtime.openrouter import OpenRouterCredentials, OpenRouterGenerator
from llm_runtime.transformers import TransformersRuntime
from llm_runtime.types import UnsupportedSettingError


class ProviderSettingsMatrixTests(unittest.TestCase):
    messages = (ChatMessage(MessageRole.USER, "Hello"),)

    def _local_runtime(self) -> tuple[TransformersRuntime, MagicMock]:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_E2B_IT,
                ProviderId.LOCAL,
                QuantizationId.BF16,
            ),
        )
        model = MagicMock()
        model.device = torch.device("cpu")

        def generate(**_: object) -> torch.Tensor:
            model.observed_initial_seed = torch.initial_seed()
            return torch.tensor([[1, 2, 3]])

        model.generate.side_effect = generate
        tokenizer = MagicMock()
        batch = MagicMock()
        batch.to.return_value = {"input_ids": torch.tensor([[1, 2]])}
        tokenizer.apply_chat_template.return_value = batch
        tokenizer.decode.return_value = "local response"
        return TransformersRuntime(route, model, tokenizer), model

    def _openrouter_generator(
        self, handler: Callable[[httpx.Request], httpx.Response]
    ) -> tuple[OpenRouterGenerator, httpx.Client]:
        route = cast(
            OpenRouterRoute,
            resolve_route(
                ModelId.DOLPHIN_MISTRAL_24B_VENICE,
                ProviderId.OPENROUTER,
                None,
            ),
        )
        client = httpx.Client(transport=httpx.MockTransport(handler))
        return OpenRouterGenerator(
            route, OpenRouterCredentials("secret"), client
        ), client

    def test_local_provider_honors_each_optional_setting(self) -> None:
        cases = (
            ("top_p", {"top_p": 0.9}, 0.9),
            ("top_k", {"top_k": 40}, 40),
            ("seed", {"seed": 123}, 123),
        )

        for setting_name, override, expected in cases:
            with self.subTest(setting=setting_name):
                runtime, model = self._local_runtime()
                runtime.generate(
                    self.messages,
                    GenerationSettings(
                        max_new_tokens=16,
                        temperature=0.7,
                        **override,  # type: ignore[arg-type]
                    ),
                )
                kwargs = model.generate.call_args.kwargs
                if setting_name == "seed":
                    self.assertNotIn("generator", kwargs)
                    self.assertEqual(model.observed_initial_seed, expected)
                else:
                    self.assertEqual(kwargs[setting_name], expected)

    def test_openrouter_honors_or_rejects_each_optional_setting(self) -> None:
        requests: list[dict[str, object]] = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(cast(dict[str, object], json.loads(request.content)))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "remote response"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                },
            )

        generator, client = self._openrouter_generator(handle)
        with client:
            response = generator.generate(
                self.messages,
                GenerationSettings(max_new_tokens=16, temperature=0.7, top_p=0.9),
            )
            self.assertEqual(response.raw_text, "remote response")
            self.assertEqual(requests[-1]["top_p"], 0.9)

            for setting_name, settings in (
                ("top_k", GenerationSettings(max_new_tokens=16, top_k=40)),
                ("seed", GenerationSettings(max_new_tokens=16, seed=123)),
            ):
                with self.subTest(setting=setting_name):
                    request_count = len(requests)
                    with self.assertRaises(UnsupportedSettingError) as rejected:
                        generator.generate(self.messages, settings)
                    self.assertEqual(rejected.exception.setting, setting_name)
                    self.assertEqual(
                        rejected.exception.provider_id, ProviderId.OPENROUTER
                    )
                    self.assertEqual(
                        rejected.exception.runtime_id,
                        RuntimeId.OPENAI_COMPATIBLE_HTTP,
                    )
                    self.assertEqual(len(requests), request_count)

    def test_unset_settings_preserve_provider_requests(self) -> None:
        runtime, model = self._local_runtime()
        runtime.generate(self.messages, GenerationSettings(max_new_tokens=16))
        self.assertEqual(
            model.generate.call_args.kwargs,
            {
                "input_ids": model.generate.call_args.kwargs["input_ids"],
                "max_new_tokens": 16,
                "do_sample": False,
            },
        )

        requests: list[dict[str, object]] = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(cast(dict[str, object], json.loads(request.content)))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "remote response"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                },
            )

        generator, client = self._openrouter_generator(handle)
        with client:
            generator.generate(self.messages, GenerationSettings(max_new_tokens=16))
        self.assertEqual(
            requests,
            [
                {
                    "model": "cognitivecomputations/dolphin-mistral-24b-venice-edition",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 16,
                    "temperature": 0.0,
                }
            ],
        )

    def test_temperature_zero_setting_edges_are_explicit(self) -> None:
        runtime, model = self._local_runtime()
        runtime.generate(
            self.messages,
            GenerationSettings(max_new_tokens=16, top_p=0.9, top_k=40),
        )
        kwargs = model.generate.call_args.kwargs
        self.assertNotIn("top_p", kwargs)
        self.assertNotIn("top_k", kwargs)
        self.assertIs(kwargs["do_sample"], False)

        requests: list[dict[str, object]] = []

        def handle(request: httpx.Request) -> httpx.Response:
            requests.append(cast(dict[str, object], json.loads(request.content)))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "remote response"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                },
            )

        generator, client = self._openrouter_generator(handle)
        with client:
            generator.generate(
                self.messages,
                GenerationSettings(max_new_tokens=16, top_p=0.9),
            )
        self.assertEqual(requests[0]["temperature"], 0.0)
        self.assertEqual(requests[0]["top_p"], 0.9)


if __name__ == "__main__":
    unittest.main()
