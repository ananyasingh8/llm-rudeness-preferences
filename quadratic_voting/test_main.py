import unittest
from collections.abc import Sequence
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from llm_runtime import (
    ChatMessage,
    GenerationSettings,
    MessageRole,
    ModelId,
    ProviderId,
    QuantizationId,
)
from quadratic_voting.conversation import run_conversation
from quadratic_voting.main import build_parser, main


class FakeGenerator:
    model_id = ModelId.GEMMA_4_E2B_IT
    provider_id = ProviderId.LOCAL
    quantization_id = QuantizationId.BF16

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[ChatMessage, ...], GenerationSettings]] = []

    def generate(
        self,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
    ) -> str:
        self.calls.append((tuple(messages), settings))
        return "hello"


class ConversationTests(unittest.TestCase):
    def test_conversation_uses_injected_text_generator(self) -> None:
        generator = FakeGenerator()
        prompts = iter(["Hi", "/exit"])
        output: list[str] = []

        run_conversation(
            generator,
            32,
            read_input=lambda _: next(prompts),
            write_output=output.append,
        )

        messages, settings = generator.calls[0]
        self.assertEqual(messages, (ChatMessage(role=MessageRole.USER, content="Hi"),))
        self.assertEqual(settings, GenerationSettings(max_new_tokens=32))
        self.assertEqual(output, ["Model: hello"])


class CliBoundaryTests(unittest.TestCase):
    def test_enum_parsing_and_local_download_dispatch(self) -> None:
        parsed = build_parser().parse_args(
            [
                "--model",
                "gemma-4-e2b-it",
                "--provider",
                "local",
                "--quantization",
                "bf16",
                "download",
            ]
        )
        self.assertIs(parsed.model, ModelId.GEMMA_4_E2B_IT)
        self.assertIs(parsed.provider, ProviderId.LOCAL)
        self.assertIs(parsed.quantization, QuantizationId.BF16)

        with patch(
            "quadratic_voting.main.download_transformers_artifact",
            return_value=Path("/cache/snapshot"),
        ) as download:
            result = main(
                [
                    "--cache-dir",
                    "/cache",
                    "--quantization",
                    "bf16",
                    "download",
                ]
            )

        self.assertEqual(result, 0)
        route, cache_dir = download.call_args.args
        self.assertIs(route.model_id, ModelId.GEMMA_4_E2B_IT)
        self.assertIs(route.quantization_id, QuantizationId.BF16)
        self.assertEqual(cache_dir, Path("/cache"))

    def test_remote_download_rejected_and_chat_closes_owned_client(self) -> None:
        remote_args = [
            "--model",
            "dolphin-mistral-24b-venice",
            "--provider",
            "openrouter",
            "--quantization",
            "none",
        ]
        stderr = StringIO()
        with patch("sys.stderr", stderr):
            result = main([*remote_args, "download"])
        self.assertEqual(result, 1)
        self.assertIn("has no reproducibly pinned local weights", stderr.getvalue())

        client = MagicMock(spec=httpx.Client)
        with (
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "secret"}, clear=True),
            patch("llm_runtime.openrouter.httpx.Client", return_value=client),
            patch("quadratic_voting.main.run_conversation") as conversation,
        ):
            result = main([*remote_args, "chat"])

        self.assertEqual(result, 0)
        conversation.assert_called_once()
        client.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
