import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import cast
from unittest.mock import patch

import httpx
import pandas as pd

from bail import config
from bail.prompts.rudeness_augmentation import extract_augmented_prompt
from bail.src.augment_bailbench import _call_one, main
from llm_runtime import (
    ChatMessage,
    GenerationError,
    GenerationFailureKind,
    GenerationSettings,
    MessageRole,
    ModelId,
    OpenRouterRoute,
    ProviderId,
    resolve_route,
)
from llm_runtime.openrouter import OpenRouterCredentials, create_openrouter_generator


class FakeGenerator:
    model_id = ModelId.DOLPHIN_MISTRAL_24B_VENICE
    provider_id = ProviderId.OPENROUTER
    quantization_id = None

    def __init__(self, outcomes: Sequence[str | Exception] | None = None) -> None:
        self.messages: tuple[ChatMessage, ...] = ()
        self.settings: GenerationSettings | None = None
        self.outcomes = list(
            outcomes
            if outcomes is not None
            else ["<augmented>You fool, decide this case.</augmented>"]
        )
        self.attempts = 0

    def generate(
        self,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
    ) -> str:
        self.messages = tuple(messages)
        self.settings = settings
        outcome = self.outcomes[self.attempts]
        self.attempts += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _write_source(path: Path) -> None:
    pd.DataFrame(
        [{"content": "Decide this case.", "subcategory": "x", "category": "y"}]
    ).to_csv(path, index=False)


class BailGeneratorTests(unittest.TestCase):
    def test_typed_prompt_flows_through_generator_and_tag_parser(self) -> None:
        generator = FakeGenerator()

        raw = _call_one(generator, 1, "Decide this case.")

        self.assertEqual(
            [message.role for message in generator.messages],
            [MessageRole.SYSTEM, MessageRole.USER],
        )
        self.assertIn("Assigned rudeness type: 1", generator.messages[1].content)
        self.assertEqual(extract_augmented_prompt(raw), "You fool, decide this case.")

    def test_retryability_controls_attempt_count_and_retry_after(self) -> None:
        transient = GenerationError(
            "temporary transport failure",
            kind=GenerationFailureKind.TRANSIENT_TRANSPORT,
            retry_after_seconds=0.25,
        )
        generator = FakeGenerator(
            [transient, "<augmented>Recovered response.</augmented>"]
        )
        with patch("bail.src.augment_bailbench.time.sleep") as sleep:
            raw = _call_one(generator, 1, "Source")
        self.assertEqual(generator.attempts, 2)
        sleep.assert_called_once_with(0.25)
        self.assertEqual(extract_augmented_prompt(raw), "Recovered response.")

        exhausted_error = GenerationError(
            "rate limit persists",
            kind=GenerationFailureKind.RETRYABLE_PROVIDER_STATUS,
            retry_after_seconds=1_000_000.0,
        )
        generator = FakeGenerator([exhausted_error] * (config.API_MAX_RETRIES + 1))
        with patch("bail.src.augment_bailbench.time.sleep") as sleep:
            raw = _call_one(generator, 1, "Source")
        self.assertEqual(generator.attempts, config.API_MAX_RETRIES + 1)
        self.assertEqual(sleep.call_count, config.API_MAX_RETRIES)
        self.assertTrue(all(call.args == (30.0,) for call in sleep.call_args_list))
        self.assertTrue(raw.startswith("API_ERROR:"))

        permanent_errors = [
            GenerationError(
                "authentication rejected",
                kind=GenerationFailureKind.PERMANENT_PROVIDER_STATUS,
            ),
            GenerationError(
                "malformed success payload",
                kind=GenerationFailureKind.MALFORMED_PROVIDER_RESPONSE,
            ),
            RuntimeError("programmer bug"),
        ]
        for error in permanent_errors:
            with self.subTest(error=type(error).__name__):
                generator = FakeGenerator([error])
                with patch("bail.src.augment_bailbench.time.sleep") as sleep:
                    raw = _call_one(generator, 1, "Source")
                self.assertEqual(generator.attempts, 1)
                self.assertTrue(raw.startswith("API_ERROR:"))
                sleep.assert_not_called()

    def test_complete_resume_avoids_credentials_client_and_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            output = root / "output.parquet"
            _write_source(source)
            pd.DataFrame(
                [{"bailbench_id": 0, "augmented_prompt": "already complete"}]
            ).to_parquet(output, index=False)

            with (
                patch.multiple(
                    config,
                    BAILBENCH_SOURCE=str(source),
                    AUGMENTED_PARQUET=str(output),
                    BAILBENCH_ID_COL="",
                    BAILBENCH_PROMPT_COL="content",
                    AUGMENT_USE_MOCK=False,
                ),
                patch(
                    "bail.src.augment_bailbench.openrouter_credentials_from_env"
                ) as credentials,
                patch(
                    "bail.src.augment_bailbench.create_openrouter_generator"
                ) as create_generator,
            ):
                main()

            credentials.assert_not_called()
            create_generator.assert_not_called()

    def test_composition_closes_owned_but_not_injected_client(self) -> None:
        route = cast(
            OpenRouterRoute,
            resolve_route(
                ModelId.DOLPHIN_MISTRAL_24B_VENICE,
                ProviderId.OPENROUTER,
                None,
            ),
        )

        def success(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "<augmented>Done.</augmented>"}}
                    ]
                },
                request=request,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            output = root / "owned.parquet"
            _write_source(source)
            owned_client = httpx.Client(transport=httpx.MockTransport(success))
            with (
                patch.multiple(
                    config,
                    BAILBENCH_SOURCE=str(source),
                    AUGMENTED_PARQUET=str(output),
                    BAILBENCH_ID_COL="",
                    BAILBENCH_PROMPT_COL="content",
                    AUGMENT_USE_MOCK=False,
                ),
                patch.dict("os.environ", {"OPENROUTER_API_KEY": "secret"}, clear=True),
                patch("llm_runtime.openrouter.httpx.Client", return_value=owned_client),
            ):
                main()
            self.assertTrue(owned_client.is_closed)
            self.assertEqual(
                pd.read_parquet(output).loc[0, "augmented_prompt"], "Done."
            )

            injected_output = root / "injected.parquet"
            external_client = httpx.Client(transport=httpx.MockTransport(success))
            injected = create_openrouter_generator(
                route,
                credentials=OpenRouterCredentials("secret"),
                http_client=external_client,
            )
            with patch.multiple(
                config,
                BAILBENCH_SOURCE=str(source),
                AUGMENTED_PARQUET=str(injected_output),
                BAILBENCH_ID_COL="",
                BAILBENCH_PROMPT_COL="content",
                AUGMENT_USE_MOCK=False,
            ):
                main(injected)
            self.assertFalse(external_client.is_closed)
            external_client.close()


if __name__ == "__main__":
    unittest.main()
