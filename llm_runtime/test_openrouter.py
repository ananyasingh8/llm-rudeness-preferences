import json
import unittest
from typing import cast
from unittest.mock import MagicMock, patch, sentinel

import httpx

from llm_runtime import (
    ChatMessage,
    GenerationFailureKind,
    GenerationSettings,
    MessageRole,
    ModelId,
    OpenRouterRoute,
    ProviderId,
    resolve_route,
)
from llm_runtime.openrouter import (
    OpenRouterCredentials,
    OpenRouterGenerator,
    OpenRouterRuntimeError,
    create_openrouter_generator,
    openrouter_credentials_from_env,
)


class OpenRouterTests(unittest.TestCase):
    def test_typed_request_parsing_and_credential_boundaries(self) -> None:
        secret = "sk-or-super-secret"

        def handle(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                request.url,
                httpx.URL("https://openrouter.ai/api/v1/chat/completions"),
            )
            self.assertEqual(request.headers["Authorization"], f"Bearer {secret}")
            self.assertEqual(
                json.loads(request.content),
                {
                    "model": (
                        "cognitivecomputations/dolphin-mistral-24b-venice-edition"
                    ),
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 20,
                    "temperature": 0.25,
                },
            )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Hi there"}}]},
            )

        route = cast(
            OpenRouterRoute,
            resolve_route(
                ModelId.DOLPHIN_MISTRAL_24B_VENICE,
                ProviderId.OPENROUTER,
                None,
            ),
        )
        credentials = OpenRouterCredentials(secret)
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            generator = create_openrouter_generator(
                route, credentials=credentials, http_client=client
            )
            response = generator.generate(
                [ChatMessage(MessageRole.USER, "Hello")],
                GenerationSettings(max_new_tokens=20, temperature=0.25),
            )

        self.assertEqual(response, "Hi there")
        self.assertNotIn(secret, repr(credentials))

        def reject(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, request=request)

        with (
            httpx.Client(transport=httpx.MockTransport(reject)) as client,
            self.assertRaises(OpenRouterRuntimeError) as rejected,
        ):
            create_openrouter_generator(
                route, credentials=credentials, http_client=client
            ).generate(
                [ChatMessage(MessageRole.USER, "Hello")],
                GenerationSettings(max_new_tokens=20),
            )
        self.assertNotIn(secret, str(rejected.exception))
        self.assertIsNone(rejected.exception.__cause__)
        self.assertEqual(
            rejected.exception.kind,
            GenerationFailureKind.PERMANENT_PROVIDER_STATUS,
        )
        self.assertFalse(rejected.exception.retryable)

        forged = OpenRouterRoute(
            model_id=ModelId.DOLPHIN_MISTRAL_24B_VENICE,
            model_slug="unreviewed/arbitrary",
        )
        with self.assertRaisesRegex(
            ValueError, "does not exactly match.*no artifact download.*arbitrary"
        ):
            create_openrouter_generator(forged, credentials=credentials)

        with self.assertRaisesRegex(
            OpenRouterRuntimeError, "OPENROUTER_API_KEY.*before HTTP"
        ) as missing:
            openrouter_credentials_from_env({})
        self.assertNotIn(secret, str(missing.exception))

    def test_malformed_transport_and_client_lifecycle(self) -> None:
        route = cast(
            OpenRouterRoute,
            resolve_route(
                ModelId.DOLPHIN_MISTRAL_24B_VENICE,
                ProviderId.OPENROUTER,
                None,
            ),
        )
        credentials = OpenRouterCredentials("secret")
        messages = [ChatMessage(MessageRole.USER, "Hello")]
        settings = GenerationSettings(max_new_tokens=5)

        def transport_failure(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline", request=request)

        with (
            httpx.Client(transport=httpx.MockTransport(transport_failure)) as client,
            self.assertRaises(OpenRouterRuntimeError) as transport_error,
        ):
            create_openrouter_generator(
                route, credentials=credentials, http_client=client
            ).generate(messages, settings)
        self.assertEqual(
            transport_error.exception.kind,
            GenerationFailureKind.TRANSIENT_TRANSPORT,
        )
        self.assertTrue(transport_error.exception.retryable)

        def malformed(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"not-json", request=request)

        with (
            httpx.Client(transport=httpx.MockTransport(malformed)) as client,
            self.assertRaises(OpenRouterRuntimeError) as malformed_error,
        ):
            create_openrouter_generator(
                route, credentials=credentials, http_client=client
            ).generate(messages, settings)
        self.assertEqual(
            malformed_error.exception.kind,
            GenerationFailureKind.MALFORMED_PROVIDER_RESPONSE,
        )
        self.assertFalse(malformed_error.exception.retryable)

        def rate_limited(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                headers={"Retry-After": "0.25"},
                request=request,
            )

        with (
            httpx.Client(transport=httpx.MockTransport(rate_limited)) as client,
            self.assertRaises(OpenRouterRuntimeError) as rate_error,
        ):
            create_openrouter_generator(
                route, credentials=credentials, http_client=client
            ).generate(messages, settings)
        self.assertEqual(
            rate_error.exception.kind,
            GenerationFailureKind.RETRYABLE_PROVIDER_STATUS,
        )
        self.assertEqual(rate_error.exception.retry_after_seconds, 0.25)
        self.assertTrue(rate_error.exception.retryable)

        for header, expected_delay in (
            ("1000000", 30.0),
            ("-1", None),
            ("inf", None),
            ("not-a-number", None),
        ):
            with self.subTest(retry_after=header):

                def bounded_retry_after(request: httpx.Request) -> httpx.Response:
                    return httpx.Response(
                        429,
                        headers={"Retry-After": header},
                        request=request,
                    )

                with (
                    httpx.Client(
                        transport=httpx.MockTransport(bounded_retry_after)
                    ) as client,
                    self.assertRaises(OpenRouterRuntimeError) as bounded_error,
                ):
                    create_openrouter_generator(
                        route, credentials=credentials, http_client=client
                    ).generate(messages, settings)
                self.assertEqual(
                    bounded_error.exception.retry_after_seconds,
                    expected_delay,
                )

        def unavailable(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, request=request)

        with (
            httpx.Client(transport=httpx.MockTransport(unavailable)) as client,
            self.assertRaises(OpenRouterRuntimeError) as unavailable_error,
        ):
            create_openrouter_generator(
                route, credentials=credentials, http_client=client
            ).generate(messages, settings)
        self.assertEqual(
            unavailable_error.exception.kind,
            GenerationFailureKind.RETRYABLE_PROVIDER_STATUS,
        )
        self.assertTrue(unavailable_error.exception.retryable)

        owned_client = MagicMock(spec=httpx.Client)
        with patch(
            "llm_runtime.openrouter.httpx.Client", return_value=owned_client
        ) as client_constructor:
            with create_openrouter_generator(route, credentials=credentials) as owned:
                self.assertIsInstance(owned, OpenRouterGenerator)
            owned.close()
        client_constructor.assert_called_once_with(timeout=60.0)
        owned_client.close.assert_called_once_with()

        external_client = MagicMock(spec=httpx.Client)
        external_client.timeout = sentinel.caller_timeout
        with patch("llm_runtime.openrouter.httpx.Client") as client_constructor:
            with create_openrouter_generator(
                route, credentials=credentials, http_client=external_client
            ):
                pass
        client_constructor.assert_not_called()
        self.assertIs(external_client.timeout, sentinel.caller_timeout)
        external_client.close.assert_not_called()


if __name__ == "__main__":
    unittest.main()
