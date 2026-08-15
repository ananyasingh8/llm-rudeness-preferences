import unittest
from typing import cast

from llm_runtime import (
    Capability,
    LocalTransformersRoute,
    ModelId,
    ModelRouteError,
    OpenRouterRoute,
    ProviderId,
    QuantizationId,
    RouteAvailability,
    registered_routes,
    resolve_route,
)


class RegistryTests(unittest.TestCase):
    def test_reviewed_routes_and_invalid_boundaries(self) -> None:
        bf16 = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_E2B_IT,
                ProviderId.LOCAL,
                QuantizationId.BF16,
                required={Capability.LOCAL_ACTIVATIONS},
            ),
        )
        self.assertIsInstance(bf16, LocalTransformersRoute)
        self.assertEqual(
            bf16.artifact.revision, "6befbaca7398925921802abd1f277b495b78b738"
        )

        w4a16 = next(
            route
            for route in registered_routes()
            if isinstance(route, LocalTransformersRoute)
            and route.quantization_id is QuantizationId.W4A16_COMPRESSED_TENSORS
        )
        self.assertEqual(w4a16.availability, RouteAvailability.UNAVAILABLE)
        self.assertEqual(w4a16.capabilities, frozenset())
        self.assertEqual(
            w4a16.artifact.revision, "971342c08f607aa7779983f6b5289778b5d271a7"
        )
        with self.assertRaisesRegex(
            ModelRouteError,
            "explicitly unavailable.*weight loading, text generation.*activation "
            "access.*no fallback",
        ):
            resolve_route(
                ModelId.GEMMA_4_E2B_IT,
                ProviderId.LOCAL,
                QuantizationId.W4A16_COMPRESSED_TENSORS,
            )

        with self.assertRaisesRegex(
            ModelRouteError, "not registered.*Supported routes.*Select"
        ):
            resolve_route(
                ModelId.DOLPHIN_MISTRAL_24B_VENICE,
                ProviderId.OPENROUTER,
                QuantizationId.BF16,
            )

        remote = resolve_route(
            ModelId.DOLPHIN_MISTRAL_24B_VENICE,
            ProviderId.OPENROUTER,
            None,
        )
        self.assertIsInstance(remote, OpenRouterRoute)
        with self.assertRaisesRegex(ModelRouteError, "local-activations.*local route"):
            resolve_route(
                ModelId.DOLPHIN_MISTRAL_24B_VENICE,
                ProviderId.OPENROUTER,
                None,
                required={Capability.LOCAL_ACTIVATIONS},
            )

    def test_gemma_4_31b_w4a16_probe_route(self) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_4_31B_IT,
                ProviderId.LOCAL,
                QuantizationId.W4A16_COMPRESSED_TENSORS,
                required={Capability.LOCAL_ACTIVATIONS},
            ),
        )
        self.assertIsInstance(route, LocalTransformersRoute)
        self.assertEqual(
            route.artifact.repository, "google/gemma-4-31B-it-qat-w4a16-ct"
        )
        self.assertEqual(
            route.artifact.revision, "52f3f65bc7a02d555763bc923bd1d9094898219d"
        )
        self.assertEqual(route.availability, RouteAvailability.ENABLED)

        with self.assertRaisesRegex(
            ModelRouteError, "not registered.*Supported routes.*Select"
        ):
            resolve_route(
                ModelId.GEMMA_4_31B_IT, ProviderId.LOCAL, QuantizationId.BF16
            )

    def test_gemma_2_2b_probe_route_and_invalid_boundaries(self) -> None:
        route = cast(
            LocalTransformersRoute,
            resolve_route(
                ModelId.GEMMA_2_2B_IT,
                ProviderId.LOCAL,
                QuantizationId.BF16,
                required={Capability.LOCAL_ACTIVATIONS},
            ),
        )
        self.assertIsInstance(route, LocalTransformersRoute)
        self.assertEqual(route.artifact.repository, "google/gemma-2-2b-it")
        self.assertEqual(
            route.artifact.revision, "299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8"
        )
        self.assertEqual(route.context_window, 8_192)
        self.assertEqual(route.availability, RouteAvailability.ENABLED)

        with self.assertRaisesRegex(
            ModelRouteError, "not registered.*Supported routes.*Select"
        ):
            resolve_route(
                ModelId.GEMMA_2_2B_IT,
                ProviderId.LOCAL,
                QuantizationId.W4A16_COMPRESSED_TENSORS,
            )
        with self.assertRaisesRegex(
            ModelRouteError, "not registered.*Supported routes.*Select"
        ):
            resolve_route(ModelId.GEMMA_2_2B_IT, ProviderId.OPENROUTER, None)


if __name__ == "__main__":
    unittest.main()
