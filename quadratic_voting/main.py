"""Download registered local artifacts or run an injected-generator chat."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

from llm_runtime import (
    LocalTransformersRoute,
    ModelId,
    ModelRoute,
    ModelRouteError,
    OpenRouterRoute,
    ProviderId,
    QuantizationId,
    TextGenerator,
    resolve_route,
)
from llm_runtime.openrouter import (
    OpenRouterGenerator,
    OpenRouterRuntimeError,
    create_openrouter_generator,
    openrouter_credentials_from_env,
)
from llm_runtime.transformers import (
    Device,
    TransformersRuntimeError,
    create_transformers_runtime,
    download_transformers_artifact,
)
from quadratic_voting.conversation import run_conversation


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not an integer; expected a positive integer"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not positive; set this option to an integer above zero"
        )
    return parsed


def quantization_id(value: str) -> QuantizationId | None:
    if value == "none":
        return None
    try:
        return QuantizationId(value)
    except ValueError as error:
        choices = ", ".join([*(item.value for item in QuantizationId), "none"])
        raise argparse.ArgumentTypeError(
            f"unknown quantization {value!r}; choose one reviewed value: {choices}"
        ) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download or chat through the closed llm_runtime registry."
    )
    parser.add_argument(
        "--model",
        type=ModelId,
        choices=list(ModelId),
        default=ModelId.GEMMA_4_E2B_IT,
        help="registered model identity (default: %(default)s)",
    )
    parser.add_argument(
        "--provider",
        type=ProviderId,
        choices=list(ProviderId),
        default=ProviderId.LOCAL,
        help="registered provider (default: %(default)s)",
    )
    parser.add_argument(
        "--quantization",
        type=quantization_id,
        default=QuantizationId.BF16,
        help="reviewed quantization or 'none' (default: %(default)s)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(HF_HUB_CACHE),
        help="Hugging Face Hub cache directory (default: %(default)s)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "download",
        help="download the exact artifact for a registered local route",
    )
    chat_parser = subparsers.add_parser(
        "chat", help="start an interactive conversation"
    )
    chat_parser.add_argument(
        "--device",
        type=Device,
        choices=list(Device),
        default=Device.AUTO,
        help="local inference device (default: %(default)s)",
    )
    chat_parser.add_argument(
        "--max-new-tokens",
        type=positive_integer,
        default=256,
        help="maximum tokens generated per response (default: %(default)s)",
    )
    return parser


def _create_generator(
    route: ModelRoute,
    *,
    cache_dir: Path,
    device: Device,
) -> TextGenerator:
    if isinstance(route, LocalTransformersRoute):
        return create_transformers_runtime(route, cache_dir=cache_dir, device=device)
    if isinstance(route, OpenRouterRoute):
        return create_openrouter_generator(
            route,
            credentials=openrouter_credentials_from_env(),
        )
    raise AssertionError(f"unhandled closed route type: {type(route).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        route = resolve_route(args.model, args.provider, args.quantization)
        if args.command == "download":
            if not isinstance(route, LocalTransformersRoute):
                raise ModelRouteError(
                    "Artifact download failed at quadratic_voting.main.main because "
                    "the selected OpenRouter route has no reproducibly pinned local "
                    "weights. No download ran. Select the local Gemma route with an "
                    "explicit registered quantization, then retry."
                )
            model_path = download_transformers_artifact(route, args.cache_dir)
            print(f"Downloaded pinned model artifact to: {model_path}")
        else:
            generator = _create_generator(
                route,
                cache_dir=args.cache_dir,
                device=args.device,
            )
            if isinstance(generator, OpenRouterGenerator):
                with generator:
                    run_conversation(generator, args.max_new_tokens)
            else:
                run_conversation(generator, args.max_new_tokens)
    except (ModelRouteError, OpenRouterRuntimeError, TransformersRuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
