"""Download and run Google's official Gemma 4 E2B IT QAT checkpoint."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

import torch
from httpx import TransportError
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


@dataclass(frozen=True)
class ModelArtifact:
    """An immutable Hugging Face model artifact."""

    repository: str
    revision: str


class Device(StrEnum):
    """Supported inference devices."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class GenerativeModel(Protocol):
    """Model operations required by the interactive text loop."""

    device: torch.device

    def generate(self, **kwargs: object) -> torch.Tensor:
        """Generate output token IDs."""


GEMMA_IT = ModelArtifact(
    repository="google/gemma-4-E2B-it-qat-q4_0-unquantized",
    revision="6befbaca7398925921802abd1f277b495b78b738",
)
MAX_CONTEXT_TOKENS = 131_072
MIN_CUDA_FREE_BYTES = 12_000_000_000


class RunnerError(RuntimeError):
    """A user-actionable failure in the download or inference workflow."""


def download_model(cache_dir: Path) -> Path:
    """Download the complete pinned Transformers checkpoint into the HF cache."""
    try:
        snapshot = snapshot_download(
            repo_id=GEMMA_IT.repository,
            revision=GEMMA_IT.revision,
            cache_dir=cache_dir,
        )
    except (
        HfHubHTTPError,
        LocalEntryNotFoundError,
        OSError,
        TransportError,
    ) as error:
        raise RunnerError(
            "Gemma download failed while fetching the pinned unquantized QAT "
            f"checkpoint from {GEMMA_IT.repository} at revision "
            f"{GEMMA_IT.revision}. The model is unavailable locally, usually "
            "because Hugging Face could not be reached or the cache is not "
            "writable. Check network access and cache permissions, then rerun "
            "`uv run python -m quadratic_voting.main download`. "
            f"Underlying error: {error}"
        ) from error
    return Path(snapshot)


def cached_model(cache_dir: Path) -> Path:
    """Resolve the complete pinned checkpoint without network access."""
    try:
        snapshot = snapshot_download(
            repo_id=GEMMA_IT.repository,
            revision=GEMMA_IT.revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except (LocalEntryNotFoundError, OSError) as error:
        raise RunnerError(
            "Gemma chat could not start because the pinned Transformers "
            f"checkpoint is missing from the Hugging Face cache at {cache_dir}. "
            "Run `uv run python -m quadratic_voting.main download` before "
            f"starting chat. Underlying error: {error}"
        ) from error
    return Path(snapshot)


def resolve_device(requested: Device) -> Device:
    """Resolve automatic device selection and validate explicit CUDA use."""
    if requested is Device.AUTO:
        return Device.AUTO
    if requested is Device.CUDA and not torch.cuda.is_available():
        raise RunnerError(
            "Gemma chat cannot use CUDA because PyTorch did not detect a CUDA "
            "device during device selection. Verify the NVIDIA driver and the "
            "installed PyTorch build, or rerun chat with `--device cpu`."
        )
    if requested is Device.CUDA:
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except RuntimeError as error:
            raise RunnerError(
                "Gemma chat could not inspect free CUDA memory during device "
                "validation. Verify the NVIDIA driver and PyTorch CUDA setup, "
                f"or use `--device auto`. Underlying error: {error}"
            ) from error
        if free_bytes < MIN_CUDA_FREE_BYTES:
            free_gb = free_bytes / 1_000_000_000
            required_gb = MIN_CUDA_FREE_BYTES / 1_000_000_000
            raise RunnerError(
                "Gemma chat cannot place the full BF16 checkpoint on CUDA "
                f"because {free_gb:.1f} GB is free and at least "
                f"{required_gb:.1f} GB is required for the explicit CUDA "
                "loading budget. Use `--device auto` for Accelerate GPU/CPU "
                "placement, close other GPU workloads, or use `--device cpu`."
            )
    return requested


def load_runtime(
    model_path: Path, device: Device
) -> tuple[GenerativeModel, PreTrainedTokenizerBase]:
    """Load the pinned local checkpoint through Transformers and PyTorch."""
    resolved_device = resolve_device(device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            padding_side="left",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map=resolved_device.value,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.eval()
    except (OSError, RuntimeError, ValueError) as error:
        raise RunnerError(
            "Gemma chat could not load the pinned Transformers checkpoint from "
            f"{model_path} on device `{resolved_device.value}`. The checkpoint "
            "may be incomplete, the selected device may lack memory, or the "
            "installed Transformers/PyTorch versions may be incompatible. "
            "Rerun `uv sync --locked`, verify the model cache, and retry with "
            f"`--device cpu` if appropriate. Underlying error: {error}"
        ) from error
    return cast(GenerativeModel, model), tokenizer


def run_chat(
    cache_dir: Path,
    device: Device,
    max_new_tokens: int,
    read_input: Callable[[str], str] = input,
    write_output: Callable[[str], None] = print,
) -> None:
    """Run an interactive text conversation through Transformers."""
    model_path = cached_model(cache_dir)
    model, tokenizer = load_runtime(model_path, device)
    messages: list[dict[str, str]] = []

    while True:
        try:
            prompt = read_input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            write_output("")
            return

        if prompt == "/exit":
            return
        if not prompt:
            continue

        messages.append({"role": "user", "content": prompt})
        try:
            inputs = cast(
                BatchEncoding,
                tokenizer.apply_chat_template(
                    [message.copy() for message in messages],
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ),
            ).to(model.device)
        except (IndexError, RuntimeError, ValueError) as error:
            raise RunnerError(
                "Gemma chat failed while tokenizing the conversation. Restart "
                "the session or shorten the prompt, then try again. "
                f"Underlying error: {error}"
            ) from error

        input_length = inputs["input_ids"].shape[-1]
        if input_length + max_new_tokens > MAX_CONTEXT_TOKENS:
            raise RunnerError(
                "Gemma chat cannot generate this response because the rendered "
                f"conversation uses {input_length} tokens and the requested "
                f"response allows {max_new_tokens}, exceeding the "
                f"{MAX_CONTEXT_TOKENS}-token model context. Restart the chat, "
                "shorten the conversation, or reduce `--max-new-tokens`."
            )

        try:
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            response = cast(
                str,
                tokenizer.decode(generated[0][input_length:], skip_special_tokens=True),
            ).strip()
        except (IndexError, RuntimeError, ValueError) as error:
            raise RunnerError(
                "Gemma chat failed during Transformers generation. The selected "
                "device may lack memory or the cached model inputs may be "
                f"incompatible. Reduce `--max-new-tokens` or use `--device cpu`. "
                f"Underlying error: {error}"
            ) from error

        messages.append({"role": "assistant", "content": response})
        write_output(f"Gemma: {response}")


def positive_integer(value: str) -> int:
    """Parse a positive integer for an argparse option."""
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not an integer; expected a positive integer"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download or chat with Google's official instruction-tuned Gemma 4 "
            "E2B unquantized QAT checkpoint through Transformers and PyTorch."
        )
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
        help="download the pinned official Transformers checkpoint",
        description="Download the pinned official Transformers checkpoint.",
    )

    chat_parser = subparsers.add_parser(
        "chat",
        help="start an interactive Transformers conversation",
        description="Start interactive text generation with Transformers.",
    )
    chat_parser.add_argument(
        "--device",
        type=Device,
        choices=list(Device),
        default=Device.AUTO,
        help="inference device: auto, cuda, or cpu (default: %(default)s)",
    )
    chat_parser.add_argument(
        "--max-new-tokens",
        type=positive_integer,
        default=256,
        help="maximum tokens generated per response (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "download":
            model_path = download_model(args.cache_dir)
            print(f"Downloaded pinned Gemma checkpoint to: {model_path}")
        else:
            run_chat(args.cache_dir, args.device, args.max_new_tokens)
    except RunnerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
