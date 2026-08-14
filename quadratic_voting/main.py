"""Download and run Google's official Gemma 4 E2B IT Q4_0 model."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError


@dataclass(frozen=True)
class ModelArtifact:
    """An immutable Hugging Face model artifact."""

    repository: str
    revision: str
    filename: str


GEMMA_IT = ModelArtifact(
    repository="google/gemma-4-E2B-it-qat-q4_0-gguf",
    revision="675cff42a74c774d6cb76f76d8eacb49b48c9b93",
    filename="gemma-4-E2B_q4_0-it.gguf",
)


class RunnerError(RuntimeError):
    """A user-actionable failure in the download or inference workflow."""


def download_model(cache_dir: Path) -> Path:
    """Download the pinned instruction-tuned model into the HF cache."""
    try:
        downloaded = hf_hub_download(
            repo_id=GEMMA_IT.repository,
            filename=GEMMA_IT.filename,
            revision=GEMMA_IT.revision,
            cache_dir=cache_dir,
        )
    except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as error:
        raise RunnerError(
            "Gemma download failed while fetching the pinned Q4_0 GGUF from "
            f"{GEMMA_IT.repository} at revision {GEMMA_IT.revision}. "
            "The model is unavailable locally, usually because Hugging Face "
            "could not be reached or the cache is not writable. Check network "
            "access and cache permissions, then rerun `download`. "
            f"Underlying error: {error}"
        ) from error
    return Path(downloaded)


def cached_model(cache_dir: Path) -> Path:
    """Resolve the pinned model without performing a network download."""
    try:
        cached = hf_hub_download(
            repo_id=GEMMA_IT.repository,
            filename=GEMMA_IT.filename,
            revision=GEMMA_IT.revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except (LocalEntryNotFoundError, OSError) as error:
        raise RunnerError(
            "Gemma chat could not start because the pinned model is missing "
            f"from the Hugging Face cache at {cache_dir}. Run "
            "`python -m quadratic_voting.main download` before starting chat. "
            f"Underlying error: {error}"
        ) from error
    return Path(cached)


def run_chat(cache_dir: Path, context_size: int, gpu_layers: int) -> None:
    """Run llama.cpp's interactive conversation mode using the GGUF template."""
    llama_cli = shutil.which("llama-cli")
    if llama_cli is None:
        raise RunnerError(
            "Gemma chat could not start because `llama-cli` was not found on "
            "PATH during runtime startup. Enter this project's Nix development "
            "shell with `nix develop`, then rerun the chat command."
        )

    model_path = cached_model(cache_dir)
    command = [
        llama_cli,
        "--model",
        str(model_path),
        "--conversation",
        "--ctx-size",
        str(context_size),
        "--gpu-layers",
        str(gpu_layers),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RunnerError(
            "Gemma chat failed after `llama-cli` started interactive inference. "
            f"The subprocess exited with status {error.returncode}; this means "
            "no further responses can be generated. Review llama.cpp's output "
            "for GPU or model-loading errors, reduce `--gpu-layers` if VRAM is "
            "insufficient, and rerun the chat command."
        ) from error


def positive_integer(value: str) -> int:
    """Parse a positive integer for an argparse option."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def nonnegative_integer(value: str) -> int:
    """Parse a nonnegative integer for an argparse option."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Download or chat with Google's official instruction-tuned Gemma 4 "
            "E2B QAT Q4_0 GGUF."
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
        help="download the pinned official instruction-tuned GGUF",
        description="Download the pinned official instruction-tuned GGUF.",
    )

    chat_parser = subparsers.add_parser(
        "chat",
        help="start llama.cpp's interactive conversation mode",
        description=(
            "Start interactive conversation mode. llama.cpp reads and applies "
            "the chat template embedded in the official GGUF."
        ),
    )
    chat_parser.add_argument(
        "--context-size",
        type=positive_integer,
        default=8192,
        help="context window passed to llama.cpp (default: %(default)s)",
    )
    chat_parser.add_argument(
        "--gpu-layers",
        type=nonnegative_integer,
        default=99,
        help="model layers to offload to CUDA; use 0 for CPU (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "download":
            model_path = download_model(args.cache_dir)
            print(f"Downloaded pinned Gemma model to: {model_path}")
        else:
            run_chat(args.cache_dir, args.context_size, args.gpu_layers)
    except RunnerError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
