"""Pinned local Transformers runtime with activation access."""

from __future__ import annotations

import importlib.util
import threading
import time
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, cast

import torch
from httpx import TransportError
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from llm_runtime.registry import (
    LocalLoaderKind,
    LocalTransformersRoute,
    require_registered_route,
)
from llm_runtime.types import (
    ChatMessage,
    GenerationSettings,
    GenerationResult,
    FinishReason,
    ModelId,
    ProviderId,
    QuantizationId,
    TextGenerator,
)

MIN_CUDA_FREE_BYTES = 12_000_000_000
_GENERATION_LOCK = threading.RLock()


class Device(StrEnum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class GenerativeModel(Protocol):
    @property
    def device(self) -> torch.device: ...

    def eval(self) -> object: ...

    def generate(self, **kwargs: object) -> torch.Tensor: ...


class LocalActivationRuntime(TextGenerator, Protocol):
    @property
    def provider_id(self) -> Literal[ProviderId.LOCAL]: ...

    @property
    def model(self) -> torch.nn.Module: ...

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase: ...


class TransformersRuntimeError(RuntimeError):
    """An actionable local download, loading, or generation failure."""


class TransformersRuntime:
    """Text generator retaining the real model and tokenizer for local probes."""

    def __init__(
        self,
        route: LocalTransformersRoute,
        model: GenerativeModel,
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self._route = require_registered_route(route)
        self._model = model
        self._tokenizer = tokenizer

    @property
    def model_id(self) -> ModelId:
        return self._route.model_id

    @property
    def provider_id(self) -> Literal[ProviderId.LOCAL]:
        return self._route.provider_id

    @property
    def quantization_id(self) -> QuantizationId:
        return self._route.quantization_id

    @property
    def model(self) -> torch.nn.Module:
        return cast(torch.nn.Module, self._model)

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    def generate(
        self,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
    ) -> GenerationResult:
        started = time.perf_counter()
        with _GENERATION_LOCK:
            return self._generate_locked(messages, settings, started)

    def _generate_locked(
        self,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        started: float,
    ) -> GenerationResult:
        cpu_state = torch.random.get_rng_state()
        cuda_devices = (
            tuple(range(torch.cuda.device_count())) if torch.cuda.is_available() else ()
        )
        cuda_states = tuple(torch.cuda.get_rng_state(device) for device in cuda_devices)
        try:
            return self._generate_scoped(messages, settings, started, cuda_devices)
        finally:
            for device, state in zip(cuda_devices, cuda_states, strict=True):
                torch.cuda.set_rng_state(state, device)
            torch.random.set_rng_state(cpu_state)

    def _generate_scoped(
        self,
        messages: Sequence[ChatMessage],
        settings: GenerationSettings,
        started: float,
        cuda_devices: tuple[int, ...],
    ) -> GenerationResult:
        model_device = self._model.device
        serialized = [
            {"role": message.role.value, "content": message.content}
            for message in messages
        ]
        try:
            batch = cast(
                BatchEncoding,
                self._tokenizer.apply_chat_template(
                    serialized,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                ),
            ).to(model_device)
        except (IndexError, RuntimeError, ValueError):
            raise TransformersRuntimeError(
                "Local generation failed while tokenizing typed chat messages in "
                "llm_runtime.transformers.TransformersRuntime.generate. The "
                "conversation may be malformed or incompatible with the pinned "
                "text tokenizer, so generation did not start and the caller has no "
                "response. Verify the registered tokenizer and retry with a shorter "
                "text-only conversation."
            ) from None

        input_length = batch["input_ids"].shape[-1]
        if input_length + settings.max_new_tokens > self._route.context_window:
            raise TransformersRuntimeError(
                "Local generation was rejected during context validation in "
                "llm_runtime.transformers.TransformersRuntime.generate because "
                f"{input_length} input tokens plus {settings.max_new_tokens} new "
                f"tokens exceed the {self._route.context_window}-token context. "
                "No model generation ran; shorten the conversation or reduce "
                "max_new_tokens and retry."
            )

        do_sample = settings.temperature > 0
        generation_options: dict[str, object] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_options["temperature"] = settings.temperature
            # Transformers only applies top-p/top-k filtering during sampling.
            if settings.top_p is not None:
                generation_options["top_p"] = settings.top_p
            if settings.top_k is not None:
                generation_options["top_k"] = settings.top_k
        try:
            if settings.seed is not None:
                torch.manual_seed(settings.seed)
                for device in cuda_devices:
                    torch.cuda.default_generators[device].manual_seed(settings.seed)
            with torch.inference_mode():
                generated = self._model.generate(**batch, **generation_options)
            completion_ids = tuple(
                int(value) for value in generated[0][input_length:].tolist()
            )
            text = cast(
                str,
                self._tokenizer.decode(list(completion_ids), skip_special_tokens=True),
            )
            eos = getattr(self._tokenizer, "eos_token_id", None)
            finish = (
                FinishReason.EOS
                if completion_ids and completion_ids[-1] == eos
                else FinishReason.LENGTH
            )
            return GenerationResult(
                text,
                int(input_length),
                len(completion_ids),
                completion_ids,
                finish,
                max(0, int((time.perf_counter() - started) * 1000)),
                {},
            )
        except (IndexError, RuntimeError, ValueError):
            raise TransformersRuntimeError(
                "Local generation failed in "
                "llm_runtime.transformers.TransformersRuntime.generate while "
                "delegating to the pinned Transformers model. The selected "
                "device may lack memory or the cached artifact may be incompatible, "
                "so no response is available to the caller. Reduce max_new_tokens, "
                "use a safer device, or refresh the pinned cache, then retry."
            ) from None


def download_transformers_artifact(
    route: LocalTransformersRoute, cache_dir: Path
) -> Path:
    """Download one exact reviewed route into the Hugging Face cache."""
    route = require_registered_route(route)
    try:
        snapshot = snapshot_download(
            repo_id=route.artifact.repository,
            revision=route.artifact.revision,
            cache_dir=cache_dir,
        )
    except (HfHubHTTPError, LocalEntryNotFoundError, OSError, TransportError):
        raise TransformersRuntimeError(
            "Local artifact download failed in "
            "llm_runtime.transformers.download_transformers_artifact while "
            f"fetching {route.artifact.repository} at pinned revision "
            f"{route.artifact.revision}. The runtime cannot be constructed, "
            "usually because the Hub is unreachable or the cache is not writable. "
            "Check network access and cache permissions, then rerun the QV "
            "download command."
        ) from None
    return Path(snapshot)


def _cached_transformers_artifact(
    route: LocalTransformersRoute, cache_dir: Path
) -> Path:
    try:
        snapshot = snapshot_download(
            repo_id=route.artifact.repository,
            revision=route.artifact.revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except (LocalEntryNotFoundError, OSError):
        raise TransformersRuntimeError(
            "Local runtime construction failed in "
            "llm_runtime.transformers._cached_transformers_artifact because "
            f"{route.artifact.repository} at revision {route.artifact.revision} "
            f"is incomplete or absent from cache {cache_dir}. No network fallback "
            "or alternate precision was attempted, so the runtime is unavailable to "
            "the caller. Run `uv run python -m quadratic_voting.main download` for "
            "this route and retry."
        ) from None
    return Path(snapshot)


def resolve_device(requested: Device) -> Device:
    if requested is Device.AUTO:
        return Device.AUTO
    if requested is Device.CUDA and not torch.cuda.is_available():
        raise TransformersRuntimeError(
            "Local runtime device validation failed in "
            "llm_runtime.transformers.resolve_device because PyTorch did not "
            "detect CUDA. The model cannot be loaded on the requested device. "
            "Verify the NVIDIA driver and PyTorch build, or select `cpu` or `auto`."
        )
    if requested is Device.CUDA:
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except RuntimeError:
            raise TransformersRuntimeError(
                "Local runtime device validation failed in "
                "llm_runtime.transformers.resolve_device while inspecting CUDA "
                "memory. Safe placement cannot be confirmed, so loading stopped and "
                "the caller cannot construct the runtime. Verify CUDA or select "
                "`auto`, then retry."
            ) from None
        if free_bytes < MIN_CUDA_FREE_BYTES:
            raise TransformersRuntimeError(
                "Local runtime device validation failed in "
                "llm_runtime.transformers.resolve_device because explicit CUDA "
                f"has {free_bytes / 1_000_000_000:.1f} GB free but the loading "
                f"budget requires {MIN_CUDA_FREE_BYTES / 1_000_000_000:.1f} GB. "
                "The model was not loaded; close GPU workloads or select `auto` "
                "or `cpu`."
            )
    return requested


def _loader_options(route: LocalTransformersRoute, device: Device) -> dict[str, object]:
    if route.loader is LocalLoaderKind.COMPRESSED_TENSORS_W4A16:
        if importlib.util.find_spec("compressed_tensors") is None:
            raise TransformersRuntimeError(
                "W4A16 runtime construction failed in "
                "llm_runtime.transformers._loader_options because the reviewed "
                "compressed-tensors loader is not installed. The selected route "
                "cannot load and will not fall back to BF16. Run `uv sync --locked` "
                "after adding and locking compressed-tensors>=0.15.0 as part of "
                "the reviewed route-enablement change, then retry."
            )
    return {
        "local_files_only": True,
        "device_map": device.value,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }


def create_transformers_runtime(
    route: LocalTransformersRoute,
    *,
    cache_dir: Path,
    device: Device = Device.AUTO,
) -> TransformersRuntime:
    """Construct a validated local route without network fallback."""
    route = require_registered_route(route)
    model_path = _cached_transformers_artifact(route, cache_dir)
    resolved_device = resolve_device(device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            padding_side="left",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **_loader_options(route, resolved_device),
        )
        model.eval()
    except (OSError, RuntimeError, ValueError):
        raise TransformersRuntimeError(
            "Local runtime loading failed in "
            "llm_runtime.transformers.create_transformers_runtime while loading "
            f"{route.quantization_id.value} from {model_path} on "
            f"{resolved_device.value}. The cache may be incomplete, the device "
            "may lack memory, or the installed runtime may be incompatible, so "
            "generation and activation access are unavailable to the caller. Run `uv "
            "sync --locked`, verify the pinned cache, and retry with `auto` or `cpu`."
        ) from None
    return TransformersRuntime(
        route,
        cast(GenerativeModel, model),
        cast(PreTrainedTokenizerBase, tokenizer),
    )
