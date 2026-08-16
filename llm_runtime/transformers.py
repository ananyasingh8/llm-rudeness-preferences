"""Pinned local Transformers runtime with activation access."""

from __future__ import annotations

import importlib.util
import shlex
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol, cast

import torch
from httpx import TransportError
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from llm_runtime.registry import (
    LocalLoaderKind,
    LocalTransformersRoute,
    TorchDTypeId,
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

# Sized to the smallest reviewed route so explicit CUDA fails fast before a
# slow load: the E4B FP4 route needs ~6.5 GB, so an empty 8 GB card must
# pass. The 31B routes need ~18 GB and OOM at load on smaller cards instead
# of being caught here — acceptable, since explicit CUDA is opt-in.
MIN_CUDA_FREE_BYTES = 6_500_000_000
_GENERATION_LOCK = threading.RLock()


class Device(StrEnum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


@dataclass(frozen=True, slots=True)
class DevicePlacement:
    """Observed placement for a loaded local model."""

    requested: Device
    resolved_device_map: tuple[tuple[str, str], ...]
    cpu_modules: tuple[str, ...]
    disk_modules: tuple[str, ...]

    @property
    def has_cpu_or_offload(self) -> bool:
        return bool(self.cpu_modules or self.disk_modules)


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

    @property
    def placement(self) -> DevicePlacement: ...


class TransformersRuntimeError(RuntimeError):
    """An actionable local download, loading, or generation failure."""


def _torch_dtype(dtype_id: TorchDTypeId) -> torch.dtype:
    """Resolve the closed registry dtype vocabulary without a silent default."""
    if dtype_id is TorchDTypeId.BFLOAT16:
        return torch.bfloat16
    if dtype_id is TorchDTypeId.UINT8:
        return torch.uint8
    raise TransformersRuntimeError(
        "Registry dtype resolution failed in "
        "llm_runtime.transformers._torch_dtype before model loading because "
        f"TorchDTypeId {dtype_id!r} has no reviewed torch.dtype mapping. The "
        "BitsAndBytes recipe cannot be trusted; add and test an explicit mapping "
        "for the new enum value before retrying."
    )


class TransformersRuntime:
    """Text generator retaining the real model and tokenizer for local probes."""

    def __init__(
        self,
        route: LocalTransformersRoute,
        model: GenerativeModel,
        tokenizer: PreTrainedTokenizerBase,
        placement: DevicePlacement | None = None,
    ) -> None:
        self._route = require_registered_route(route)
        self._model = model
        self._tokenizer = tokenizer
        self._placement = placement or DevicePlacement(Device.AUTO, (), (), ())

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

    @property
    def placement(self) -> DevicePlacement:
        return self._placement

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
            finish = (
                FinishReason.LENGTH
                if len(completion_ids) == settings.max_new_tokens
                else FinishReason.EOS
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
    except (HfHubHTTPError, LocalEntryNotFoundError, OSError, TransportError) as error:
        raise TransformersRuntimeError(
            "Local artifact download failed in "
            "llm_runtime.transformers.download_transformers_artifact while "
            f"fetching {route.artifact.repository} at pinned revision "
            f"{route.artifact.revision}. The runtime cannot be constructed, "
            "usually because the Hub is unreachable or the cache is not writable. "
            "Check network access and cache permissions. "
            f"{_download_remedy(route, cache_dir)} "
            f"Underlying error: {error}"
        ) from error
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
    except (LocalEntryNotFoundError, OSError) as error:
        raise TransformersRuntimeError(
            "Local runtime construction failed in "
            "llm_runtime.transformers._cached_transformers_artifact because "
            f"{route.artifact.repository} at revision {route.artifact.revision} "
            f"is incomplete or absent from cache {cache_dir}. No network fallback "
            "or alternate precision was attempted. "
            f"{_download_remedy(route, cache_dir)} "
            f"Underlying error: {error}"
        ) from error
    return Path(snapshot)


def _download_remedy(route: LocalTransformersRoute, cache_dir: Path) -> str:
    cache_argument = shlex.quote(str(cache_dir))
    if route.loader is LocalLoaderKind.BITSANDBYTES_4BIT:
        return (
            "Run `uv run python -m emotion_probing.main --experiment "
            f"convabuse-31b-local-quant --cache-dir {cache_argument} download` "
            "after accepting any applicable Gemma license and authenticating "
            "with Hugging Face, then retry the same run command."
        )
    if (
        route.model_id is ModelId.GEMMA_2_2B_IT
        and route.quantization_id is QuantizationId.BF16
    ):
        return (
            "Run `uv run python -m emotion_probing.main --experiment "
            f"bailbench-2b --cache-dir {cache_argument} download` after accepting "
            "the Gemma license and authenticating with Hugging Face, then retry."
        )
    return (
        "Run `uv run python -m quadratic_voting.main "
        f"--model {route.model_id.value} --provider {route.provider_id.value} "
        f"--quantization {route.quantization_id.value} "
        f"--cache-dir {cache_argument} download` for this exact route and retry."
    )


def resolve_device(requested: Device, route: LocalTransformersRoute) -> Device:
    bitsandbytes = route.loader is LocalLoaderKind.BITSANDBYTES_4BIT
    if requested is Device.AUTO:
        return Device.AUTO
    if requested is Device.CUDA and not torch.cuda.is_available():
        remedy = (
            "Repair the NVIDIA driver and CUDA-enabled PyTorch installation, free "
            "the requested CUDA device, and retry with `--device cuda`; this route "
            "does not support CPU, automatic, precision, or model fallback."
            if bitsandbytes
            else "Verify the NVIDIA driver and PyTorch build, or select `cpu` or `auto`."
        )
        raise TransformersRuntimeError(
            "Local runtime device validation failed in "
            "llm_runtime.transformers.resolve_device because PyTorch did not "
            "detect CUDA. The model cannot be loaded on the requested device. "
            f"{remedy}"
        )
    if requested is Device.CUDA:
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
        except RuntimeError as error:
            remedy = (
                "Repair CUDA diagnostics and retry with `--device cuda`; this "
                "route does not support CPU, automatic, precision, or model fallback."
                if bitsandbytes
                else "Verify CUDA or select `auto`."
            )
            raise TransformersRuntimeError(
                "Local runtime device validation failed in "
                "llm_runtime.transformers.resolve_device while inspecting CUDA "
                "memory. Safe placement cannot be confirmed, so loading stopped. "
                f"{remedy} "
                f"Underlying error: {error}"
            ) from error
        if free_bytes < MIN_CUDA_FREE_BYTES:
            remedy = (
                "Close other GPU workloads until the required CUDA budget is free "
                "and retry with `--device cuda`; this route does not support CPU, "
                "automatic, precision, or model fallback."
                if bitsandbytes
                else "Close GPU workloads or select `auto` or `cpu`."
            )
            raise TransformersRuntimeError(
                "Local runtime device validation failed in "
                "llm_runtime.transformers.resolve_device because explicit CUDA "
                f"has {free_bytes / 1_000_000_000:.1f} GB free but the loading "
                f"budget requires {MIN_CUDA_FREE_BYTES / 1_000_000_000:.1f} GB. "
                f"The model was not loaded. {remedy}"
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
    if route.loader is LocalLoaderKind.BITSANDBYTES_4BIT:
        if importlib.util.find_spec("bitsandbytes") is None:
            raise TransformersRuntimeError(
                "BitsAndBytes runtime construction failed during dependency "
                "validation in llm_runtime.transformers._loader_options because "
                "the direct bitsandbytes dependency is absent. The pinned FP4 "
                "route cannot load and no precision or loader fallback will run. "
                "Run `uv sync --locked` to install the reviewed dependency, then "
                "retry before starting the experiment."
            )
        if device is Device.AUTO and not torch.cuda.is_available():
            raise TransformersRuntimeError(
                "BitsAndBytes runtime construction rejected automatic placement in "
                "llm_runtime.transformers._loader_options before cache or model "
                "loading because PyTorch did not detect CUDA. The reviewed Gemma "
                "4 31B FP4 route cannot run and no CPU, precision, or model fallback "
                "will be attempted. Repair the NVIDIA driver/PyTorch CUDA runtime "
                "and retry with `--device cuda`."
            )
        if device is Device.CPU:
            raise TransformersRuntimeError(
                "BitsAndBytes runtime construction rejected CPU placement in "
                "llm_runtime.transformers._loader_options before cache or model "
                "loading because the reviewed Gemma 4 31B FP4 route requires CUDA. "
                "The caller cannot run activation probing on this placement. Use "
                "`--device cuda` (or `auto` on a CUDA host) and retry; no CPU "
                "offload fallback is enabled."
            )
        settings = route.bitsandbytes
        if settings is None:
            raise TransformersRuntimeError(
                "BitsAndBytes runtime construction failed closed in "
                "llm_runtime.transformers._loader_options because the registered "
                "route has no explicit 4-bit settings. Loading stopped before "
                "cache or model access; add reviewed static settings to the route."
            )
        compute_dtype = _torch_dtype(settings.compute_dtype)
        quant_storage = _torch_dtype(settings.quant_storage)
        return {
            "local_files_only": True,
            "device_map": device.value,
            "dtype": compute_dtype,
            "attn_implementation": "sdpa",
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=settings.load_in_4bit,
                bnb_4bit_quant_type=settings.quant_type.value,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_storage=quant_storage,
                bnb_4bit_use_double_quant=settings.use_double_quant,
            ),
        }
    return {
        "local_files_only": True,
        "device_map": device.value,
        "dtype": torch.bfloat16,
        "attn_implementation": "sdpa",
    }


def _placement_for_model(
    route: LocalTransformersRoute, model: object, requested: Device
) -> DevicePlacement:
    raw_map = getattr(model, "hf_device_map", None)
    if not isinstance(raw_map, dict) or not raw_map:
        if route.loader is LocalLoaderKind.BITSANDBYTES_4BIT:
            named_parameters = getattr(model, "named_parameters", None)
            named_buffers = getattr(model, "named_buffers", None)
            if not callable(named_parameters) or not callable(named_buffers):
                raise TransformersRuntimeError(
                    "BitsAndBytes placement validation failed in "
                    "llm_runtime.transformers._placement_for_model immediately "
                    "after model loading because Transformers exposed neither a "
                    "non-empty hf_device_map nor inspectable named tensors. The "
                    "caller cannot verify that all weights stayed on CUDA, so the "
                    "experiment will not start. Use compatible locked dependencies "
                    "and retry; do not bypass placement validation."
                )
            tensor_devices: set[str] = set()
            invalid_tensors: list[tuple[str, str]] = []
            tensor_count = 0
            for kind, tensors in (
                ("parameter", named_parameters()),
                ("buffer", named_buffers()),
            ):
                for name, tensor in tensors:
                    tensor_count += 1
                    target = str(tensor.device)
                    tensor_devices.add(target)
                    if target not in {"cuda", "cuda:0"} and len(invalid_tensors) < 10:
                        invalid_tensors.append((f"{kind}:{name}", target))
            if tensor_count == 0:
                raise TransformersRuntimeError(
                    "BitsAndBytes placement validation failed in "
                    "llm_runtime.transformers._placement_for_model immediately "
                    "after model loading because Transformers exposed neither a "
                    "non-empty hf_device_map nor any named parameters or buffers. "
                    "The caller cannot verify CUDA-only placement, so activation "
                    "probing will not start."
                )
            if invalid_tensors:
                details = ", ".join(
                    f"{name}={target}" for name, target in invalid_tensors
                )
                raise TransformersRuntimeError(
                    "BitsAndBytes placement validation failed in "
                    "llm_runtime.transformers._placement_for_model immediately "
                    "after model loading because direct tensor inspection found "
                    f"non-CUDA placement: {details}. Activation probing has not "
                    "started and silent CPU/disk/meta offload is not accepted for "
                    "this reviewed route. Free enough RTX 4090 memory and retry."
                )
            verified = tuple(
                ("<all-parameters-and-buffers>", target)
                for target in sorted(tensor_devices)
            )
            return DevicePlacement(requested, verified, (), ())
        return DevicePlacement(requested, (), (), ())

    normalized = tuple(
        sorted((str(name), str(target)) for name, target in raw_map.items())
    )
    cpu_modules = tuple(name for name, target in normalized if target == "cpu")
    disk_modules = tuple(name for name, target in normalized if target == "disk")
    if route.loader is LocalLoaderKind.BITSANDBYTES_4BIT:
        invalid = tuple(
            (name, target)
            for name, target in normalized
            if target not in {"0", "cuda", "cuda:0"}
        )
        if invalid:
            details = ", ".join(f"{name}={target}" for name, target in invalid)
            raise TransformersRuntimeError(
                "BitsAndBytes placement validation failed in "
                "llm_runtime.transformers._placement_for_model immediately after "
                f"model loading because non-CUDA placement was resolved: {details}. "
                "Activation probing has not started and silent CPU/disk offload is "
                "not accepted for this reviewed route. Free enough RTX 4090 memory, "
                "select `--device cuda`, and retry."
            )
    return DevicePlacement(requested, normalized, cpu_modules, disk_modules)


def create_transformers_runtime(
    route: LocalTransformersRoute,
    *,
    cache_dir: Path,
    device: Device = Device.AUTO,
) -> TransformersRuntime:
    """Construct a validated local route without network fallback."""
    route = require_registered_route(route)
    resolved_device = resolve_device(device, route)
    loader_options = _loader_options(route, resolved_device)
    model_path = _cached_transformers_artifact(route, cache_dir)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            padding_side="left",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **loader_options,
        )
        model.eval()
    except (OSError, RuntimeError, ValueError) as error:
        if route.loader is LocalLoaderKind.BITSANDBYTES_4BIT:
            remedy = (
                "Run `uv sync --locked`, verify the pinned cache and NVIDIA CUDA "
                "runtime, free enough GPU memory for all modules, and retry with "
                "`--device cuda`; CPU/offload and alternate precision are disabled."
            )
        else:
            remedy = (
                "Run `uv sync --locked`, verify the pinned cache, and retry with "
                "`auto` or `cpu`."
            )
        raise TransformersRuntimeError(
            "Local runtime loading failed in "
            "llm_runtime.transformers.create_transformers_runtime while loading "
            f"{route.quantization_id.value} from {model_path} on "
            f"{resolved_device.value}. The cache may be incomplete, the device "
            "may lack memory, or the installed runtime may be incompatible, so "
            f"generation and activation access are unavailable. {remedy} "
            f"Underlying error: {error}"
        ) from error
    placement = _placement_for_model(route, model, device)
    return TransformersRuntime(
        route,
        cast(GenerativeModel, model),
        cast(PreTrainedTokenizerBase, tokenizer),
        placement,
    )
