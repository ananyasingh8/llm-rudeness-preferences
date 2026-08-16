"""Probe Gemma's emotion activations on rude/abusive vs normal user messages.

The experiment: run each dataset item through the model with a single forward
pass (no generation), read the residual stream at the last prompt token — the
position right before the model starts its reply, the analog of the ":" after
"Assistant" in Anthropic's emotion-concepts paper — and take the cosine
similarity of that activation against pre-extracted emotion vectors. Comparing
scores between rude/abusive and normal inputs shows which emotion
representations mistreatment activates.

Four registered experiment configurations (see EXPERIMENTS below):

- **bailbench-2b**: google/gemma-2-2b-it scored against EmotionScope's 20
  emotion vectors (layer 22) on 1,630 synthetic normal/rude prompt pairs.
- **convabuse-31b**: Google's pre-quantized W4A16 Gemma 4 31B route.
- **convabuse-31b-local-quant**: google/gemma-4-31B-it quantized locally with
  BitsAndBytes FP4 and scored against
  the 171 gemotions emotion vectors (layer 40) on 4,185 real, human-annotated
  user-bot conversation snippets from ConvAbuse.
- **convabuse-31b-base**: the base (non-instruction-tuned) google/gemma-4-31B,
  BitsAndBytes FP4, scored against our own extracted vectors
  (emotion_probing.extract) on the same ConvAbuse data. Prompts render as a
  plain "User:/Assistant:" transcript because base models have no chat
  template. Blocked with instructions until BASE_PROBE_LAYER and
  BASE_VECTORS_RUN are pinned from the extraction sweep's layer_quality.json.

Emotion vectors are model-specific: each configuration pairs a model route
from the shared llm_runtime registry with vectors extracted from that same
model. The gemotions vectors and cluster analysis are read from the vendored
emotion_probing/gemotions/ folder (a committed subset of the
dejanseo/gemotions HF repo — see gemotions/VENDORED.md for provenance).

Every run writes into a fresh results/<timestamp>_<experiment>/ folder
(scores.csv + run_info.json + clusters.json where applicable) so previous
results are never overwritten. Use --resume to continue the latest run of an
experiment instead of starting a new folder.

Usage:
    uv run python -m emotion_probing.main [--experiment NAME] [--cache-dir PATH] download
    uv run python -m emotion_probing.main [--experiment NAME] [--cache-dir PATH]
        run [--device auto|cuda|cpu] [--limit N] [--resume]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import tempfile
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Iterator, cast

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.constants import HF_HUB_CACHE
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding

from emotion_probing.datasets import (
    DatasetError,
    load_bailbench,
    load_convabuse,
)
from llm_runtime import (
    Capability,
    LocalTransformersRoute,
    ModelId,
    ModelRouteError,
    ProviderId,
    QuantizationId,
    resolve_route,
)
from llm_runtime.transformers import (
    Device,
    LocalActivationRuntime,
    TransformersRuntimeError,
    create_transformers_runtime,
    download_transformers_artifact,
)


class ExperimentId(StrEnum):
    BAILBENCH_2B = "bailbench-2b"
    CONVABUSE_31B = "convabuse-31b"
    CONVABUSE_31B_LOCAL_QUANT = "convabuse-31b-local-quant"
    CONVABUSE_31B_BASE = "convabuse-31b-base"
    CONVABUSE_E4B = "convabuse-e4b"


class VectorSource(StrEnum):
    EMOTIONSCOPE = "emotionscope"
    GEMOTIONS = "gemotions"
    EXTRACTED_BASE = "extracted-base"
    EXTRACTED_E4B = "extracted-e4b"


class DatasetId(StrEnum):
    BAILBENCH = "bailbench"
    CONVABUSE = "convabuse"


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """One reviewed pairing of model route, emotion vectors, and dataset."""

    name: ExperimentId
    model_id: ModelId
    quantization_id: QuantizationId
    probe_layer: int
    vectors: VectorSource
    dataset: DatasetId
    expected_width: int
    token_limit: int
    batch_size: int = 1
    use_cache: bool = False
    # "chat" renders via the tokenizer's chat template; "transcript" renders a
    # plain "User: ...\nAssistant:" transcript for models with no chat
    # template (base models), ending at the ":" measurement position.
    prompt_style: str = "chat"


# --- Base-model probing: pin BOTH constants after the extraction sweep. ------
# Run the sweep (`uv run python -m emotion_probing.extract run`, see
# extract_vectors.md), read layer_quality.json, pick the winning layer
# (prefer a stable plateau over a lone spike), then set the layer and the
# extraction run folder name here. Until both are set, selecting the
# convabuse-31b-base experiment fails with instructions; download still works.
BASE_PROBE_LAYER: int | None = 13  # picked from the extraction sweep's scorecard
BASE_VECTORS_RUN: str | None = "2026-08-16_000402_extract-gemma4-e4b-it"

# --- E4B probing: same pinning pattern, from the gemma4-e4b extraction -------
# (`uv run python -m emotion_probing.extract run` — the default extraction —
# then pick the layer from its layer_quality.json scorecard).
E4B_PROBE_LAYER: int | None = None
E4B_VECTORS_RUN: str | None = None

# --- Experiment constants: edit these to change the model setups. ------------
EXPERIMENTS: dict[ExperimentId, ExperimentConfig] = {
    ExperimentId.BAILBENCH_2B: ExperimentConfig(
        name=ExperimentId.BAILBENCH_2B,
        model_id=ModelId.GEMMA_2_2B_IT,
        quantization_id=QuantizationId.BF16,
        probe_layer=22,
        vectors=VectorSource.EMOTIONSCOPE,
        dataset=DatasetId.BAILBENCH,
        expected_width=2_304,
        token_limit=8_192,
    ),
    ExperimentId.CONVABUSE_31B: ExperimentConfig(
        name=ExperimentId.CONVABUSE_31B,
        model_id=ModelId.GEMMA_4_31B_IT,
        quantization_id=QuantizationId.W4A16_COMPRESSED_TENSORS,
        probe_layer=40,
        vectors=VectorSource.GEMOTIONS,
        dataset=DatasetId.CONVABUSE,
        expected_width=5_376,
        token_limit=512,
    ),
    ExperimentId.CONVABUSE_31B_LOCAL_QUANT: ExperimentConfig(
        name=ExperimentId.CONVABUSE_31B_LOCAL_QUANT,
        model_id=ModelId.GEMMA_4_31B_IT,
        quantization_id=QuantizationId.BITSANDBYTES_FP4,
        probe_layer=40,
        vectors=VectorSource.GEMOTIONS,
        dataset=DatasetId.CONVABUSE,
        expected_width=5_376,
        token_limit=512,
    ),
    ExperimentId.CONVABUSE_31B_BASE: ExperimentConfig(
        name=ExperimentId.CONVABUSE_31B_BASE,
        model_id=ModelId.GEMMA_4_31B,
        quantization_id=QuantizationId.BITSANDBYTES_FP4,
        probe_layer=BASE_PROBE_LAYER if BASE_PROBE_LAYER is not None else -1,
        vectors=VectorSource.EXTRACTED_BASE,
        dataset=DatasetId.CONVABUSE,
        expected_width=5_376,
        token_limit=512,
        prompt_style="transcript",
    ),
    ExperimentId.CONVABUSE_E4B: ExperimentConfig(
        name=ExperimentId.CONVABUSE_E4B,
        model_id=ModelId.GEMMA_4_E4B_IT,
        quantization_id=QuantizationId.BITSANDBYTES_FP4,
        probe_layer=E4B_PROBE_LAYER if E4B_PROBE_LAYER is not None else -1,
        vectors=VectorSource.EXTRACTED_E4B,
        dataset=DatasetId.CONVABUSE,
        expected_width=2_560,
        token_limit=512,
    ),
}
DEFAULT_EXPERIMENT = ExperimentId.CONVABUSE_E4B

PACKAGE_DIR = Path(__file__).parent
RESULTS_DIR = PACKAGE_DIR / "results"
EMOTIONSCOPE_VECTORS_FILE = (
    PACKAGE_DIR / "EmotionScope" / "results" / "vectors" / "google_gemma-2-2b-it.pt"
)
GEMOTIONS_DIR = PACKAGE_DIR / "gemotions"
# Provenance of the vendored files (see gemotions/VENDORED.md).
GEMOTIONS_SOURCE_REVISION = "4fd2ac63551f1be37e6e6c2eacd1b1898c9af656"
GEMOTIONS_ANALYSIS_FILE = (
    GEMOTIONS_DIR / "results" / "gemma4-31b" / "analysis" / "analysis_results.json"
)
PROBE_SOURCE_FILE = Path(__file__)
RUN_WRITER_LOCK_NAME = ".writer.lock"
HASH_CHUNK_BYTES = 1024 * 1024


class ProbeError(RuntimeError):
    """A user-actionable failure in the probing workflow."""


def resolve_probe_route(config: ExperimentConfig) -> LocalTransformersRoute:
    """Resolve the experiment's route, requiring local activation access."""
    route = resolve_route(
        config.model_id,
        ProviderId.LOCAL,
        config.quantization_id,
        required={Capability.LOCAL_ACTIVATIONS},
    )
    if not isinstance(route, LocalTransformersRoute):
        raise ProbeError(
            f"The resolved route is {type(route).__name__}, not a local "
            "Transformers route. Emotion probing reads activations and can only "
            "run against a registered local route."
        )
    return route


def _load_emotionscope_vectors(
    route: LocalTransformersRoute, probe_layer: int
) -> tuple[list[str], torch.Tensor]:
    if not EMOTIONSCOPE_VECTORS_FILE.exists():
        raise ProbeError(f"Emotion vectors file not found: {EMOTIONSCOPE_VECTORS_FILE}")
    saved = torch.load(
        EMOTIONSCOPE_VECTORS_FILE, weights_only=False, map_location="cpu"
    )
    vectors_model = saved["model_info"]["model_name"]
    if vectors_model != route.artifact.repository:
        raise ProbeError(
            f"The emotion vectors were extracted from {vectors_model} but the "
            f"configured route loads {route.artifact.repository}. Emotion "
            "vectors are model-specific; use the matching model or re-extract."
        )
    if int(saved["probe_layer_used"]) != probe_layer:
        raise ProbeError(
            f"The vectors file was extracted at layer {saved['probe_layer_used']} "
            f"but the experiment is configured for layer {probe_layer}."
        )
    names = list(saved["vectors"].keys())
    matrix = F.normalize(
        torch.stack([saved["vectors"][name].float() for name in names]), dim=1
    )
    return names, matrix


def _load_gemotions_vectors(probe_layer: int) -> tuple[list[str], torch.Tensor]:
    path = (
        GEMOTIONS_DIR
        / "results"
        / "gemma4-31b"
        / f"emotion_vectors_layer{probe_layer}.npz"
    )
    if not path.exists():
        raise ProbeError(
            f"Emotion vectors file not found: {path}. Only layer 40 is "
            "vendored; other layers can be fetched from the dejanseo/gemotions "
            "HF repo (see gemotions/VENDORED.md)."
        )
    saved = np.load(path)
    names = list(saved.files)
    matrix = F.normalize(
        torch.stack([torch.from_numpy(saved[name]).float() for name in names]),
        dim=1,
    )
    return names, matrix


def _extraction_pins(
    vectors: VectorSource,
) -> tuple[int | None, str | None, str, str, str]:
    """(layer, run, npz prefix, pin-constant names, extraction CLI name)."""
    if vectors is VectorSource.EXTRACTED_BASE:
        return (
            BASE_PROBE_LAYER,
            BASE_VECTORS_RUN,
            "gemma4-31b-base",
            "BASE_PROBE_LAYER and BASE_VECTORS_RUN",
            "gemma4-31b-base",
        )
    return (
        E4B_PROBE_LAYER,
        E4B_VECTORS_RUN,
        "gemma4-e4b-it",
        "E4B_PROBE_LAYER and E4B_VECTORS_RUN",
        "gemma4-e4b",
    )


def _extracted_vectors_path(vectors: VectorSource) -> Path:
    """Resolve the pinned extracted-vectors file, or explain how to pin it."""
    layer, run, prefix, pin_names, extraction = _extraction_pins(vectors)
    if layer is None or run is None:
        raise ProbeError(
            "This experiment is not pinned yet. Run the extraction sweep "
            f"(`uv run python -m emotion_probing.extract --extraction "
            f"{extraction} run`, see extract_vectors.md), pick the winning "
            "layer from layer_quality.json (prefer a plateau over a lone "
            f"spike), then set {pin_names} at the top of "
            "emotion_probing/main.py."
        )
    return RESULTS_DIR / run / f"{prefix}_emotion_vectors_layer{layer}.npz"


def _load_extracted_vectors(
    vectors: VectorSource, route: LocalTransformersRoute
) -> tuple[list[str], torch.Tensor]:
    layer, _, _, pin_names, _ = _extraction_pins(vectors)
    path = _extracted_vectors_path(vectors)
    if not path.exists():
        raise ProbeError(
            f"Emotion vectors file not found: {path}. Check {pin_names} "
            "against the extraction run's actual outputs."
        )
    run_info_file = path.parent / "run_info.json"
    if run_info_file.exists():
        info = json.loads(run_info_file.read_text(encoding="utf-8"))
        if info.get("repository") != route.artifact.repository:
            raise ProbeError(
                "The pinned vectors were extracted from "
                f"{info.get('repository')} but the configured route loads "
                f"{route.artifact.repository}. Emotion vectors are "
                "model-specific; fix the pinning."
            )
        if layer not in info.get("probe_layers", []):
            raise ProbeError(
                f"Pinned probe layer {layer} was not part of the pinned "
                "extraction run's probe_layers; fix the pinning."
            )
    saved = np.load(path)
    names = list(saved.files)
    matrix = F.normalize(
        torch.stack([torch.from_numpy(saved[name]).float() for name in names]),
        dim=1,
    )
    return names, matrix


def load_vectors(
    config: ExperimentConfig, route: LocalTransformersRoute
) -> tuple[list[str], torch.Tensor]:
    """Load the unit-normalized emotion-vector matrix for this experiment."""
    if config.vectors is VectorSource.EMOTIONSCOPE:
        return _load_emotionscope_vectors(route, config.probe_layer)
    if config.vectors is VectorSource.GEMOTIONS:
        return _load_gemotions_vectors(config.probe_layer)
    if config.vectors in (VectorSource.EXTRACTED_BASE, VectorSource.EXTRACTED_E4B):
        return _load_extracted_vectors(config.vectors, route)
    raise ProbeError(f"Unknown vectors source {config.vectors!r}.")


def load_clusters(config: ExperimentConfig) -> dict | None:
    """Load the vendored gemotions cluster analysis for the configured layer."""
    if config.vectors is VectorSource.EMOTIONSCOPE:
        return None
    if not GEMOTIONS_ANALYSIS_FILE.exists():
        raise ProbeError(f"Cluster analysis file not found: {GEMOTIONS_ANALYSIS_FILE}.")
    with GEMOTIONS_ANALYSIS_FILE.open(encoding="utf-8") as handle:
        analysis = json.load(handle)
    # Base-model runs reuse the IT layer-40 clustering purely as an
    # emotion-name grouping aid for analysis (the analysis file only has
    # entries for the IT model's swept layers).
    layer = (
        str(config.probe_layer)
        if config.vectors is VectorSource.GEMOTIONS
        else "40"
    )
    if layer not in analysis:
        raise ProbeError(f"The gemotions analysis file has no layer {layer} entry.")
    return {"clusters": analysis[layer]["clusters"]}


def load_dataset(config: ExperimentConfig, limit: int | None):
    """Load the experiment's dataset as (key_columns, metadata_columns, tasks)."""
    if config.dataset is DatasetId.BAILBENCH:
        return load_bailbench(limit)
    if config.dataset is DatasetId.CONVABUSE:
        return load_convabuse(limit)
    raise ProbeError(f"Unknown dataset {config.dataset!r}.")


def emotion_scores(
    runtime: LocalActivationRuntime,
    messages: list[dict[str, str]],
    vector_matrix: torch.Tensor,
    probe_layer: int,
    *,
    expected_width: int,
    token_limit: int,
    prompt_style: str = "chat",
) -> tuple[list[float], int]:
    """Run one forward pass and score the response-start activation.

    Returns the cosine similarity to each emotion vector, plus the prompt's
    token count. hidden_states[0] is the embedding output, so the residual
    stream after block `probe_layer` is hidden_states[probe_layer + 1] — the
    same convention both vector sources used during extraction.
    """
    inputs, token_count = _tokenize_probe_prompt(
        runtime, messages, token_limit, prompt_style=prompt_style
    )
    return _score_prepared_prompt(
        runtime,
        inputs,
        token_count,
        vector_matrix,
        probe_layer,
        expected_width=expected_width,
    )


def _render_transcript(messages: list[dict[str, str]]) -> str:
    """Plain-text transcript for models with no chat template (base models).

    Ends with "Assistant:" so the last prompt token is the ":" after
    "Assistant" — literally the measurement position from Anthropic's paper.
    """
    speaker = {"user": "User", "assistant": "Assistant"}
    lines = [
        f"{speaker[message['role']]}: {message['content']}"
        for message in messages
    ]
    return "\n".join(lines) + "\nAssistant:"


def _tokenize_probe_prompt(
    runtime: LocalActivationRuntime,
    messages: list[dict[str, str]],
    token_limit: int,
    *,
    prompt_style: str = "chat",
    move_to_model: bool = True,
) -> tuple[dict[str, torch.Tensor], int]:
    """Tokenize and enforce batch/token bounds before any model call."""
    try:
        if prompt_style == "transcript":
            encoded = cast(
                BatchEncoding,
                runtime.tokenizer(
                    _render_transcript(messages), return_tensors="pt"
                ),
            )
        else:
            encoded = cast(
                BatchEncoding,
                runtime.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    truncation=False,
                ),
            )
        inputs = (
            encoded.to(cast(torch.device, getattr(runtime.model, "device")))
            if move_to_model
            else encoded
        )
        input_ids = inputs["input_ids"]
    except (
        AttributeError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise ProbeError(
            "Prompt tokenization failed in emotion_probing.main.emotion_scores "
            "before the model call because the chat messages or tokenizer output "
            "were invalid. No activation was measured; correct the text-only chat "
            f"turns and retry. Underlying error: {error}"
        ) from error
    if not isinstance(input_ids, torch.Tensor):
        raise ProbeError(
            "Prompt validation failed in emotion_probing.main._tokenize_probe_prompt "
            "before the model call because the locked tokenizer did not return a "
            "tensor named input_ids. No forward pass ran; reinstall the locked "
            "Transformers runtime and retry."
        )
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ProbeError(
            "Prompt validation failed in emotion_probing.main.emotion_scores "
            f"before the model call because tokenization produced shape "
            f"{tuple(input_ids.shape)} instead of batch one. No forward pass ran; "
            "submit exactly one conversation per probe."
        )
    token_count = int(input_ids.shape[-1])
    if token_count > token_limit:
        raise ProbeError(
            "Prompt validation failed in emotion_probing.main.emotion_scores "
            f"before the model call because {token_count} tokens exceed the "
            f"reviewed {token_limit}-token bound. No forward pass ran; shorten the "
            "conversation and retry."
        )

    return cast(dict[str, torch.Tensor], inputs), token_count


def validate_prompt_bounds_before_iteration(
    runtime: LocalActivationRuntime,
    prompts: Sequence[list[dict[str, str]]],
    token_limit: int,
    prompt_style: str = "chat",
) -> None:
    """Validate all prompts on CPU before any experiment model forward."""
    for index, messages in enumerate(prompts):
        try:
            _tokenize_probe_prompt(
                runtime,
                messages,
                token_limit,
                prompt_style=prompt_style,
                move_to_model=False,
            )
        except ProbeError as error:
            raise ProbeError(
                "Experiment preflight failed in "
                "emotion_probing.main.validate_prompt_bounds_before_iteration "
                f"while validating zero-based task {index}. No experiment model "
                f"forward ran; correct or remove that prompt and retry. {error}"
            ) from error


def _score_prepared_prompt(
    runtime: LocalActivationRuntime,
    inputs: dict[str, torch.Tensor],
    token_count: int,
    vector_matrix: torch.Tensor,
    probe_layer: int,
    *,
    expected_width: int,
) -> tuple[list[float], int]:
    """Score one already-validated batch-one prompt through the production hook."""
    if vector_matrix.ndim != 2 or vector_matrix.shape[1] != expected_width:
        raise ProbeError(
            "Emotion-vector validation failed in "
            "emotion_probing.main._score_prepared_prompt before the model call "
            f"because shape {tuple(vector_matrix.shape)} is not a rank-two matrix "
            f"of width {expected_width}. No hook or forward ran; restore the "
            "reviewed vectors for this route and retry."
        )
    block = _decoder_block(runtime.model, probe_layer)
    captured: list[torch.Tensor] = []

    def capture_output(
        _module: torch.nn.Module, _args: tuple[object, ...], output: object
    ) -> None:
        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor):
            raise ProbeError(
                "Activation capture failed in emotion_probing.main.emotion_scores "
                f"during decoder block {probe_layer} because its output was "
                f"{type(value).__name__}, not a tensor. Scoring cannot continue; "
                "use the locked Transformers model implementation."
            )
        captured.append(value)

    handle = block.register_forward_hook(capture_output)
    try:
        with torch.inference_mode():
            runtime.model(**inputs, use_cache=False)
        if len(captured) != 1:
            raise ProbeError(
                "Activation capture validation failed in "
                "emotion_probing.main.emotion_scores after the forward pass "
                f"because decoder block {probe_layer} fired {len(captured)} times, "
                "not once. No scores are valid; use the locked model architecture."
            )
        hidden = captured[0]
        expected_shape = (1, token_count, expected_width)
        if hidden.ndim != 3 or tuple(hidden.shape) != expected_shape:
            raise ProbeError(
                "Activation shape validation failed in "
                "emotion_probing.main.emotion_scores after decoder block "
                f"{probe_layer}: observed {tuple(hidden.shape)}, expected "
                f"{expected_shape}. Scores were not computed; verify "
                "the exact pinned model and layer before retrying."
            )
        activation = hidden[0, -1, :].float().cpu()
    except ProbeError:
        raise
    except (IndexError, RuntimeError, TypeError, ValueError) as error:
        raise ProbeError(
            "Activation forward failed in emotion_probing.main.emotion_scores "
            f"while measuring decoder block {probe_layer} with use_cache=False. "
            "No scores were produced; verify model compatibility and available "
            f"CUDA memory, then retry. Underlying error: {error}"
        ) from error
    finally:
        handle.remove()

    scores = F.normalize(activation, dim=0) @ vector_matrix.T
    return [float(score) for score in scores], token_count


def _decoder_block(model: torch.nn.Module, probe_layer: int) -> torch.nn.Module:
    """Resolve one exact decoder block without enabling all hidden states."""
    base = getattr(model, "model", None)
    language_model = getattr(base, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        layers = getattr(base, "layers", None)
    if not isinstance(layers, (torch.nn.ModuleList, list, tuple)):
        raise ProbeError(
            "Decoder-layer validation failed in emotion_probing.main._decoder_block "
            "before the experiment loop because the loaded model exposes neither "
            "model.language_model.layers nor model.layers. Activation probing "
            "cannot start; use the exact pinned Gemma Transformers architecture."
        )
    if probe_layer < 0 or probe_layer >= len(layers):
        raise ProbeError(
            "Decoder-layer validation failed in emotion_probing.main._decoder_block "
            f"before activation capture because layer {probe_layer} is outside the "
            f"loaded model's {len(layers)} blocks. No probe ran; select the reviewed "
            "experiment route and layer."
        )
    block = layers[probe_layer]
    if not isinstance(block, torch.nn.Module):
        raise ProbeError(
            "Decoder-layer validation failed in emotion_probing.main._decoder_block "
            f"because block {probe_layer} is not a torch module. No hook was "
            "registered; use the locked model implementation."
        )
    return block


def latest_run_dir(experiment: ExperimentId) -> Path | None:
    """Return the most recent run folder for an experiment, if any."""
    if not RESULTS_DIR.exists():
        return None
    runs = sorted(
        path
        for path in RESULTS_DIR.iterdir()
        if path.is_dir() and path.name.endswith(f"_{experiment}")
    )
    return runs[-1] if runs else None


def prepare_run_dir(
    config: ExperimentConfig,
    resume: bool,
) -> Path:
    """Reserve a fresh run directory, or locate the latest directory for resume."""
    if resume:
        run_dir = latest_run_dir(config.name)
        if run_dir is None:
            raise ProbeError(
                f"--resume was passed but no previous {config.name} run exists "
                f"under {RESULTS_DIR}. Start a run without --resume first."
            )
        return run_dir
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RESULTS_DIR / f"{stamp}_{config.name}"
    try:
        run_dir.mkdir(parents=True)
    except FileExistsError as error:
        raise ProbeError(
            "New run directory reservation failed in "
            "emotion_probing.main.prepare_run_dir before metadata or scores were "
            f"written because {run_dir} already exists. Another run may have "
            "started in the same second. Wait one second and retry without "
            "`--resume`, or explicitly resume the existing compatible run."
        ) from error
    return run_dir


def _stage_new_run(
    config: ExperimentConfig,
    run_dir: Path,
    provenance: dict[str, object],
) -> None:
    """Atomically stage new-run metadata while holding its writer lock."""
    stamp = run_dir.name.removesuffix(f"_{config.name}")
    run_info = dict(provenance)
    run_info["started"] = stamp
    clusters = load_clusters(config)
    if clusters is not None:
        if not _is_cluster_snapshot(clusters):
            raise ProbeError(
                "Cluster snapshot staging failed in emotion_probing.main._stage_new_run "
                "before scores were created because the selected cluster payload "
                "does not contain a clusters object mapping IDs to non-empty emotion "
                "name lists. Restore the reviewed analysis file and retry."
            )
        expected_digest = provenance.get("cluster_snapshot_sha256")
        actual_digest = _canonical_json_sha256(clusters)
        if actual_digest != expected_digest:
            raise ProbeError(
                "Cluster snapshot staging failed in emotion_probing.main._stage_new_run "
                "before scores were created because the canonical cluster payload "
                "changed after input fingerprinting. No measurements started; keep "
                "the reviewed analysis file stable and retry."
            )
        _atomic_write_json(
            run_dir / "clusters.json",
            cast(dict[str, object], clusters),
            operation="staging cluster analysis for a new run",
            canonical=True,
        )
    _atomic_write_json(
        run_dir / "run_info.json",
        run_info,
        operation="creating new run provenance",
    )


def _canonical_json_bytes(data: object) -> bytes:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(data: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(data)).hexdigest()


def _is_cluster_snapshot(payload: object) -> bool:
    if not isinstance(payload, dict) or set(payload) != {"clusters"}:
        return False
    clusters = payload.get("clusters")
    return isinstance(clusters, dict) and all(
        isinstance(cluster_id, str)
        and isinstance(names, list)
        and all(isinstance(name, str) and name for name in names)
        for cluster_id, names in clusters.items()
    )


def _atomic_write_json(
    path: Path,
    data: dict[str, object],
    *,
    operation: str,
    canonical: bool = False,
) -> None:
    """Replace authoritative JSON atomically using a same-directory temp file."""
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            if canonical:
                handle.write(_canonical_json_bytes(data).decode("utf-8"))
            else:
                json.dump(data, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except (OSError, TypeError) as error:
        raise ProbeError(
            "Atomic provenance update failed in "
            "emotion_probing.main._atomic_write_json while "
            f"{operation} at {path}. The replacement did not complete, so callers "
            "must treat the attempted update as unrecorded; the previous "
            "run_info.json remains authoritative. Check directory permissions and "
            f"free disk space, then retry. Underlying error: {error}"
        ) from error
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


@contextmanager
def _run_writer_lock(run_dir: Path) -> Iterator[None]:
    """Acquire one atomic, fail-fast writer lock for a single run directory."""
    lock_path = run_dir / RUN_WRITER_LOCK_NAME
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise ProbeError(
            "Run writer lock acquisition failed immediately in "
            "emotion_probing.main._run_writer_lock before resume checks or result "
            f"mutation because {lock_path} already exists. Another process may be "
            "writing this run, so appending now could corrupt CSV/provenance "
            "consistency. Retry after that writer finishes; if no process owns the "
            "stale lock, remove that exact lock file and retry."
        ) from error
    except OSError as error:
        raise ProbeError(
            "Run writer lock acquisition failed in "
            "emotion_probing.main._run_writer_lock before result mutation "
            f"because {lock_path} could not be created atomically. No scores "
            "or provenance were changed. Check directory permissions and free "
            f"disk space, then retry. Underlying error: {error}"
        ) from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
    except OSError as error:
        os.close(descriptor)
        try:
            lock_path.unlink()
        except OSError:
            pass
        raise ProbeError(
            "Run writer lock initialization failed in "
            "emotion_probing.main._run_writer_lock after atomic acquisition "
            f"because ownership metadata could not be written to {lock_path}. "
            "The lock was released and no run files were changed. Check filesystem "
            f"health and retry. Underlying error: {error}"
        ) from error
    try:
        yield
    finally:
        close_error: OSError | None = None
        try:
            os.close(descriptor)
        except OSError as error:
            close_error = error
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as error:
            raise ProbeError(
                "Run writer lock cleanup failed in "
                "emotion_probing.main._run_writer_lock after the writer stopped "
                f"because {lock_path} could not be removed. The run files are "
                "closed, but future writers will be blocked. Remove that stale "
                f"lock after confirming no writer is active. Underlying error: {error}"
            ) from error
        if close_error is not None:
            raise ProbeError(
                "Run writer lock descriptor cleanup failed in "
                "emotion_probing.main._run_writer_lock after the lock file was "
                "released. Run files are closed and future writers are not "
                "blocked, but the caller should verify operating-system resource "
                f"health before retrying. Underlying error: {close_error}"
            ) from close_error


_RESUME_INVARIANTS = (
    "experiment",
    "dataset",
    "model_id",
    "provider_id",
    "runtime_id",
    "loader",
    "route",
    "repository",
    "revision",
    "quantization",
    "quantization_settings",
    "model_compute_dtype",
    "attention_implementation",
    "package_versions",
    "python_version",
    "platform",
    "pytorch_version",
    "cuda_runtime_version",
    "requested_placement",
    "resolved_device_map",
    "cpu_modules",
    "disk_offload_modules",
    "has_cpu_or_offload",
    "probe_layer",
    "activation_width",
    "token_limit",
    "batch_size",
    "use_cache",
    "inference_mode",
    "vectors",
    "vectors_revision",
    "dataset_manifest_sha256",
    "vector_source_path",
    "vector_source_sha256",
    "cluster_analysis_path",
    "cluster_analysis_sha256",
    "cluster_snapshot_expected",
    "cluster_snapshot_sha256",
    "probe_implementation_source",
    "probe_implementation_sha256",
    "probe_implementation_revision",
    "historical_extraction_model_revision",
    "limit",
)


def _require_resume_compatible(
    run_dir: Path,
    current: dict[str, object],
    expected_columns: Sequence[str],
    key_columns: Sequence[str],
) -> set[tuple[str, ...]]:
    """Validate a prior run without mutating its provenance or measurements."""
    info_path = run_dir / "run_info.json"
    try:
        previous = json.loads(info_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ProbeError(
            "Resume compatibility validation failed in "
            "emotion_probing.main._require_resume_compatible before any run file "
            f"was modified because {info_path} is missing, unreadable, or invalid "
            "JSON. Existing CSV rows cannot be assigned trustworthy provenance. "
            "Keep this run unchanged and start a new run without `--resume`, or "
            f"restore its original run_info.json. Underlying error: {error}"
        ) from error
    if not isinstance(previous, dict):
        raise ProbeError(
            "Resume compatibility validation failed in "
            "emotion_probing.main._require_resume_compatible before any run file "
            f"was modified because {info_path} is not a JSON object. Existing "
            "scores cannot be resumed safely; restore the original provenance or "
            "start a new run without `--resume`."
        )
    mismatches = [
        key for key in _RESUME_INVARIANTS if previous.get(key) != current.get(key)
    ]
    scores_path = run_dir / "scores.csv"
    if scores_path.exists():
        try:
            with scores_path.open(encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle), None)
        except OSError as error:
            raise ProbeError(
                "Resume compatibility validation failed in "
                "emotion_probing.main._require_resume_compatible before any run "
                f"file was modified because {scores_path} could not be read. The "
                "caller cannot safely append scores; repair file permissions or "
                f"start a new run. Underlying error: {error}"
            ) from error
        if header != list(expected_columns):
            mismatches.append("scores.csv columns")
    if mismatches:
        details = ", ".join(mismatches)
        raise ProbeError(
            "Resume compatibility validation failed in "
            "emotion_probing.main._require_resume_compatible before any run file "
            f"was modified because these scientific invariants differ: {details}. "
            "Appending would mix incompatible measurements while retaining old "
            "CSV rows. Start a new run without `--resume`; the existing run and "
            "its provenance were left unchanged."
        )

    _validate_cluster_snapshot(run_dir, current)
    return _validated_completed_keys(scores_path, expected_columns, key_columns)


def _validate_cluster_snapshot(run_dir: Path, current: dict[str, object]) -> None:
    path = run_dir / "clusters.json"
    expected = current.get("cluster_snapshot_expected") is True
    expected_digest = current.get("cluster_snapshot_sha256")
    if not expected:
        if path.exists():
            raise ProbeError(
                "Resume cluster validation failed in "
                "emotion_probing.main._validate_cluster_snapshot before mutation "
                f"because unexpected {path} exists for a route with no cluster "
                "snapshot. Start a new run; the existing run was left unchanged."
            )
        return
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise ProbeError(
            "Resume cluster validation failed in "
            "emotion_probing.main._validate_cluster_snapshot before mutation "
            f"because required {path} is missing, unreadable, or invalid JSON. "
            "Restore the original canonical snapshot or start a new run; no run "
            f"file was changed. Underlying error: {error}"
        ) from error
    if not _is_cluster_snapshot(payload):
        raise ProbeError(
            "Resume cluster validation failed in "
            "emotion_probing.main._validate_cluster_snapshot before mutation "
            f"because {path} does not have the canonical clusters object schema. "
            "Restore it or start a new run; no run file was changed."
        )
    canonical = _canonical_json_bytes(payload)
    actual_digest = hashlib.sha256(canonical).hexdigest()
    if raw != canonical or actual_digest != expected_digest:
        raise ProbeError(
            "Resume cluster validation failed in "
            "emotion_probing.main._validate_cluster_snapshot before mutation "
            f"because {path} is non-canonical or its digest differs from "
            "cluster_snapshot_sha256. Restore the exact original snapshot or start "
            "a new run; no run file was changed."
        )


def _validated_completed_keys(
    scores_path: Path,
    expected_columns: Sequence[str],
    key_columns: Sequence[str],
) -> set[tuple[str, ...]]:
    if not scores_path.exists():
        return set()
    expected = list(expected_columns)
    key_indexes = [expected.index(column) for column in key_columns]
    token_index = expected.index("n_tokens")
    score_indexes = [
        index for index, column in enumerate(expected) if column.startswith("score_")
    ]
    completed: set[tuple[str, ...]] = set()
    try:
        with scores_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header != expected:
                raise ValueError("header/schema mismatch")
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(expected):
                    raise ValueError(
                        f"row {line_number} has {len(row)} fields; expected {len(expected)}"
                    )
                key = tuple(row[index] for index in key_indexes)
                if any(not value.strip() for value in key):
                    raise ValueError(f"row {line_number} has a blank task key")
                if key in completed:
                    raise ValueError(f"row {line_number} duplicates task key {key!r}")
                try:
                    token_count = int(row[token_index])
                except ValueError as error:
                    raise ValueError(
                        f"row {line_number} has a non-integer n_tokens value"
                    ) from error
                if token_count <= 0:
                    raise ValueError(f"row {line_number} has non-positive n_tokens")
                for index in score_indexes:
                    try:
                        score = float(row[index])
                    except ValueError as error:
                        raise ValueError(
                            f"row {line_number} has a non-numeric {expected[index]} value"
                        ) from error
                    if not math.isfinite(score):
                        raise ValueError(
                            f"row {line_number} has a non-finite {expected[index]} value"
                        )
                completed.add(key)
    except (OSError, ValueError) as error:
        raise ProbeError(
            "Resume CSV validation failed in "
            "emotion_probing.main._validated_completed_keys before mutation "
            f"because {scores_path} is not a complete, unique, finite instance of "
            f"the recorded schema: {error}. Repair or restore the CSV, or start a "
            "new run; existing files were left byte-identical."
        ) from error
    return completed


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise ProbeError(
            "Runtime provenance collection failed in "
            "emotion_probing.main._package_version before the experiment loop "
            f"because required distribution {distribution!r} is not installed. "
            "No measurements started; run `uv sync --locked` and retry."
        ) from error


def _sha256_file(path: Path, *, purpose: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as error:
        raise ProbeError(
            "Input fingerprinting failed in emotion_probing.main._sha256_file "
            f"before run creation while hashing {purpose} at {path}. Scientific "
            "input identity cannot be recorded, so no experiment loop started. "
            f"Restore/read-enable the reviewed file and retry. Underlying error: {error}"
        ) from error
    return digest.hexdigest()


def _dataset_manifest_sha256(
    key_columns: Sequence[str],
    metadata_columns: Sequence[str],
    tasks: Sequence[dict[str, object]],
) -> str:
    digest = hashlib.sha256()
    values: Sequence[object] = [list(key_columns), list(metadata_columns), *tasks]
    try:
        for value in values:
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    except (TypeError, ValueError) as error:
        raise ProbeError(
            "Dataset fingerprinting failed in "
            "emotion_probing.main._dataset_manifest_sha256 before run creation "
            "because the finalized task manifest is not deterministic JSON. No "
            "experiment loop started; normalize task values to finite JSON scalar "
            f"types and retry. Underlying error: {error}"
        ) from error
    return digest.hexdigest()


def build_input_fingerprints(
    config: ExperimentConfig,
    key_columns: Sequence[str],
    metadata_columns: Sequence[str],
    tasks: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Fingerprint finalized scientific inputs with bounded streaming hashes."""
    if config.vectors is VectorSource.GEMOTIONS:
        vector_path = (
            GEMOTIONS_DIR
            / "results"
            / "gemma4-31b"
            / f"emotion_vectors_layer{config.probe_layer}.npz"
        )
        cluster_path: Path | None = GEMOTIONS_ANALYSIS_FILE
    elif config.vectors in (VectorSource.EXTRACTED_BASE, VectorSource.EXTRACTED_E4B):
        vector_path = _extracted_vectors_path(config.vectors)
        cluster_path = GEMOTIONS_ANALYSIS_FILE
    else:
        vector_path = EMOTIONSCOPE_VECTORS_FILE
        cluster_path = None
    probe_hash = _sha256_file(PROBE_SOURCE_FILE, purpose="probe implementation")
    clusters = load_clusters(config)
    return {
        "dataset_manifest_sha256": _dataset_manifest_sha256(
            key_columns, metadata_columns, tasks
        ),
        "vector_source_path": str(vector_path.relative_to(PACKAGE_DIR.parent)),
        "vector_source_sha256": _sha256_file(vector_path, purpose="emotion vectors"),
        "cluster_analysis_path": (
            str(cluster_path.relative_to(PACKAGE_DIR.parent))
            if cluster_path is not None
            else None
        ),
        "cluster_analysis_sha256": (
            _sha256_file(cluster_path, purpose="cluster analysis")
            if cluster_path is not None
            else None
        ),
        "cluster_snapshot_expected": clusters is not None,
        "cluster_snapshot_sha256": (
            _canonical_json_sha256(clusters) if clusters is not None else None
        ),
        "probe_implementation_source": str(
            PROBE_SOURCE_FILE.relative_to(PACKAGE_DIR.parent)
        ),
        "probe_implementation_sha256": probe_hash,
        "probe_implementation_revision": f"sha256:{probe_hash}",
        "historical_extraction_model_revision": "unknown",
    }


def build_run_provenance(
    config: ExperimentConfig,
    route: LocalTransformersRoute,
    runtime: LocalActivationRuntime,
    limit: int | None,
    input_fingerprints: dict[str, object],
) -> dict[str, object]:
    """Build exact static/runtime provenance before measured work starts."""
    bnb = route.bitsandbytes
    quantization_settings: dict[str, object] | None = None
    if bnb is not None:
        quantization_settings = {
            "load_in_4bit": bnb.load_in_4bit,
            "bnb_4bit_quant_type": bnb.quant_type.value,
            "bnb_4bit_compute_dtype": bnb.compute_dtype.value,
            "bnb_4bit_quant_storage": bnb.quant_storage.value,
            "bnb_4bit_use_double_quant": bnb.use_double_quant,
        }
    return {
        "experiment": config.name.value,
        "dataset": config.dataset.value,
        "model_id": config.model_id.value,
        "provider_id": ProviderId.LOCAL.value,
        "runtime_id": route.runtime_id.value,
        "loader": route.loader.value,
        "route": (
            f"{config.model_id.value}/{ProviderId.LOCAL.value}/"
            f"{config.quantization_id.value}"
        ),
        "repository": route.artifact.repository,
        "revision": route.artifact.revision,
        "quantization": config.quantization_id.value,
        "quantization_settings": quantization_settings,
        "model_compute_dtype": "bfloat16",
        "attention_implementation": "sdpa",
        "package_versions": {
            name: _package_version(name)
            for name in ("torch", "transformers", "accelerate", "bitsandbytes")
        },
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "requested_placement": runtime.placement.requested.value,
        "resolved_device_map": dict(runtime.placement.resolved_device_map),
        "cpu_modules": list(runtime.placement.cpu_modules),
        "disk_offload_modules": list(runtime.placement.disk_modules),
        "has_cpu_or_offload": runtime.placement.has_cpu_or_offload,
        "probe_layer": config.probe_layer,
        "activation_width": config.expected_width,
        "token_limit": config.token_limit,
        "batch_size": config.batch_size,
        "use_cache": config.use_cache,
        "prompt_style": config.prompt_style,
        "inference_mode": True,
        "vectors": config.vectors.value,
        "vectors_revision": (
            GEMOTIONS_SOURCE_REVISION
            if config.vectors is VectorSource.GEMOTIONS
            else BASE_VECTORS_RUN
            if config.vectors is VectorSource.EXTRACTED_BASE
            else E4B_VECTORS_RUN
            if config.vectors is VectorSource.EXTRACTED_E4B
            else None
        ),
        "limit": limit,
        "cuda_peak_memory_measured": False,
        "cuda_peak_allocated_bytes": None,
        "cuda_peak_reserved_bytes": None,
        **input_fingerprints,
    }


def validate_probe_setup(
    runtime: LocalActivationRuntime,
    config: ExperimentConfig,
    vector_matrix: torch.Tensor,
) -> None:
    """Reject incompatible architecture/vector metadata before iteration."""
    _decoder_block(runtime.model, config.probe_layer)
    model_config = getattr(runtime.model, "config", None)
    text_config = getattr(model_config, "text_config", model_config)
    configured_width = getattr(text_config, "hidden_size", None)
    if not isinstance(configured_width, int):
        raise ProbeError(
            "Probe setup validation failed in emotion_probing.main.validate_probe_setup "
            "before the experiment loop because the loaded model config does not "
            "declare an integer text hidden_size. Activation width cannot be "
            "verified, so no tasks ran; install the lock and load the exact "
            "registered revision."
        )
    if configured_width != config.expected_width:
        raise ProbeError(
            "Probe setup validation failed in emotion_probing.main.validate_probe_setup "
            f"before the experiment loop because the loaded model reports width "
            f"{configured_width}, not the reviewed {config.expected_width}. No "
            "tasks ran; load the exact registered model revision and retry."
        )
    if config.batch_size != 1 or config.use_cache:
        raise ProbeError(
            "Probe setup validation failed in emotion_probing.main.validate_probe_setup "
            "before the experiment loop because the reviewed path requires batch "
            "one and use_cache=False. No tasks ran; restore the static experiment "
            "configuration."
        )
    if vector_matrix.ndim != 2 or vector_matrix.shape[1] != config.expected_width:
        raise ProbeError(
            "Probe setup validation failed in emotion_probing.main.validate_probe_setup "
            f"before the experiment loop because vectors have shape "
            f"{tuple(vector_matrix.shape)}, not (emotions, {config.expected_width}). "
            "No tasks ran; use the reviewed vectors for the exact model and layer."
        )


def _uses_cuda(runtime: LocalActivationRuntime) -> bool:
    return any(
        target in {"0", "cuda", "cuda:0"}
        for _, target in runtime.placement.resolved_device_map
    )


def reset_cuda_peaks(runtime: LocalActivationRuntime) -> bool:
    """Reset CUDA peaks immediately before measured forwards, if CUDA is used."""
    if not _uses_cuda(runtime):
        return False
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    except RuntimeError as error:
        raise ProbeError(
            "CUDA peak reset failed in emotion_probing.main.reset_cuda_peaks "
            "immediately before the measured workload. The run did not start and "
            "memory provenance would be invalid; repair the CUDA runtime and retry. "
            f"Underlying error: {error}"
        ) from error
    return True


def persist_cuda_peaks(run_dir: Path, measured: bool) -> None:
    """Synchronize and persist genuine CUDA peaks; retain nulls off CUDA."""
    if not measured:
        return
    info_path = run_dir / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    try:
        torch.cuda.synchronize()
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
    except RuntimeError as error:
        raise ProbeError(
            "CUDA peak collection failed in "
            "emotion_probing.main.persist_cuda_peaks after the measured "
            "workload. Results may exist but memory acceptance is unknown; "
            "repair CUDA and rerun the smoke test. "
            f"Underlying error: {error}"
        ) from error
    old_allocated = info.get("cuda_peak_allocated_bytes")
    old_reserved = info.get("cuda_peak_reserved_bytes")
    info["cuda_peak_memory_measured"] = True
    info["cuda_peak_allocated_bytes"] = max(
        allocated, old_allocated if isinstance(old_allocated, int) else 0
    )
    info["cuda_peak_reserved_bytes"] = max(
        reserved, old_reserved if isinstance(old_reserved, int) else 0
    )
    _atomic_write_json(
        info_path,
        info,
        operation="persisting synchronized CUDA peak measurements",
    )


def run_probe(
    experiment: ExperimentId,
    cache_dir: Path,
    device: Device,
    limit: int | None,
    resume: bool,
) -> None:
    """Score every dataset task and write scores.csv into the run folder."""
    config = EXPERIMENTS[experiment]
    route = resolve_probe_route(config)
    names, vector_matrix = load_vectors(config, route)
    runtime = create_transformers_runtime(route, cache_dir=cache_dir, device=device)
    validate_probe_setup(runtime, config, vector_matrix)
    key_columns, metadata_columns, tasks = load_dataset(config, limit)
    validate_prompt_bounds_before_iteration(
        runtime,
        [task["messages"] for task in tasks],
        config.token_limit,
        prompt_style=config.prompt_style,
    )
    input_fingerprints = build_input_fingerprints(
        config, key_columns, metadata_columns, tasks
    )
    provenance = build_run_provenance(config, route, runtime, limit, input_fingerprints)
    columns = metadata_columns + ["n_tokens"] + [f"score_{name}" for name in names]
    run_dir = prepare_run_dir(config, resume)
    scores_file = run_dir / "scores.csv"
    with _run_writer_lock(run_dir):
        if resume:
            done = _require_resume_compatible(run_dir, provenance, columns, key_columns)
        else:
            _stage_new_run(config, run_dir, provenance)
            done = set()
        pending_tasks = [
            task
            for task in tasks
            if tuple(str(task["row"][column]) for column in key_columns) not in done
        ]
        completed_count = len(tasks) - len(pending_tasks)
        print(f"Run folder: {run_dir} ({completed_count} tasks already scored)")

        write_header = not scores_file.exists()
        has_pending_tasks = bool(pending_tasks)
        measured_cuda = reset_cuda_peaks(runtime) if has_pending_tasks else False
        try:
            with scores_file.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                if write_header:
                    writer.writeheader()
                with tqdm(
                    pending_tasks,
                    total=len(tasks),
                    initial=completed_count,
                    desc="Scoring",
                    unit="task",
                ) as progress:
                    for task in progress:
                        row = dict(task["row"])
                        scores, n_tokens = emotion_scores(
                            runtime,
                            task["messages"],
                            vector_matrix,
                            config.probe_layer,
                            expected_width=config.expected_width,
                            token_limit=config.token_limit,
                            prompt_style=config.prompt_style,
                        )
                        row["n_tokens"] = n_tokens
                        row |= dict(zip([f"score_{name}" for name in names], scores))
                        writer.writerow(row)
                        handle.flush()
        finally:
            persist_cuda_peaks(run_dir, measured_cuda)
    print(f"Results written to {scores_file}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Probe emotion-vector activations on rude vs normal inputs."
    )
    parser.add_argument(
        "--experiment",
        type=ExperimentId,
        choices=sorted(EXPERIMENTS),
        default=DEFAULT_EXPERIMENT,
        help="experiment configuration (default: %(default)s)",
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
        help="download the experiment's pinned checkpoint (the 2B repo is "
        "gated: accept the Gemma license and `hf auth login` first)",
    )
    run_parser = subparsers.add_parser("run", help="run the probing experiment")
    run_parser.add_argument(
        "--device",
        type=Device,
        choices=list(Device),
        default=Device.AUTO,
        help="inference device (default: %(default)s)",
    )
    run_parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="only process the first N tasks (for smoke tests)",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the latest run folder instead of creating a new one",
    )
    return parser


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--limit must be a positive integer"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("--limit must be a positive integer")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "download":
            route = resolve_probe_route(EXPERIMENTS[args.experiment])
            model_path = download_transformers_artifact(route, args.cache_dir)
            print(f"Downloaded pinned checkpoint to: {model_path}")
        else:
            run_probe(
                args.experiment, args.cache_dir, args.device, args.limit, args.resume
            )
    except (ModelRouteError, TransformersRuntimeError, ProbeError, DatasetError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
