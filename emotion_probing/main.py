"""Probe Gemma's emotion activations on rude/abusive vs normal user messages.

The experiment: run each dataset item through the model with a single forward
pass (no generation), read the residual stream at the last prompt token — the
position right before the model starts its reply, the analog of the ":" after
"Assistant" in Anthropic's emotion-concepts paper — and take the cosine
similarity of that activation against pre-extracted emotion vectors. Comparing
scores between rude/abusive and normal inputs shows which emotion
representations mistreatment activates.

Two registered experiment configurations (see EXPERIMENTS below):

- **bailbench-2b**: google/gemma-2-2b-it scored against EmotionScope's 20
  emotion vectors (layer 22) on 1,630 synthetic normal/rude prompt pairs.
- **convabuse-31b**: google/gemma-4-31B-it (W4A16 quantized) scored against
  the 171 gemotions emotion vectors (layer 40) on 4,185 real, human-annotated
  user-bot conversation snippets from ConvAbuse.

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
    uv run python -m emotion_probing.main download [--experiment NAME]
    uv run python -m emotion_probing.main run [--experiment NAME]
        [--device auto|cuda|cpu] [--limit N] [--resume]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub.constants import HF_HUB_CACHE

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


@dataclass(frozen=True)
class ExperimentConfig:
    """One reviewed pairing of model route, emotion vectors, and dataset."""

    name: str
    model_id: ModelId
    quantization_id: QuantizationId
    probe_layer: int
    vectors: str  # "emotionscope" | "gemotions"
    dataset: str  # "bailbench" | "convabuse"


# --- Experiment constants: edit these to change the model setups. ------------
EXPERIMENTS = {
    "bailbench-2b": ExperimentConfig(
        name="bailbench-2b",
        model_id=ModelId.GEMMA_2_2B_IT,
        quantization_id=QuantizationId.BF16,
        probe_layer=22,
        vectors="emotionscope",
        dataset="bailbench",
    ),
    "convabuse-31b": ExperimentConfig(
        name="convabuse-31b",
        model_id=ModelId.GEMMA_4_31B_IT,
        quantization_id=QuantizationId.W4A16_COMPRESSED_TENSORS,
        probe_layer=40,
        vectors="gemotions",
        dataset="convabuse",
    ),
}
DEFAULT_EXPERIMENT = "convabuse-31b"

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
        raise ProbeError(
            f"Emotion vectors file not found: {EMOTIONSCOPE_VECTORS_FILE}"
        )
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
        torch.stack(
            [torch.from_numpy(saved[name]).float() for name in names]
        ),
        dim=1,
    )
    return names, matrix


def load_vectors(
    config: ExperimentConfig, route: LocalTransformersRoute
) -> tuple[list[str], torch.Tensor]:
    """Load the unit-normalized emotion-vector matrix for this experiment."""
    if config.vectors == "emotionscope":
        return _load_emotionscope_vectors(route, config.probe_layer)
    if config.vectors == "gemotions":
        return _load_gemotions_vectors(config.probe_layer)
    raise ProbeError(f"Unknown vectors source {config.vectors!r}.")


def load_clusters(config: ExperimentConfig) -> dict | None:
    """Load the vendored gemotions cluster analysis for the configured layer."""
    if config.vectors != "gemotions":
        return None
    if not GEMOTIONS_ANALYSIS_FILE.exists():
        raise ProbeError(
            f"Cluster analysis file not found: {GEMOTIONS_ANALYSIS_FILE}."
        )
    with GEMOTIONS_ANALYSIS_FILE.open(encoding="utf-8") as handle:
        analysis = json.load(handle)
    layer = str(config.probe_layer)
    if layer not in analysis:
        raise ProbeError(
            f"The gemotions analysis file has no layer {layer} entry."
        )
    return {"clusters": analysis[layer]["clusters"]}


def load_dataset(config: ExperimentConfig, limit: int | None):
    """Load the experiment's dataset as (key_columns, metadata_columns, tasks)."""
    if config.dataset == "bailbench":
        return load_bailbench(limit)
    if config.dataset == "convabuse":
        return load_convabuse(limit)
    raise ProbeError(f"Unknown dataset {config.dataset!r}.")


def emotion_scores(
    runtime: LocalActivationRuntime,
    messages: list[dict[str, str]],
    vector_matrix: torch.Tensor,
    probe_layer: int,
) -> tuple[list[float], int]:
    """Run one forward pass and score the response-start activation.

    Returns the cosine similarity to each emotion vector, plus the prompt's
    token count. hidden_states[0] is the embedding output, so the residual
    stream after block `probe_layer` is hidden_states[probe_layer + 1] — the
    same convention both vector sources used during extraction.
    """
    inputs = runtime.tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(runtime.model.device)
    with torch.inference_mode():
        output = runtime.model(**inputs, output_hidden_states=True, use_cache=False)
    activation = output.hidden_states[probe_layer + 1][0, -1, :].float().cpu()
    scores = F.normalize(activation, dim=0) @ vector_matrix.T
    return [float(score) for score in scores], int(inputs["input_ids"].shape[-1])


def latest_run_dir(experiment: str) -> Path | None:
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
    route: LocalTransformersRoute,
    limit: int | None,
    resume: bool,
) -> Path:
    """Create a fresh timestamped run folder, or return the latest for resume."""
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
    run_dir.mkdir(parents=True)
    run_info = {
        "experiment": config.name,
        "dataset": config.dataset,
        "model_id": config.model_id.value,
        "repository": route.artifact.repository,
        "revision": route.artifact.revision,
        "quantization": config.quantization_id.value,
        "probe_layer": config.probe_layer,
        "vectors": config.vectors,
        "vectors_revision": (
            GEMOTIONS_SOURCE_REVISION if config.vectors == "gemotions" else None
        ),
        "limit": limit,
        "started": stamp,
    }
    (run_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    clusters = load_clusters(config)
    if clusters is not None:
        (run_dir / "clusters.json").write_text(
            json.dumps(clusters), encoding="utf-8"
        )
    return run_dir


def completed_keys(scores_file: Path, key_columns: list[str]) -> set[tuple[str, ...]]:
    """Return task keys already present in a run's scores.csv."""
    if not scores_file.exists():
        return set()
    with scores_file.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {tuple(row[column] for column in key_columns) for row in reader}


def run_probe(
    experiment: str,
    cache_dir: Path,
    device: Device,
    limit: int | None,
    resume: bool,
) -> None:
    """Score every dataset task and write scores.csv into the run folder."""
    config = EXPERIMENTS[experiment]
    route = resolve_probe_route(config)
    key_columns, metadata_columns, tasks = load_dataset(config, limit)
    names, vector_matrix = load_vectors(config, route)
    run_dir = prepare_run_dir(config, route, limit, resume)
    scores_file = run_dir / "scores.csv"
    done = completed_keys(scores_file, key_columns)
    print(f"Run folder: {run_dir} ({len(done)} tasks already scored)")

    runtime = create_transformers_runtime(route, cache_dir=cache_dir, device=device)
    columns = metadata_columns + ["n_tokens"] + [f"score_{name}" for name in names]
    write_header = not scores_file.exists()
    with scores_file.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if write_header:
            writer.writeheader()
        for index, task in enumerate(tasks):
            row = dict(task["row"])
            key = tuple(str(row[column]) for column in key_columns)
            if key in done:
                continue
            scores, n_tokens = emotion_scores(
                runtime, task["messages"], vector_matrix, config.probe_layer
            )
            row["n_tokens"] = n_tokens
            row |= dict(zip([f"score_{name}" for name in names], scores))
            writer.writerow(row)
            handle.flush()
            if (index + 1) % 50 == 0 or index + 1 == len(tasks):
                print(f"Scored {index + 1}/{len(tasks)} tasks")
    print(f"Results written to {scores_file}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Probe emotion-vector activations on rude vs normal inputs."
    )
    parser.add_argument(
        "--experiment",
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
        type=int,
        default=None,
        help="only process the first N tasks (for smoke tests)",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the latest run folder instead of creating a new one",
    )
    return parser


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
