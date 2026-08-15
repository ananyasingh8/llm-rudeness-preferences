"""Probe Gemma's emotion activations on normal vs rude BailBench prompts.

The experiment: for every BailBench prompt we have a normal version and a rude
version (read from bail/data/bailbench_augmented.csv at the repo root, so
dataset updates in the bail workstream are picked up automatically). Each
version is formatted with the chat
template and run through the model with a single forward pass (no generation).
We read the residual stream at the last prompt token — the position right before
the model starts its reply, the analog of the ":" after "Assistant" in
Anthropic's emotion-concepts paper — and take the cosine similarity of that
activation against the 20 pre-extracted EmotionScope emotion vectors. Comparing
the scores between the rude and normal version of the same prompt shows which
emotion representations rudeness activates.

Models are selected through the shared llm_runtime closed registry. The emotion
vectors were extracted from google/gemma-2-2b-it at layer 22, so this experiment
resolves that model's registered local BF16 route and requires local activation
access. Change the constants below to target a different registered route (a
new model also needs its own matching vectors file — the run refuses a
vectors/model mismatch).

Usage (same shape as quadratic_voting):
    uv run python -m emotion_probing.main download
    uv run python -m emotion_probing.main run [--device auto|cuda|cpu] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub.constants import HF_HUB_CACHE

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

# --- Experiment constants: edit these to change the model setup. -------------
# The triple must be a registered llm_runtime route with local activation
# access, and VECTORS_FILE must hold vectors extracted from that same model.
MODEL_ID = ModelId.GEMMA_2_2B_IT
PROVIDER_ID = ProviderId.LOCAL
QUANTIZATION_ID = QuantizationId.BF16

PACKAGE_DIR = Path(__file__).parent
VECTORS_FILE = (
    PACKAGE_DIR / "EmotionScope" / "results" / "vectors" / "google_gemma-2-2b-it.pt"
)
DATASET_FILE = PACKAGE_DIR.parent / "bail" / "data" / "bailbench_augmented.csv"
RESULTS_FILE = PACKAGE_DIR / "results" / "scores.csv"

METADATA_COLUMNS = [
    "bailbench_id",
    "condition",
    "rudeness_type",
    "rudeness_name",
    "category",
    "subcategory",
    "n_tokens",
]


class ProbeError(RuntimeError):
    """A user-actionable failure in the probing workflow."""


def resolve_probe_route() -> LocalTransformersRoute:
    """Resolve the configured route, requiring local activation access."""
    route = resolve_route(
        MODEL_ID,
        PROVIDER_ID,
        QUANTIZATION_ID,
        required={Capability.LOCAL_ACTIVATIONS},
    )
    if not isinstance(route, LocalTransformersRoute):
        raise ProbeError(
            f"The resolved route is {type(route).__name__}, not a local "
            "Transformers route. Emotion probing reads activations and can only "
            "run against a registered local route."
        )
    return route


def load_vectors(route: LocalTransformersRoute) -> tuple[list[str], torch.Tensor, int]:
    """Load the EmotionScope emotion vectors and check they match the model.

    Returns the 20 emotion names, a (20, 2304) unit-normalized matrix, and the
    layer the vectors were extracted from (22 for gemma-2-2b-it).
    """
    if not VECTORS_FILE.exists():
        raise ProbeError(f"Emotion vectors file not found: {VECTORS_FILE}")
    saved = torch.load(VECTORS_FILE, weights_only=False, map_location="cpu")
    vectors_model = saved["model_info"]["model_name"]
    if vectors_model != route.artifact.repository:
        raise ProbeError(
            f"The emotion vectors were extracted from {vectors_model} but the "
            f"configured route loads {route.artifact.repository}. Emotion "
            "vectors are model-specific; use the matching model or re-extract "
            "vectors with EmotionScope."
        )
    names = list(saved["vectors"].keys())
    matrix = F.normalize(
        torch.stack([saved["vectors"][name].float() for name in names]), dim=1
    )
    return names, matrix, int(saved["probe_layer_used"])


def emotion_scores(
    runtime: LocalActivationRuntime,
    prompt: str,
    vector_matrix: torch.Tensor,
    probe_layer: int,
) -> tuple[list[float], int]:
    """Run one forward pass and score the response-start activation.

    Returns the cosine similarity to each emotion vector, plus the prompt's
    token count. hidden_states[0] is the embedding output, so the residual
    stream after block `probe_layer` is hidden_states[probe_layer + 1].
    """
    inputs = runtime.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
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


def load_dataset(limit: int | None) -> list[dict[str, str]]:
    """Read the paired normal/rude dataset."""
    if not DATASET_FILE.exists():
        raise ProbeError(f"Dataset file not found: {DATASET_FILE}")
    with DATASET_FILE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit is not None else rows


def completed_keys() -> set[tuple[str, str]]:
    """Return (bailbench_id, condition) pairs already present in the results."""
    if not RESULTS_FILE.exists():
        return set()
    with RESULTS_FILE.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {(row["bailbench_id"], row["condition"]) for row in reader}


def run_probe(cache_dir: Path, device: Device, limit: int | None) -> None:
    """Score every prompt pair and append rows to results/scores.csv."""
    route = resolve_probe_route()
    names, vector_matrix, probe_layer = load_vectors(route)
    rows = load_dataset(limit)
    done = completed_keys()
    runtime = create_transformers_runtime(route, cache_dir=cache_dir, device=device)

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    columns = METADATA_COLUMNS + [f"score_{name}" for name in names]
    write_header = not RESULTS_FILE.exists()
    with RESULTS_FILE.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if write_header:
            writer.writeheader()
        for index, row in enumerate(rows):
            for condition, prompt in (
                ("normal", row["original_prompt"]),
                ("rude", row["augmented_prompt"]),
            ):
                if (row["bailbench_id"], condition) in done:
                    continue
                scores, n_tokens = emotion_scores(
                    runtime, prompt, vector_matrix, probe_layer
                )
                writer.writerow(
                    {
                        "bailbench_id": row["bailbench_id"],
                        "condition": condition,
                        "rudeness_type": row["rudeness_type"],
                        "rudeness_name": row["rudeness_name"],
                        "category": row["category"],
                        "subcategory": row["subcategory"],
                        "n_tokens": n_tokens,
                    }
                    | dict(zip([f"score_{name}" for name in names], scores))
                )
            handle.flush()
            if (index + 1) % 50 == 0 or index + 1 == len(rows):
                print(f"Scored {index + 1}/{len(rows)} prompt pairs")
    print(f"Results written to {RESULTS_FILE}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Probe emotion-vector activations on normal vs rude prompts."
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
        help="download the pinned checkpoint (gated repo: accept the Gemma "
        "license on Hugging Face and `hf auth login` first)",
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
        help="only process the first N prompt pairs (for smoke tests)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "download":
            route = resolve_probe_route()
            model_path = download_transformers_artifact(route, args.cache_dir)
            print(f"Downloaded pinned checkpoint to: {model_path}")
        else:
            run_probe(args.cache_dir, args.device, args.limit)
    except (ModelRouteError, TransformersRuntimeError, ProbeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
