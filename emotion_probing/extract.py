"""Extract emotion vectors from the base (non-instruction-tuned) Gemma 4 31B.

Replicates the gemotions extraction method (vendored in gemotions/, see
extract_vectors.py there) at reduced scale, against the base model instead of
the instruction-tuned one. Per story: tokenize the raw prose (no chat
template — base models have none, and the original extraction also used raw
text), run one forward pass, and capture every decoder block in PROBE_LAYERS
at once — a forward pass computes all layers anyway, so sweeping layers costs
no extra GPU time. Each captured activation is mean-pooled (from START_TOKEN
onward) on the GPU before leaving it. Then, independently per layer:

    vector = mean(activation over its stories) - global mean over ALL stories

The global mean pools every sampled story from all 171 emotions (the 20 target
emotions at TARGET_STORIES each, the remaining 151 at GLOBAL_MEAN_STORIES each)
so the centering matches the original method's meaning. Finally the top
principal components of NEUTRAL_SAMPLE neutral-dialogue activations (explaining
NEUTRAL_VARIANCE of variance) are projected out, and each layer's 20 target
vectors are saved un-normalized in the same npz format the probe loader
consumes, under base-model-prefixed names so they can never be confused with
the vendored IT-model files.

The run ends with layer_quality.json: per layer, how cleanly opposite-valence
emotions point apart, how tightly synonyms agree, and (where a vendored
IT-model file exists for that layer) the cosine to the IT vectors. Pick the
best layer from that scorecard — prefer a layer on a stable plateau of good
scores over a lone spike — and use that layer's npz downstream.

Usage:
    uv run python -m emotion_probing.extract download   # base model + stories.db
    uv run python -m emotion_probing.extract run [--device auto|cuda|cpu]
        [--limit N] [--resume]
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
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

# --- Extraction constants: the scientific knobs, all in one place. -----------
MODEL_ID = ModelId.GEMMA_4_31B
QUANTIZATION_ID = QuantizationId.BITSANDBYTES_FP4
PROBE_LAYERS = tuple(range(20, 60))  # swept; pick the winner from layer_quality
HIDDEN_WIDTH = 5376
OUTPUT_PREFIX = "gemma4-31b-base"

# Top 10 risers and top 10 fallers by shift_band_avg (pooled bands -1..-3
# minus band 0) in run results/2026-08-15_052621_convabuse-31b-local-quant.
TARGET_RISERS = (
    "grumpy",
    "hostile",
    "mad",
    "insulted",
    "offended",
    "angry",
    "hateful",
    "upset",
    "sullen",
    "furious",
)
TARGET_FALLERS = (
    "jubilant",
    "awestruck",
    "amused",
    "delighted",
    "elated",
    "ecstatic",
    "excited",
    "thrilled",
    "euphoric",
    "amazed",
)
TARGET_EMOTIONS = TARGET_RISERS + TARGET_FALLERS
# Near-synonym pairs within the targets; a good layer keeps them aligned.
SYNONYM_PAIRS = (
    ("angry", "mad"),
    ("mad", "furious"),
    ("angry", "furious"),
    ("ecstatic", "elated"),
    ("delighted", "elated"),
    ("excited", "thrilled"),
    ("jubilant", "elated"),
)
TARGET_STORIES = 100  # stories per target emotion
GLOBAL_MEAN_STORIES = 10  # stories per non-target emotion (global mean only)
NEUTRAL_SAMPLE = 300  # neutral dialogues for denoising
SAMPLE_SEED = 42
# Both values below mirror the vendored gemotions extraction exactly.
MAX_STORY_TOKENS = 512
START_TOKEN = 50
NEUTRAL_VARIANCE = 0.5

PACKAGE_DIR = Path(__file__).parent
RESULTS_DIR = PACKAGE_DIR / "results"
GEMOTIONS_REPO = "dejanseo/gemotions"
GEMOTIONS_REVISION = "4fd2ac63551f1be37e6e6c2eacd1b1898c9af656"
STORIES_DB_FILE = "data/stories.db"
NEUTRAL_DB = PACKAGE_DIR / "gemotions" / "data" / "neutral.db"
IT_VECTORS_DIR = PACKAGE_DIR / "gemotions" / "results" / "gemma4-31b"


class ExtractError(RuntimeError):
    """A user-actionable failure in the extraction workflow."""


def resolve_extract_route() -> LocalTransformersRoute:
    """Resolve the base-model route, requiring local activation access."""
    route = resolve_route(
        MODEL_ID,
        ProviderId.LOCAL,
        QUANTIZATION_ID,
        required={Capability.LOCAL_ACTIVATIONS},
    )
    if not isinstance(route, LocalTransformersRoute):
        raise ExtractError(
            f"The resolved route is {type(route).__name__}, not a local "
            "Transformers route; extraction reads activations locally."
        )
    return route


def stories_db_path(cache_dir: Path, download: bool) -> Path:
    """Resolve (or fetch) the 433 MB gemotions story corpus in the HF cache."""
    try:
        return Path(
            hf_hub_download(
                GEMOTIONS_REPO,
                STORIES_DB_FILE,
                revision=GEMOTIONS_REVISION,
                cache_dir=cache_dir,
                local_files_only=not download,
            )
        )
    except Exception as error:  # huggingface_hub raises several types here
        raise ExtractError(
            f"The story corpus ({STORIES_DB_FILE}, ~433 MB) is not in the "
            f"cache at {cache_dir}. Run `uv run python -m "
            "emotion_probing.extract download` first. "
            f"Underlying error: {error}"
        ) from error


def load_story_samples(db_path: Path) -> dict[str, list[str]]:
    """Load a seeded story sample per emotion (targets get TARGET_STORIES)."""
    connection = sqlite3.connect(db_path, timeout=30)
    try:
        rows = connection.execute(
            "SELECT emotion, text FROM stories_clean ORDER BY emotion, story_idx"
        ).fetchall()
    finally:
        connection.close()
    stories: dict[str, list[str]] = {}
    for emotion, text in rows:
        stories.setdefault(emotion, []).append(text)
    missing = [e for e in TARGET_EMOTIONS if e not in stories]
    if missing:
        raise ExtractError(
            f"The story corpus has no stories for: {', '.join(missing)}. "
            "Check TARGET_EMOTIONS against the gemotions emotion names."
        )
    # Seeded sampling in a fixed iteration order keeps the run reproducible
    # and topic-diverse (first-N would over-sample the corpus's first topics).
    rng = random.Random(SAMPLE_SEED)
    samples: dict[str, list[str]] = {}
    for emotion in sorted(stories):
        count = (
            TARGET_STORIES if emotion in TARGET_EMOTIONS else GLOBAL_MEAN_STORIES
        )
        pool = stories[emotion]
        samples[emotion] = rng.sample(pool, min(count, len(pool)))
    return samples


def load_neutral_samples() -> list[str]:
    """Load a seeded sample of neutral dialogues from the vendored neutral.db."""
    if not NEUTRAL_DB.exists():
        raise ExtractError(f"Vendored neutral dialogues not found: {NEUTRAL_DB}")
    connection = sqlite3.connect(NEUTRAL_DB, timeout=30)
    try:
        rows = connection.execute(
            "SELECT text FROM dialogues ORDER BY topic_idx, dialogue_idx"
        ).fetchall()
    finally:
        connection.close()
    texts = [text for (text,) in rows]
    rng = random.Random(SAMPLE_SEED)
    return rng.sample(texts, min(NEUTRAL_SAMPLE, len(texts)))


def _decoder_blocks(model: torch.nn.Module) -> dict[int, torch.nn.Module]:
    """Resolve every probed decoder block (same layouts as the probe runner)."""
    base = getattr(model, "model", None)
    language_model = getattr(base, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        layers = getattr(base, "layers", None)
    if not isinstance(layers, (torch.nn.ModuleList, list, tuple)):
        raise ExtractError(
            "The loaded model exposes neither model.language_model.layers nor "
            "model.layers; extraction needs the pinned Gemma architecture."
        )
    out_of_range = [layer for layer in PROBE_LAYERS if layer >= len(layers)]
    if out_of_range:
        raise ExtractError(
            f"Layers {out_of_range} are outside the model's {len(layers)} blocks."
        )
    return {layer: layers[layer] for layer in PROBE_LAYERS}


def pooled_activations(
    runtime: LocalActivationRuntime,
    blocks: dict[int, torch.nn.Module],
    text: str,
) -> np.ndarray:
    """One forward pass; mean-pooled output of every probed block.

    Returns an array of shape (len(PROBE_LAYERS), HIDDEN_WIDTH), rows in
    PROBE_LAYERS order. Pooling happens on the GPU inside each hook so only
    one small vector per layer crosses to the CPU.
    """
    inputs = runtime.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=MAX_STORY_TOKENS
    )
    inputs = {key: value.to(runtime.model.device) for key, value in inputs.items()}
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def capture_output(
            _module: torch.nn.Module, _args: tuple[object, ...], output: object
        ) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if not isinstance(value, torch.Tensor):
                return
            sequence = value[0]
            pooled = (
                sequence[START_TOKEN:]
                if sequence.shape[0] > START_TOKEN
                else sequence
            )
            if layer in captured:
                raise ExtractError(
                    f"Decoder block {layer} fired more than once; extraction "
                    "requires the locked model architecture."
                )
            captured[layer] = pooled.mean(dim=0).detach().float().cpu()

        return capture_output

    handles = [
        block.register_forward_hook(make_hook(layer))
        for layer, block in blocks.items()
    ]
    try:
        with torch.inference_mode():
            runtime.model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(PROBE_LAYERS):
        missing = sorted(set(PROBE_LAYERS) - set(captured))
        raise ExtractError(
            f"Decoder blocks {missing} produced no tensor output; extraction "
            "requires the locked model architecture."
        )
    stacked = np.stack(
        [captured[layer].numpy() for layer in PROBE_LAYERS]
    ).astype(np.float32)
    if stacked.shape != (len(PROBE_LAYERS), HIDDEN_WIDTH):
        raise ExtractError(
            f"Unexpected pooled activation shape {stacked.shape}; expected "
            f"({len(PROBE_LAYERS)}, {HIDDEN_WIDTH})."
        )
    return stacked


def denoise_vectors(
    vectors: dict[str, np.ndarray], neutral_matrix: np.ndarray
) -> tuple[dict[str, np.ndarray], int]:
    """Project out the top neutral PCs (NEUTRAL_VARIANCE of variance)."""
    centered = neutral_matrix - neutral_matrix.mean(axis=0)
    _, singular, rows = np.linalg.svd(centered, full_matrices=False)
    cumulative = np.cumsum(singular**2) / (singular**2).sum()
    n_components = int(np.searchsorted(cumulative, NEUTRAL_VARIANCE)) + 1
    noise_basis = rows[:n_components].T  # (HIDDEN_WIDTH, n_components)
    denoised = {
        name: vector - noise_basis @ (noise_basis.T @ vector)
        for name, vector in vectors.items()
    }
    return denoised, n_components


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def layer_quality(vectors: dict[str, np.ndarray]) -> dict[str, float]:
    """Score one layer's vector set.

    valence_separation: mean cosine across all riser-faller pairs (opposite
    emotions should point apart, so more negative is better).
    synonym_coherence: mean cosine across SYNONYM_PAIRS (higher is better).
    score: synonym_coherence - valence_separation (higher is better).
    """
    cross = [
        _cosine(vectors[r], vectors[f])
        for r in TARGET_RISERS
        for f in TARGET_FALLERS
    ]
    synonyms = [_cosine(vectors[a], vectors[b]) for a, b in SYNONYM_PAIRS]
    valence_separation = sum(cross) / len(cross)
    synonym_coherence = sum(synonyms) / len(synonyms)
    return {
        "valence_separation": round(valence_separation, 4),
        "synonym_coherence": round(synonym_coherence, 4),
        "score": round(synonym_coherence - valence_separation, 4),
    }


def latest_extract_dir() -> Path | None:
    """Most recent extraction run folder, if any."""
    if not RESULTS_DIR.exists():
        return None
    runs = sorted(
        path
        for path in RESULTS_DIR.iterdir()
        if path.is_dir() and path.name.endswith("_extract-gemma4-31b-base")
    )
    return runs[-1] if runs else None


def prepare_run_dir(
    route: LocalTransformersRoute, resume: bool, limit: int | None
) -> Path:
    """Create a fresh extraction run folder, or return the latest for resume."""
    if resume:
        run_dir = latest_extract_dir()
        if run_dir is None:
            raise ExtractError(
                "--resume was passed but no previous extraction run exists "
                f"under {RESULTS_DIR}."
            )
        stored = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
        if stored.get("limit") != limit:
            raise ExtractError(
                f"--resume limit mismatch: the run was created with "
                f"limit={stored.get('limit')} but this invocation passed "
                f"limit={limit}. Cached per-emotion means would mix sample "
                "sizes; repeat the original value or start a fresh run."
            )
        if stored.get("probe_layers") != list(PROBE_LAYERS):
            raise ExtractError(
                "--resume probe-layer mismatch: the run folder was created "
                "with different PROBE_LAYERS. Start a fresh run."
            )
        return run_dir
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RESULTS_DIR / f"{stamp}_extract-gemma4-31b-base"
    (run_dir / "means").mkdir(parents=True)
    run_info = {
        "kind": "vector-extraction",
        "model_id": MODEL_ID.value,
        "repository": route.artifact.repository,
        "revision": route.artifact.revision,
        "quantization": QUANTIZATION_ID.value,
        "probe_layers": list(PROBE_LAYERS),
        "target_emotions": list(TARGET_EMOTIONS),
        "target_stories": TARGET_STORIES,
        "global_mean_stories": GLOBAL_MEAN_STORIES,
        "neutral_sample": NEUTRAL_SAMPLE,
        "sample_seed": SAMPLE_SEED,
        "max_story_tokens": MAX_STORY_TOKENS,
        "start_token": START_TOKEN,
        "neutral_variance": NEUTRAL_VARIANCE,
        "stories_source": f"{GEMOTIONS_REPO}@{GEMOTIONS_REVISION}",
        "limit": limit,
        "started": stamp,
    }
    (run_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    return run_dir


def _load_cached_mean(path: Path) -> np.ndarray:
    array = np.load(path)
    if array.shape != (len(PROBE_LAYERS), HIDDEN_WIDTH):
        raise ExtractError(
            f"Cached mean {path.name} has shape {array.shape}, expected "
            f"({len(PROBE_LAYERS)}, {HIDDEN_WIDTH}); it predates the layer "
            "sweep. Start a fresh run (omit --resume)."
        )
    return array


def run_extraction(
    cache_dir: Path, device: Device, limit: int | None, resume: bool
) -> None:
    """Extract the 20 target vectors at every probed layer of the base model."""
    route = resolve_extract_route()
    samples = load_story_samples(stories_db_path(cache_dir, download=False))
    neutral_texts = load_neutral_samples()
    if limit is not None:
        samples = {name: texts[:limit] for name, texts in samples.items()}
        neutral_texts = neutral_texts[:limit]
    run_dir = prepare_run_dir(route, resume, limit)
    means_dir = run_dir / "means"
    total = sum(len(texts) for texts in samples.values()) + len(neutral_texts)
    print(
        f"Run folder: {run_dir} ({total} forward passes when starting fresh, "
        f"{len(PROBE_LAYERS)} layers per pass)"
    )

    runtime = create_transformers_runtime(route, cache_dir=cache_dir, device=device)
    blocks = _decoder_blocks(runtime.model)

    # Per-emotion means are cached as they complete, so --resume skips them.
    counts: dict[str, int] = {}
    done = 0
    for emotion in sorted(samples):
        texts = samples[emotion]
        counts[emotion] = len(texts)
        mean_file = means_dir / f"{emotion}.npy"
        done += 1
        if mean_file.exists():
            continue
        activations = [
            pooled_activations(runtime, blocks, text) for text in texts
        ]
        np.save(mean_file, np.mean(activations, axis=0))
        print(f"[{done}/{len(samples)}] {emotion} ({len(texts)} stories)")

    neutral_file = means_dir / "_neutral_matrix.npy"
    if not neutral_file.exists():
        neutral_matrix = np.stack(
            [pooled_activations(runtime, blocks, text) for text in neutral_texts]
        )  # (n_neutral, n_layers, width)
        np.save(neutral_file, neutral_matrix)
        print(f"neutral dialogues done ({len(neutral_texts)})")
    neutral_matrix = np.load(neutral_file)

    emotion_means = {
        name: _load_cached_mean(means_dir / f"{name}.npy") for name in samples
    }
    total_stories = sum(counts.values())
    global_means = (
        sum(emotion_means[name] * counts[name] for name in samples)
        / total_stories
    )  # (n_layers, width)

    it_files = {
        layer: IT_VECTORS_DIR / f"emotion_vectors_layer{layer}.npz"
        for layer in PROBE_LAYERS
    }
    quality: dict[str, dict[str, float]] = {}
    details_components: dict[str, int] = {}
    for index, layer in enumerate(PROBE_LAYERS):
        raw_vectors = {
            name: emotion_means[name][index] - global_means[index]
            for name in TARGET_EMOTIONS
        }
        vectors, n_components = denoise_vectors(
            raw_vectors, neutral_matrix[:, index, :]
        )
        details_components[str(layer)] = n_components
        np.savez(
            run_dir / f"{OUTPUT_PREFIX}_emotion_vectors_layer{layer}.npz",
            **{n: v.astype(np.float32) for n, v in vectors.items()},
        )
        quality[str(layer)] = layer_quality(vectors)
        if it_files[layer].exists():
            it_vectors = np.load(it_files[layer])
            cosines = [
                _cosine(vectors[name], it_vectors[name])
                for name in TARGET_EMOTIONS
                if name in it_vectors.files
            ]
            quality[str(layer)]["it_cosine_mean"] = round(
                sum(cosines) / len(cosines), 4
            )
    np.savez(
        run_dir / f"{OUTPUT_PREFIX}_global_means.npz",
        **{str(layer): global_means[i] for i, layer in enumerate(PROBE_LAYERS)},
    )

    best = max(quality, key=lambda layer: quality[layer]["score"])
    print("\nlayer  valence_sep  synonym_coh  score  it_cosine")
    for layer in PROBE_LAYERS:
        entry = quality[str(layer)]
        marker = "  <- best score" if str(layer) == best else ""
        it_part = (
            f"  {entry['it_cosine_mean']:+.3f}"
            if "it_cosine_mean" in entry
            else "      -"
        )
        print(
            f"{layer:>5}  {entry['valence_separation']:>+11.3f}"
            f"  {entry['synonym_coherence']:>+11.3f}"
            f"  {entry['score']:>+5.3f}{it_part}{marker}"
        )
    print(
        f"\nBest score: layer {best}. Prefer a layer on a stable plateau of "
        "good scores over a lone spike; see layer_quality.json."
    )

    (run_dir / "layer_quality.json").write_text(
        json.dumps(
            {"best_by_score": int(best), "layers": quality}, indent=2
        ),
        encoding="utf-8",
    )
    (run_dir / "extraction_details.json").write_text(
        json.dumps(
            {
                "stories_per_emotion": counts,
                "neutral_dialogues": int(neutral_matrix.shape[0]),
                "denoise_components_per_layer": details_components,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Vectors written to {run_dir}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Extract emotion vectors from the base Gemma 4 31B."
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
        help="download the pinned base checkpoint (~60 GB) and the story "
        "corpus (~433 MB)",
    )
    run_parser = subparsers.add_parser("run", help="run the extraction")
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
        help="cap stories per emotion and neutral dialogues (smoke tests)",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the latest extraction folder (skips cached emotions)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    try:
        if args.command == "download":
            route = resolve_extract_route()
            model_path = download_transformers_artifact(route, args.cache_dir)
            print(f"Downloaded pinned base checkpoint to: {model_path}")
            stories = stories_db_path(args.cache_dir, download=True)
            print(f"Downloaded story corpus to: {stories}")
        else:
            run_extraction(args.cache_dir, args.device, args.limit, args.resume)
    except (ModelRouteError, TransformersRuntimeError, ExtractError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
