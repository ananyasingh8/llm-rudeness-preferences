"""Extract emotion vectors from a pinned model (E4B or the base 31B).

Replicates the gemotions/Anthropic extraction method: per story, tokenize the
raw prose (no chat template — the original extraction also used raw text),
run one forward pass, and capture every probed decoder block at once — a
forward pass computes all layers anyway, so sweeping layers costs no extra
GPU time. Each captured activation is mean-pooled (from START_TOKEN onward)
on the GPU before leaving it. Then, independently per layer:

    vector = mean(activation over its stories) - global mean over ALL stories

Finally the top principal components of the neutral-text activations
(explaining NEUTRAL_VARIANCE of variance) are projected out, and each layer's
vectors are saved un-normalized in the npz format the probe loader consumes,
under model-prefixed names so vector sets can never be confused across models.

Two extraction configs (EXTRACTIONS below; --extraction picks one):

- gemma4-e4b (default): the IT Gemma 4 E4B, all 171 emotions, using the story
  corpus and neutral paragraphs from the vendored sinievanderben/
  emotion_experiment replication (emotion_experiment/ submodule) — Gemma-
  generated and Apertus-generated stories combined, ~18 stories per emotion.
- gemma4-31b-base: the original base-31B run (20 target emotions from the
  gemotions stories.db, kept exactly as it was when it produced
  results/2026-08-15_182042_extract-gemma4-31b-base).

The run ends with layer_quality.json: per layer, how cleanly opposite-valence
emotions point apart, how tightly synonyms agree, and (for the 31B, whose
width matches the vendored IT files) the cosine to the IT vectors. Pick the
best layer from that scorecard — prefer a layer on a stable plateau of good
scores over a lone spike — and use that layer's npz downstream.

Usage:
    uv run python -m emotion_probing.extract download
    uv run python -m emotion_probing.extract run [--device auto|cuda|cpu]
        [--limit N] [--resume] [--extraction gemma4-e4b|gemma4-31b-base]
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass
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


class StorySource:
    GEMOTIONS_DB = "gemotions-db"
    EMOTION_EXPERIMENT = "emotion-experiment"


@dataclass(frozen=True)
class ExtractionConfig:
    """One reviewed pairing of model route, story corpus, and output naming."""

    model_id: ModelId
    quantization_id: QuantizationId
    output_prefix: str
    stories: str  # StorySource
    neutral: str  # StorySource
    save_all_emotions: bool  # False = only TARGET_EMOTIONS get vectors
    # None sweeps every decoder block of the loaded model.
    probe_layers: tuple[int, ...] | None
    # None accepts whatever width the model produces (recorded in run_info).
    hidden_width: int | None
    target_stories: int | None  # per target emotion; None = all available
    global_mean_stories: int | None  # per non-target emotion; None = all
    neutral_sample: int | None  # None = all
    compare_it: bool  # cosine vs the vendored 31B IT vectors (width must match)


# --- Extraction constants: the scientific knobs, all in one place. -----------
EXTRACTIONS: dict[str, ExtractionConfig] = {
    "gemma4-e4b": ExtractionConfig(
        model_id=ModelId.GEMMA_4_E4B_IT,
        quantization_id=QuantizationId.BITSANDBYTES_FP4,
        output_prefix="gemma4-e4b-it",
        stories=StorySource.EMOTION_EXPERIMENT,
        neutral=StorySource.EMOTION_EXPERIMENT,
        save_all_emotions=True,  # the corpus is small; extract all 171
        probe_layers=None,  # sweep every block; layer count read at load
        hidden_width=None,
        target_stories=None,  # all ~18 stories per emotion
        global_mean_stories=None,
        neutral_sample=None,  # all 40 neutral paragraphs
        compare_it=False,  # different width than the vendored 31B files
    ),
    "gemma4-31b-base": ExtractionConfig(
        model_id=ModelId.GEMMA_4_31B,
        quantization_id=QuantizationId.BITSANDBYTES_FP4,
        output_prefix="gemma4-31b-base",
        stories=StorySource.GEMOTIONS_DB,
        neutral=StorySource.GEMOTIONS_DB,
        save_all_emotions=False,
        probe_layers=tuple(range(20, 60)),
        hidden_width=5376,
        target_stories=100,
        global_mean_stories=10,
        neutral_sample=300,
        compare_it=True,
    ),
}
DEFAULT_EXTRACTION = "gemma4-e4b"

# Top 10 risers and top 10 fallers by shift_band_avg (pooled bands -1..-3
# minus band 0) in run results/2026-08-15_052621_convabuse-31b-local-quant.
# Used as the extraction targets for the trimmed 31B run and as the
# valence/synonym scorecard emotions for every run.
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
SAMPLE_SEED = 42
# All three values below mirror the vendored gemotions extraction exactly.
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
# The vendored sinievanderben/emotion_experiment replication (submodule).
EMOTION_EXPERIMENT_DIR = PACKAGE_DIR / "emotion_experiment"
EMOTION_EXPERIMENT_STORY_FILES = (
    Path("output_gemma_stories") / "stories.jsonl",
    Path("output_apertus_stories") / "stories_dedup.jsonl",
)
EMOTION_EXPERIMENT_NEUTRAL = Path("prompts") / "neutral_texts.txt"


class ExtractError(RuntimeError):
    """A user-actionable failure in the extraction workflow."""


def resolve_extract_route(config: ExtractionConfig) -> LocalTransformersRoute:
    """Resolve the extraction model's route, requiring activation access."""
    route = resolve_route(
        config.model_id,
        ProviderId.LOCAL,
        config.quantization_id,
        required={Capability.LOCAL_ACTIVATIONS},
    )
    if not isinstance(route, LocalTransformersRoute):
        raise ExtractError(
            f"The resolved route is {type(route).__name__}, not a local "
            "Transformers route; extraction reads activations locally."
        )
    return route


# --- Story and neutral-text sources ------------------------------------------


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


def _emotion_experiment_file(relative: Path) -> Path:
    path = EMOTION_EXPERIMENT_DIR / relative
    if not path.exists():
        raise ExtractError(
            f"{path} not found. The emotion_experiment submodule is not "
            "checked out; run `git submodule update --init "
            "emotion_probing/emotion_experiment` and retry."
        )
    return path


def load_story_samples_jsonl() -> dict[str, list[str]]:
    """All stories per emotion from the vendored emotion_experiment corpora.

    Combines the Gemma-generated and (deduplicated) Apertus-generated story
    files — the reference repo's own cross-condition setup extracts Gemma
    vectors from either corpus. ~18 stories per emotion, 171 emotions.
    """
    stories: dict[str, list[str]] = {}
    for relative in EMOTION_EXPERIMENT_STORY_FILES:
        path = _emotion_experiment_file(relative)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                parsed = row["stories"]
                if isinstance(parsed, str):
                    parsed = json.loads(parsed)
                stories.setdefault(row["emotion"], []).extend(
                    text for text in parsed if text.strip()
                )
    if not stories:
        raise ExtractError("The emotion_experiment story files are empty.")
    return stories


def load_story_samples_db(db_path: Path) -> dict[str, list[str]]:
    """All stories per emotion from the gemotions stories.db."""
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
    return stories


def load_story_samples(
    config: ExtractionConfig, cache_dir: Path
) -> dict[str, list[str]]:
    """A seeded story sample per emotion, per the extraction config."""
    if config.stories == StorySource.EMOTION_EXPERIMENT:
        stories = load_story_samples_jsonl()
    else:
        stories = load_story_samples_db(stories_db_path(cache_dir, download=False))
    missing = [e for e in TARGET_EMOTIONS if e not in stories]
    if missing:
        raise ExtractError(
            f"The story corpus has no stories for: {', '.join(missing)}. "
            "Check TARGET_EMOTIONS against the corpus emotion names."
        )
    # Seeded sampling in a fixed iteration order keeps the run reproducible
    # and topic-diverse (first-N would over-sample the corpus's first topics).
    rng = random.Random(SAMPLE_SEED)
    samples: dict[str, list[str]] = {}
    for emotion in sorted(stories):
        count = (
            config.target_stories
            if emotion in TARGET_EMOTIONS
            else config.global_mean_stories
        )
        pool = stories[emotion]
        if count is None or count >= len(pool):
            samples[emotion] = list(pool)
        else:
            samples[emotion] = rng.sample(pool, count)
    return samples


def load_neutral_samples(config: ExtractionConfig) -> list[str]:
    """A seeded sample of neutral texts, per the extraction config."""
    if config.neutral == StorySource.EMOTION_EXPERIMENT:
        path = _emotion_experiment_file(EMOTION_EXPERIMENT_NEUTRAL)
        texts = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        if not NEUTRAL_DB.exists():
            raise ExtractError(
                f"Vendored neutral dialogues not found: {NEUTRAL_DB}"
            )
        connection = sqlite3.connect(NEUTRAL_DB, timeout=30)
        try:
            rows = connection.execute(
                "SELECT text FROM dialogues ORDER BY topic_idx, dialogue_idx"
            ).fetchall()
        finally:
            connection.close()
        texts = [text for (text,) in rows]
    if config.neutral_sample is None or config.neutral_sample >= len(texts):
        return texts
    rng = random.Random(SAMPLE_SEED)
    return rng.sample(texts, config.neutral_sample)


# --- Activation capture -------------------------------------------------------


def _decoder_blocks(
    model: torch.nn.Module, requested: tuple[int, ...] | None
) -> dict[int, torch.nn.Module]:
    """Resolve every probed decoder block (same layouts as the probe runner).

    With requested=None, every block of the loaded model is swept.
    """
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
    probe_layers = (
        requested if requested is not None else tuple(range(len(layers)))
    )
    out_of_range = [layer for layer in probe_layers if layer >= len(layers)]
    if out_of_range:
        raise ExtractError(
            f"Layers {out_of_range} are outside the model's {len(layers)} blocks."
        )
    return {layer: layers[layer] for layer in probe_layers}


def pooled_activations(
    runtime: LocalActivationRuntime,
    blocks: dict[int, torch.nn.Module],
    text: str,
    expected_width: int | None,
) -> np.ndarray:
    """One forward pass; mean-pooled output of every probed block.

    Returns an array of shape (len(blocks), width), rows in blocks order.
    Pooling happens on the GPU inside each hook so only one small vector per
    layer crosses to the CPU.
    """
    probe_layers = tuple(blocks)
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
    if set(captured) != set(probe_layers):
        missing = sorted(set(probe_layers) - set(captured))
        raise ExtractError(
            f"Decoder blocks {missing} produced no tensor output; extraction "
            "requires the locked model architecture."
        )
    stacked = np.stack(
        [captured[layer].numpy() for layer in probe_layers]
    ).astype(np.float32)
    if expected_width is not None and stacked.shape != (
        len(probe_layers),
        expected_width,
    ):
        raise ExtractError(
            f"Unexpected pooled activation shape {stacked.shape}; expected "
            f"({len(probe_layers)}, {expected_width})."
        )
    return stacked


# --- Vector math and quality --------------------------------------------------


def denoise_vectors(
    vectors: dict[str, np.ndarray], neutral_matrix: np.ndarray
) -> tuple[dict[str, np.ndarray], int]:
    """Project out the top neutral PCs (NEUTRAL_VARIANCE of variance)."""
    centered = neutral_matrix - neutral_matrix.mean(axis=0)
    _, singular, rows = np.linalg.svd(centered, full_matrices=False)
    cumulative = np.cumsum(singular**2) / (singular**2).sum()
    n_components = int(np.searchsorted(cumulative, NEUTRAL_VARIANCE)) + 1
    noise_basis = rows[:n_components].T  # (width, n_components)
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


# --- Run folder ---------------------------------------------------------------


def latest_extract_dir(config: ExtractionConfig) -> Path | None:
    """Most recent extraction run folder for this config, if any."""
    if not RESULTS_DIR.exists():
        return None
    runs = sorted(
        path
        for path in RESULTS_DIR.iterdir()
        if path.is_dir()
        and path.name.endswith(f"_extract-{config.output_prefix}")
    )
    return runs[-1] if runs else None


def prepare_run_dir(
    config: ExtractionConfig,
    route: LocalTransformersRoute,
    resume: bool,
    limit: int | None,
    probe_layers: tuple[int, ...],
    stories_source: str,
) -> Path:
    """Create a fresh extraction run folder, or return the latest for resume."""
    if resume:
        run_dir = latest_extract_dir(config)
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
        if stored.get("probe_layers") != list(probe_layers):
            raise ExtractError(
                "--resume probe-layer mismatch: the run folder was created "
                "with different probe layers. Start a fresh run."
            )
        return run_dir
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RESULTS_DIR / f"{stamp}_extract-{config.output_prefix}"
    (run_dir / "means").mkdir(parents=True)
    run_info = {
        "kind": "vector-extraction",
        "model_id": config.model_id.value,
        "repository": route.artifact.repository,
        "revision": route.artifact.revision,
        "quantization": config.quantization_id.value,
        "output_prefix": config.output_prefix,
        "probe_layers": list(probe_layers),
        "save_all_emotions": config.save_all_emotions,
        "target_emotions": list(TARGET_EMOTIONS),
        "target_stories": config.target_stories,
        "global_mean_stories": config.global_mean_stories,
        "neutral_sample": config.neutral_sample,
        "sample_seed": SAMPLE_SEED,
        "max_story_tokens": MAX_STORY_TOKENS,
        "start_token": START_TOKEN,
        "neutral_variance": NEUTRAL_VARIANCE,
        "stories_source": stories_source,
        "limit": limit,
        "started": stamp,
    }
    (run_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    return run_dir


def _load_cached_mean(
    path: Path, n_layers: int, expected_width: int | None
) -> np.ndarray:
    array = np.load(path)
    good = array.ndim == 2 and array.shape[0] == n_layers
    if good and expected_width is not None:
        good = array.shape[1] == expected_width
    if not good:
        raise ExtractError(
            f"Cached mean {path.name} has shape {array.shape}, expected "
            f"({n_layers}, width); it predates a config change. Start a "
            "fresh run (omit --resume)."
        )
    return array


def _stories_source_label(config: ExtractionConfig) -> str:
    if config.stories == StorySource.EMOTION_EXPERIMENT:
        files = ", ".join(str(f) for f in EMOTION_EXPERIMENT_STORY_FILES)
        return f"emotion_experiment submodule ({files})"
    return f"{GEMOTIONS_REPO}@{GEMOTIONS_REVISION}"


# --- The run ------------------------------------------------------------------


def run_extraction(
    config: ExtractionConfig,
    cache_dir: Path,
    device: Device,
    limit: int | None,
    resume: bool,
) -> None:
    """Extract emotion vectors at every probed layer of the configured model."""
    route = resolve_extract_route(config)
    samples = load_story_samples(config, cache_dir)
    neutral_texts = load_neutral_samples(config)
    if limit is not None:
        samples = {name: texts[:limit] for name, texts in samples.items()}
        neutral_texts = neutral_texts[:limit]

    print(f"Loading {route.artifact.repository} ({config.quantization_id.value})...")
    runtime = create_transformers_runtime(route, cache_dir=cache_dir, device=device)
    blocks = _decoder_blocks(runtime.model, config.probe_layers)
    probe_layers = tuple(blocks)

    run_dir = prepare_run_dir(
        config, route, resume, limit, probe_layers, _stories_source_label(config)
    )
    means_dir = run_dir / "means"
    total = sum(len(texts) for texts in samples.values()) + len(neutral_texts)
    print(
        f"Run folder: {run_dir} ({total} forward passes when starting fresh, "
        f"{len(probe_layers)} layers per pass)"
    )

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
            pooled_activations(runtime, blocks, text, config.hidden_width)
            for text in texts
        ]
        np.save(mean_file, np.mean(activations, axis=0))
        print(f"[{done}/{len(samples)}] {emotion} ({len(texts)} stories)")

    neutral_file = means_dir / "_neutral_matrix.npy"
    if not neutral_file.exists():
        neutral_matrix = np.stack(
            [
                pooled_activations(runtime, blocks, text, config.hidden_width)
                for text in neutral_texts
            ]
        )  # (n_neutral, n_layers, width)
        np.save(neutral_file, neutral_matrix)
        print(f"neutral texts done ({len(neutral_texts)})")
    neutral_matrix = np.load(neutral_file)

    emotion_means = {
        name: _load_cached_mean(
            means_dir / f"{name}.npy", len(probe_layers), config.hidden_width
        )
        for name in samples
    }
    total_stories = sum(counts.values())
    global_means = (
        sum(emotion_means[name] * counts[name] for name in samples)
        / total_stories
    )  # (n_layers, width)

    save_names = (
        sorted(samples) if config.save_all_emotions else list(TARGET_EMOTIONS)
    )
    it_files = {
        layer: IT_VECTORS_DIR / f"emotion_vectors_layer{layer}.npz"
        for layer in probe_layers
    }
    quality: dict[str, dict[str, float]] = {}
    details_components: dict[str, int] = {}
    for index, layer in enumerate(probe_layers):
        raw_vectors = {
            name: emotion_means[name][index] - global_means[index]
            for name in save_names
        }
        vectors, n_components = denoise_vectors(
            raw_vectors, neutral_matrix[:, index, :]
        )
        details_components[str(layer)] = n_components
        np.savez(
            run_dir / f"{config.output_prefix}_emotion_vectors_layer{layer}.npz",
            **{n: v.astype(np.float32) for n, v in vectors.items()},
        )
        quality[str(layer)] = layer_quality(vectors)
        if config.compare_it and it_files[layer].exists():
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
        run_dir / f"{config.output_prefix}_global_means.npz",
        **{str(layer): global_means[i] for i, layer in enumerate(probe_layers)},
    )

    best = max(quality, key=lambda layer: quality[layer]["score"])
    print("\nlayer  valence_sep  synonym_coh  score  it_cosine")
    for layer in probe_layers:
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
                "saved_emotions": len(save_names),
                "neutral_texts": int(neutral_matrix.shape[0]),
                "hidden_width": int(neutral_matrix.shape[2]),
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
        description="Extract emotion vectors from a pinned Gemma model."
    )
    parser.add_argument(
        "--extraction",
        choices=sorted(EXTRACTIONS),
        default=DEFAULT_EXTRACTION,
        help="extraction configuration (default: %(default)s)",
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
        help="download the pinned checkpoint (and, for the 31B config, the "
        "gemotions story corpus)",
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
        help="cap stories per emotion and neutral texts (smoke tests)",
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
    config = EXTRACTIONS[args.extraction]
    try:
        if args.command == "download":
            route = resolve_extract_route(config)
            model_path = download_transformers_artifact(route, args.cache_dir)
            print(f"Downloaded pinned checkpoint to: {model_path}")
            if config.stories == StorySource.GEMOTIONS_DB:
                stories = stories_db_path(args.cache_dir, download=True)
                print(f"Downloaded story corpus to: {stories}")
            else:
                for relative in EMOTION_EXPERIMENT_STORY_FILES:
                    _emotion_experiment_file(relative)
                _emotion_experiment_file(EMOTION_EXPERIMENT_NEUTRAL)
                print("Story corpus present in the emotion_experiment submodule.")
        else:
            run_extraction(
                config, args.cache_dir, args.device, args.limit, args.resume
            )
    except (ModelRouteError, TransformersRuntimeError, ExtractError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
