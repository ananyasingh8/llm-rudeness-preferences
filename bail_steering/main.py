"""Bail-with-steering: does the rude-conversation emotional state cause bailing?

The emotion probing (emotion_probing/) showed that rude ConvAbuse
conversations shift Gemma 4 31B IT toward "insulted/hostile" and away from
"joy" emotions. This experiment tests whether that internal state is CAUSAL
for wanting to leave a conversation: while the model answers the bail
wellbeing check (bail/), we steer its residual stream with one emotion vector
at a time and measure how the bail rate moves.

Design (one condition per emotion, plus an unsteered baseline):
  - the model is Gemma 4 E4B IT (4-bit at load), and the emotion vectors are
    the ones extracted from that same model by emotion_probing/extract.py
    (--extraction gemma4-e4b, stories from the vendored sinievanderben/
    emotion_experiment replication)
  - conditions: risers (emotions that went UP on rude conversations in the
    convabuse-e4b probing run) steered at +0.1, fallers at -0.1 — pinned in
    STEER_RISERS/STEER_FALLERS after that run's analysis
  - steering strength follows Anthropic's convention: units are fractions of
    the residual-stream norm, so each generated token gets a push of exactly
    |coefficient| x (its own residual norm) in the emotion's direction. The
    stored vectors are unit-normalized first (their raw lengths are
    extraction artifacts and would otherwise change the dose per emotion).
  - conversations: the frozen VERIFIED ConvAbuse bail sample (bail/data/
    convabuse_sample_verified.csv, 1,023 rows built by
    scripts/build_bail_sample_verified.py from the collapsed parquet:
    annotator-agreement filter severity_std <= 0.5, all rows of bands
    0/-1/-2/-3 [122/98/241/62] plus a seeded 500-row draw of band 1),
    replayed exactly like bail/src/convabuse_run.py.
  - elicitation: the paper-verbatim methods from bail/prompts/
    bail_methods.py — any subset of the two prompt orderings and the tool
    method via --methods — generated locally instead of over the OpenRouter
    API so the hook can reach the activations.

Runs are resumable per generation; --conditions selects/prioritizes
conditions (baseline first), --seed takes one or more seeds (one results
folder per seed), --steer overrides the coefficient magnitude, and --limit
takes a severity-stratified subset of conversations for smoke tests.

Until the three pin blocks below are set (steering layer + extraction run,
then the riser/faller lists), `run` fails with instructions.

Usage (from the repo root, on the machine with the GPU):
  uv run python -m bail_steering.main download
  uv run python -m bail_steering.main run --device cuda --limit 10 --conditions baseline,enraged
  uv run python -m bail_steering.main run --device cuda --seed 42,43,44,45,46
  uv run python -m bail_steering.main run --device cuda --conditions enraged --steer 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from huggingface_hub.constants import HF_HUB_CACHE

from bail.prompts.bail_methods import (
    BAIL_TOOL_NAME,
    ORDERINGS,
    get_bail_prompt,
    get_bail_prompt_status,
    get_bail_tool,
    remove_thinking,
)
from llm_runtime import (
    Capability,
    LocalTransformersRoute,
    ModelId,
    ProviderId,
    QuantizationId,
    resolve_route,
)
from llm_runtime.transformers import (
    Device,
    create_transformers_runtime,
    download_transformers_artifact,
)

# --- Experiment constants -----------------------------------------------------
MODEL_ID = ModelId.GEMMA_4_E4B_IT
QUANTIZATION_ID = QuantizationId.BITSANDBYTES_FP4
COEFFICIENT = 0.1  # fraction of residual-stream norm (Anthropic's units)

# --- Pins from the upstream E4B pipeline steps (human-picked, in order): ------
# 1. STEER_LAYER + STEER_VECTORS_RUN: after the extraction sweep
#    (`uv run python -m emotion_probing.extract run`), pick the layer from
#    its layer_quality.json (prefer a plateau over a lone spike) and name the
#    extraction results folder. Use the same layer the probing experiment
#    pins as E4B_PROBE_LAYER in emotion_probing/main.py.
# 2. STEER_RISERS + STEER_FALLERS: after the convabuse-e4b probing run, the
#    top-10 risers and top-10 fallers by shift vs band 0 (pooled avg band)
#    from its analysis. Risers get +COEFFICIENT, fallers get -COEFFICIENT.
STEER_LAYER: int | None = 13
STEER_VECTORS_RUN: str | None = "2026-08-16_000402_extract-gemma4-e4b-it"
# Set by --layer: the decoder blocks steered this invocation (default: the
# pinned STEER_LAYER alone). Multi-layer bands steer each block with that
# block's OWN vector -- the emotion code drifts across layers, so a band
# must stay inside one coherent region (11-16 early, 30-33 late).
STEER_LAYERS: tuple[int, ...] | None = None
# Top-10 risers/fallers by shift_band_avg per PROBED layer (verified sample:
# annotator agreement severity_std <= 0.5). The steered layer set must
# contain exactly one layer pinned here; its lists define the conditions
# and steering signs. Layer 13: results/2026-08-17_015634_convabuse-e4b.
# Layer 30: pin after probing it (`uv run python -m emotion_probing.main
# run --layer 30` + analyze).
STEER_PINS: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] = {
    13: (
        (
            "enraged",
            "spiteful",
            "angry",
            "irate",
            "insulted",
            "defiant",
            "indignant",
            "outraged",
            "mortified",
            "vindictive",
        ),
        (
            "excited",
            "enthusiastic",
            "refreshed",
            "amazed",
            "serene",
            "sentimental",
            "content",
            "peaceful",
            "elated",
            "nostalgic",
        ),
    ),
}

BASELINE_CONDITION = "baseline"
VECTORS_PREFIX = "gemma4-e4b-it"  # npz prefix written by the extraction

# Inline tool-call shapes seen in raw generations: the bail paper's
# OpenAI-style JSON scan, plus Gemma's own "call:<name>{...}" syntax
# (observed in the first E4B run; the strict JSON scan alone missed it).
TOOL_CALL_MARKERS = (
    '"name": "switchconversation_tool"',
    "call:switchconversation_tool",
)

# The three bail elicitation methods, individually selectable via --methods.
METHODS = ("bail_first", "continue_first", "tool")
BAIL_MODEL_NAME = "Gemma"  # own-model name in the bail tool text (bail/config.py)
MAX_NEW_TOKENS = 1000  # matches bail/config.py BAIL_MAX_TOKENS
# Gemma's recommended sampling settings, applied explicitly (the API run used
# provider defaults; locally we pin them so the run is fully described).
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 64
SEED = 42  # documentation/default only; the run always records its resolved seed (see main())
BATCH_SIZE = 32  # conversations generated at once; lower this if CUDA OOMs
# (Sized for an 8 GB RTX 3070: ~4.5 GB of 4-bit weights + desktop overhead
# leaves ~2 GB for the KV cache; E4B's 2-head GQA keeps that cheap.)

REPO_ROOT = Path(__file__).parent.parent
SAMPLE_FILE = REPO_ROOT / "bail" / "data" / "convabuse_sample_verified.csv"
SAMPLE_BANDS = ("1", "0", "-1", "-2", "-3")  # stratification order for --limit
EXTRACTION_RESULTS_DIR = REPO_ROOT / "emotion_probing" / "results"
RESULTS_DIR = Path(__file__).parent / "results"
RESPONSES_FILE = "responses.csv"
RESPONSE_COLUMNS = (
    "custom_id",
    "condition",
    "phase",
    "ordering",
    "snippet_id",
    "abuse_severity",
    "group",
    "wellbeing",
    "tool_called",
    "tool_mentioned",
    "n_new_tokens",
    "response_text",
)

# Conversation turns replayed as chat messages, exactly as in
# bail/src/convabuse_run.py.
TRANSCRIPT_TURNS = (
    ("prev_agent", "assistant"),
    ("prev_user", "user"),
    ("agent", "assistant"),
    ("user", "user"),  # the annotated user message
)


class SteeringError(RuntimeError):
    """A user-actionable failure in the bail-steering workflow."""


# --- Steering -----------------------------------------------------------------


class SteeringHook:
    """Forward hook adding a fixed-fraction emotion push at token positions.

    Registered on a steered decoder block. Each steered position's hidden
    state gets `coefficient x (its residual norm)` added in the emotion's
    unit direction -- Anthropic's "fraction of residual stream norm" units.

    scope "response" (the original design): prefill passes (sequence length
    > 1, i.e. the conversation being read) are left untouched; with KV-cache
    generation every later pass computes exactly one generated token, which
    is steered. (The first response token is decided during prefill and is
    therefore unsteered.) scope "all": every position of every pass is
    steered, prefill included -- the model reads the conversation already in
    the induced state, matching Anthropic's all-positions interventions.
    """

    def __init__(
        self, direction: torch.Tensor, coefficient: float, scope: str
    ) -> None:
        self.direction = direction  # unit-normalized, float32
        self.coefficient = coefficient  # signed: +0.1 risers, -0.1 fallers
        self.scope = scope  # "response" or "all"

    def __call__(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.scope == "response" and hidden.shape[1] != 1:
            return output  # prompt prefill: only steer generated tokens
        direction = self.direction.to(hidden.device)
        norms = hidden.float().norm(dim=-1, keepdim=True)
        steered = hidden + (self.coefficient * norms * direction).to(hidden.dtype)
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered


def require_pins() -> tuple[tuple[int, ...], str, tuple[str, ...], tuple[str, ...]]:
    """The pinned (layers, extraction run, risers, fallers), or instructions."""
    if STEER_LAYER is None or STEER_VECTORS_RUN is None:
        raise SteeringError(
            "STEER_LAYER and STEER_VECTORS_RUN are not pinned yet. Run the "
            "E4B extraction sweep (`uv run python -m emotion_probing.extract "
            "run`), pick the layer from its layer_quality.json (prefer a "
            "plateau over a lone spike), then set both constants at the top "
            "of bail_steering/main.py."
        )
    layers = STEER_LAYERS if STEER_LAYERS is not None else (STEER_LAYER,)
    pinned = [layer for layer in layers if layer in STEER_PINS]
    if len(pinned) != 1:
        raise SteeringError(
            f"The steered layer set {list(layers)} must contain exactly one "
            f"layer with pinned riser/faller lists (pinned: "
            f"{sorted(STEER_PINS)}); found {pinned}. Probe the new layer "
            "(`uv run python -m emotion_probing.main run --layer N`), then "
            "add its top-10 movers to STEER_PINS in bail_steering/main.py."
        )
    risers, fallers = STEER_PINS[pinned[0]]
    return tuple(layers), STEER_VECTORS_RUN, risers, fallers


def conditions() -> tuple[str, ...]:
    """All 21 run conditions: baseline plus the pinned steered emotions."""
    _, _, risers, fallers = require_pins()
    return (BASELINE_CONDITION,) + risers + fallers


def vectors_path(layer: int) -> Path:
    _, run, _, _ = require_pins()
    return (
        EXTRACTION_RESULTS_DIR
        / run
        / f"{VECTORS_PREFIX}_emotion_vectors_layer{layer}.npz"
    )


def validate_vector_pins(route) -> None:
    """The pinned extraction run must match the steering model exactly."""
    layers, _, _, _ = require_pins()
    for layer in layers:
        path = vectors_path(layer)
        if not path.exists():
            raise SteeringError(
                f"Emotion vectors file not found: {path}. Check "
                "STEER_VECTORS_RUN and the steered layers against the "
                "extraction run's actual outputs."
            )
        run_info_file = path.parent / "run_info.json"
        if run_info_file.exists():
            info = json.loads(run_info_file.read_text(encoding="utf-8"))
            if info.get("repository") != route.artifact.repository:
                raise SteeringError(
                    f"The pinned vectors were extracted from "
                    f"{info.get('repository')} but the steering route loads "
                    f"{route.artifact.repository}. Emotion vectors are "
                    "model-specific; fix the pinning."
                )
            if layer not in info.get("probe_layers", []):
                raise SteeringError(
                    f"Steered layer {layer} was not part of the pinned "
                    "extraction run's probe_layers; fix the pinning."
                )


def load_direction(
    condition: str, magnitude: float, layer: int
) -> tuple[torch.Tensor, float]:
    """The unit steering direction (at `layer`) and the signed coefficient.

    The sign comes from the pinned lists (risers +, fallers -); the
    magnitude comes from --steer (default COEFFICIENT). Each layer uses its
    own extracted vector: the emotion code drifts with depth.
    """
    _, _, risers, _ = require_pins()
    path = vectors_path(layer)
    data = np.load(path)
    if condition not in data.files:
        raise SteeringError(
            f"Emotion '{condition}' not in {path.name} "
            f"({len(data.files)} emotions available)."
        )
    vector = torch.from_numpy(data[condition]).float()
    direction = vector / vector.norm()
    sign = 1.0 if condition in risers else -1.0
    return direction, sign * magnitude


def _decoder_block(model: torch.nn.Module, layer: int) -> torch.nn.Module:
    """Resolve the decoder block to hook (same access path as emotion_probing)."""
    base = getattr(model, "model", None)
    language_model = getattr(base, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        layers = getattr(base, "layers", None)
    if not isinstance(layers, (torch.nn.ModuleList, list, tuple)):
        raise SteeringError(
            "The loaded model exposes neither model.language_model.layers nor "
            "model.layers; steering needs the pinned Gemma architecture."
        )
    if layer < 0 or layer >= len(layers):
        raise SteeringError(
            f"STEER_LAYER {layer} is outside the model's {len(layers)} blocks."
        )
    return layers[layer]


# --- Dataset and prompts ------------------------------------------------------


def load_sample() -> list[dict[str, str]]:
    """The frozen verified ConvAbuse bail sample (see the module docstring)."""
    if not SAMPLE_FILE.exists():
        raise SteeringError(
            f"Frozen sample not found: {SAMPLE_FILE}. Build it with "
            "`python scripts/build_bail_sample_verified.py` (needs the "
            "collapsed parquet committed in bail/data/)."
        )
    with SAMPLE_FILE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SteeringError(f"{SAMPLE_FILE} has no rows.")
    missing = {"snippet_id", "severity_band", "group"} - set(rows[0])
    if missing:
        raise SteeringError(
            f"{SAMPLE_FILE} is missing columns {sorted(missing)}; rebuild it "
            "with scripts/build_bail_sample_verified.py."
        )
    return rows


def stratified_rows(
    rows: list[dict[str, str]], limit: int
) -> list[dict[str, str]]:
    """A severity-stratified subset of `limit` conversations.

    The limit is split evenly over the five bands (remainder to the earliest
    bands in 1, 0, -1, -2, -3 order), taking each band's first rows in the
    frozen sample's deterministic order — so --limit 20 means 4 conversations
    per band, not 20 friendly ones.
    """
    quota = {
        band: limit // len(SAMPLE_BANDS) + (1 if i < limit % len(SAMPLE_BANDS) else 0)
        for i, band in enumerate(SAMPLE_BANDS)
    }
    picked = []
    for row in rows:
        if quota.get(row["severity_band"], 0) > 0:
            quota[row["severity_band"]] -= 1
            picked.append(row)
    return picked


def build_messages(row: dict[str, str], ordering: str | None) -> list[dict]:
    """Conversation turns; wellbeing check appended for the prompt method,
    bare conversation for the tool method (the tool rides along instead)."""
    messages = []
    for column, role in TRANSCRIPT_TURNS:
        text = (row.get(column) or "").strip()
        if text and text.lower() != "nan":
            messages.append({"role": role, "content": text})
    if ordering is not None:
        messages.append({"role": "user", "content": get_bail_prompt(ordering)})
    return messages


def build_tasks(
    rows: list[dict[str, str]], methods: Sequence[str]
) -> list[dict[str, str]]:
    """One task per generation for each selected elicitation method."""
    tasks = []
    for row in rows:
        meta = {
            "snippet_id": row["snippet_id"],
            "abuse_severity": row["severity_band"],
            "group": row["group"],
        }
        for ordering in ORDERINGS:
            if ordering not in methods:
                continue
            tasks.append(
                {
                    "task_id": f"prompt|{row['snippet_id']}|{ordering}",
                    "phase": "prompt",
                    "ordering": ordering,
                    "row": row,
                    **meta,
                }
            )
        if "tool" in methods:
            tasks.append(
                {
                    "task_id": f"tool|{row['snippet_id']}",
                    "phase": "tool",
                    "ordering": "",
                    "row": row,
                    **meta,
                }
            )
    return tasks


def render_prompt(tokenizer, task: dict) -> str:
    """Chat-template text for one task (tool attached for the tool phase)."""
    messages = build_messages(task["row"], task["ordering"] or None)
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if task["phase"] == "tool":
        kwargs["tools"] = [get_bail_tool(BAIL_MODEL_NAME)]
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception as error:
        raise SteeringError(
            f"Chat template rendering failed for {task['task_id']}: {error}. "
            "If this only happens in the tool phase, the pinned tokenizer's "
            "template may not support tools -- rerun with "
            "--methods bail_first,continue_first."
        ) from error


# --- Parsing ------------------------------------------------------------------


def parse_response(phase: str, text: str) -> dict[str, str]:
    """Bail signal per method: wellbeing emoji or tool-call detection."""
    if phase == "prompt":
        return {
            "wellbeing": get_bail_prompt_status(text),
            "tool_called": "",
            "tool_mentioned": "",
        }
    cleaned = remove_thinking(text)
    called = any(marker in cleaned for marker in TOOL_CALL_MARKERS)
    return {
        "wellbeing": "",
        "tool_called": str(called),
        "tool_mentioned": str(BAIL_TOOL_NAME in cleaned.lower()),
    }


# --- Run folder ---------------------------------------------------------------


def prepare_run_dir(resume: bool, run_info: dict, seed: int) -> Path:
    """New timestamped folder, or the latest one when resuming."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if resume:
        runs = [
            path
            for path in RESULTS_DIR.iterdir()
            if path.is_dir() and (path / "run_info.json").exists()
        ]
        if not runs:
            raise SteeringError("No run folder to resume; run without --resume.")
        run_dir = max(
            runs,
            key=lambda p: json.loads(
                (p / "run_info.json").read_text(encoding="utf-8")
            ).get("started", ""),
        )
        print(f"Resuming {run_dir}")
        return run_dir
    # Microsecond resolution plus the resolved seed in the name: a Slurm array
    # can launch dozens of `run` processes within the same wall-clock second,
    # and second-resolution timestamps alone would collide.
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    base_name = f"{stamp}_bail-steering_seed{seed}"
    # Belt-and-suspenders collision guard: even microsecond stamps can tie
    # under enough concurrency, so retry with a numeric suffix until mkdir
    # (which fails atomically if the directory already exists) succeeds.
    run_dir = RESULTS_DIR / base_name
    suffix = 0
    while True:
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            suffix += 1
            run_dir = RESULTS_DIR / f"{base_name}_{suffix}"
    (run_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    print(f"New run folder: {run_dir}")
    return run_dir


def load_done(run_dir: Path) -> set[str]:
    """custom_ids already generated (for resume)."""
    path = run_dir / RESPONSES_FILE
    if not path.exists():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["custom_id"] for row in csv.DictReader(handle)}


def append_rows(run_dir: Path, rows: list[dict[str, str]]) -> None:
    path = run_dir / RESPONSES_FILE
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


# --- Generation ---------------------------------------------------------------


def generate_batch(
    model, tokenizer, texts: list[str], max_new_tokens: int
) -> tuple[list[str], list[int]]:
    """Batched sampling; returns decoded responses and their token counts."""
    # The chat template already includes BOS, so no extra special tokens.
    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
        )
    input_length = encoded["input_ids"].shape[1]
    generated = output[:, input_length:]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    pad_id = tokenizer.pad_token_id
    counts = [int((seq != pad_id).sum()) for seq in generated]
    return decoded, counts


def run(
    cache_dir: Path,
    device: Device,
    limit: int | None,
    resume: bool,
    selected: Sequence[str] | None,
    methods: Sequence[str],
    seed: int,
    coefficient: float,
    batch_size: int,
    prompt_max_tokens: int,
    tool_max_tokens: int,
    steer_scope: str,
) -> None:
    route = resolve_steering_route()
    validate_vector_pins(route)
    steer_layers, steer_run, risers, fallers = require_pins()
    all_conditions = conditions()
    run_conditions = list(selected) if selected else list(all_conditions)
    unknown = [name for name in run_conditions if name not in all_conditions]
    if unknown:
        raise SteeringError(
            f"unknown condition(s) {unknown}; choose from "
            f"{', '.join(all_conditions)}"
        )
    rows = load_sample()
    if limit is not None:
        rows = stratified_rows(rows, limit)
    run_info = {
        "experiment": "bail-steering",
        "started": datetime.now(timezone.utc).isoformat(),
        "model_id": route.model_id.value,
        "quantization_id": route.quantization_id.value,
        "repository": route.artifact.repository,
        "revision": route.artifact.revision,
        # steer_layer stays for single-layer runs (old analyses read it);
        # steer_layers is the full set actually hooked.
        "steer_layer": steer_layers[0] if len(steer_layers) == 1 else None,
        "steer_layers": list(steer_layers),
        "steer_scope": steer_scope,
        "coefficient": coefficient,
        "coefficient_units": "fraction of residual stream norm (unit direction)",
        "risers": list(risers),
        "fallers": list(fallers),
        "vectors_run": steer_run,
        "vectors_files": [
            str(vectors_path(layer).relative_to(REPO_ROOT))
            for layer in steer_layers
        ],
        "sample_file": str(SAMPLE_FILE.relative_to(REPO_ROOT)),
        "sample_rows": len(rows),
        "methods": list(methods),
        "prompt_max_tokens": prompt_max_tokens,
        "tool_max_tokens": tool_max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        # The ACTUAL seed used (resolved in main() from --seed or randomness),
        # never the SEED constant -- recording this is what makes a randomly
        # seeded run reproducible after the fact.
        "seed": seed,
        "batch_size": batch_size,
    }
    run_dir = prepare_run_dir(resume, run_info, seed)
    done = load_done(run_dir)

    print(f"Loading {route.artifact.repository} ({route.quantization_id.value})...")
    runtime = create_transformers_runtime(route, cache_dir=cache_dir, device=device)
    if runtime.placement.has_cpu_or_offload:
        print(
            "WARNING: parts of the model were placed on CPU/disk "
            f"(cpu={list(runtime.placement.cpu_modules)}); generation will be "
            "very slow. Free GPU memory and restart with --resume."
        )
    model, tokenizer = runtime.model, runtime.tokenizer
    blocks = [_decoder_block(model, layer) for layer in steer_layers]
    torch.manual_seed(seed)

    all_tasks = build_tasks(rows, methods)
    for condition in run_conditions:
        tasks = [
            task
            for task in all_tasks
            if f"{condition}|{task['task_id']}" not in done
        ]
        if not tasks:
            print(f"[{condition}] nothing to do")
            continue
        for task in tasks:
            task["text"] = render_prompt(tokenizer, task)
        # Similar-length prompts batched together -> less padding waste.
        tasks.sort(key=lambda task: len(task["text"]))

        hook_handles = []
        if condition != BASELINE_CONDITION:
            for layer, block in zip(steer_layers, blocks):
                direction, signed = load_direction(condition, coefficient, layer)
                hook_handles.append(
                    block.register_forward_hook(
                        SteeringHook(direction, signed, steer_scope)
                    )
                )
        print(f"[{condition}] {len(tasks)} generations")
        started = time.perf_counter()
        finished = 0
        try:
            for start in range(0, len(tasks), batch_size):
                batch = tasks[start : start + batch_size]
                # Tasks are sorted by prompt length, so batches are mostly
                # single-phase already (tool prompts, carrying the tool spec,
                # are systematically longer than prompt-method ones). Taking
                # the max phase cap over the batch therefore captures nearly
                # all the token-limit savings while never truncating a task
                # below its own phase's cap on the rare mixed batch.
                batch_max_new_tokens = max(
                    tool_max_tokens if task["phase"] == "tool" else prompt_max_tokens
                    for task in batch
                )
                decoded, counts = generate_batch(
                    model,
                    tokenizer,
                    [task["text"] for task in batch],
                    batch_max_new_tokens,
                )
                out_rows = []
                for task, text, count in zip(batch, decoded, counts):
                    out_rows.append(
                        {
                            "custom_id": f"{condition}|{task['task_id']}",
                            "condition": condition,
                            "phase": task["phase"],
                            "ordering": task["ordering"],
                            "snippet_id": task["snippet_id"],
                            "abuse_severity": task["abuse_severity"],
                            "group": task["group"],
                            "n_new_tokens": count,
                            "response_text": text,
                            **parse_response(task["phase"], text),
                        }
                    )
                append_rows(run_dir, out_rows)
                finished += len(batch)
                # Print after every batch (not just every 10) so progress on
                # a Slurm log stays visible even for short/smoke runs.
                rate = finished / (time.perf_counter() - started)
                remaining = (len(tasks) - finished) / rate if rate else 0
                print(
                    f"[{condition}] {finished}/{len(tasks)} "
                    f"({rate * 60:.1f}/min, ~{remaining / 3600:.1f} h left)"
                )
        finally:
            for handle in hook_handles:
                handle.remove()
    print(f"Done. Results in {run_dir}")


def resolve_steering_route() -> LocalTransformersRoute:
    route = resolve_route(
        MODEL_ID,
        ProviderId.LOCAL,
        QUANTIZATION_ID,
        required={Capability.TEXT_GENERATION, Capability.LOCAL_ACTIVATIONS},
    )
    if not isinstance(route, LocalTransformersRoute):
        raise SteeringError(
            "Steering needs a local Transformers route (hook access)."
        )
    return route


# --- CLI ----------------------------------------------------------------------


def _conditions_arg(value: str) -> tuple[str, ...]:
    # Validated against the pinned condition list inside run().
    return tuple(name.strip() for name in value.split(",") if name.strip())


def _methods_arg(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = [name for name in names if name not in METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown method(s) {unknown}; choose from {', '.join(METHODS)}"
        )
    return names


def _seeds_arg(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--seed takes an integer or a comma-separated list, e.g. 42,43,44"
        ) from error


def _layers_arg(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--layer must be an integer or comma-separated integers"
        ) from error
    if len(layers) != len(set(layers)):
        raise argparse.ArgumentTypeError("--layer has duplicate layers")
    return layers


def _steer_arg(value: str) -> float:
    try:
        magnitude = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("--steer must be a number") from error
    if magnitude == 0:
        raise argparse.ArgumentTypeError(
            "--steer 0 would make every steered condition identical to "
            "baseline; use the baseline condition instead"
        )
    return magnitude


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bail experiment under per-emotion activation steering."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(HF_HUB_CACHE),
        help="Hugging Face Hub cache directory (default: %(default)s)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download", help="download the pinned checkpoint")
    run_parser = subparsers.add_parser("run", help="run the experiment")
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
        help=(
            "run only N conversations, split evenly across the five severity "
            "bands (e.g. 20 -> 4 per band); for smoke tests"
        ),
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="continue the latest run folder instead of creating a new one",
    )
    run_parser.add_argument(
        "--conditions",
        type=_conditions_arg,
        default=None,
        help="comma-separated subset of conditions (default: all 21)",
    )
    run_parser.add_argument(
        "--methods",
        type=_methods_arg,
        default=METHODS,
        help=(
            "comma-separated subset of bail elicitation methods: "
            f"{', '.join(METHODS)} (default: all three)"
        ),
    )
    run_parser.add_argument(
        "--seed",
        type=_seeds_arg,
        default=None,
        help=(
            "RNG seed(s) for generation; a comma-separated list runs one "
            "full repeat per seed (one results folder each). Default draws "
            "a single random 32-bit seed. The resolved seed is always "
            "recorded in run_info.json."
        ),
    )
    run_parser.add_argument(
        "--steer",
        type=_steer_arg,
        default=COEFFICIENT,
        help=(
            "steering coefficient in fractions of residual-stream norm; "
            "risers get +STEER, fallers -STEER, so a negative value flips "
            "both (suppresses risers / enhances fallers). Baseline is never "
            "steered (default: %(default)s)"
        ),
    )
    run_parser.add_argument(
        "--layer",
        type=_layers_arg,
        default=None,
        help=(
            "steer at these decoder block(s) instead of the pinned "
            "STEER_LAYER; a comma list (e.g. 11,12,13,14,15) hooks every "
            "listed block, each with its own layer's vector. The set must "
            "contain exactly one STEER_PINS layer, whose riser/faller lists "
            "apply (default: STEER_LAYER)"
        ),
    )
    run_parser.add_argument(
        "--steer-scope",
        choices=("response", "all"),
        default="response",
        help=(
            "which token positions get steered: 'response' = generated "
            "tokens only (the conversation is read unsteered; original "
            "design), 'all' = every position including prefill, so the "
            "model reads the conversation already in the induced state "
            "(default: %(default)s)"
        ),
    )
    run_parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="conversations generated per batch (default: %(default)s)",
    )
    run_parser.add_argument(
        "--prompt-max-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="max new tokens for prompt-method generations (default: %(default)s)",
    )
    run_parser.add_argument(
        "--tool-max-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="max new tokens for tool-method generations (default: %(default)s)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    global STEER_LAYERS
    args = build_parser().parse_args(argv)
    # --layer overrides the pin for this invocation; everything downstream
    # (vector files, hook blocks, run_info's steer_layers) reads the global.
    if getattr(args, "layer", None) is not None:
        STEER_LAYERS = args.layer
    try:
        if args.command == "download":
            route = resolve_steering_route()
            model_path = download_transformers_artifact(route, args.cache_dir)
            print(f"Downloaded pinned checkpoint to: {model_path}")
        else:
            # Resolve the actual seed(s) now: explicit --seed values are used
            # verbatim (a list runs one full repeat per seed), otherwise draw
            # one fresh random seed so parallel Slurm array tasks each get an
            # independent RNG stream. The resolved integer (never the SEED
            # constant) is what gets used and recorded per run.
            seeds = args.seed if args.seed is not None else (secrets.randbits(32),)
            if args.resume and len(seeds) > 1:
                raise SteeringError(
                    "--resume continues one existing folder and cannot be "
                    "combined with multiple seeds; resume one seed at a time."
                )
            for index, seed in enumerate(seeds):
                if len(seeds) > 1:
                    print(f"=== seed {seed} ({index + 1}/{len(seeds)}) ===")
                run(
                    args.cache_dir,
                    args.device,
                    args.limit,
                    args.resume,
                    args.conditions,
                    args.methods,
                    seed,
                    args.steer,
                    args.batch_size,
                    args.prompt_max_tokens,
                    args.tool_max_tokens,
                    args.steer_scope,
                )
    except SteeringError as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
