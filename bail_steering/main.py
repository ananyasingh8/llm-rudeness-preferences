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
  - conversations: the frozen ConvAbuse bail sample (bail/data/
    convabuse_sample.csv, 1501 rows: 500 friendly / 500 neutral / 501 rude
    split evenly over severity -1/-2/-3), replayed exactly like
    bail/src/convabuse_run.py.
  - elicitation: the paper-verbatim prompt method (both orderings) and tool
    method from bail/prompts/bail_methods.py, generated locally instead of
    over the OpenRouter API so the hook can reach the activations.

The full grid is 21 conditions x 1501 conversations x 3 generations
= 94,563 generations. Runs are resumable per generation, and --conditions
lets you run/prioritize a subset (baseline first).

Until the three pin blocks below are set (steering layer + extraction run,
then the riser/faller lists), `run` fails with instructions.

Usage (from the repo root, on the machine with the GPU):
  uv run python -m bail_steering.main download
  uv run python -m bail_steering.main run --device cuda --limit 6   # smoke
  uv run python -m bail_steering.main run --device cuda --conditions baseline
  uv run python -m bail_steering.main run --device cuda --resume
"""

from __future__ import annotations

import argparse
import csv
import json
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
# Top-10 movers by shift_band_avg in results/2026-08-16_002652_convabuse-e4b.
STEER_RISERS: tuple[str, ...] | None = (
    "enraged",
    "spiteful",
    "vindictive",
    "outraged",
    "angry",
    "insulted",
    "hostile",
    "defiant",
    "sullen",
    "hateful",
)
STEER_FALLERS: tuple[str, ...] | None = (
    "content",
    "refreshed",
    "elated",
    "serene",
    "at ease",
    "hope",
    "peaceful",
    "dependent",
    "happy",
    "nostalgic",
)

BASELINE_CONDITION = "baseline"
VECTORS_PREFIX = "gemma4-e4b-it"  # npz prefix written by the extraction

PHASES = ("prompt", "tool")
BAIL_MODEL_NAME = "Gemma"  # own-model name in the bail tool text (bail/config.py)
MAX_NEW_TOKENS = 1000  # matches bail/config.py BAIL_MAX_TOKENS
# Gemma's recommended sampling settings, applied explicitly (the API run used
# provider defaults; locally we pin them so the run is fully described).
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 64
SEED = 42
BATCH_SIZE = 8  # conversations generated at once; lower this if CUDA OOMs
# (Sized for an 8 GB RTX 3070: ~4.5 GB of 4-bit weights + desktop overhead
# leaves ~2 GB for the KV cache; E4B's 2-head GQA keeps that cheap.)

REPO_ROOT = Path(__file__).parent.parent
SAMPLE_FILE = REPO_ROOT / "bail" / "data" / "convabuse_sample.csv"
EXTRACTION_RESULTS_DIR = REPO_ROOT / "emotion_probing" / "results"
RESULTS_DIR = Path(__file__).parent / "results"
RESPONSES_FILE = "responses.csv"
RESPONSE_COLUMNS = (
    "custom_id",
    "condition",
    "phase",
    "ordering",
    "example_no",
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
    """Forward hook adding a fixed-fraction emotion push to generated tokens.

    Registered on the steered decoder block. Prefill passes (sequence length
    > 1, i.e. the conversation being read) are left untouched; with KV-cache
    generation every later pass computes exactly one generated token, and
    that token's hidden state gets `coefficient x (its residual norm)` added
    in the emotion's unit direction -- Anthropic's "fraction of residual
    stream norm" units. (The first response token is decided during prefill
    and is therefore unsteered; the remaining hundreds are steered.)
    """

    def __init__(self, direction: torch.Tensor, coefficient: float) -> None:
        self.direction = direction  # unit-normalized, float32
        self.coefficient = coefficient  # signed: +0.1 risers, -0.1 fallers

    def __call__(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] != 1:
            return output  # prompt prefill: only steer generated tokens
        direction = self.direction.to(hidden.device)
        norms = hidden.float().norm(dim=-1, keepdim=True)
        steered = hidden + (self.coefficient * norms * direction).to(hidden.dtype)
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered


def require_pins() -> tuple[int, str, tuple[str, ...], tuple[str, ...]]:
    """The pinned (layer, extraction run, risers, fallers), or instructions."""
    if STEER_LAYER is None or STEER_VECTORS_RUN is None:
        raise SteeringError(
            "STEER_LAYER and STEER_VECTORS_RUN are not pinned yet. Run the "
            "E4B extraction sweep (`uv run python -m emotion_probing.extract "
            "run`), pick the layer from its layer_quality.json (prefer a "
            "plateau over a lone spike), then set both constants at the top "
            "of bail_steering/main.py."
        )
    if STEER_RISERS is None or STEER_FALLERS is None:
        raise SteeringError(
            "STEER_RISERS and STEER_FALLERS are not pinned yet. Run the "
            "probing experiment (`uv run python -m emotion_probing.main "
            "--experiment convabuse-e4b run`) and its analysis, then set the "
            "top-10 risers and fallers (by shift vs band 0) at the top of "
            "bail_steering/main.py."
        )
    return STEER_LAYER, STEER_VECTORS_RUN, STEER_RISERS, STEER_FALLERS


def conditions() -> tuple[str, ...]:
    """All 21 run conditions: baseline plus the pinned steered emotions."""
    _, _, risers, fallers = require_pins()
    return (BASELINE_CONDITION,) + risers + fallers


def vectors_path() -> Path:
    layer, run, _, _ = require_pins()
    return (
        EXTRACTION_RESULTS_DIR
        / run
        / f"{VECTORS_PREFIX}_emotion_vectors_layer{layer}.npz"
    )


def validate_vector_pins(route) -> None:
    """The pinned extraction run must match the steering model exactly."""
    layer, _, _, _ = require_pins()
    path = vectors_path()
    if not path.exists():
        raise SteeringError(
            f"Emotion vectors file not found: {path}. Check STEER_VECTORS_RUN "
            "and STEER_LAYER against the extraction run's actual outputs."
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
                f"STEER_LAYER={layer} was not part of the pinned extraction "
                "run's probe_layers; fix the pinning."
            )


def load_direction(condition: str) -> tuple[torch.Tensor, float]:
    """The unit steering direction and signed coefficient for a condition."""
    _, _, risers, _ = require_pins()
    path = vectors_path()
    data = np.load(path)
    if condition not in data.files:
        raise SteeringError(
            f"Emotion '{condition}' not in {path.name} "
            f"({len(data.files)} emotions available)."
        )
    vector = torch.from_numpy(data[condition]).float()
    direction = vector / vector.norm()
    sign = 1.0 if condition in risers else -1.0
    return direction, sign * COEFFICIENT


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
    """The frozen ConvAbuse bail sample (severity-stratified, seed-42)."""
    if not SAMPLE_FILE.exists():
        raise SteeringError(
            f"Frozen sample not found: {SAMPLE_FILE}. It is committed with the "
            "bail workstream; check the repo checkout."
        )
    with SAMPLE_FILE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SteeringError(f"{SAMPLE_FILE} has no rows.")
    return rows


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
    rows: list[dict[str, str]], phases: Sequence[str]
) -> list[dict[str, str]]:
    """One task per generation: 2 prompt orderings + 1 tool call per row."""
    tasks = []
    for row in rows:
        meta = {
            "example_no": row["example_no"],
            "abuse_severity": row["abuse_severity"],
            "group": row["group"],
        }
        if "prompt" in phases:
            for ordering in ORDERINGS:
                tasks.append(
                    {
                        "task_id": f"prompt|{row['example_no']}|{ordering}",
                        "phase": "prompt",
                        "ordering": ordering,
                        "row": row,
                        **meta,
                    }
                )
        if "tool" in phases:
            tasks.append(
                {
                    "task_id": f"tool|{row['example_no']}",
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
            "template may not support tools -- rerun with --phase prompt."
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
    # The paper's text-scan for an OpenAI-style inline call, plus a lenient
    # mention flag so analysis can audit other call syntaxes in the raw text.
    called = f'"name": "{BAIL_TOOL_NAME}"' in cleaned
    return {
        "wellbeing": "",
        "tool_called": str(called),
        "tool_mentioned": str(BAIL_TOOL_NAME in cleaned.lower()),
    }


# --- Run folder ---------------------------------------------------------------


def prepare_run_dir(resume: bool, run_info: dict) -> Path:
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
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = RESULTS_DIR / f"{stamp}_bail-steering"
    run_dir.mkdir()
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


def generate_batch(model, tokenizer, texts: list[str]) -> tuple[list[str], list[int]]:
    """Batched sampling; returns decoded responses and their token counts."""
    # The chat template already includes BOS, so no extra special tokens.
    encoded = tokenizer(
        texts, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
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
    phases: Sequence[str],
) -> None:
    route = resolve_steering_route()
    validate_vector_pins(route)
    steer_layer, steer_run, risers, fallers = require_pins()
    all_conditions = conditions()
    run_conditions = list(selected) if selected else list(all_conditions)
    unknown = [name for name in run_conditions if name not in all_conditions]
    if unknown:
        raise SteeringError(
            f"unknown condition(s) {unknown}; choose from "
            f"{', '.join(all_conditions)}"
        )
    rows = load_sample()
    run_info = {
        "experiment": "bail-steering",
        "started": datetime.now(timezone.utc).isoformat(),
        "model_id": route.model_id.value,
        "quantization_id": route.quantization_id.value,
        "repository": route.artifact.repository,
        "revision": route.artifact.revision,
        "steer_layer": steer_layer,
        "coefficient": COEFFICIENT,
        "coefficient_units": "fraction of residual stream norm (unit direction)",
        "risers": list(risers),
        "fallers": list(fallers),
        "vectors_run": steer_run,
        "vectors_file": str(vectors_path().relative_to(REPO_ROOT)),
        "sample_file": str(SAMPLE_FILE.relative_to(REPO_ROOT)),
        "sample_rows": len(rows),
        "orderings": list(ORDERINGS),
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
    }
    run_dir = prepare_run_dir(resume, run_info)
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
    block = _decoder_block(model, steer_layer)
    torch.manual_seed(SEED)

    all_tasks = build_tasks(rows, phases)
    for condition in run_conditions:
        tasks = [
            task
            for task in all_tasks
            if f"{condition}|{task['task_id']}" not in done
        ]
        if limit is not None:
            tasks = tasks[:limit]
        if not tasks:
            print(f"[{condition}] nothing to do")
            continue
        for task in tasks:
            task["text"] = render_prompt(tokenizer, task)
        # Similar-length prompts batched together -> less padding waste.
        tasks.sort(key=lambda task: len(task["text"]))

        hook_handle = None
        if condition != BASELINE_CONDITION:
            direction, coefficient = load_direction(condition)
            hook_handle = block.register_forward_hook(
                SteeringHook(direction, coefficient)
            )
        print(f"[{condition}] {len(tasks)} generations")
        started = time.perf_counter()
        finished = 0
        try:
            for start in range(0, len(tasks), BATCH_SIZE):
                batch = tasks[start : start + BATCH_SIZE]
                decoded, counts = generate_batch(
                    model, tokenizer, [task["text"] for task in batch]
                )
                out_rows = []
                for task, text, count in zip(batch, decoded, counts):
                    out_rows.append(
                        {
                            "custom_id": f"{condition}|{task['task_id']}",
                            "condition": condition,
                            "phase": task["phase"],
                            "ordering": task["ordering"],
                            "example_no": task["example_no"],
                            "abuse_severity": task["abuse_severity"],
                            "group": task["group"],
                            "n_new_tokens": count,
                            "response_text": text,
                            **parse_response(task["phase"], text),
                        }
                    )
                append_rows(run_dir, out_rows)
                finished += len(batch)
                if finished % (BATCH_SIZE * 10) == 0 or finished == len(tasks):
                    rate = finished / (time.perf_counter() - started)
                    remaining = (len(tasks) - finished) / rate if rate else 0
                    print(
                        f"[{condition}] {finished}/{len(tasks)} "
                        f"({rate * 60:.1f}/min, ~{remaining / 3600:.1f} h left)"
                    )
        finally:
            if hook_handle is not None:
                hook_handle.remove()
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
        help="only the first N generations per condition (smoke tests)",
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
        "--phase",
        choices=["prompt", "tool", "both"],
        default="both",
        help="bail elicitation method(s) to run (default: both)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "download":
            route = resolve_steering_route()
            model_path = download_transformers_artifact(route, args.cache_dir)
            print(f"Downloaded pinned checkpoint to: {model_path}")
        else:
            phases = PHASES if args.phase == "both" else (args.phase,)
            run(
                args.cache_dir,
                args.device,
                args.limit,
                args.resume,
                args.conditions,
                phases,
            )
    except SteeringError as error:
        print(f"error: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
