"""Deterministic, resumable execution loop for one experiment run."""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from llm_runtime.types import ChatMessage, UnsupportedSettingError
from quadratic_voting.experiment import ballots, seeds, statements, transcript
from quadratic_voting.experiment.config import RuntimeRetryPolicy
from quadratic_voting.experiment.store import (
    AcceptedBallot,
    AcceptedStatement,
    BallotAbstention,
    ExperimentStore,
    StatementInvalidMissing,
    TerminalWrite,
)
from quadratic_voting.experiment.types import (
    BarrierReady,
    Clock,
    ExecutionEnvironment,
    NextUnit,
    RunComplete,
    RunId,
    RunStatus,
    RuntimeFailureKind,
    TurnKind,
    VoterGenerator,
    WorkUnit,
)
from quadratic_voting.experiment.ballots import ValidationFailure


def _version(module_name: str) -> str:
    """Return an installed package version, or ``unknown`` when unavailable."""
    try:
        module = __import__(module_name)
        return str(module.__version__)
    except (ImportError, AttributeError):
        return "unknown"


def _uv_lock_hash() -> str:
    for parent in (Path.cwd(), *Path(__file__).resolve().parents):
        lock = parent / "uv.lock"
        if lock.is_file():
            try:
                return hashlib.sha256(lock.read_bytes()).hexdigest()
            except OSError:
                return "unknown"
    return "unknown"


def _git_output(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        timeout=10,
    ).stdout


def _manifest_hash(root: Path, paths: Sequence[bytes], *, include_content: bool) -> str:
    digest = hashlib.sha256()
    for encoded_path in paths:
        path = root / encoded_path.decode("utf-8", errors="surrogateescape")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        if include_content:
            try:
                content = path.read_bytes()
            except OSError:
                content = b"<unreadable>"
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool, str, str, str, str]:
    try:
        root = Path(__file__).resolve().parents[2]
        commit = _git_output(root, "rev-parse", "HEAD").decode().strip()
        status = _git_output(root, "status", "--porcelain=v1", "-z")
        tracked = tuple(
            item for item in _git_output(root, "ls-files", "-z").split(b"\0") if item
        )
        untracked = tuple(
            item
            for item in _git_output(
                root, "ls-files", "--others", "--exclude-standard", "-z"
            ).split(b"\0")
            if item
        )
        diff = _git_output(root, "diff", "--binary", "HEAD", "--")
        return (
            commit or "unknown",
            bool(status),
            _manifest_hash(root, tracked, include_content=True),
            hashlib.sha256(diff).hexdigest(),
            _manifest_hash(root, untracked, include_content=False),
            _manifest_hash(root, untracked, include_content=True),
        )
    except (OSError, subprocess.SubprocessError):
        return ("unknown", False, "unknown", "unknown", "unknown", "unknown")


def _nvidia_smi(field: str) -> tuple[str, ...]:
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def collect_execution_environment(*, dtype: str = "bf16") -> ExecutionEnvironment:
    """Collect best-effort execution provenance without making GPU work mandatory.

    ``unknown`` is used only when an optional package, lock file, host facility, or
    git checkout cannot be queried. The immutable BF16 route remains explicit even
    when torch is unavailable, while the device describes detected execution hardware.
    """
    commit, dirty, tree_hash, diff_hash, untracked_manifest, untracked_tree = (
        _git_state()
    )
    cuda_runtime = "unknown"
    cudnn_version = "unknown"
    gpu_model = "unknown"
    gpu_count = 0
    gpu_capability = "unknown"
    deterministic = False
    tf32 = False
    cudnn_benchmark = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        device = "cuda:0" if cuda_available else "cpu"
        if cuda_available:
            gpu_model = ";".join(
                torch.cuda.get_device_name(index) for index in range(gpu_count)
            )
            gpu_capability = ";".join(
                ".".join(str(part) for part in torch.cuda.get_device_capability(index))
                for index in range(gpu_count)
            )
        cuda_runtime = str(torch.version.cuda or "unknown")
        cudnn = torch.backends.cudnn.version()
        cudnn_version = "unknown" if cudnn is None else str(cudnn)
        deterministic = bool(torch.are_deterministic_algorithms_enabled())
        tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    except (ImportError, RuntimeError, AttributeError):
        device = "unknown"
    try:
        hostname = socket.gethostname() or "unknown"
    except OSError:
        hostname = "unknown"
    uuids = _nvidia_smi("uuid")
    drivers = _nvidia_smi("driver_version")
    try:
        os_release = platform.freedesktop_os_release()
    except OSError:
        os_release = {}
    return ExecutionEnvironment(
        python_version=platform.python_version() or sys.version.split()[0],
        torch_version=_version("torch"),
        transformers_version=_version("transformers"),
        uv_lock_hash=_uv_lock_hash(),
        device=device,
        dtype=dtype,
        hostname=hostname,
        git_commit=commit,
        git_dirty=dirty,
        cuda_runtime_version=cuda_runtime,
        nvidia_driver_version=";".join(drivers) if drivers else "unknown",
        cudnn_version=cudnn_version,
        gpu_model=gpu_model,
        gpu_count=gpu_count,
        gpu_compute_capability=gpu_capability,
        gpu_uuid_hash=(
            hashlib.sha256("\n".join(uuids).encode()).hexdigest()
            if uuids
            else "unknown"
        ),
        os_name=os_release.get("NAME", platform.system() or "unknown"),
        os_version=os_release.get("VERSION_ID", platform.version() or "unknown"),
        kernel_version=platform.release() or "unknown",
        cpu_architecture=platform.machine() or "unknown",
        deterministic_algorithms=deterministic,
        tf32_enabled=tf32,
        cudnn_benchmark=cudnn_benchmark,
        tracked_tree_hash=tree_hash,
        binary_diff_sha256=diff_hash,
        untracked_manifest_hash=untracked_manifest,
        untracked_tree_hash=untracked_tree,
        hostname_hash=hashlib.sha256(hostname.encode()).hexdigest(),
    )


def _canonical_messages(messages: Sequence[ChatMessage]) -> str:
    return json.dumps(
        [
            {"content": message.content, "role": message.role.value}
            for message in messages
        ],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def classify_runtime_failure(error: BaseException) -> RuntimeFailureKind:
    """Classify generation machinery failures without importing CUDA eagerly."""
    try:
        import torch

        if isinstance(error, torch.cuda.OutOfMemoryError):
            return RuntimeFailureKind.OOM
    except (ImportError, AttributeError):
        pass
    if isinstance(error, UnsupportedSettingError):
        return RuntimeFailureKind.PROVIDER_REJECTED
    if isinstance(error, TimeoutError):
        return RuntimeFailureKind.TIMEOUT
    name = type(error).__name__.casefold()
    message = str(error).casefold()
    if isinstance(error, ValueError) or "tokeniz" in name or "tokeniz" in message:
        return RuntimeFailureKind.TOKENIZER
    if "cuda" in message and ("driver" in message or "device" in message):
        return RuntimeFailureKind.DRIVER
    return RuntimeFailureKind.UNKNOWN


def _diagnostics(error: BaseException) -> Mapping[str, str]:
    """Return only approved scalar diagnostics; exception text may contain secrets."""
    return {
        "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "operation": "generate",
    }


def _terminal(
    unit: WorkUnit,
    parsed: ballots.ParsedBallot
    | statements.ParsedStatement
    | tuple[ValidationFailure, ...],
    final_attempt: bool,
) -> tuple[TerminalWrite | None, tuple[ValidationFailure, ...]]:
    if unit.kind is TurnKind.BALLOT:
        if isinstance(parsed, ballots.ParsedBallot):
            terminal = AcceptedBallot(
                parsed.rationale,
                dict(parsed.allocations),
                ballots.ballot_cost(parsed.allocations),
            )
            return terminal, ()
        assert isinstance(parsed, tuple)
        failures = parsed
        return (BallotAbstention() if final_attempt else None), failures
    if isinstance(parsed, statements.ParsedStatement):
        return AcceptedStatement(
            {candidate: (rating, text) for candidate, rating, text in parsed.items}
        ), ()
    assert isinstance(parsed, tuple)
    failures = parsed
    return (StatementInvalidMissing() if final_attempt else None), failures


def run_experiment(
    run_id: RunId,
    *,
    store: ExperimentStore,
    generator: VoterGenerator,
    clock: Clock,
    sleep: Callable[[float], None] = time.sleep,
    runtime_retry: RuntimeRetryPolicy = RuntimeRetryPolicy(),
) -> RunStatus:
    """Execute deterministic units until complete or bounded failures pause the run."""
    del (
        clock
    )  # Store transactions own durable timestamps; the seam is intentionally inert.
    first = store.next_incomplete_unit(run_id)
    if isinstance(first, RunComplete):
        return RunStatus.COMPLETE
    store.set_run_in_progress(run_id)
    total_runtime_failures = 0
    unit_or_barrier: NextUnit = first
    while True:
        if isinstance(unit_or_barrier, RunComplete):
            return RunStatus.COMPLETE
        if isinstance(unit_or_barrier, BarrierReady):
            sealed = store.aggregate_and_seal_round(run_id)
            if isinstance(sealed, RunComplete):
                return RunStatus.COMPLETE
            unit_or_barrier = store.next_incomplete_unit(run_id)
            continue

        unit = unit_or_barrier
        view = store.voter_round_view(run_id, unit.voter_id)
        messages = transcript.render_transcript(view)
        info = store.run_info(run_id)
        seed = seeds.call_seed(
            info.master_seed,
            info.arm,
            info.regime,
            unit.voter_index,
            unit.round_index,
            unit.kind,
            unit.attempt_index,
        )
        prompt_json = _canonical_messages(messages)
        turn_id = store.resolve_turn_id(unit)
        call_id = store.begin_call(
            turn_id,
            unit.attempt_index,
            prompt_json,
            hashlib.sha256(prompt_json.encode("utf-8")).hexdigest(),
            seed,
        )
        try:
            result = generator.generate(messages, info.sampling, seed)
        except Exception as error:
            kind = classify_runtime_failure(error)
            store.interrupt_call_with_failure(call_id, kind, _diagnostics(error))
            total_runtime_failures += 1
            if total_runtime_failures >= runtime_retry.max_failures_per_execution:
                reason = (
                    f"Run {run_id} paused after {total_runtime_failures} total runtime "
                    f"failures at call {call_id}, turn {turn_id}, voter {unit.voter_id} "
                    f"(index {unit.voter_index}), round {unit.round_index}: {kind.value} "
                    f"({type(error).__name__}). The model-visible attempt "
                    "was not consumed. Fix the runtime/provider/device cause, then rerun "
                    f"`qv run --run-id {run_id}` to regenerate the same attempt."
                )
                store.pause_run(run_id, reason)
                return RunStatus.PAUSED
            delay_ms = min(
                runtime_retry.initial_backoff_ms
                * runtime_retry.multiplier ** (total_runtime_failures - 1),
                runtime_retry.max_backoff_ms,
            )
            sleep(float(delay_ms / 1000))
            unit_or_barrier = store.next_incomplete_unit(run_id)
            continue

        if unit.kind is TurnKind.BALLOT:
            parsed_ballot = ballots.parse_and_validate_ballot(
                result.text, view.pending.active, info.credit_budget, known=None
            )
            terminal, failures = _terminal(
                unit,
                parsed_ballot,
                unit.attempt_index == info.max_correction_attempts,
            )
        else:
            parsed_statement = statements.parse_and_validate_statement(
                result.text, view.pending.active
            )
            terminal, failures = _terminal(
                unit,
                parsed_statement,
                unit.attempt_index == info.max_correction_attempts,
            )
        store.commit_call(call_id, result, failures, terminal)
        unit_or_barrier = store.next_incomplete_unit(run_id)
