"""Matched-set creation, execution, and transcript inspection commands."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from llm_runtime import (
    ChatMessage,
    LocalTransformersRoute,
    ModelId,
    ProviderId,
    QuantizationId,
    resolve_route,
)
from llm_runtime.transformers import Device, create_transformers_runtime
from quadratic_voting.experiment import gemma
from quadratic_voting.experiment.artifacts import read_frozen_sample
from quadratic_voting.experiment.config import MatchedSetConfigV1
from quadratic_voting.experiment.runner import (
    collect_execution_environment,
    run_experiment,
)
from quadratic_voting.experiment.seeds import support_removal_draw, tie_break_draw
from quadratic_voting.experiment.store import (
    RunDefinition,
    SqliteExperimentStore,
    open_readonly_sqlite_store,
    open_sqlite_store,
)
from quadratic_voting.experiment.transcript import SETUP_TITLE, render_transcript
from quadratic_voting.experiment.types import (
    CandidateId,
    Clock,
    GenerationResult,
    RunId,
    RunStatus,
    SamplingProfile,
    TemplateId,
    TemplateKind,
    VoterGenerator,
)


class _SystemClock(Clock):
    def now(self) -> datetime:
        return datetime.now(UTC)


class _ExecutionExitReason(StrEnum):
    COMPLETED = "completed"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    ERROR = "error"


class _LazyGenerator(VoterGenerator):
    """Delay model construction until the first actually incomplete call."""

    def __init__(self, factory: Callable[[], VoterGenerator]) -> None:
        self._factory = factory
        self._generator: VoterGenerator | None = None

    def generate(
        self, messages: Sequence[ChatMessage], profile: SamplingProfile, seed: int
    ) -> GenerationResult:
        if self._generator is None:
            self._generator = self._factory()
        return self._generator.generate(messages, profile, seed)


def _positive(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not an integer; expected a positive integer"
        ) from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not positive; use an integer greater than zero"
        )
    return parsed


def _non_negative(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{value!r} is not an integer; expected a non-negative seed"
        ) from error
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"{value!r} is negative; use zero or a positive seed"
        )
    return parsed


def _matched_set_create(args: argparse.Namespace) -> int:
    config = MatchedSetConfigV1.from_json_file(args.config)
    with open_sqlite_store(
        args.db, writer_lock=args.writer_lock, require_writer_lock=True
    ) as store:
        route_json = _canonical_json(config.route.model_dump(mode="json"))
        store.register_static_route(
            RunDefinition(
                model_id=config.route.model_id,
                provider_id=config.route.provider_id,
                quantization_id=config.route.quantization_id,
                artifact_repository=config.route.artifact_repository,
                artifact_revision=config.route.artifact_revision,
                presentation_template_id=TemplateId(
                    config.sample.presentation_template.template_id
                ),
                presentation_template_hash=config.sample.presentation_template.expected_sha256,
                instruction_templates={},
                dataset_release_hash=config.sample.release.expected_sha256,
                sample_artifact_hash=config.sample.expected_sha256,
                runtime_id=config.route.runtime_id,
                tokenizer_repository=config.route.tokenizer_repository,
                tokenizer_revision=config.route.tokenizer_revision,
                dtype=config.route.dtype,
                route_registry_hash=hashlib.sha256(route_json.encode()).hexdigest(),
            )
        )
        creation = store.create_matched_set_v1(config)
    print(f"matched_set_id={creation.matched_set_id}")
    for (arm, regime), run_id in creation.run_ids.items():
        print(f"run_id={run_id} arm={arm.value} regime={regime.value}")
    return 0


def _default_generator(
    args: argparse.Namespace, definition: RunDefinition
) -> VoterGenerator:
    route = resolve_route(
        ModelId(definition.model_id),
        ProviderId(definition.provider_id),
        QuantizationId(definition.quantization_id),
    )
    if not isinstance(route, LocalTransformersRoute):
        raise AssertionError("the closed Gemma BF16 route must be local")
    runtime = create_transformers_runtime(
        route, cache_dir=args.cache_dir, device=args.device
    )
    return gemma.GemmaVoterGenerator(runtime)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _current_run_definition(
    store: SqliteExperimentStore, run_id: RunId
) -> RunDefinition:
    """Resolve the complete current immutable graph for core drift verification."""
    persisted = store.connection.execute(
        "SELECT d.*,c.temperature,c.top_p,c.top_k,c.max_new_tokens,s.artifact_path,"
        "s.artifact_sha256,r.file_sha256,lp.version AS current_label_version,"
        "lp.rule_sha256 AS current_label_hash,pt.body_sha256 AS current_presentation_hash "
        ",ip.reviewed AS current_prompt_reviewed,"
        "ip.review_version AS current_prompt_review_version,"
        "ip.review_sha256 AS current_prompt_review_sha256 "
        "FROM run_definition d JOIN experiment_run er ON er.run_id=d.run_id "
        "JOIN matched_set m ON m.matched_set_id=er.matched_set_id "
        "JOIN experiment_config c ON c.config_id=m.config_id "
        "JOIN experiment_definition ed ON ed.definition_id=c.definition_id "
        "JOIN instruction_profile ip ON ip.profile_id=ed.instruction_profile_id "
        "JOIN candidate_sample s ON s.sample_id=c.sample_id "
        "JOIN dataset_release r ON r.release_id=s.release_id "
        "JOIN label_policy lp ON lp.label_policy_id=s.label_policy_id "
        "JOIN presentation_template pt ON pt.template_id=s.template_id WHERE d.run_id=?",
        (run_id,),
    ).fetchone()
    if persisted is None:
        raise ValueError(
            f"Run preflight refused {run_id} because run_definition is absent. Validation "
            "failed in cli.run_cmds._validate_immutable_route before execution creation, "
            "so no model was loaded. Recreate the matched set from its strict config."
        )
    try:
        route = resolve_route(
            ModelId(persisted["model_id"]),
            ProviderId(persisted["provider_id"]),
            QuantizationId(persisted["quantization_id"]),
        )
    except ValueError as error:
        raise ValueError(
            f"Run preflight refused {run_id} because its persisted route identifiers no "
            "longer map to the closed runtime registry. Resolution failed in "
            "cli.run_cmds._current_run_definition before execution creation. Create a "
            "linked fork using an enabled route; no drift override exists."
        ) from error
    if not isinstance(route, LocalTransformersRoute):
        raise ValueError(
            f"Run preflight refused {run_id} because its route is not a local Transformers "
            "route. The execution CLI cannot honor its seeded local-generation definition. "
            "Create a linked local-route fork and retry."
        )
    serialized_templates = json.loads(persisted["instruction_templates_json"])
    instructions: dict[TemplateKind, tuple[TemplateId, str]] = {}
    for kind in TemplateKind:
        selected = serialized_templates.get(kind.value)
        if not isinstance(selected, list) or len(selected) != 2:
            raise ValueError(
                f"Run preflight refused {run_id} because persisted {kind.value} template "
                "binding is malformed. Verification failed before execution creation. "
                "Restore the immutable definition or recreate the matched set."
            )
        template_id = TemplateId(str(selected[0]))
        current = store.connection.execute(
            "SELECT body_sha256 FROM instruction_template WHERE template_id=? AND name=?",
            (template_id, kind.value),
        ).fetchone()
        if current is None:
            raise ValueError(
                f"Run preflight refused {run_id} because {kind.value} template "
                f"{template_id} no longer resolves. Verification failed before execution. "
                "Restore the immutable catalog row or create a linked fork."
            )
        instructions[kind] = (template_id, str(current["body_sha256"]))
    route_values = {
        "model_id": route.model_id.value,
        "provider_id": route.provider_id.value,
        "quantization_id": route.quantization_id.value,
        "runtime_id": route.runtime_id.value,
        "artifact_repository": route.artifact.repository,
        "artifact_revision": route.artifact.revision,
        "tokenizer_repository": route.artifact.repository,
        "tokenizer_revision": route.artifact.revision,
        "dtype": route.quantization_id.value,
    }
    instruction_json = _canonical_json(
        {kind.value: [str(value[0]), value[1]] for kind, value in instructions.items()}
    )
    sampling_json = _canonical_json(
        {
            "temperature": persisted["temperature"],
            "top_p": persisted["top_p"],
            "top_k": persisted["top_k"],
            "max_new_tokens": persisted["max_new_tokens"],
        }
    )
    _sample, artifact_hash = read_frozen_sample(Path(persisted["artifact_path"]))
    return RunDefinition(
        model_id=route_values["model_id"],
        provider_id=route_values["provider_id"],
        quantization_id=route_values["quantization_id"],
        artifact_repository=route_values["artifact_repository"],
        artifact_revision=route_values["artifact_revision"],
        presentation_template_id=TemplateId(persisted["presentation_template_id"]),
        presentation_template_hash=persisted["current_presentation_hash"],
        instruction_templates=instructions,
        dataset_release_hash=persisted["file_sha256"],
        sample_artifact_hash=artifact_hash,
        runtime_id=route_values["runtime_id"],
        tokenizer_repository=route_values["tokenizer_repository"],
        tokenizer_revision=route_values["tokenizer_revision"],
        dtype=route_values["dtype"],
        route_registry_hash=hashlib.sha256(
            _canonical_json(route_values).encode()
        ).hexdigest(),
        sampling_profile_hash=hashlib.sha256(sampling_json.encode()).hexdigest(),
        instruction_profile_hash=hashlib.sha256(instruction_json.encode()).hexdigest(),
        canonical_json_version=persisted["canonical_json_version"],
        prompt_encoding_version=persisted["prompt_encoding_version"],
        seed_version=persisted["seed_version"],
        source_release_id=persisted["source_release_id"],
        label_policy_id=persisted["label_policy_id"],
        label_policy_version=persisted["current_label_version"],
        label_policy_hash=persisted["current_label_hash"],
        sample_id=persisted["sample_id"],
        prompt_reviewed=bool(persisted["current_prompt_reviewed"]),
        prompt_review_version=persisted["current_prompt_review_version"],
        prompt_review_sha256=persisted["current_prompt_review_sha256"],
    )


def _verify_complete_definition(
    store: SqliteExperimentStore, run_id: RunId, current: RunDefinition
) -> None:
    persisted = store.connection.execute(
        "SELECT * FROM run_definition WHERE run_id=?", (run_id,)
    ).fetchone()
    if persisted is None:
        raise ValueError(
            f"Run definition verification failed because {run_id} has no normalized "
            "definition. The read-only preflight stopped before execution creation. "
            "Recreate the matched set from its strict config."
        )
    current_templates = {
        kind.value: [str(value[0]), value[1]]
        for kind, value in current.instruction_templates.items()
    }
    persisted_fields = set(persisted.keys())
    values: dict[str, object] = {
        field: getattr(current, field)
        for field in RunDefinition.__dataclass_fields__
        if field != "instruction_templates" and field in persisted_fields
    }
    mismatches = {
        field: (persisted[field], value)
        for field, value in values.items()
        if persisted[field] != value
    }
    persisted_templates = json.loads(persisted["instruction_templates_json"])
    if persisted_templates != current_templates:
        mismatches["instruction_templates"] = (
            persisted_templates,
            current_templates,
        )
    if mismatches:
        raise ValueError(
            f"Run preflight refused {run_id} because immutable definition fields drifted: "
            f"{mismatches}. Complete verification failed in cli.run_cmds before execution "
            "creation. Restore the exact route/templates/profile/sample graph or create a "
            "linked fork; no override exists."
        )


def _run(args: argparse.Namespace) -> int:
    run_id = RunId(args.run_id)
    with _open_readonly_store(args.db) as preflight:
        if preflight.run_info(run_id).status is RunStatus.COMPLETE:
            print(f"run_id={run_id} status={RunStatus.COMPLETE.value}")
            return 0
        current_definition = _current_run_definition(preflight, run_id)
        assert current_definition.dtype is not None
        environment = collect_execution_environment(dtype=current_definition.dtype)
        _verify_complete_definition(preflight, run_id, current_definition)
        preflight.preflight_execution(run_id, environment)
    with open_sqlite_store(
        args.db, writer_lock=args.writer_lock, require_writer_lock=True
    ) as store:
        info = store.run_info(run_id)
        factory: Callable[[SamplingProfile], VoterGenerator] | None = (
            args.generator_factory
        )
        generator = _LazyGenerator(
            (lambda: factory(info.sampling))
            if factory
            else (lambda: _default_generator(args, current_definition))
        )
        execution_id = store.begin_execution(run_id, environment)
        exit_reason = _ExecutionExitReason.ERROR
        try:
            status = run_experiment(
                run_id, store=store, generator=generator, clock=_SystemClock()
            )
            exit_reason = (
                _ExecutionExitReason.COMPLETED
                if status is RunStatus.COMPLETE
                else _ExecutionExitReason.PAUSED
            )
        finally:
            store.end_execution(execution_id, exit_reason.value)
        print(f"run_id={run_id} status={status.value}")
        if status is RunStatus.PAUSED:
            reason = store.connection.execute(
                "SELECT pause_reason FROM experiment_run WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            print(f"pause_reason={reason}")
            return 1
    return 0


def complete_without_writer_lock(args: argparse.Namespace) -> bool:
    """Return a complete run before the dispatcher mutates writer-lock metadata."""
    if not args.db.is_file():
        return False
    run_id = RunId(args.run_id)
    with _open_readonly_store(args.db) as store:
        complete = store.run_info(run_id).status is RunStatus.COMPLETE
    if complete:
        print(f"run_id={run_id} status={RunStatus.COMPLETE.value}")
    return complete


def _open_readonly_store(path: Path) -> SqliteExperimentStore:
    """Use the durable core's no-create, no-migration read-only boundary."""
    if not path.is_file():
        raise ValueError(
            f"Read-only database open failed because {path} does not exist. The check "
            "failed in cli.run_cmds._open_readonly_store before SQLite open, so inspect "
            "and verify cannot create or migrate state. Supply an existing database path."
        )
    return open_readonly_sqlite_store(path.resolve())


def _verify(store: Any, run_id: RunId) -> bool:
    from quadratic_voting.experiment.engine import aggregate_round

    info = store.run_info(run_id)
    ok = True
    rounds = store.connection.execute(
        "SELECT r.round_id,r.round_index,o.protected_candidate_id,o.removed_candidate_id "
        'FROM "round" r JOIN round_outcome o ON o.round_id=r.round_id '
        "WHERE r.run_id=? ORDER BY r.round_index",
        (run_id,),
    ).fetchall()
    for row in rounds:
        active = tuple(
            CandidateId(value[0])
            for value in store.connection.execute(
                "SELECT candidate_id FROM round_candidate WHERE round_id=? "
                "ORDER BY sample_position",
                (row["round_id"],),
            )
        )
        ballot_rows = store.connection.execute(
            "SELECT b.ballot_id,a.candidate_id,a.votes FROM turn t JOIN ballot b "
            "ON b.turn_id=t.turn_id LEFT JOIN allocation a ON a.ballot_id=b.ballot_id "
            "WHERE t.round_id=? ORDER BY b.ballot_id",
            (row["round_id"],),
        ).fetchall()
        allocations: dict[str, dict[CandidateId, int]] = {}
        for ballot in ballot_rows:
            values = allocations.setdefault(str(ballot["ballot_id"]), {})
            if ballot["candidate_id"] is not None:
                values[CandidateId(ballot["candidate_id"])] = int(ballot["votes"])
        recomputed = aggregate_round(
            info.regime,
            active,
            tuple(allocations.values()),
            tie_break_draw(info.master_seed, info.arm, info.regime, row["round_index"]),
            support_removal_draw(
                info.master_seed, info.arm, info.regime, row["round_index"]
            ),
        ).result
        expected_protected = row["protected_candidate_id"]
        if (
            recomputed.protected != expected_protected
            or recomputed.removed != row["removed_candidate_id"]
        ):
            ok = False
            print(
                f"VERIFY MISMATCH run={run_id} round={row['round_index']}: persisted "
                f"protected={expected_protected} removed={row['removed_candidate_id']}; "
                f"recomputed protected={recomputed.protected} removed={recomputed.removed}. "
                "Inspect ballots, allocations, and persisted RNG draws."
            )
        else:
            print(f"VERIFY OK run={run_id} round={row['round_index']}")
    return ok


def _inspect(args: argparse.Namespace) -> int:
    run_id = RunId(args.run_id)
    with _open_readonly_store(args.db) as store:
        voters = store.voters(run_id)
        selected = (
            tuple(item for item in voters if item[0] == args.voter_index)
            if args.voter_index is not None
            else voters
        )
        if not selected:
            raise ValueError(
                f"Inspection failed because voter index {args.voter_index} does not exist "
                f"in run {run_id}. Select one of: {', '.join(str(i) for i, _ in voters)}."
            )
        for voter_index, voter_id in selected:
            print(f"=== voter_index={voter_index} voter_id={voter_id} ===")
            for message in render_transcript(store.voter_round_view(run_id, voter_id)):
                if args.round_index is None or (
                    f"Round {args.round_index} " in message.content
                    or SETUP_TITLE in message.content
                    or "Final result" in message.content
                ):
                    print(f"{message.role.value}: {message.content}")
        return 0 if not args.verify or _verify(store, run_id) else 1


def _verify_command(args: argparse.Namespace) -> int:
    run_id = RunId(args.run_id)
    with _open_readonly_store(args.db) as store:
        return 0 if _verify(store, run_id) else 1


def register(
    subparsers: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> None:
    matched_set = subparsers.add_parser(
        "matched-set", help="matched six-run definitions"
    )
    matched_sub = matched_set.add_subparsers(required=True)
    matched = matched_sub.add_parser("create", help="create the six-run matched matrix")
    matched.add_argument(
        "--config",
        type=Path,
        required=True,
        help="strict qv-run-config/v1 JSON file containing the complete definition",
    )
    matched.set_defaults(handler=_matched_set_create, mutates_db=True)

    legacy_matched = subparsers.add_parser("matched-set-create", help=argparse.SUPPRESS)
    legacy_matched.add_argument("--config", type=Path, required=True)
    legacy_matched.set_defaults(handler=_matched_set_create, mutates_db=True)

    run = subparsers.add_parser("run", help="run or resume one experiment")
    run.add_argument("--run-id", required=True)
    run.add_argument(
        "--cache-dir", type=Path, default=Path("~/.cache/huggingface").expanduser()
    )
    run.add_argument("--device", type=Device, choices=list(Device), default=Device.AUTO)
    run.set_defaults(handler=_run, mutates_db=True)

    inspect = subparsers.add_parser(
        "inspect", help="render persisted model-visible transcripts"
    )
    inspect.add_argument("--run-id", required=True)
    inspect.add_argument("--voter-index", type=_non_negative)
    inspect.add_argument("--round", dest="round_index", type=_positive)
    inspect.add_argument("--verify", action="store_true")
    inspect.set_defaults(handler=_inspect)

    verify = subparsers.add_parser(
        "verify", help="verify persisted outcomes without migrating the database"
    )
    verify.add_argument("--run-id", required=True)
    verify.set_defaults(handler=_verify_command)
