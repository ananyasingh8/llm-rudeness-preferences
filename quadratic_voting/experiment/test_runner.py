from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections.abc import Sequence
from contextlib import redirect_stderr
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

from llm_runtime.types import (
    ChatMessage,
    ProviderId,
    RuntimeId,
    UnsupportedSettingError,
)
from llm_runtime import LocalTransformersRoute, ModelId, QuantizationId, resolve_route
from quadratic_voting.experiment.runner import (
    classify_runtime_failure,
    collect_execution_environment,
    run_experiment,
)
from quadratic_voting.experiment.config import MatchedSetConfigV1
from quadratic_voting.experiment.seeds import call_seed
from quadratic_voting.experiment.store import (
    CandidateRecord,
    RunDefinition,
    acquire_writer_lock,
    open_sqlite_store,
)
from quadratic_voting.experiment.types import (
    Clock,
    ElicitationArm,
    GenerationResult,
    RudenessLabel,
    RunStatus,
    RuntimeFailureKind,
    SamplerPolicy,
    SamplingProfile,
    StopReason,
    TemplateKind,
    TurnKind,
    VotingRegime,
)


class FixedClock(Clock):
    def now(self) -> datetime:
        return datetime(2026, 1, 1, tzinfo=UTC)


def make_runs(path: Path, *, execution_class: str = "fixture"):
    lock = acquire_writer_lock(path, command="runner-fixture-create")
    store = open_sqlite_store(path, writer_lock=lock, require_writer_lock=True)
    candidates = tuple(
        CandidateRecord(
            str(index),
            RudenessLabel.RUDE if index == 0 else RudenessLabel.NON_RUDE,
            (("user", f"u{index}"), ("agent", f"a{index}")),
            hashlib.sha256(f"candidate-{index}".encode()).hexdigest(),
        )
        for index in range(3)
    )
    release_hash = "a" * 64
    reviewed = execution_class != "fixture"
    review_hash = hashlib.sha256(b"fixture review").hexdigest()
    release = store.ingest_release(
        "fixture",
        "v1",
        "fixture.csv",
        release_hash,
        candidates,
        label_policy_reviewed=reviewed,
        label_policy_review_version="review/v1" if reviewed else None,
        label_policy_review_sha256=review_hash if reviewed else None,
    )
    presentation = store.register_template("card", "v1", "{text}")
    store.render_presentations(release, presentation, lambda record: record.turns[0][1])
    candidate_ids = tuple(
        row[0]
        for row in store.connection.execute(
            "SELECT candidate_id FROM candidate ORDER BY source_row_id"
        )
    )
    sample_id = store.create_sample(
        release, presentation, SamplerPolicy.BALANCED_MATCHED, 7, candidate_ids
    )
    artifact = path.with_suffix(".sample.json")
    store.freeze_sample(sample_id, artifact)
    instructions = {}
    for kind in TemplateKind:
        template_id = store.register_template(
            kind, "v1", f"fixture {kind.value} {{budget}}"
        )
        digest = store.connection.execute(
            "SELECT body_sha256 FROM instruction_template WHERE template_id=?",
            (template_id,),
        ).fetchone()[0]
        instructions[kind] = (template_id, digest)
    route = resolve_route(ModelId.GEMMA_4_E2B_IT, ProviderId.LOCAL, QuantizationId.BF16)
    assert isinstance(route, LocalTransformersRoute)
    sample_row = store.connection.execute(
        "SELECT s.*,r.dataset_name,r.version AS release_version,r.file_sha256,"
        "lp.name AS policy_name,lp.version AS policy_version,lp.rule_sha256,"
        "pt.name AS template_name,pt.version AS template_version,pt.body_sha256 "
        "FROM candidate_sample s JOIN dataset_release r ON r.release_id=s.release_id "
        "JOIN label_policy lp ON lp.label_policy_id=s.label_policy_id "
        "JOIN presentation_template pt ON pt.template_id=s.template_id WHERE s.sample_id=?",
        (sample_id,),
    ).fetchone()
    prompt_config: dict[str, object] = {}
    for kind, (template_id, digest) in instructions.items():
        version = store.connection.execute(
            "SELECT version FROM instruction_template WHERE template_id=?",
            (template_id,),
        ).fetchone()[0]
        prompt_config[
            "final_result" if kind is TemplateKind.FINAL_RESULT else kind.value
        ] = {
            "template_id": template_id,
            "name": kind.value,
            "version": version,
            "expected_sha256": digest,
        }
    strict = MatchedSetConfigV1.model_validate(
        {
            "schema_version": "qv-run-config/v1",
            "canonical_json_version": "qv-canonical-json/v1",
            "prompt_encoding_version": "qv-prompt/v1",
            "seed_version": "qv-seed/v1",
            "sample": {
                "sample_id": sample_id,
                "artifact_path": artifact.resolve(),
                "expected_sha256": sample_row["artifact_sha256"],
                "release": {
                    "release_id": sample_row["release_id"],
                    "dataset_name": sample_row["dataset_name"],
                    "version": sample_row["release_version"],
                    "expected_sha256": sample_row["file_sha256"],
                },
                "label_policy": {
                    "label_policy_id": sample_row["label_policy_id"],
                    "name": sample_row["policy_name"],
                    "version": sample_row["policy_version"],
                    "expected_sha256": sample_row["rule_sha256"],
                    "reviewed": reviewed,
                    "review_version": "review/v1" if reviewed else None,
                    "review_sha256": review_hash if reviewed else None,
                },
                "presentation_template": {
                    "template_id": sample_row["template_id"],
                    "name": sample_row["template_name"],
                    "version": sample_row["template_version"],
                    "expected_sha256": sample_row["body_sha256"],
                },
            },
            "route": {
                "model_id": route.model_id.value,
                "provider_id": route.provider_id.value,
                "quantization_id": route.quantization_id.value,
                "runtime_id": route.runtime_id.value,
                "artifact_repository": route.artifact.repository,
                "artifact_revision": route.artifact.revision,
                "tokenizer_repository": route.artifact.repository,
                "tokenizer_revision": route.artifact.revision,
                "dtype": "bf16",
            },
            "prompts": {
                **prompt_config,
                "reviewed": reviewed,
                "review_version": "review/v1" if reviewed else None,
                "review_sha256": review_hash if reviewed else None,
            },
            "sampling": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 10,
                "max_new_tokens": 100,
            },
            "ballot_retry": {"max_corrections": 3},
            "statement_retry": {"max_corrections": 3},
            "runtime_retry": {
                "max_failures_per_execution": 3,
                "initial_backoff_ms": 1000,
                "multiplier": 2.0,
                "max_backoff_ms": 2000,
            },
            "master_seed": 42,
            "voter_count": 2,
            "credit_budget": 100,
            "sampler_policy": "balanced-matched/v1",
            "presentation_policy": "setup-once-ids-later/v1",
            "tie_policy": "uniform-seeded/v1",
            "action_format": "json-with-rationale/v1",
            "execution_class": execution_class,
        }
    )
    if execution_class != "fixture":
        route_json = json.dumps(
            strict.route.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        store.register_static_route(
            RunDefinition(
                route.model_id.value,
                route.provider_id.value,
                route.quantization_id.value,
                route.artifact.repository,
                route.artifact.revision,
                presentation,
                sample_row["body_sha256"],
                {},
                sample_row["file_sha256"],
                sample_row["artifact_sha256"],
                runtime_id=route.runtime_id.value,
                tokenizer_repository=route.artifact.repository,
                tokenizer_revision=route.artifact.revision,
                dtype="bf16",
                route_registry_hash=hashlib.sha256(route_json.encode()).hexdigest(),
            )
        )
    creation = store.create_matched_set_v1(strict)
    lock.release()
    return store, creation


class ScriptedGenerator:
    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.messages: list[tuple[ChatMessage, ...]] = []
        self.seeds: list[int] = []

    def generate(
        self, messages: Sequence[ChatMessage], profile: SamplingProfile, seed: int
    ) -> GenerationResult:
        del profile
        self.messages.append(tuple(messages))
        self.seeds.append(seed)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("scripted CUDA-free failure")
        final = messages[-1].content
        active_text = final.split("stable order: ", 1)[-1].split(".", 1)[0]
        active = [item.strip() for item in active_text.split(",") if item.strip()]
        if "statement turn" in final:
            text = json.dumps(
                {
                    "statements": [
                        {"candidate_id": item, "rating": "neutral", "statement": "ok"}
                        for item in active
                    ]
                }
            )
        else:
            text = json.dumps(
                {
                    "rationale": "fixture",
                    "allocations": [{"candidate_id": active[0], "votes": 1}],
                }
            )
        return GenerationResult(text, 1, 1, None, StopReason.EOS, 1, {})


class RunnerTests(unittest.TestCase):
    def test_execution_environment_collects_complete_cpu_safe_provenance(self) -> None:
        environment = collect_execution_environment(dtype="bf16")
        self.assertEqual(environment.dtype, "bf16")
        self.assertTrue(environment.os_name)
        self.assertTrue(environment.kernel_version)
        self.assertTrue(environment.cpu_architecture)
        self.assertGreaterEqual(environment.gpu_count, 0)
        for value in (
            environment.uv_lock_hash,
            environment.tracked_tree_hash,
            environment.binary_diff_sha256,
            environment.untracked_manifest_hash,
            environment.untracked_tree_hash,
            environment.hostname_hash,
        ):
            self.assertEqual(len(value), 64)

    def test_seed_and_terminal_selection_drive_real_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store, creation = make_runs(Path(directory) / "qv.sqlite3")
            self.addCleanup(store.close)
            run_id = creation.run_ids[
                (ElicitationArm.ACTION_ONLY, VotingRegime.OPPOSITION)
            ]
            generator = ScriptedGenerator()
            self.assertIs(
                run_experiment(
                    run_id, store=store, generator=generator, clock=FixedClock()
                ),
                RunStatus.COMPLETE,
            )
            expected = call_seed(
                42,
                ElicitationArm.ACTION_ONLY,
                VotingRegime.OPPOSITION,
                0,
                1,
                TurnKind.BALLOT,
                0,
            )
            self.assertEqual(generator.seeds[0], expected)
            self.assertEqual(
                store.connection.execute(
                    "SELECT COUNT(*) FROM ballot WHERE status='accepted'"
                ).fetchone()[0],
                4,
            )

    def test_live_stderr_prints_only_the_current_prompt_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store, creation = make_runs(Path(directory) / "qv.sqlite3")
            self.addCleanup(store.close)
            run_id = creation.run_ids[
                (ElicitationArm.ACTION_ONLY, VotingRegime.OPPOSITION)
            ]
            generator = ScriptedGenerator()
            stderr = StringIO()

            with redirect_stderr(stderr):
                self.assertIs(
                    run_experiment(
                        run_id, store=store, generator=generator, clock=FixedClock()
                    ),
                    RunStatus.COMPLETE,
                )

            output = stderr.getvalue()
            self.assertTrue(generator.messages)
            self.assertNotIn(generator.messages[0][0].content, output)
            for messages in generator.messages:
                current_prompt = messages[-1].content
                self.assertIn(current_prompt, output)
            self.assertIn("prompt:\nRound 1 ballot turn", output)
            self.assertIn("attempt=0 generating", output)
            self.assertIn("response:\n{", output)

    def test_runtime_failures_backoff_pause_and_resume_same_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store, creation = make_runs(Path(directory) / "qv.sqlite3")
            self.addCleanup(store.close)
            run_id = creation.run_ids[
                (ElicitationArm.ACTION_ONLY, VotingRegime.SUPPORT)
            ]
            sleeps: list[float] = []
            generator = ScriptedGenerator(failures=2)
            self.assertIs(
                run_experiment(
                    run_id,
                    store=store,
                    generator=generator,
                    clock=FixedClock(),
                    sleep=sleeps.append,
                ),
                RunStatus.COMPLETE,
            )
            self.assertEqual(sleeps, [1.0, 2.0])
            self.assertEqual(len(set(generator.seeds[:3])), 1)
            self.assertEqual(
                store.connection.execute(
                    "SELECT COUNT(*) FROM runtime_failure"
                ).fetchone()[0],
                2,
            )
            retry_rows = store.connection.execute(
                "SELECT attempt_index,prompt_sha256,seed FROM model_call "
                "ORDER BY started_at LIMIT 3"
            ).fetchall()
            self.assertEqual({row["attempt_index"] for row in retry_rows}, {0})
            self.assertEqual(len({row["prompt_sha256"] for row in retry_rows}), 1)
            self.assertEqual(len({bytes(row["seed"]) for row in retry_rows}), 1)

    def test_runtime_failure_budget_never_resets_after_model_responses(self) -> None:
        class InterleavedGenerator(ScriptedGenerator):
            def __init__(self) -> None:
                super().__init__()
                self.events: list[BaseException | str] = [
                    RuntimeError("failure one"),
                    "{}",
                    RuntimeError("failure two"),
                    "{}",
                    RuntimeError("failure three"),
                ]

            def generate(
                self,
                messages: Sequence[ChatMessage],
                profile: SamplingProfile,
                seed: int,
            ) -> GenerationResult:
                event = self.events.pop(0)
                self.seeds.append(seed)
                if isinstance(event, BaseException):
                    raise event
                return GenerationResult(event, 1, 1, None, StopReason.EOS, 1, {})

        with tempfile.TemporaryDirectory() as directory:
            store, creation = make_runs(Path(directory) / "qv.sqlite3")
            self.addCleanup(store.close)
            run_id = creation.run_ids[
                (ElicitationArm.ACTION_ONLY, VotingRegime.OPPOSITION)
            ]
            sleeps: list[float] = []
            self.assertIs(
                run_experiment(
                    run_id,
                    store=store,
                    generator=InterleavedGenerator(),
                    clock=FixedClock(),
                    sleep=sleeps.append,
                ),
                RunStatus.PAUSED,
            )
            self.assertEqual(sleeps, [1.0, 2.0])
            self.assertEqual(
                store.connection.execute(
                    "SELECT COUNT(*) FROM runtime_failure"
                ).fetchone()[0],
                3,
            )
            self.assertEqual(
                store.connection.execute(
                    "SELECT COUNT(*) FROM model_call WHERE status='committed'"
                ).fetchone()[0],
                2,
            )

    def test_classification(self) -> None:
        self.assertIs(
            classify_runtime_failure(TimeoutError()), RuntimeFailureKind.TIMEOUT
        )
        self.assertIs(
            classify_runtime_failure(ValueError("tokenization")),
            RuntimeFailureKind.TOKENIZER,
        )
        self.assertIs(
            classify_runtime_failure(
                UnsupportedSettingError(
                    setting="seed",
                    value=1,
                    provider_id=ProviderId.OPENROUTER,
                    runtime_id=RuntimeId.OPENAI_COMPATIBLE_HTTP,
                    location="fixture",
                    alternative_provider_id=ProviderId.LOCAL,
                )
            ),
            RuntimeFailureKind.PROVIDER_REJECTED,
        )
        self.assertIs(
            classify_runtime_failure(RuntimeError("other")), RuntimeFailureKind.UNKNOWN
        )

    def test_failure_budget_pauses_actionably_and_healthy_rerun_completes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store, creation = make_runs(Path(directory) / "qv.sqlite3")
            self.addCleanup(store.close)
            run_id = creation.run_ids[
                (ElicitationArm.ACTION_ONLY, VotingRegime.OPPOSITION)
            ]
            self.assertIs(
                run_experiment(
                    run_id,
                    store=store,
                    generator=ScriptedGenerator(failures=3),
                    clock=FixedClock(),
                    sleep=lambda _delay: None,
                ),
                RunStatus.PAUSED,
            )
            reason = store.connection.execute(
                "SELECT pause_reason FROM experiment_run WHERE run_id=?", (run_id,)
            ).fetchone()[0]
            for detail in ("call", "turn", "voter", "round", "Fix"):
                self.assertIn(detail.casefold(), reason.casefold())
            failures = store.connection.execute(
                "SELECT diagnostics_json FROM runtime_failure ORDER BY occurred_at"
            ).fetchall()
            self.assertEqual(len(failures), 3)
            for row in failures:
                diagnostics = json.loads(row[0])
                self.assertEqual(set(diagnostics), {"error_type", "operation"})
                self.assertNotIn("scripted CUDA-free failure", row[0])
            self.assertIs(
                run_experiment(
                    run_id,
                    store=store,
                    generator=ScriptedGenerator(),
                    clock=FixedClock(),
                ),
                RunStatus.COMPLETE,
            )

    def test_four_invalid_ballots_atomically_abstain_then_run_advances(self) -> None:
        class InvalidFirstTurn(ScriptedGenerator):
            def __init__(self) -> None:
                super().__init__()
                self.responses = 0

            def generate(
                self,
                messages: Sequence[ChatMessage],
                profile: SamplingProfile,
                seed: int,
            ) -> GenerationResult:
                if self.responses < 4:
                    self.responses += 1
                    self.seeds.append(seed)
                    return GenerationResult("{}", 1, 1, None, StopReason.EOS, 1, {})
                return super().generate(messages, profile, seed)

        with tempfile.TemporaryDirectory() as directory:
            store, creation = make_runs(Path(directory) / "qv.sqlite3")
            self.addCleanup(store.close)
            run_id = creation.run_ids[
                (ElicitationArm.ACTION_ONLY, VotingRegime.SUPPORT)
            ]
            generator = InvalidFirstTurn()
            self.assertIs(
                run_experiment(
                    run_id, store=store, generator=generator, clock=FixedClock()
                ),
                RunStatus.COMPLETE,
            )
            self.assertEqual(
                store.connection.execute(
                    "SELECT COUNT(*) FROM ballot WHERE status='abstained'"
                ).fetchone()[0],
                1,
            )
            first_turn = store.connection.execute(
                "SELECT turn_id FROM model_call WHERE raw_text='{}' LIMIT 1"
            ).fetchone()[0]
            self.assertEqual(
                store.connection.execute(
                    "SELECT COUNT(*) FROM model_call WHERE turn_id=? AND status='committed'",
                    (first_turn,),
                ).fetchone()[0],
                4,
            )

    def test_four_invalid_statements_atomically_commit_invalid_missing(self) -> None:
        class InvalidFirstStatement(ScriptedGenerator):
            def __init__(self) -> None:
                super().__init__()
                self.invalid_responses = 0
                self.active: tuple[str, ...] = ()

            def generate(
                self,
                messages: Sequence[ChatMessage],
                profile: SamplingProfile,
                seed: int,
            ) -> GenerationResult:
                final = messages[-1].content
                if self.invalid_responses == 0 and "statement turn" in final:
                    active_text = final.split("stable order: ", 1)[-1].split(".", 1)[0]
                    self.active = tuple(
                        item.strip() for item in active_text.split(",") if item.strip()
                    )
                if self.active and self.invalid_responses < 4:
                    attempt = self.invalid_responses
                    self.invalid_responses += 1
                    self.seeds.append(seed)
                    items = [
                        {
                            "candidate_id": candidate,
                            "rating": "neutral",
                            "statement": "valid fixture statement",
                        }
                        for candidate in self.active
                    ]
                    if attempt == 0:
                        items.pop()
                    elif attempt == 1:
                        items[0]["rating"] = "unknown rating"
                    elif attempt == 2:
                        items[0]["statement"] = "   "
                    else:
                        return GenerationResult("{}", 1, 1, None, StopReason.EOS, 1, {})
                    return GenerationResult(
                        json.dumps({"statements": items}),
                        1,
                        1,
                        None,
                        StopReason.EOS,
                        1,
                        {},
                    )
                return super().generate(messages, profile, seed)

        with tempfile.TemporaryDirectory() as directory:
            store, creation = make_runs(Path(directory) / "qv.sqlite3")
            self.addCleanup(store.close)
            run_id = creation.run_ids[
                (ElicitationArm.STATEMENT_THEN_ACTION, VotingRegime.OPPOSITION)
            ]
            generator = InvalidFirstStatement()
            self.assertIs(
                run_experiment(
                    run_id, store=store, generator=generator, clock=FixedClock()
                ),
                RunStatus.COMPLETE,
            )
            statement = store.connection.execute(
                "SELECT s.statement_id,s.turn_id,s.status,t.status AS turn_status "
                "FROM statement s JOIN turn t ON t.turn_id=s.turn_id "
                "WHERE s.status='invalid-missing'"
            ).fetchone()
            self.assertIsNotNone(statement)
            assert statement is not None
            self.assertEqual(statement["turn_status"], "committed")
            calls = store.connection.execute(
                "SELECT call_id,attempt_index,prompt_messages_json FROM model_call "
                "WHERE turn_id=? AND status='committed' ORDER BY attempt_index",
                (statement["turn_id"],),
            ).fetchall()
            self.assertEqual([row["attempt_index"] for row in calls], [0, 1, 2, 3])
            codes = {
                row[0]
                for row in store.connection.execute(
                    "SELECT vf.error_code FROM validation_failure vf JOIN model_call c "
                    "ON c.call_id=vf.call_id WHERE c.turn_id=?",
                    (statement["turn_id"],),
                )
            }
            self.assertTrue(
                {"missing-candidate", "unknown-rating", "empty-statement"} <= codes
            )
            for previous, correction in zip(calls[:-1], calls[1:], strict=True):
                messages = tuple(
                    row[0]
                    for row in store.connection.execute(
                        "SELECT message FROM validation_failure WHERE call_id=? "
                        "ORDER BY ordinal",
                        (previous["call_id"],),
                    )
                )
                self.assertTrue(messages)
                for message in messages:
                    self.assertIn(message, correction["prompt_messages_json"])


if __name__ == "__main__":
    unittest.main()
