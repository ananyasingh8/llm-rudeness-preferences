from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

from quadratic_voting.experiment.store import (
    AcceptedBallot,
    CandidateRecord,
    RunDefinition,
    SourceAnnotation,
    acquire_writer_lock,
    open_sqlite_store,
)
from quadratic_voting.experiment.errors import FreezeMismatchError
from quadratic_voting.experiment.config import MatchedSetConfigV1
from quadratic_voting.experiment.test_config import valid_config
from quadratic_voting.experiment.types import (
    CandidateId,
    ElicitationArm,
    ExecutionEnvironment,
    FinalResultEvent,
    FreezePoint,
    MatchedSetConfig,
    GenerationResult,
    ReleaseId,
    RudenessLabel,
    RunComplete,
    RunId,
    RunStatus,
    SamplerPolicy,
    SamplingProfile,
    StopReason,
    TemplateId,
    TemplateKind,
    VotingRegime,
    WorkUnit,
)


class StoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = open_sqlite_store(self.root / "qv.sqlite3")

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def _catalog(
        self, *, reviewed: bool = False
    ) -> tuple[ReleaseId, TemplateId, tuple[CandidateId, ...]]:
        digest = "a" * 64
        release_id = self.store.ingest_release(
            "fixture",
            "v1",
            "fixture.json",
            digest,
            (
                CandidateRecord(
                    "one", RudenessLabel.RUDE, (("user", "u1"), ("agent", "a1")), digest
                ),
                CandidateRecord(
                    "two",
                    RudenessLabel.NON_RUDE,
                    (("user", "u2"), ("agent", "a2")),
                    "b" * 64,
                ),
            ),
            label_policy_reviewed=reviewed,
            label_policy_review_version="review/v1" if reviewed else None,
            label_policy_review_sha256="e" * 64 if reviewed else None,
        )
        presentation = self.store.register_template("card", "v1", "{text}")
        self.store.render_presentations(
            release_id, presentation, lambda record: record.turns[0][1]
        )
        candidates = tuple(
            CandidateId(row[0])
            for row in self.store.connection.execute(
                "SELECT candidate_id FROM candidate ORDER BY source_row_id"
            )
        )
        return release_id, presentation, candidates

    def _definition(
        self,
        presentation: TemplateId,
        artifact_hash: str,
        instruction_bodies: Mapping[TemplateKind, str] | None = None,
    ) -> RunDefinition:
        instructions = {}
        for kind in TemplateKind:
            body = (
                f"{kind.value} {{budget}}"
                if instruction_bodies is None
                else instruction_bodies[kind]
            )
            template_id = self.store.register_template(kind, "v1", body)
            template_hash = self.store.connection.execute(
                "SELECT body_sha256 FROM instruction_template WHERE template_id=?",
                (template_id,),
            ).fetchone()[0]
            instructions[kind] = (template_id, template_hash)
        return RunDefinition(
            "model",
            "provider",
            "bf16",
            "repo",
            "revision",
            presentation,
            self.store.connection.execute(
                "SELECT body_sha256 FROM presentation_template WHERE template_id=?",
                (presentation,),
            ).fetchone()[0],
            instructions,
            "a" * 64,
            artifact_hash,
        )

    def test_migration_and_strict_checks(self) -> None:
        self.assertEqual(
            self.store.connection.execute(
                "SELECT version FROM schema_version"
            ).fetchone()[0],
            1,
        )
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.connection.execute(
                "INSERT INTO experiment_run VALUES ('x','missing','bad','support','created',NULL)"
            )
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.connection.execute(
                "INSERT INTO allocation VALUES ('missing','candidate',0)"
            )

    def test_newer_schema_is_actionably_refused(self) -> None:
        path = self.root / "newer.sqlite3"
        other = open_sqlite_store(path)
        other.connection.execute("INSERT INTO schema_version VALUES (2)")
        other.close()
        with self.assertRaisesRegex(RuntimeError, "newer.*known version 1"):
            open_sqlite_store(path)

    def test_freeze_validate_and_six_matched_runs(self) -> None:
        release, presentation, candidates = self._catalog(reviewed=True)
        sample_id = self.store.create_sample(
            release, presentation, SamplerPolicy.BALANCED_MATCHED, 4, candidates
        )
        config = MatchedSetConfig(2, 19, SamplingProfile(0.7, 0.9, 20, 50))
        draft_definition = RunDefinition(
            "model",
            "provider",
            "bf16",
            "repo",
            "revision",
            presentation,
            self.store.connection.execute(
                "SELECT body_sha256 FROM presentation_template WHERE template_id=?",
                (presentation,),
            ).fetchone()[0],
            {},
            "a" * 64,
            "unused-until-frozen",
        )
        from quadratic_voting.experiment.artifacts import FrozenCandidateSample

        with self.assertRaisesRegex(ValueError, "status is draft.*qv sample freeze"):
            self.store.create_matched_set(
                FrozenCandidateSample(tuple(candidates)),
                sample_id,
                config,
                draft_definition,
            )
        artifact = self.root / "sample.json"
        sample = self.store.freeze_sample(sample_id, artifact)
        self.assertTrue(Path(f"{artifact}.provenance.json").exists())
        self.assertEqual(self.store.validate_sample(artifact), sample_id)
        definition = self._definition(
            presentation, hashlib.sha256(artifact.read_bytes()).hexdigest()
        )
        creation = self.store.create_matched_set(sample, sample_id, config, definition)
        self.assertEqual(len(creation.run_ids), 6)
        self.assertEqual(
            set(creation.run_ids),
            {(arm, regime) for arm in ElicitationArm for regime in VotingRegime},
        )
        selected_run = creation.run_ids[
            (ElicitationArm.STATEMENT_THEN_ACTION, VotingRegime.OPPOSITION)
        ]
        info = self.store.run_info(selected_run)
        self.assertEqual(info.run_id, selected_run)
        self.assertEqual(info.matched_set_id, creation.matched_set_id)
        self.assertEqual(info.arm, ElicitationArm.STATEMENT_THEN_ACTION)
        self.assertEqual(info.regime, VotingRegime.OPPOSITION)
        self.assertEqual(info.status, RunStatus.CREATED)
        self.assertEqual(info.master_seed, config.master_seed)
        self.assertEqual(info.voter_count, config.voter_count)
        self.assertEqual(info.credit_budget, config.credit_budget)
        self.assertEqual(
            info.max_correction_attempts, config.retry_policy.max_correction_attempts
        )
        self.assertEqual(
            info.max_consecutive_runtime_failures,
            config.max_consecutive_runtime_failures,
        )
        self.assertEqual(info.sampling, config.sampling)
        run_voters = self.store.voters(selected_run)
        self.assertEqual(tuple(index for index, _voter_id in run_voters), (0, 1))
        self.assertEqual(len({voter_id for _index, voter_id in run_voters}), 2)
        with self.assertRaisesRegex(
            ValueError, "Run-info lookup failed.*does not exist"
        ):
            self.store.run_info(RunId("missing-run"))
        with self.assertRaisesRegex(ValueError, "Voter lookup failed.*does not exist"):
            self.store.voters(RunId("missing-run"))
        permutations: dict[int, set[tuple[str, ...]]] = {0: set(), 1: set()}
        for row in self.store.connection.execute(
            "SELECT voter_id,voter_index FROM voter"
        ):
            permutation = tuple(
                item[0]
                for item in self.store.connection.execute(
                    "SELECT candidate_id FROM voter_permutation WHERE voter_id=? ORDER BY position",
                    (row[0],),
                )
            )
            permutations[row[1]].add(permutation)
        self.assertEqual(
            {index: len(values) for index, values in permutations.items()}, {0: 1, 1: 1}
        )
        self.assertEqual(
            self.store.connection.execute('SELECT COUNT(*) FROM "round"').fetchone()[0],
            6,
        )
        candidate_rows = self.store.candidate_rows()
        self.assertEqual(
            {row["rudeness_label"] for row in candidate_rows}, {"rude", "non_rude"}
        )
        self.assertEqual(
            {row["candidate_id"] for row in candidate_rows}, set(candidates)
        )
        snapshots = self.store.round_candidate_rows()
        self.assertEqual(len(snapshots), 12)
        self.assertEqual(
            set(snapshots[0]),
            {"round_id", "run_id", "round_index", "candidate_id", "sample_position"},
        )

    def test_validate_detects_changed_artifact(self) -> None:
        release, presentation, candidates = self._catalog()
        sample_id = self.store.create_sample(
            release, presentation, SamplerPolicy.BALANCED_MATCHED, 4, candidates
        )
        artifact = self.root / "sample.json"
        self.store.freeze_sample(sample_id, artifact)
        artifact.write_text('["changed"]', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "do not match"):
            self.store.validate_sample(artifact)

    def test_freeze_reconciliation_never_overwrites_mismatching_final(self) -> None:
        release, presentation, candidates = self._catalog()
        sample_id = self.store.create_sample(
            release, presentation, SamplerPolicy.BALANCED_MATCHED, 4, candidates
        )
        artifact = self.root / "pending.json"
        fail_at = self.store._commit_ordinal + 2

        def fail_f2(ordinal: int) -> None:
            if ordinal == fail_at:
                raise RuntimeError("simulated F2 crash")

        self.store._commit_hook = fail_f2
        with self.assertRaisesRegex(RuntimeError, "simulated F2 crash"):
            self.store.freeze_sample(sample_id, artifact)
        self.store._commit_hook = lambda _ordinal: None
        state = self.store.connection.execute(
            "SELECT status FROM candidate_sample WHERE sample_id=?", (sample_id,)
        ).fetchone()[0]
        self.assertEqual(state, "freeze_pending")
        artifact.write_bytes(b'["FOREIGN"]')
        with self.assertRaisesRegex(FreezeMismatchError, "not overwritten"):
            self.store.reconcile_sample(sample_id, artifact)
        self.assertEqual(artifact.read_bytes(), b'["FOREIGN"]')
        self.assertEqual(
            self.store.connection.execute(
                "SELECT status FROM candidate_sample WHERE sample_id=?", (sample_id,)
            ).fetchone()[0],
            "freeze_pending",
        )

    def test_adversarial_sql_rejects_seed_type_and_cross_run_turn(self) -> None:
        release, presentation, candidates = self._catalog()
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.connection.execute(
                "INSERT INTO candidate_sample VALUES "
                "('bad',?,?,?,?,1,2,'draft',NULL,NULL,NULL,NULL)",
                (
                    release,
                    self.store.connection.execute(
                        "SELECT label_policy_id FROM label_policy LIMIT 1"
                    ).fetchone()[0],
                    presentation,
                    SamplerPolicy.BALANCED_MATCHED.value,
                ),
            )
        sample_id = self.store.create_sample(
            release, presentation, SamplerPolicy.BALANCED_MATCHED, 4, candidates
        )
        artifact = self.root / "lineage.json"
        sample = self.store.freeze_sample(sample_id, artifact)
        creation = self.store.create_matched_set(
            sample,
            sample_id,
            MatchedSetConfig(1, (1 << 64) - 1, SamplingProfile(0.7, 0.9, 20, 50)),
            self._definition(
                presentation, hashlib.sha256(artifact.read_bytes()).hexdigest()
            ),
        )
        runs = tuple(creation.run_ids.values())
        voter = self.store.connection.execute(
            "SELECT voter_id FROM voter WHERE run_id=?", (runs[0],)
        ).fetchone()[0]
        round_id = self.store.connection.execute(
            'SELECT round_id FROM "round" WHERE run_id=?', (runs[1],)
        ).fetchone()[0]
        with self.assertRaisesRegex(sqlite3.IntegrityError, "share run"):
            self.store.connection.execute(
                "INSERT INTO turn VALUES ('cross-run',?,?, 'ballot','pending')",
                (round_id, voter),
            )

    def test_canonical_setup_template_is_rendered_once(self) -> None:
        from quadratic_voting.experiment.transcript import (
            TEMPLATE_BODIES,
            render_transcript,
        )

        release, presentation, candidates = self._catalog()
        sample_id = self.store.create_sample(
            release, presentation, SamplerPolicy.BALANCED_MATCHED, 4, candidates
        )
        artifact = self.root / "canonical-sample.json"
        sample = self.store.freeze_sample(sample_id, artifact)
        definition = self._definition(
            presentation,
            hashlib.sha256(artifact.read_bytes()).hexdigest(),
            TEMPLATE_BODIES,
        )
        creation = self.store.create_matched_set(
            sample,
            sample_id,
            MatchedSetConfig(1, 19, SamplingProfile(0.7, 0.9, 20, 50)),
            definition,
        )
        run_id = creation.run_ids[(ElicitationArm.ACTION_ONLY, VotingRegime.SUPPORT)]
        voter_id = self.store.voters(run_id)[0][1]
        view = self.store.voter_round_view(run_id, voter_id)
        self.assertEqual(view.setup.instructions, "")
        transcript = render_transcript(view)
        rendered = "\n".join(message.content for message in transcript)
        self.assertEqual(
            rendered.count("Quadratic voting experiment instructions (v3)"), 1
        )
        self.assertNotIn("Additional frozen instructions", rendered)

    def test_composed_setup_template_drift_is_actionably_rejected(self) -> None:
        from quadratic_voting.experiment.transcript import TEMPLATE_BODIES

        release, presentation, candidates = self._catalog()
        sample_id = self.store.create_sample(
            release, presentation, SamplerPolicy.BALANCED_MATCHED, 4, candidates
        )
        artifact = self.root / "drift-sample.json"
        sample = self.store.freeze_sample(sample_id, artifact)
        drifted = dict(TEMPLATE_BODIES)
        drifted[TemplateKind.SETUP] = (
            TEMPLATE_BODIES[TemplateKind.SETUP] + "\nSilently altered setup."
        )
        definition = self._definition(
            presentation,
            hashlib.sha256(artifact.read_bytes()).hexdigest(),
            drifted,
        )
        creation = self.store.create_matched_set(
            sample,
            sample_id,
            MatchedSetConfig(1, 19, SamplingProfile(0.7, 0.9, 20, 50)),
            definition,
        )
        run_id = creation.run_ids[(ElicitationArm.ACTION_ONLY, VotingRegime.SUPPORT)]
        voter_id = self.store.voters(run_id)[0][1]
        with self.assertRaisesRegex(RuntimeError, "setup drift detected.*Re-register"):
            self.store.voter_round_view(run_id, voter_id)

    def test_t1_t3_barrier_and_terminal_aggregation(self) -> None:
        release, presentation, candidates = self._catalog()
        sample_id = self.store.create_sample(
            release, presentation, SamplerPolicy.BALANCED_MATCHED, 4, candidates
        )
        artifact = self.root / "sample.json"
        sample = self.store.freeze_sample(sample_id, artifact)
        definition = self._definition(
            presentation, hashlib.sha256(artifact.read_bytes()).hexdigest()
        )
        creation = self.store.create_matched_set(
            sample,
            sample_id,
            MatchedSetConfig(1, 19, SamplingProfile(0.7, 0.9, 20, 50)),
            definition,
        )
        run_id = creation.run_ids[(ElicitationArm.ACTION_ONLY, VotingRegime.SUPPORT)]
        with self.assertRaisesRegex(RuntimeError, "incomplete ballot turn"):
            self.store.aggregate_and_seal_round(run_id)
        self.store.set_run_in_progress(run_id)
        unit = self.store.next_incomplete_unit(run_id)
        self.assertIsInstance(unit, WorkUnit)
        assert isinstance(unit, WorkUnit)
        prompt = json.dumps(
            [{"role": "user", "content": "vote"}], separators=(",", ":")
        )
        call_id = self.store.begin_call(
            self.store.resolve_turn_id(unit),
            unit.attempt_index,
            prompt,
            hashlib.sha256(prompt.encode()).hexdigest(),
            7,
        )
        self.store.commit_call(
            call_id,
            GenerationResult("raw", 3, 2, (1, 2), StopReason.EOS, 10, {}),
            (),
            AcceptedBallot("why", {candidates[0]: 2}, 4),
        )
        self.assertIsInstance(self.store.aggregate_and_seal_round(run_id), RunComplete)
        self.assertIsInstance(self.store.next_incomplete_unit(run_id), RunComplete)
        self.assertEqual(
            self.store.connection.execute(
                "SELECT COUNT(*) FROM round_outcome"
            ).fetchone()[0],
            1,
        )
        self.assertEqual(
            self.store.connection.execute(
                "SELECT COUNT(*) FROM rng_draw WHERE stream_domain='support-removal'"
            ).fetchone()[0],
            1,
        )
        voter_id = self.store.connection.execute(
            "SELECT voter_id FROM voter WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        view = self.store.voter_round_view(run_id, voter_id)
        self.assertEqual(view.setup.instructions, "setup 100")
        self.assertIsInstance(view.history[-1], FinalResultEvent)
        final = view.history[-1]
        assert isinstance(final, FinalResultEvent)
        self.assertEqual(final.winner, candidates[0])

        def strings(value: object) -> tuple[str, ...]:
            if isinstance(value, str):
                return (value,)
            if is_dataclass(value) and not isinstance(value, type):
                return tuple(
                    text
                    for field in fields(value)
                    for text in strings(getattr(value, field.name))
                )
            if isinstance(value, (tuple, list, set, frozenset)):
                return tuple(text for item in value for text in strings(item))
            if isinstance(value, dict):
                return tuple(
                    text
                    for key, item in value.items()
                    for text in (*strings(key), *strings(item))
                )
            return ()

        visible = "\n".join(strings(view)).casefold()
        for internal_marker in (
            "started_at",
            "committed_at",
            "interrupted",
            "invocation",
            "resume marker",
            "restart artifact",
        ):
            self.assertNotIn(internal_marker, visible)

    def _strict_config(
        self,
        sample_id: str,
        artifact: Path,
        definition: RunDefinition,
        *,
        execution_class: str = "fixture",
    ) -> MatchedSetConfigV1:
        data = valid_config()
        sample = self.store.connection.execute(
            "SELECT s.*,r.dataset_name,r.version AS release_version,r.file_sha256,"
            "lp.name AS policy_name,lp.version AS policy_version,lp.rule_sha256,"
            "lp.reviewed AS policy_reviewed,lp.review_version AS policy_review_version,"
            "lp.review_sha256 AS policy_review_sha256,"
            "pt.name AS template_name,pt.version AS template_version,pt.body_sha256 "
            "FROM candidate_sample s JOIN dataset_release r ON r.release_id=s.release_id "
            "JOIN label_policy lp ON lp.label_policy_id=s.label_policy_id "
            "JOIN presentation_template pt ON pt.template_id=s.template_id "
            "WHERE s.sample_id=?",
            (sample_id,),
        ).fetchone()
        data["sample"] = {
            "sample_id": sample_id,
            "artifact_path": artifact.resolve(),
            "expected_sha256": sample["artifact_sha256"],
            "release": {
                "release_id": sample["release_id"],
                "dataset_name": sample["dataset_name"],
                "version": sample["release_version"],
                "expected_sha256": sample["file_sha256"],
            },
            "label_policy": {
                "label_policy_id": sample["label_policy_id"],
                "name": sample["policy_name"],
                "version": sample["policy_version"],
                "expected_sha256": sample["rule_sha256"],
                "reviewed": bool(sample["policy_reviewed"]),
                "review_version": sample["policy_review_version"],
                "review_sha256": sample["policy_review_sha256"],
            },
            "presentation_template": {
                "template_id": sample["template_id"],
                "name": sample["template_name"],
                "version": sample["template_version"],
                "expected_sha256": sample["body_sha256"],
            },
        }
        prompts: dict[str, object] = {}
        for kind, (template_id, digest) in definition.instruction_templates.items():
            row = self.store.connection.execute(
                "SELECT version FROM instruction_template WHERE template_id=?",
                (template_id,),
            ).fetchone()
            key = "final_result" if kind is TemplateKind.FINAL_RESULT else kind.value
            prompts[key] = {
                "template_id": template_id,
                "name": kind.value,
                "version": row["version"],
                "expected_sha256": digest,
            }
        prompts.update(
            {
                "reviewed": execution_class != "fixture",
                "review_version": "prompt-review/v1"
                if execution_class != "fixture"
                else None,
                "review_sha256": "f" * 64 if execution_class != "fixture" else None,
            }
        )
        data["prompts"] = prompts
        data["route"] = {
            "model_id": definition.model_id,
            "provider_id": definition.provider_id,
            "quantization_id": definition.quantization_id,
            "runtime_id": "transformers",
            "artifact_repository": definition.artifact_repository,
            "artifact_revision": definition.artifact_revision,
            "tokenizer_repository": definition.artifact_repository,
            "tokenizer_revision": definition.artifact_revision,
            "dtype": "bf16",
        }
        data["execution_class"] = execution_class
        return MatchedSetConfigV1.model_validate(data)

    def test_strict_config_maps_to_normalized_graph_and_exports_lineage(self) -> None:
        release, presentation, candidates = self._catalog(reviewed=True)
        self.store.connection.execute(
            "INSERT INTO source_annotation VALUES (?,?,?,?,?)",
            (candidates[0], 0, "f" * 64, "is_abuse.-1", "1"),
        )
        sample_id = self.store.create_sample(
            release, presentation, SamplerPolicy.BALANCED_MATCHED, 4, candidates
        )
        artifact = self.root / "strict.json"
        self.store.freeze_sample(sample_id, artifact)
        definition = self._definition(
            presentation, hashlib.sha256(artifact.read_bytes()).hexdigest()
        )
        config = self._strict_config(str(sample_id), artifact, definition)
        path = self.store.path
        self.store.close()
        with acquire_writer_lock(path) as lock:
            self.store = open_sqlite_store(
                path, writer_lock=lock, require_writer_lock=True
            )
            creation = self.store.create_matched_set_v1(config)
            self.assertEqual(len(creation.run_ids), 6)
            config_row = self.store.experiment_config_rows()[0]
            self.assertEqual(config_row["schema_version"], "qv-run-config/v1")
            self.assertEqual(
                config_row["canonical_json_version"], "qv-canonical-json/v1"
            )
            self.assertEqual(config_row["model_id"], "model")
            self.assertEqual(config_row["tokenizer_revision"], "revision")
            self.assertNotIn("config_json", config_row)
            candidate = self.store.candidate_rows()[0]
            self.assertEqual(candidate["label_policy_version"], "v1")
            self.assertIn("presentation_template_version", candidate)
            self.assertEqual(
                self.store.source_annotation_rows()[0]["source_label"], "is_abuse.-1"
            )
            self.assertEqual(len(self.store.candidate_presentation_rows()), 2)
            permutations = self.store.voter_permutation_rows()
            self.assertEqual(len(permutations), 24)
            self.assertEqual({row["position"] for row in permutations}, {0, 1})
            definitions = self.store.run_definition_rows()
            self.assertEqual(definitions, self.store.run_definition_rows())
            self.assertEqual(
                tuple(row["run_id"] for row in definitions),
                tuple(sorted(creation.run_ids.values())),
            )
            self.assertEqual(
                set(definitions[0]),
                {
                    "run_id",
                    "model_id",
                    "provider_id",
                    "quantization_id",
                    "artifact_repository",
                    "artifact_revision",
                    "presentation_template_id",
                    "presentation_template_hash",
                    "instruction_templates_json",
                    "dataset_release_hash",
                    "sample_artifact_hash",
                    "runtime_id",
                    "tokenizer_repository",
                    "tokenizer_revision",
                    "dtype",
                    "route_registry_hash",
                    "sampling_profile_hash",
                    "instruction_profile_hash",
                    "canonical_json_version",
                    "prompt_encoding_version",
                    "seed_version",
                    "source_release_id",
                    "label_policy_id",
                    "label_policy_version",
                    "label_policy_hash",
                    "sample_id",
                },
            )
            serialized = json.dumps(definitions, sort_keys=True).casefold()
            for secret_marker in (
                "authorization",
                "api_key",
                "access_token",
                "password",
                "diagnostics_json",
                "raw_text",
            ):
                self.assertNotIn(secret_marker, serialized)
            self.store.close()
        self.store = open_sqlite_store(path)

    def test_execution_provenance_and_primary_dirty_preflight(self) -> None:
        release, presentation, candidates = self._catalog(reviewed=True)
        sample_id = self.store.create_sample(
            release, presentation, SamplerPolicy.BALANCED_MATCHED, 4, candidates
        )
        artifact = self.root / "primary.json"
        self.store.freeze_sample(sample_id, artifact)
        definition = self._definition(
            presentation, hashlib.sha256(artifact.read_bytes()).hexdigest()
        )
        config = self._strict_config(
            str(sample_id), artifact, definition, execution_class="primary"
        )
        path = self.store.path
        self.store.close()
        lock = acquire_writer_lock(path)
        self.store = open_sqlite_store(path, writer_lock=lock, require_writer_lock=True)
        route_definition = replace(
            definition,
            runtime_id=config.route.runtime_id,
            tokenizer_repository=config.route.tokenizer_repository,
            tokenizer_revision=config.route.tokenizer_revision,
            dtype=config.route.dtype,
            route_registry_hash=hashlib.sha256(
                json.dumps(
                    config.route.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        )
        self.store.register_static_route(route_definition)
        creation = self.store.create_matched_set_v1(config)
        run_id = next(iter(creation.run_ids.values()))
        dirty = ExecutionEnvironment(
            "3.12",
            "2.13",
            "5.5",
            "u" * 64,
            "cuda:0",
            "bf16",
            "host",
            "deadbeef",
            True,
            tracked_tree_hash="t" * 64,
            binary_diff_sha256="d" * 64,
            untracked_manifest_hash="m" * 64,
            untracked_tree_hash="n" * 64,
        )
        with self.assertRaisesRegex(RuntimeError, "dirty-tree preflight"):
            self.store.preflight_execution(run_id, dirty)
        self.assertEqual(
            self.store.connection.execute(
                "SELECT COUNT(*) FROM run_execution"
            ).fetchone()[0],
            0,
        )
        clean = replace(dirty, git_dirty=False)
        execution_id = self.store.begin_execution(run_id, clean)
        row = self.store.connection.execute(
            "SELECT * FROM run_execution WHERE execution_id=?", (execution_id,)
        ).fetchone()
        self.assertEqual(row["tracked_tree_hash"], "t" * 64)
        self.assertEqual(row["gpu_count"], 0)
        self.store.close()
        lock.release()
        self.store = open_sqlite_store(path)

    def test_real_process_freeze_kill_points_reconcile(self) -> None:
        for point in FreezePoint:
            with self.subTest(point=point):
                root = self.root / point.value
                root.mkdir()
                db = root / "qv.sqlite3"
                artifact = root / "sample.json"
                store = open_sqlite_store(db)
                release = store.ingest_release(
                    "kill-fixture",
                    point.value,
                    "fixture",
                    "a" * 64,
                    (
                        CandidateRecord(
                            "one",
                            RudenessLabel.RUDE,
                            (("user", "one"),),
                            "b" * 64,
                            (SourceAnnotation("c" * 64, "label", "rude"),),
                        ),
                        CandidateRecord(
                            "two",
                            RudenessLabel.NON_RUDE,
                            (("user", "two"),),
                            "d" * 64,
                        ),
                    ),
                )
                template = store.register_template("card", point.value, "{text}")
                store.render_presentations(
                    release, template, lambda row: row.source_row_id
                )
                members = tuple(
                    CandidateId(row[0])
                    for row in store.connection.execute(
                        "SELECT candidate_id FROM candidate ORDER BY source_row_id"
                    )
                )
                sample_id = store.create_sample(
                    release, template, SamplerPolicy.BALANCED_MATCHED, 7, members
                )
                store.close()
                code = (
                    "import os,signal,sys; from pathlib import Path; "
                    "from quadratic_voting.experiment.types import FreezePoint,SampleId; "
                    "from quadratic_voting.experiment.store import acquire_writer_lock,open_sqlite_store; "
                    "p=FreezePoint(sys.argv[3]); "
                    "hook=lambda hit: os.kill(os.getpid(),signal.SIGKILL) if hit is p else None; "
                    "lock=acquire_writer_lock(Path(sys.argv[1])); "
                    "store=open_sqlite_store(Path(sys.argv[1]),writer_lock=lock,"
                    "require_writer_lock=True,freeze_hook=hook); "
                    "store.freeze_sample(SampleId(sys.argv[2]),Path(sys.argv[4]))"
                )
                child = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        code,
                        str(db),
                        sample_id,
                        point.value,
                        str(artifact),
                    ],
                    check=False,
                    timeout=10,
                )
                self.assertEqual(child.returncode, -9)
                with acquire_writer_lock(db) as lock:
                    reopened = open_sqlite_store(
                        db, writer_lock=lock, require_writer_lock=True
                    )
                    frozen = reopened.freeze_sample(sample_id, artifact)
                    status = reopened.connection.execute(
                        "SELECT status FROM candidate_sample WHERE sample_id=?",
                        (sample_id,),
                    ).fetchone()[0]
                    reopened.close()
                self.assertEqual(status, "frozen")
                self.assertEqual(frozen.root, tuple(str(value) for value in members))


if __name__ == "__main__":
    unittest.main()
