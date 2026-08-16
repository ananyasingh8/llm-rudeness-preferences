"""Integration tests for normalized and hand-calculated analysis exports."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from unittest import mock
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
from collections.abc import Mapping, Sequence

from quadratic_voting.experiment.ballots import ValidationFailure
from quadratic_voting.experiment.cli.export_cmds import register
from quadratic_voting.experiment.export import export_parquet
from quadratic_voting.experiment.store import (
    AcceptedBallot,
    AcceptedStatement,
    BallotAbstention,
    CandidateRecord,
    RunDefinition,
    SourceAnnotation,
    StatementInvalidMissing,
    TerminalWrite,
    open_sqlite_store,
)
from quadratic_voting.experiment.types import (
    BarrierReady,
    CandidateId,
    ElicitationArm,
    ExportDataset,
    GenerationResult,
    LikertRating,
    MatchedSetConfig,
    RudenessLabel,
    RetryPolicy,
    RunId,
    SamplerPolicy,
    SamplingProfile,
    StopReason,
    TemplateKind,
    ValidationErrorCode,
    VotingRegime,
    WorkUnit,
)


WriteFactory = Callable[
    [WorkUnit], tuple[tuple[ValidationFailure, ...], TerminalWrite | None]
]


class MissingRunDefinitionAccessorStore:
    """Delegating test double that deliberately omits one required accessor."""

    run_definition_rows = None

    def __init__(self, store: Any) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)


class TwoMatchedSetExportStore:
    """Adds a foreign run to prove export filtering cannot leak another set."""

    def __init__(self, store: Any) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def export_rows(self, dataset: ExportDataset) -> tuple[dict[str, object], ...]:
        rows = self._store.export_rows(dataset)
        if dataset is not ExportDataset.RUNS:
            return rows
        foreign = dict(rows[0])
        foreign["run_id"] = "foreign-run"
        foreign["matched_set_id"] = "foreign-matched-set"
        return (*rows, foreign)


class AnalysisFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = open_sqlite_store(self.root / "fixture.sqlite3")
        self.export_store = self.store
        digest = "a" * 64
        release = self.store.ingest_release(
            "fixture",
            "v1",
            "fixture.json",
            digest,
            tuple(
                CandidateRecord(
                    f"c{index}",
                    RudenessLabel.RUDE if index == 1 else RudenessLabel.NON_RUDE,
                    (("user", f"u{index}"), ("agent", f"a{index}")),
                    str(index) * 64,
                    (
                        SourceAnnotation(
                            "b" * 64,
                            "rudeness",
                            "rude" if index == 1 else "non_rude",
                        ),
                    ),
                )
                for index in range(1, 4)
            ),
        )
        presentation = self.store.register_template("card", "v1", "{text}")
        self.store.render_presentations(
            release, presentation, lambda record: record.turns[0][1]
        )
        self.candidates = tuple(
            CandidateId(row[0])
            for row in self.store.connection.execute(
                "SELECT candidate_id FROM candidate ORDER BY source_row_id"
            )
        )
        sample_id = self.store.create_sample(
            release, presentation, SamplerPolicy.BALANCED_MATCHED, 4, self.candidates
        )
        artifact = self.root / "sample.json"
        sample = self.store.freeze_sample(sample_id, artifact)
        instructions = {}
        for kind in TemplateKind:
            template_id = self.store.register_template(
                kind, "v1", f"{kind.value} {{budget}}"
            )
            template_hash = self.store.connection.execute(
                "SELECT body_sha256 FROM instruction_template WHERE template_id=?",
                (template_id,),
            ).fetchone()[0]
            instructions[kind] = (template_id, template_hash)
        definition = RunDefinition(
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
            digest,
            hashlib.sha256(artifact.read_bytes()).hexdigest(),
        )
        self.creation = self.store.create_matched_set(
            sample,
            sample_id,
            MatchedSetConfig(
                2,
                19,
                SamplingProfile(0.7, 0.9, 20, 50),
                retry_policy=RetryPolicy(3),
            ),
            definition,
        )
        self._populate_runs()
        provenance_run = self.creation.run_ids[
            (ElicitationArm.STATEMENT_THEN_ACTION, VotingRegime.SUPPORT)
        ]
        self.store.connection.execute(
            "INSERT INTO run_execution VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,"
            "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "execution-fixture",
                provenance_run,
                "3.12.0",
                "2.13.0",
                "5.5.0",
                "u" * 64,
                "cuda:0",
                "bf16",
                "fixture-host",
                "g" * 40,
                1,
                "2026-08-15T00:00:00Z",
                "2026-08-15T00:00:01Z",
                "completed",
                0,
                "{}",
                "13.0",
                "590.00",
                "9.0",
                "Fixture GPU",
                1,
                "10.0",
                "q" * 64,
                "Linux",
                "fixture",
                "6.0.0",
                "x86_64",
                1,
                0,
                0,
                "t" * 64,
                "d" * 64,
                "m" * 64,
                "n" * 64,
                "h" * 64,
            ),
        )

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    @staticmethod
    def _result() -> GenerationResult:
        return GenerationResult("raw", 3, 2, (1, 2), StopReason.EOS, 10, {})

    def _commit_round(self, run_id: RunId, factory: WriteFactory) -> None:
        self.store.set_run_in_progress(run_id)
        for _ in range(20):
            unit = self.store.next_incomplete_unit(run_id)
            if isinstance(unit, BarrierReady):
                break
            self.assertIsInstance(unit, WorkUnit)
            assert isinstance(unit, WorkUnit)
            prompt = json.dumps(
                [{"role": "user", "content": "fixture"}], separators=(",", ":")
            )
            call_id = self.store.begin_call(
                self.store.resolve_turn_id(unit),
                unit.attempt_index,
                prompt,
                hashlib.sha256(prompt.encode()).hexdigest(),
                7,
            )
            failures, terminal = factory(unit)
            self.store.commit_call(call_id, self._result(), failures, terminal)
        else:
            self.fail(
                f"fixture run {run_id} did not reach its round barrier in 20 turns"
            )
        self.store.aggregate_and_seal_round(run_id)

    def _populate_runs(self) -> None:
        c1, c2, c3 = self.candidates
        support_ratings = (
            {
                c1: (LikertRating.STRONGLY_PREFER_NOT_TO_CONTINUE, "one"),
                c2: (LikertRating.NEUTRAL, "two"),
                c3: (LikertRating.STRONGLY_PREFER_TO_CONTINUE, "three"),
            },
            {
                c1: (LikertRating.STRONGLY_PREFER_TO_CONTINUE, "one"),
                c2: (LikertRating.STRONGLY_PREFER_TO_CONTINUE, "two"),
                c3: (LikertRating.STRONGLY_PREFER_NOT_TO_CONTINUE, "three"),
            },
        )
        support_votes = ({c1: 1, c2: 2}, {c1: 3, c3: 1})

        def support(
            unit: WorkUnit,
        ) -> tuple[tuple[ValidationFailure, ...], TerminalWrite]:
            if unit.kind.value == "statement":
                return (), AcceptedStatement(support_ratings[unit.voter_index])
            votes = support_votes[unit.voter_index]
            return (), AcceptedBallot(
                "support", votes, sum(value * value for value in votes.values())
            )

        self._commit_round(
            self.creation.run_ids[
                (ElicitationArm.STATEMENT_THEN_ACTION, VotingRegime.SUPPORT)
            ],
            support,
        )

        def opposition(
            unit: WorkUnit,
        ) -> tuple[tuple[ValidationFailure, ...], TerminalWrite]:
            if unit.kind.value == "statement":
                return (), AcceptedStatement(support_ratings[unit.voter_index])
            votes = support_votes[unit.voter_index]
            return (), AcceptedBallot(
                "oppose", votes, sum(value * value for value in votes.values())
            )

        self._commit_round(
            self.creation.run_ids[
                (ElicitationArm.STATEMENT_THEN_ACTION, VotingRegime.OPPOSITION)
            ],
            opposition,
        )

        failure = (
            ValidationFailure(ValidationErrorCode.MALFORMED_JSON, 0, "bad fixture"),
        )

        def missing(
            unit: WorkUnit,
        ) -> tuple[tuple[ValidationFailure, ...], TerminalWrite | None]:
            if unit.voter_index == 0:
                if unit.kind.value == "statement":
                    return (), AcceptedStatement(support_ratings[0])
                return failure, BallotAbstention() if unit.attempt_index == 3 else None
            if unit.kind.value == "ballot":
                return (), AcceptedBallot("valid", {c1: 1}, 1)
            return (
                failure,
                StatementInvalidMissing() if unit.attempt_index == 3 else None,
            )

        self._commit_round(
            self.creation.run_ids[
                (ElicitationArm.ACTION_THEN_STATEMENT, VotingRegime.SUPPORT)
            ],
            missing,
        )

        def action_only(
            unit: WorkUnit,
        ) -> tuple[tuple[ValidationFailure, ...], TerminalWrite]:
            return (), AcceptedBallot("baseline", {c2: 1}, 1)

        self._commit_round(
            self.creation.run_ids[(ElicitationArm.ACTION_ONLY, VotingRegime.SUPPORT)],
            action_only,
        )
        self._commit_round(
            self.creation.run_ids[
                (ElicitationArm.ACTION_ONLY, VotingRegime.OPPOSITION)
            ],
            action_only,
        )
        self._commit_round(
            self.creation.run_ids[
                (ElicitationArm.ACTION_THEN_STATEMENT, VotingRegime.OPPOSITION)
            ],
            opposition,
        )


class ExportTests(AnalysisFixture):
    def test_matched_set_filter_is_required_and_applied_before_analysis(self) -> None:
        store = TwoMatchedSetExportStore(self.store)
        with self.assertRaisesRegex(ValueError, "explicit matched_set_id"):
            export_parquet(store, self.root / "ambiguous")
        destination = self.root / "filtered"
        export_parquet(
            store,
            destination,
            matched_set_id=str(self.creation.matched_set_id),
        )
        runs = pq.read_table(destination / "runs.parquet").to_pylist()
        self.assertTrue(runs)
        self.assertEqual(
            {row["matched_set_id"] for row in runs}, {str(self.creation.matched_set_id)}
        )
        self.assertFalse(any(row["run_id"] == "foreign-run" for row in runs))

    def test_export_failure_never_publishes_partial_directory(self) -> None:
        from quadratic_voting.experiment import export as export_module

        destination = self.root / "failed-export"
        original = export_module._write_parquet
        calls = 0

        def fail_after_first(
            path: Path, rows: Sequence[Mapping[str, object]], schema: pa.Schema
        ) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("injected export write failure")
            original(path, rows, schema)

        with mock.patch.object(export_module, "_write_parquet", fail_after_first):
            with self.assertRaisesRegex(OSError, "injected export"):
                export_parquet(self.export_store, destination)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".failed-export.staging-*")), [])

    def test_missing_model_definition_accessor_fails_before_partial_files(self) -> None:
        export_dir = self.root / "missing-accessor"
        with self.assertRaisesRegex(RuntimeError, r"run_definition_rows"):
            export_parquet(MissingRunDefinitionAccessorStore(self.store), export_dir)
        self.assertEqual(list(export_dir.glob("*.parquet")), [])

    def test_hand_calculated_agreement_and_sign(self) -> None:
        export_dir = self.root / "exports"
        manifest = export_parquet(self.export_store, export_dir)
        self.assertEqual(len(manifest.files), len(ExportDataset) + 19)
        pairs = pq.read_table(
            export_dir / "preference_action_pairs.parquet"
        ).to_pylist()
        agreement = pq.read_table(
            export_dir / "preference_action_agreement.parquet"
        ).to_pylist()
        support_run = str(
            self.creation.run_ids[
                (ElicitationArm.STATEMENT_THEN_ACTION, VotingRegime.SUPPORT)
            ]
        )
        support_pairs = [row for row in pairs if row["run_id"] == support_run]
        by_voter = {
            voter: {
                row["candidate_id"]: (row["rating_code"], row["signed_action"])
                for row in support_pairs
                if row["voter_index"] == voter
            }
            for voter in (0, 1)
        }
        c1, c2, c3 = map(str, self.candidates)
        self.assertEqual(by_voter[0], {c1: (-2, 1), c2: (0, 2), c3: (2, 0)})
        self.assertEqual(by_voter[1], {c1: (2, 3), c2: (2, 0), c3: (-2, 1)})

        # V0: rating ranks [1,2,3], action ranks [2,3,1], hence rho=-1/2.
        # V1: rating mid-ranks [2.5,2.5,1], action ranks [3,1,2], hence rho=0.
        support_summary = sorted(
            (
                row
                for row in agreement
                if row["run_id"] == support_run and row["scope"] == "overall"
            ),
            key=lambda row: row["voter_index"],
        )
        self.assertAlmostEqual(support_summary[0]["spearman_rho"], -0.5, places=14)
        self.assertAlmostEqual(support_summary[1]["spearman_rho"], 0.0, places=14)
        self.assertEqual(
            [row["n_eligible_candidates"] for row in support_summary], [3, 3]
        )
        self.assertTrue(all(row["null_reason"] is None for row in support_summary))

        opposition_run = str(
            self.creation.run_ids[
                (ElicitationArm.STATEMENT_THEN_ACTION, VotingRegime.OPPOSITION)
            ]
        )
        opposition_actions = {
            row["candidate_id"]: row["signed_action"]
            for row in pairs
            if row["run_id"] == opposition_run and row["voter_index"] == 0
        }
        self.assertEqual(opposition_actions, {c1: -1, c2: -2, c3: 0})
        contrasts = pq.read_table(export_dir / "paired_contrasts.parquet").to_pylist()
        order_actions = [
            row
            for row in contrasts
            if row["regime"] == "opposition"
            and row["round_index"] == 1
            and row["metric"] == "mean-signed-action"
            and row["contrast_kind"] == "elicitation-order"
        ]
        self.assertEqual(
            {(row["left_arm"], row["right_arm"]) for row in order_actions},
            {
                ("action-only", "statement-then-action"),
                ("action-only", "action-then-statement"),
                ("statement-then-action", "action-then-statement"),
            },
        )
        rudeness = [
            row
            for row in contrasts
            if str(row["contrast_kind"]).startswith("rudeness-label:")
        ]
        self.assertTrue(rudeness)
        self.assertTrue(
            all("associational" in row["estimand_language"] for row in rudeness)
        )

    def test_action_only_exclusion_and_terminal_missingness(self) -> None:
        export_dir = self.root / "exports"
        export_parquet(self.export_store, export_dir)
        pairs = pq.read_table(
            export_dir / "preference_action_pairs.parquet"
        ).to_pylist()
        agreement = pq.read_table(
            export_dir / "preference_action_agreement.parquet"
        ).to_pylist()
        self.assertFalse(any(row["arm"] == "action-only" for row in pairs + agreement))
        missing_run = str(
            self.creation.run_ids[
                (ElicitationArm.ACTION_THEN_STATEMENT, VotingRegime.SUPPORT)
            ]
        )
        self.assertFalse(any(row["run_id"] == missing_run for row in pairs))
        missing_rows = sorted(
            (
                row
                for row in agreement
                if row["run_id"] == missing_run and row["scope"] == "overall"
            ),
            key=lambda row: row["voter_index"],
        )
        self.assertEqual(len(missing_rows), 2)
        self.assertEqual(
            [row["null_reason"] for row in missing_rows],
            ["ABSTAINED_BALLOT", "MISSING_STATEMENT"],
        )
        self.assertTrue(all(row["spearman_rho"] is None for row in missing_rows))

        candidate_rows = pq.read_table(
            export_dir / "candidate_analysis.parquet"
        ).to_pylist()
        action_only = [row for row in candidate_rows if row["arm"] == "action-only"]
        self.assertTrue(action_only)
        self.assertTrue(all(row["rating_code"] is None for row in action_only))
        self.assertTrue(
            all(
                row["missing_reason"] == "NOT_APPLICABLE_ACTION_ONLY"
                for row in action_only
            )
        )

    def test_label_scopes_denominators_and_deterministic_values(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        export_parquet(self.export_store, first)
        export_parquet(self.export_store, second)
        for name in (
            "candidate_metadata",
            "source_annotations",
            "candidate_presentations",
            "voter_permutations",
            "experiment_configurations",
            "model_definitions",
            "run_executions",
            "candidate_analysis",
            "preference_action_agreement",
            "preference_action_summary",
            "paired_contrasts",
        ):
            first_table = pq.read_table(first / f"{name}.parquet")
            second_table = pq.read_table(second / f"{name}.parquet")
            self.assertEqual(first_table.schema, second_table.schema)
            self.assertEqual(first_table.to_pylist(), second_table.to_pylist())
        summaries = pq.read_table(
            first / "preference_action_summary.parquet"
        ).to_pylist()
        self.assertTrue(any(row["scope"] == "overall" for row in summaries))
        self.assertTrue(any(row["scope"] == "rudeness-label" for row in summaries))
        self.assertTrue(
            all(
                row["n_total_eligible_cells"]
                == row["n_defined_cells"]
                + row["n_null_missing_statement"]
                + row["n_null_abstained_ballot"]
                + row["n_null_n_lt_2"]
                + row["n_null_constant_rating"]
                + row["n_null_constant_action"]
                for row in summaries
            )
        )

    def test_normalized_provenance_is_complete_and_versioned(self) -> None:
        export_dir = self.root / "exports"
        export_parquet(self.export_store, export_dir)
        expected_columns = {
            "candidate_metadata": {
                "release_sha256",
                "label_policy_id",
                "label_policy_version",
                "label_policy_sha256",
                "presentation_template_version",
                "presentation_sha256",
            },
            "source_annotations": {
                "annotator_hash",
                "source_label",
                "source_value",
            },
            "candidate_presentations": {
                "template_id",
                "template_version",
                "body_sha256",
                "rendered_text",
                "rendered_sha256",
            },
            "voter_permutations": {
                "permutation_seed",
                "permutation_algorithm",
                "permutation_coordinates_json",
                "position",
                "candidate_id",
            },
            "experiment_configurations": {
                "definition_hash",
                "master_seed",
                "ballot_max_corrections",
                "statement_max_corrections",
                "schema_version",
                "canonical_json_version",
                "prompt_encoding_version",
                "sampler_policy_version",
                "execution_class",
            },
            "model_definitions": {
                "artifact_repository",
                "artifact_revision",
                "tokenizer_repository",
                "tokenizer_revision",
                "route_registry_hash",
                "sampling_profile_hash",
                "instruction_profile_hash",
                "label_policy_id",
                "label_policy_hash",
            },
            "run_executions": {
                "cuda_runtime_version",
                "nvidia_driver_version",
                "cudnn_version",
                "gpu_model",
                "gpu_count",
                "gpu_compute_capability",
                "gpu_uuid_hash",
                "os_name",
                "os_version",
                "kernel_version",
                "cpu_architecture",
                "deterministic_algorithms",
                "tf32_enabled",
                "cudnn_benchmark",
                "tracked_tree_hash",
                "binary_diff_sha256",
                "untracked_manifest_hash",
                "untracked_tree_hash",
                "hostname_hash",
            },
        }
        for name, required in expected_columns.items():
            schema = pq.read_schema(export_dir / f"{name}.parquet")
            self.assertTrue(required.issubset(schema.names), name)
            self.assertEqual(
                schema.metadata[b"qv_schema_version"], b"qv-analysis-export/v1"
            )
            self.assertEqual(
                schema.metadata[b"qv_sort_key"].decode().split(","), schema.names
            )

        candidate_rows = pq.read_table(
            export_dir / "candidate_analysis.parquet"
        ).to_pylist()
        self.assertTrue(candidate_rows)
        self.assertFalse(
            any(
                "unversioned-store-label" in str(value)
                for row in candidate_rows
                for value in row.values()
            )
        )
        self.assertTrue(all(row["label_policy_id"] for row in candidate_rows))
        self.assertTrue(all(row["label_policy_sha256"] for row in candidate_rows))
        self.assertTrue(
            all(json.loads(row["source_annotations_json"]) for row in candidate_rows)
        )
        self.assertTrue(all(row["presentation_sha256"] for row in candidate_rows))
        annotations = pq.read_table(
            export_dir / "source_annotations.parquet"
        ).to_pylist()
        self.assertEqual(len(annotations), 3)
        self.assertEqual(
            {row["source_value"] for row in annotations}, {"rude", "non_rude"}
        )
        presentations = pq.read_table(
            export_dir / "candidate_presentations.parquet"
        ).to_pylist()
        self.assertEqual(len(presentations), 3)
        self.assertEqual(
            {row["candidate_id"]: row["rendered_text"] for row in presentations},
            dict(zip(map(str, self.candidates), ("u1", "u2", "u3"), strict=True)),
        )
        permutations = pq.read_table(
            export_dir / "voter_permutations.parquet"
        ).to_pylist()
        self.assertEqual(len(permutations), 6 * 2 * 3)
        self.assertTrue(
            all(
                row["permutation_algorithm"] == "fisher-yates-pyrandom/v1"
                for row in permutations
            )
        )
        first_run = min(str(row["run_id"]) for row in permutations)
        positioned = sorted(
            (
                row
                for row in permutations
                if row["run_id"] == first_run and row["voter_index"] == 0
            ),
            key=lambda row: row["position"],
        )
        presentation_by_candidate = {
            row["candidate_id"]: row["rendered_text"] for row in presentations
        }
        self.assertEqual([row["position"] for row in positioned], [0, 1, 2])
        self.assertEqual(
            [presentation_by_candidate[row["candidate_id"]] for row in positioned],
            ["u2", "u1", "u3"],
        )
        configs = pq.read_table(
            export_dir / "experiment_configurations.parquet"
        ).to_pylist()
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["schema_version"], "qv-run-config/v1")
        self.assertEqual(configs[0]["ballot_max_corrections"], 3)
        self.assertEqual(configs[0]["statement_max_corrections"], 3)
        definitions = pq.read_table(
            export_dir / "model_definitions.parquet"
        ).to_pylist()
        self.assertEqual(len(definitions), 6)
        self.assertTrue(
            all(row["artifact_repository"] == "repo" for row in definitions)
        )
        executions = pq.read_table(export_dir / "run_executions.parquet").to_pylist()
        self.assertEqual(len(executions), 1)
        self.assertEqual(executions[0]["cuda_runtime_version"], "13.0")
        self.assertEqual(executions[0]["gpu_model"], "Fixture GPU")
        self.assertEqual(executions[0]["kernel_version"], "6.0.0")
        self.assertTrue(executions[0]["git_dirty"])
        self.assertEqual(executions[0]["binary_diff_sha256"], "d" * 64)
        draws = pq.read_table(export_dir / "rng_draws.parquet").to_pylist()
        populations = pq.read_table(
            export_dir / "rng_draw_populations.parquet"
        ).to_pylist()
        self.assertTrue(draws)
        self.assertTrue(populations)
        self.assertTrue(all(row["seed_version"] == "qv-seed/v1" for row in draws))
        self.assertTrue(all(row["coordinates_json"] for row in draws))
        for draw in draws:
            members = sorted(
                (row for row in populations if row["draw_id"] == draw["draw_id"]),
                key=lambda row: row["position"],
            )
            self.assertEqual(
                [row["position"] for row in members], list(range(len(members)))
            )
            self.assertEqual(
                members[draw["draw_index"]]["candidate_id"],
                draw["chosen_candidate_id"],
            )
        export_manifest = json.loads(
            (export_dir / "export-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(export_manifest["version"], "qv-export-manifest/v1")
        self.assertEqual(
            export_manifest["matched_set_id"], str(self.creation.matched_set_id)
        )
        for entry in export_manifest["files"]:
            path = export_dir / entry["name"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"]
            )
            self.assertEqual(path.stat().st_size, entry["bytes"])

    def test_all_normalized_files_have_pinned_columns(self) -> None:
        export_dir = self.root / "exports"
        export_parquet(self.export_store, export_dir)
        for dataset in ExportDataset:
            path = export_dir / f"{dataset.value}.parquet"
            self.assertTrue(path.exists())
            self.assertGreater(len(pq.read_schema(path).names), 0)

    def test_cli_registers_documented_flags(self) -> None:
        parser = ArgumentParser()
        parser.add_argument("--db")
        subparsers = parser.add_subparsers()
        register(subparsers)
        export_args = parser.parse_args(
            ["--db", "fixture.sqlite3", "export", "--matched-set", "M1", "--out", "out"]
        )
        self.assertEqual(export_args.out, Path("out"))
        self.assertEqual(export_args.matched_set, "M1")
        plot_args = parser.parse_args(
            ["plot", "--export-dir", "exports", "--out", "plots"]
        )
        self.assertEqual(plot_args.export_dir, Path("exports"))
        self.assertEqual(plot_args.out, Path("plots"))


if __name__ == "__main__":
    unittest.main()
