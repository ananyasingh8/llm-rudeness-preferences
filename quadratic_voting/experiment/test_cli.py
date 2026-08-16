from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import subprocess
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from llm_runtime import (
    LocalTransformersRoute,
    ModelId,
    ProviderId,
    QuantizationId,
    resolve_route,
)
from llm_runtime.registry import HuggingFaceArtifact
from quadratic_voting.experiment.cli import pipeline_cmds
from quadratic_voting.experiment.cli.main import main
from quadratic_voting.experiment.runner import collect_execution_environment
from quadratic_voting.experiment.store import acquire_writer_lock
from quadratic_voting.experiment.test_catalog import write_fixture
from quadratic_voting.experiment.test_runner import ScriptedGenerator, make_runs
from quadratic_voting.experiment.types import ElicitationArm, VotingRegime


class ExperimentCliTests(unittest.TestCase):
    def _default_pipeline_command(self, root: Path) -> tuple[list[str], Path, Path]:
        db = root / "pipeline.sqlite3"
        dataset = root / "convabuse.csv"
        output = root / "default-pilot"
        write_fixture(dataset)
        return (
            [
                "--db",
                str(db),
                "pipeline",
                "run",
                "--dataset-path",
                str(dataset),
                "--dataset-version",
                "fixture-default-v2",
                "--output-dir",
                str(output),
                "--sample-seed",
                "17",
                "--master-seed",
                "29",
                "--repeat",
                "2",
                "--voters",
                "2",
                "--device",
                "cpu",
            ],
            db,
            output,
        )

    def _write_run_config(
        self, db: Path, sample_id: str, artifact: Path, destination: Path
    ) -> None:
        connection = sqlite3.connect(db)
        connection.row_factory = sqlite3.Row
        sample = connection.execute(
            "SELECT s.*,r.dataset_name,r.version AS release_version,r.file_sha256,"
            "lp.name AS policy_name,lp.version AS policy_version,lp.rule_sha256,"
            "pt.name AS template_name,pt.version AS template_version,pt.body_sha256 "
            "FROM candidate_sample s JOIN dataset_release r ON r.release_id=s.release_id "
            "JOIN label_policy lp ON lp.label_policy_id=s.label_policy_id "
            "JOIN presentation_template pt ON pt.template_id=s.template_id "
            "WHERE s.sample_id=?",
            (sample_id,),
        ).fetchone()
        prompts: dict[str, object] = {}
        for row in connection.execute(
            "SELECT template_id,name,version,body_sha256 FROM instruction_template"
        ):
            prompts[
                "final_result" if row["name"] == "final-result" else row["name"]
            ] = {
                "template_id": row["template_id"],
                "name": row["name"],
                "version": row["version"],
                "expected_sha256": row["body_sha256"],
            }
        connection.close()
        route = resolve_route(
            ModelId.GEMMA_4_E2B_IT, ProviderId.LOCAL, QuantizationId.BF16
        )
        assert isinstance(route, LocalTransformersRoute)
        config = {
            "schema_version": "qv-run-config/v1",
            "canonical_json_version": "qv-canonical-json/v1",
            "prompt_encoding_version": "qv-prompt/v1",
            "seed_version": "qv-seed/v1",
            "sample": {
                "sample_id": sample_id,
                "artifact_path": str(artifact.resolve()),
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
                },
                "presentation_template": {
                    "template_id": sample["template_id"],
                    "name": sample["template_name"],
                    "version": sample["template_version"],
                    "expected_sha256": sample["body_sha256"],
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
            "prompts": prompts,
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
            "master_seed": 29,
            "voter_count": 2,
            "credit_budget": 100,
            "sampler_policy": "balanced-matched/v1",
            "presentation_policy": "setup-once-ids-later/v1",
            "tie_policy": "uniform-seeded/v1",
            "action_format": "json-with-rationale/v1",
            "execution_class": "fixture",
        }
        destination.write_text(json.dumps(config), encoding="utf-8")

    def _invoke(
        self,
        argv: list[str],
        *,
        generator: bool = False,
    ) -> str:
        output = io.StringIO()
        error = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            status = main(
                argv,
                generator_factory=(
                    (lambda _profile: ScriptedGenerator()) if generator else None
                ),
            )
        self.assertEqual(status, 0, error.getvalue())
        return output.getvalue()

    def test_run_resume_and_verified_inspect_use_production_dispatcher(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "qv.sqlite3"
            store, creation = make_runs(db)
            run_id = creation.run_ids[
                (ElicitationArm.ACTION_ONLY, VotingRegime.SUPPORT)
            ]
            store.close()
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(
                    main(
                        ["--db", str(db), "run", "--run-id", str(run_id)],
                        generator_factory=lambda _profile: ScriptedGenerator(),
                    ),
                    0,
                )
                execution_connection = sqlite3.connect(db)
                execution_count = execution_connection.execute(
                    "SELECT COUNT(*) FROM run_execution WHERE run_id=?", (run_id,)
                ).fetchone()[0]
                execution_connection.close()
                self.assertEqual(
                    main(
                        ["--db", str(db), "run", "--run-id", str(run_id)],
                        generator_factory=lambda _profile: ScriptedGenerator(),
                    ),
                    0,
                )
                connection = sqlite3.connect(db)
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM run_execution WHERE run_id=?", (run_id,)
                    ).fetchone()[0],
                    execution_count,
                )
                connection.close()
                self.assertEqual(
                    main(
                        [
                            "--db",
                            str(db),
                            "inspect",
                            "--run-id",
                            str(run_id),
                            "--verify",
                        ]
                    ),
                    0,
                )
            text = output.getvalue()
            self.assertIn("status=complete", text)
            self.assertIn("# Conversation voting task", text)
            self.assertIn("VERIFY OK", text)

    def test_dirty_primary_is_rejected_before_execution_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "primary.sqlite3"
            store, creation = make_runs(db, execution_class="primary")
            run_id = next(iter(creation.run_ids.values()))
            store.close()
            error = io.StringIO()
            dirty_environment = replace(collect_execution_environment(), git_dirty=True)
            with (
                patch(
                    "quadratic_voting.experiment.cli.run_cmds.collect_execution_environment",
                    return_value=dirty_environment,
                ),
                contextlib.redirect_stderr(error),
            ):
                status = main(
                    ["--db", str(db), "run", "--run-id", str(run_id)],
                    generator_factory=lambda _profile: ScriptedGenerator(),
                )
            self.assertEqual(status, 1)
            self.assertIn("dirty-tree preflight", error.getvalue())
            connection = sqlite3.connect(db)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM run_execution").fetchone()[0],
                0,
            )
            connection.close()

    def test_concurrent_writer_is_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "qv.sqlite3"
            error = io.StringIO()
            with acquire_writer_lock(db), contextlib.redirect_stderr(error):
                self.assertEqual(
                    main(["--db", str(db), "migrate"]),
                    1,
                )
            self.assertIn("another process holds", error.getvalue())
            self.assertIn("delete the lock file", error.getvalue())

    def test_every_database_writer_locks_before_handler_or_database_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "absent.sqlite3"
            commands = (
                ["migrate"],
                ["catalog", "ingest", "--dataset-version", "v1"],
                ["template", "register"],
                [
                    "sample",
                    "create",
                    "--release-id",
                    "release",
                    "--template-id",
                    "template",
                    "--seed",
                    "1",
                ],
                [
                    "sample",
                    "freeze",
                    "--sample-id",
                    "sample",
                    "--out",
                    str(root / "sample.json"),
                ],
                [
                    "matched-set",
                    "create",
                    "--config",
                    str(root / "run-config.json"),
                ],
                ["run", "--run-id", "run"],
                [
                    "export",
                    "--matched-set",
                    "matched",
                    "--out",
                    str(root / "export"),
                ],
            )
            holder_code = (
                "import sys,time; from pathlib import Path; "
                "from quadratic_voting.experiment.store import acquire_writer_lock; "
                "lock=acquire_writer_lock(Path(sys.argv[1]),command='subprocess-holder'); "
                "print('READY',flush=True); time.sleep(3600)"
            )
            holder = subprocess.Popen(
                [sys.executable, "-u", "-c", holder_code, str(db)],
                cwd=Path(__file__).resolve().parents[2],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                assert holder.stdout is not None
                self.assertEqual(holder.stdout.readline().strip(), "READY")
                for command in commands:
                    error = io.StringIO()
                    with contextlib.redirect_stderr(error):
                        self.assertEqual(main(["--db", str(db), *command]), 1, command)
                    self.assertIn("another process holds", error.getvalue(), command)
            finally:
                holder.kill()
                holder.wait(timeout=10)
                if holder.stdout is not None:
                    holder.stdout.close()
                if holder.stderr is not None:
                    holder.stderr.close()
            self.assertFalse(db.exists())

            Path(f"{db}.lock").write_text(
                '{"pid":999999,"command":"stale"}', encoding="utf-8"
            )
            self.assertEqual(main(["--db", str(db), "migrate"]), 0)
            self.assertTrue(db.exists())

    def test_read_only_commands_never_create_or_migrate_missing_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "missing.sqlite3"
            for command in (
                ["inspect", "--run-id", "run"],
                ["verify", "--run-id", "run"],
            ):
                error = io.StringIO()
                with contextlib.redirect_stderr(error):
                    self.assertEqual(main(["--db", str(db), *command]), 1)
                self.assertIn("does not exist", error.getvalue())
                self.assertFalse(db.exists())

    def test_parser_rejects_old_permissive_matched_set_flags(self) -> None:
        with self.assertRaises(SystemExit):
            main(
                [
                    "matched-set-create",
                    "--sample-file",
                    "x",
                    "--sample-id",
                    "s",
                    "--master-seed",
                    "1",
                    "--voters",
                    "0",
                ]
            )

    def test_full_production_cli_chain_creates_run_exports_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "pipeline.sqlite3"
            dataset = root / "convabuse.csv"
            columns = (
                "conv_id",
                "prev_agent",
                "prev_user",
                "agent",
                "user",
                "is_abuse.1",
                "is_abuse.0",
                "is_abuse.-1",
                "is_abuse.-2",
                "is_abuse.-3",
            )
            with dataset.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "conv_id": "non-rude",
                        "prev_agent": "Previous helpful response",
                        "prev_user": "Previous request",
                        "agent": "Helpful response",
                        "user": "Thank you",
                        "is_abuse.1": "1",
                        "is_abuse.0": "0",
                        "is_abuse.-1": "0",
                        "is_abuse.-2": "0",
                        "is_abuse.-3": "0",
                    }
                )
                writer.writerow(
                    {
                        "conv_id": "rude",
                        "prev_agent": "Previous unhelpful response",
                        "prev_user": "Previous rude request",
                        "agent": "Unhelpful response",
                        "user": "Rude response",
                        "is_abuse.1": "0",
                        "is_abuse.0": "0",
                        "is_abuse.-1": "1",
                        "is_abuse.-2": "0",
                        "is_abuse.-3": "0",
                    }
                )

            prefix = ["--db", str(db)]
            ingest = self._invoke(
                [
                    *prefix,
                    "catalog",
                    "ingest",
                    "--dataset-path",
                    str(dataset),
                    "--dataset-version",
                    "fixture-v1",
                ]
            )
            release_id = ingest.strip().split("release_id=", 1)[1]
            templates = self._invoke([*prefix, "template", "register"])
            presentation_id = templates.split("candidate-card=", 1)[1].split()[0]
            created = self._invoke(
                [
                    *prefix,
                    "sample",
                    "create",
                    "--release-id",
                    release_id,
                    "--template-id",
                    presentation_id,
                    "--size",
                    "2",
                    "--seed",
                    "17",
                ]
            )
            sample_id = created.split("sample_id=", 1)[1].split()[0]
            sample_file = root / "sample.json"
            self._invoke(
                [
                    *prefix,
                    "sample",
                    "freeze",
                    "--sample-id",
                    sample_id,
                    "--out",
                    str(sample_file),
                ]
            )
            validated = self._invoke(
                [
                    *prefix,
                    "sample",
                    "verify",
                    "--sample-id",
                    sample_id,
                    "--artifact",
                    str(sample_file),
                ]
            )
            self.assertIn(f"sample_id={sample_id}", validated)
            run_config = root / "run-config.json"
            self._write_run_config(db, sample_id, sample_file, run_config)
            matched = self._invoke(
                [*prefix, "matched-set", "create", "--config", str(run_config)]
            )
            run_line = next(
                line for line in matched.splitlines() if line.startswith("run_id=")
            )
            run_id = run_line.split("run_id=", 1)[1].split()[0]
            matched_set_id = matched.splitlines()[0].split("matched_set_id=", 1)[1]
            first_run = self._invoke(
                [*prefix, "run", "--run-id", run_id], generator=True
            )
            self.assertIn("status=complete", first_run)
            connection = sqlite3.connect(db)
            model_visible_setup = json.loads(
                connection.execute(
                    "SELECT prompt_messages_json FROM model_call ORDER BY rowid LIMIT 1"
                ).fetchone()[0]
            )[0]["content"]
            connection.close()
            self.assertIn(
                "User: Previous request\nAssistant: Previous helpful response\n"
                "User: Thank you\nAssistant: Helpful response",
                model_visible_setup,
            )
            self.assertIn(
                "User: Previous rude request\nAssistant: Previous unhelpful response\n"
                "User: Rude response\nAssistant: Unhelpful response",
                model_visible_setup,
            )
            replay = self._invoke([*prefix, "run", "--run-id", run_id], generator=True)
            self.assertIn("status=complete", replay)
            inspected = self._invoke(
                [*prefix, "inspect", "--run-id", run_id, "--verify"]
            )
            self.assertIn("VERIFY OK", inspected)

            export_dir = root / "exports"
            self._invoke(
                [
                    *prefix,
                    "export",
                    "--matched-set",
                    matched_set_id,
                    "--out",
                    str(export_dir),
                ]
            )
            parquet_files = tuple(export_dir.glob("*.parquet"))
            self.assertTrue(parquet_files)
            self.assertTrue(all(path.is_file() for path in parquet_files))
            plot_dir = root / "plots"
            self._invoke(
                [
                    *prefix,
                    "plot",
                    "--export-dir",
                    str(export_dir),
                    "--out",
                    str(plot_dir),
                ]
            )
            png_files = tuple(plot_dir.glob("*.png"))
            self.assertTrue(png_files)
            self.assertTrue(all(path.is_file() for path in png_files))

    def test_default_pipeline_runs_seed_repeats_and_rerun_reuses_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command, db, output = self._default_pipeline_command(root)

            first = self._invoke(command, generator=True)
            connection = sqlite3.connect(db)
            first_counts = (
                connection.execute("SELECT COUNT(*) FROM matched_set").fetchone()[0],
                connection.execute("SELECT COUNT(*) FROM experiment_run").fetchone()[0],
                connection.execute("SELECT COUNT(*) FROM run_execution").fetchone()[0],
            )
            connection.close()

            second = self._invoke(command, generator=True)
            connection = sqlite3.connect(db)
            second_counts = (
                connection.execute("SELECT COUNT(*) FROM matched_set").fetchone()[0],
                connection.execute("SELECT COUNT(*) FROM experiment_run").fetchone()[0],
                connection.execute("SELECT COUNT(*) FROM run_execution").fetchone()[0],
            )
            connection.close()

            self.assertIn("pipeline=complete", first)
            self.assertIn("status=resuming", second)
            # Two seed-repeat replicates, action-only x {support, opposition}:
            # 2 matched sets and 2 x 2 = 4 runs/executions.
            self.assertEqual(first_counts, (2, 4, 4))
            self.assertEqual(second_counts, first_counts)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "sample.json").is_file())
            self.assertTrue((output / "run-config.repeat-0.json").is_file())
            self.assertTrue((output / "run-config.repeat-1.json").is_file())
            config = json.loads(
                (output / "run-config.repeat-0.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["sampling"]["max_new_tokens"], 2048)
            self.assertEqual(config["sampler_policy"], "level-stratified/v1")
            self.assertEqual(config["arms"], ["action-only"])
            self.assertEqual(config["regimes"], ["support", "opposition"])
            self.assertTrue(tuple((output / "export" / "repeat-0").glob("*.parquet")))
            self.assertTrue(tuple((output / "export" / "aggregate").glob("*.parquet")))
            self.assertTrue(tuple((output / "plots").glob("*.png")))

            db.unlink()
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                self.assertEqual(
                    main(
                        command,
                        generator_factory=lambda _profile: ScriptedGenerator(),
                    ),
                    1,
                )
            self.assertIn("references missing database", error.getvalue())
            self.assertFalse(db.exists())

    def test_default_pipeline_lazily_shares_one_generator_across_nested_runs(
        self,
    ) -> None:
        """The default path must not reload Gemma for every matched condition."""
        with tempfile.TemporaryDirectory() as directory:
            command, db, _output = self._default_pipeline_command(Path(directory))
            generator = ScriptedGenerator()
            constructions = 0

            def create_generator(_args: argparse.Namespace) -> ScriptedGenerator:
                nonlocal constructions
                constructions += 1
                return generator

            with (
                patch.object(pipeline_cmds, "_bind_model_provenance"),
                patch.object(
                    pipeline_cmds,
                    "_default_generator",
                    side_effect=create_generator,
                ),
            ):
                self._invoke(command)
                self._invoke(command)

            connection = sqlite3.connect(db)
            completed_runs = connection.execute(
                "SELECT COUNT(*) FROM experiment_run WHERE status='complete'"
            ).fetchone()[0]
            connection.close()
            # Two replicates x action-only x {support, opposition} = 4 runs.
            self.assertEqual(completed_runs, 4)
            self.assertEqual(constructions, 1)

    def test_default_pipeline_recovers_commit_to_checkpoint_interruptions(self) -> None:
        boundaries = (
            ("catalog", "ingest"),
            ("sample", "create"),
            ("matched-set", "create"),
            ("export",),
            ("aggregate",),
            ("plot",),
        )
        for boundary in boundaries:
            with (
                self.subTest(boundary=boundary),
                tempfile.TemporaryDirectory() as directory,
            ):
                command, db, _output = self._default_pipeline_command(Path(directory))
                original_invoke = pipeline_cmds._invoke
                interrupted = False

                def interrupt_after_success(
                    args: argparse.Namespace, nested_command: list[str]
                ) -> str:
                    nonlocal interrupted
                    result = original_invoke(args, nested_command)
                    if (
                        not interrupted
                        and tuple(nested_command[: len(boundary)]) == boundary
                    ):
                        interrupted = True
                        raise RuntimeError("injected post-commit interruption")
                    return result

                with patch.object(
                    pipeline_cmds, "_invoke", side_effect=interrupt_after_success
                ):
                    self.assertEqual(
                        main(
                            command,
                            generator_factory=lambda _profile: ScriptedGenerator(),
                        ),
                        1,
                    )
                self.assertTrue(interrupted)

                resumed = self._invoke(command, generator=True)
                connection = sqlite3.connect(db)
                counts = (
                    connection.execute(
                        "SELECT COUNT(*) FROM dataset_release"
                    ).fetchone()[0],
                    connection.execute(
                        "SELECT COUNT(*) FROM candidate_sample"
                    ).fetchone()[0],
                    connection.execute("SELECT COUNT(*) FROM matched_set").fetchone()[
                        0
                    ],
                    connection.execute(
                        "SELECT COUNT(*) FROM experiment_run"
                    ).fetchone()[0],
                )
                connection.close()
                self.assertIn("pipeline=complete", resumed)
                # 1 release, 1 reused sample, 2 replicate matched sets, 4 runs.
                self.assertEqual(counts, (1, 1, 2, 4))

    def test_default_pipeline_rejects_changed_dataset_bytes_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command, _db, _output = self._default_pipeline_command(root)
            self._invoke(command, generator=True)
            dataset = root / "convabuse.csv"
            dataset.write_bytes(dataset.read_bytes() + b"\n")
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                self.assertEqual(
                    main(
                        command, generator_factory=lambda _profile: ScriptedGenerator()
                    ),
                    1,
                )
            self.assertIn("dataset", error.getvalue())
            self.assertIn("changed", error.getvalue())

    def test_default_pipeline_rejects_missing_database_for_partial_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command, db, output = self._default_pipeline_command(root)
            self._invoke(command, generator=True)
            manifest_path = output / "manifest.json"
            self.assertTrue(manifest_path.is_file())
            db.unlink()
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                self.assertEqual(
                    main(
                        command, generator_factory=lambda _profile: ScriptedGenerator()
                    ),
                    1,
                )
            self.assertIn("references missing database", error.getvalue())

    def test_default_pipeline_rejects_fresh_replacement_database_before_catalog_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command, db, output = self._default_pipeline_command(root)
            with patch.object(
                pipeline_cmds,
                "_ensure_release",
                side_effect=RuntimeError("early interruption"),
            ):
                self.assertEqual(
                    main(
                        command, generator_factory=lambda _profile: ScriptedGenerator()
                    ),
                    1,
                )
            self.assertTrue((output / "manifest.json").is_file())
            db.unlink()
            self.assertEqual(main(["--db", str(db), "migrate"]), 0)
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                self.assertEqual(
                    main(
                        command, generator_factory=lambda _profile: ScriptedGenerator()
                    ),
                    1,
                )
            self.assertIn("database identity changed", error.getvalue())
            connection = sqlite3.connect(db)
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM dataset_release").fetchone()[
                    0
                ],
                0,
            )
            connection.close()

    def test_default_pipeline_rejects_replacement_sample_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command, _db, output = self._default_pipeline_command(root)
            self._invoke(command, generator=True)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["sample_id"] = "replacement-sample-id"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                self.assertEqual(
                    main(
                        command, generator_factory=lambda _profile: ScriptedGenerator()
                    ),
                    1,
                )
            self.assertIn("missing sample ID", error.getvalue())

    def test_model_provenance_requires_complete_snapshot_and_detects_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot"
            snapshot.mkdir()
            for name in (
                ".gitattributes",
                "README.md",
                "chat_template.jinja",
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "processor_config.json",
                "tokenizer_config.json",
                "tokenizer.json",
            ):
                (snapshot / name).write_bytes(name.encode())
            provenance = pipeline_cmds._model_provenance(
                snapshot, repository="repo", revision="revision"
            )
            self.assertEqual(provenance["repository"], "repo")
            self.assertNotIn("unavailable", json.dumps(provenance))
            (snapshot / "special_tokens_map.json").write_bytes(b"unexpected")
            with self.assertRaisesRegex(ValueError, "unexpected files"):
                pipeline_cmds._model_provenance(
                    snapshot, repository="repo", revision="revision"
                )
            (snapshot / "special_tokens_map.json").unlink()
            (snapshot / "model.safetensors").write_bytes(b"changed")
            changed = pipeline_cmds._model_provenance(
                snapshot, repository="repo", revision="revision"
            )
            self.assertNotEqual(provenance, changed)
            (snapshot / "config.json").unlink()
            with self.assertRaisesRegex(ValueError, "essential files"):
                pipeline_cmds._model_provenance(
                    snapshot, repository="repo", revision="revision"
                )

    def test_bind_model_provenance_is_durable_and_rejects_snapshot_or_route_drift(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            for name in pipeline_cmds._MODEL_SNAPSHOT_FILES:
                (snapshot / name).write_bytes(name.encode())
            route = resolve_route(
                ModelId.GEMMA_4_E4B_IT,
                ProviderId.LOCAL,
                QuantizationId.BITSANDBYTES_FP4,
            )
            assert isinstance(route, LocalTransformersRoute)
            args = argparse.Namespace(cache_dir=root)
            manifest: dict[str, object] = {}
            manifest_path = root / "manifest.json"
            with patch.object(
                pipeline_cmds,
                "download_transformers_artifact",
                return_value=snapshot,
            ):
                pipeline_cmds._bind_model_provenance(args, manifest, manifest_path)
                first = manifest_path.read_bytes()
                pipeline_cmds._bind_model_provenance(args, manifest, manifest_path)
                self.assertEqual(first, manifest_path.read_bytes())
                (snapshot / "README.md").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "provenance drift"):
                    pipeline_cmds._bind_model_provenance(args, manifest, manifest_path)
                (snapshot / "README.md").write_bytes(b"README.md")
                drifted = replace(
                    route,
                    artifact=HuggingFaceArtifact(
                        route.artifact.repository, "different-revision"
                    ),
                )
                with (
                    patch.object(pipeline_cmds, "resolve_route", return_value=drifted),
                    self.assertRaisesRegex(ValueError, "provenance drift"),
                ):
                    pipeline_cmds._bind_model_provenance(args, manifest, manifest_path)
                second_snapshot = root / "second-snapshot"
                second_snapshot.mkdir()
                for name in pipeline_cmds._MODEL_SNAPSHOT_FILES:
                    (second_snapshot / name).write_bytes(name.encode())
                with (
                    patch.object(
                        pipeline_cmds,
                        "download_transformers_artifact",
                        return_value=second_snapshot,
                    ),
                    self.assertRaisesRegex(ValueError, "provenance drift"),
                ):
                    pipeline_cmds._bind_model_provenance(args, manifest, manifest_path)

    def test_derived_artifact_reuse_requires_source_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "export"
            manifest_path = root / "manifest.json"
            manifest: dict[str, object] = {}
            args = argparse.Namespace()
            calls: list[list[str]] = []

            def generate(_args: argparse.Namespace, command: list[str]) -> str:
                calls.append(command)
                output.mkdir(exist_ok=True)
                (output / "result.txt").write_text(str(len(calls)), encoding="utf-8")
                return ""

            binding = {"matched_set_id": "matched", "source_fingerprint": "one"}
            with patch.object(pipeline_cmds, "_invoke", side_effect=generate):
                self.assertTrue(
                    pipeline_cmds._ensure_derived_artifact(
                        args,
                        manifest,
                        manifest_path,
                        output=output,
                        manifest_key="export_files",
                        binding_key="export_binding",
                        source_binding=binding,
                        command=["export"],
                    )
                )
                self.assertFalse(
                    pipeline_cmds._ensure_derived_artifact(
                        args,
                        manifest,
                        manifest_path,
                        output=output,
                        manifest_key="export_files",
                        binding_key="export_binding",
                        source_binding=binding,
                        command=["export"],
                    )
                )
                changed = {"matched_set_id": "matched", "source_fingerprint": "two"}
                self.assertTrue(
                    pipeline_cmds._ensure_derived_artifact(
                        args,
                        manifest,
                        manifest_path,
                        output=output,
                        manifest_key="export_files",
                        binding_key="export_binding",
                        source_binding=changed,
                        command=["export"],
                    )
                )
            self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
