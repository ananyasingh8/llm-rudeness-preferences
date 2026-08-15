from __future__ import annotations

import contextlib
import csv
import io
import json
import subprocess
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

from llm_runtime import (
    LocalTransformersRoute,
    ModelId,
    ProviderId,
    QuantizationId,
    resolve_route,
)
from quadratic_voting.experiment.cli.main import main
from quadratic_voting.experiment.store import acquire_writer_lock
from quadratic_voting.experiment.test_runner import ScriptedGenerator, make_runs
from quadratic_voting.experiment.types import ElicitationArm, VotingRegime


class ExperimentCliTests(unittest.TestCase):
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
            self.assertIn("Candidate cards", text)
            self.assertIn("VERIFY OK", text)

    def test_dirty_primary_is_rejected_before_execution_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "primary.sqlite3"
            store, creation = make_runs(db, execution_class="primary")
            run_id = next(iter(creation.run_ids.values()))
            store.close()
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
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
                        "prev_agent": "",
                        "prev_user": "",
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
                        "prev_agent": "",
                        "prev_user": "",
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


if __name__ == "__main__":
    unittest.main()
