"""Child-process integration gates for the public ``python -m`` CLI path."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from quadratic_voting.experiment.store import acquire_writer_lock


class ModuleCliIntegrationTests(unittest.TestCase):
    def invoke(self, db: Path, *command: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-m",
                "quadratic_voting.experiment.cli",
                "--db",
                str(db),
                *command,
            ],
            cwd=Path(__file__).resolve().parents[2],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    def test_module_migrate_uses_real_store_and_kernel_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "module.sqlite3"
            migrated = self.invoke(db, "migrate")
            self.assertEqual(migrated.returncode, 0, migrated.stderr)
            self.assertTrue(db.is_file())
            self.assertIn("schema=current", migrated.stdout)

            with acquire_writer_lock(db):
                blocked = self.invoke(db, "migrate")
            self.assertEqual(blocked.returncode, 1)
            self.assertIn("another process holds", blocked.stderr)

    def test_module_read_only_dispatch_does_not_create_database(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "missing.sqlite3"
            inspected = self.invoke(db, "inspect", "--run-id", "missing")
            self.assertEqual(inspected.returncode, 1)
            self.assertIn("does not exist", inspected.stderr)
            self.assertFalse(db.exists())

    def test_module_matched_set_requires_strict_versioned_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = root / "missing.sqlite3"
            config = root / "run.json"
            config.write_text('{"schema_version":"wrong"}', encoding="utf-8")
            created = self.invoke(db, "matched-set", "create", "--config", str(config))
            self.assertEqual(created.returncode, 1)
            self.assertIn("Run-config validation failed", created.stderr)
            self.assertFalse(db.exists())


if __name__ == "__main__":
    unittest.main()
