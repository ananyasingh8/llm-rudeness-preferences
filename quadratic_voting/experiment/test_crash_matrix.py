"""Real-process crash gates for durable sample-freeze integration points."""

from __future__ import annotations

import hashlib
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from quadratic_voting.experiment.store import (
    CandidateRecord,
    acquire_writer_lock,
    open_sqlite_store,
)
from quadratic_voting.experiment.types import (
    CandidateId,
    FreezePoint,
    RudenessLabel,
    SamplerPolicy,
)


@unittest.skipUnless(hasattr(signal, "SIGKILL"), "SIGKILL requires POSIX")
class FreezeCrashMatrixTests(unittest.TestCase):
    def test_every_core_freeze_hook_reconciles_after_real_sigkill(self) -> None:
        child_code = (
            "import os,signal,sys; from pathlib import Path; "
            "from quadratic_voting.experiment.types import FreezePoint,SampleId; "
            "from quadratic_voting.experiment.store import acquire_writer_lock,open_sqlite_store; "
            "db=Path(sys.argv[1]); point=FreezePoint(sys.argv[3]); "
            "hook=lambda hit: os.kill(os.getpid(),signal.SIGKILL) if hit is point else None; "
            "lock=acquire_writer_lock(db,command='freeze-crash-child'); "
            "store=open_sqlite_store(db,writer_lock=lock,require_writer_lock=True,freeze_hook=hook); "
            "store.freeze_sample(SampleId(sys.argv[2]),Path(sys.argv[4]))"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for point in FreezePoint:
                with self.subTest(point=point):
                    case = root / point.value
                    case.mkdir()
                    db = case / "qv.sqlite3"
                    artifact = case / "sample.json"
                    with acquire_writer_lock(db, command="freeze-crash-setup") as lock:
                        store = open_sqlite_store(
                            db, writer_lock=lock, require_writer_lock=True
                        )
                        candidates = tuple(
                            CandidateRecord(
                                str(index),
                                RudenessLabel.RUDE
                                if index == 0
                                else RudenessLabel.NON_RUDE,
                                (("user", f"candidate {index}"),),
                                hashlib.sha256(str(index).encode()).hexdigest(),
                            )
                            for index in range(2)
                        )
                        release = store.ingest_release(
                            "freeze-fixture",
                            point.value,
                            "fixture",
                            "a" * 64,
                            candidates,
                        )
                        template = store.register_template(
                            "card", point.value, "{text}"
                        )
                        store.render_presentations(
                            release, template, lambda record: record.source_row_id
                        )
                        members = tuple(
                            CandidateId(row[0])
                            for row in store.connection.execute(
                                "SELECT candidate_id FROM candidate ORDER BY source_row_id"
                            )
                        )
                        sample_id = store.create_sample(
                            release,
                            template,
                            SamplerPolicy.BALANCED_MATCHED,
                            7,
                            members,
                        )
                        store.close()
                    child = subprocess.run(
                        [
                            sys.executable,
                            "-c",
                            child_code,
                            str(db),
                            str(sample_id),
                            point.value,
                            str(artifact),
                        ],
                        cwd=Path(__file__).resolve().parents[2],
                        check=False,
                        timeout=15,
                    )
                    self.assertEqual(child.returncode, -signal.SIGKILL)
                    with acquire_writer_lock(
                        db, command="freeze-crash-reconcile"
                    ) as lock:
                        reopened = open_sqlite_store(
                            db, writer_lock=lock, require_writer_lock=True
                        )
                        frozen = reopened.freeze_sample(sample_id, artifact)
                        self.assertEqual(
                            frozen.root, tuple(str(item) for item in members)
                        )
                        self.assertEqual(
                            reopened.connection.execute(
                                "SELECT status FROM candidate_sample WHERE sample_id=?",
                                (sample_id,),
                            ).fetchone()[0],
                            "frozen",
                        )
                        reopened.close()


if __name__ == "__main__":
    unittest.main()
