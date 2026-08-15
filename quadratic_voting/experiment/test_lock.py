from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from quadratic_voting.experiment.store import acquire_writer_lock


@unittest.skipUnless(os.name == "posix", "fcntl writer lock requires POSIX")
class WriterLockTest(unittest.TestCase):
    def test_sigkill_releases_kernel_lock_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qv.sqlite3"
            code = (
                "from pathlib import Path; import sys,time; "
                "from quadratic_voting.experiment.store import acquire_writer_lock; "
                "lock=acquire_writer_lock(Path(sys.argv[1])); print('ready',flush=True); time.sleep(60)"
            )
            child = subprocess.Popen(
                [sys.executable, "-c", code, str(path)],
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                assert child.stdout is not None
                self.assertEqual(child.stdout.readline().strip(), "ready")
                with self.assertRaisesRegex(RuntimeError, str(path) + r"\.lock"):
                    acquire_writer_lock(path)
                os.kill(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
                with acquire_writer_lock(path):
                    pass
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)
                if child.stdout is not None:
                    child.stdout.close()


if __name__ == "__main__":
    unittest.main()
