"""Reusable kernel writer lock acquired before database open or migration."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

from quadratic_voting.experiment.errors import WriterLockError


@dataclass(frozen=True, slots=True)
class LockHolder:
    pid: int
    command: str
    started_at: str
    execution_id: str | None = None


class WriterLock:
    """Exclusive nonblocking flock whose metadata is advisory only."""

    def __init__(self, stream: IO[str], path: Path, holder: LockHolder) -> None:
        self._stream = stream
        self.path = path
        self.holder = holder
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        import fcntl

        # Metadata is cleared while ownership is still kernel-enforced.
        self._stream.seek(0)
        self._stream.truncate()
        self._stream.flush()
        os.fsync(self._stream.fileno())
        fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()
        self._released = True

    def __enter__(self) -> WriterLock:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()


def acquire_writer_lock(
    db_path: Path,
    *,
    command: str = "experiment-writer",
    execution_id: str | None = None,
) -> WriterLock:
    """Acquire ``<db>.lock`` before callers open or migrate SQLite."""
    try:
        import fcntl
    except ImportError as error:  # pragma: no cover - non-POSIX only
        raise WriterLockError(
            "Writer-lock acquisition failed because fcntl is unavailable in "
            "experiment.lock.acquire_writer_lock before database open. Mutation is unsafe. "
            "Run on a POSIX platform with flock support."
        ) from error
    db_path.parent.mkdir(parents=True, exist_ok=True)
    path = Path(f"{db_path}.lock")
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        stream.seek(0)
        advisory = stream.read().strip() or "metadata unavailable or stale"
        stream.close()
        raise WriterLockError(
            f"Writer-lock acquisition failed because another process holds {path}. The kernel "
            "reported contention in experiment.lock.acquire_writer_lock before database open; "
            f"advisory holder metadata: {advisory}. This writer made no database or artifact "
            "change. Wait for the holder to exit and retry; never delete the lock file."
        ) from error
    holder = LockHolder(
        pid=os.getpid(),
        command=command,
        started_at=datetime.now(UTC).isoformat(),
        execution_id=execution_id,
    )
    stream.seek(0)
    stream.truncate()
    json.dump(asdict(holder), stream, sort_keys=True, separators=(",", ":"))
    stream.flush()
    os.fsync(stream.fileno())
    return WriterLock(stream, path, holder)


def lock_matches_database(lock: WriterLock, db_path: Path) -> bool:
    return lock.path == Path(f"{db_path}.lock") and not lock._released
