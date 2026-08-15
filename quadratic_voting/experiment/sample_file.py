"""Durable filesystem primitives for the two-transaction sample freeze protocol."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def write_fsynced_temp(final_path: Path, content: bytes) -> Path:
    """Create and fsync a unique temp in the final artifact's directory."""
    final_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{final_path.name}.", suffix=".freeze.tmp", dir=final_path.parent
    )
    path = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def replace_and_fsync_directory(temp_path: Path, final_path: Path) -> None:
    os.replace(temp_path, final_path)
    descriptor = os.open(final_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def remove_and_fsync_directory(path: Path) -> None:
    path.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
