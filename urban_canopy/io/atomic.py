"""Small atomic-write primitives shared by caches and exported artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["atomic_write_bytes", "atomic_write_text"]


def atomic_write_bytes(path: str | Path, content: bytes) -> Path:
    """Write bytes beside the target, then atomically replace the destination."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    except BaseException:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return target


def atomic_write_text(path: str | Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Atomically write text using an explicit encoding."""
    return atomic_write_bytes(path, content.encode(encoding))
