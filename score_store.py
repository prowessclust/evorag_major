"""Append-only score store for EvoRAG evolution data."""

import json
import os
from pathlib import Path
from typing import Any

from config import SCORE_STORE_FILE


class _FileLock:
    """Cross-platform exclusive file lock for append-safe JSON updates."""

    def __init__(self, handle: Any) -> None:
        self.handle = handle

    def __enter__(self) -> None:
        if os.name == "nt":
            import msvcrt

            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if os.name == "nt":
            import msvcrt

            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)


def append_score_entry(entry: dict[str, Any], path: Path = SCORE_STORE_FILE) -> None:
    """Append one query score entry to a JSON array file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]", encoding="utf-8")

    with path.open("r+", encoding="utf-8") as handle:
        with _FileLock(handle):
            try:
                handle.seek(0)
                existing = json.load(handle)
            except json.JSONDecodeError:
                existing = []
            if not isinstance(existing, list):
                existing = []
            existing.append(entry)
            handle.seek(0)
            json.dump(existing, handle, ensure_ascii=False, indent=2)
            handle.truncate()
