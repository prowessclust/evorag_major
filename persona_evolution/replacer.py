"""
persona_evolution/replacer.py — Execute Replacement & Version History

Executes persona replacement in memory and persists a complete version history
in persona_versions.json so every experiment is reproducible.

Public API
----------
    load_persona_versions() -> dict
    save_persona_versions(versions) -> None
    execute_replacement(weak_persona_id, candidate, reason, personas_module, logger) -> dict
    get_replacement_history() -> list[dict]
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PERSONA_VERSIONS_PATH = Path(__file__).parent / "persona_versions.json"

_DEFAULT_VERSIONS: dict[str, Any] = {
    "active_personas":   [],
    "version_counter":   0,
    "replacements":      [],
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. FILE I/O WITH LOCKING
# ══════════════════════════════════════════════════════════════════════════════

class _FileLock:
    """Minimal cross-platform exclusive file lock (mirrors score_store.py)."""

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


def load_persona_versions(path: Path = PERSONA_VERSIONS_PATH) -> dict[str, Any]:
    """Load persona_versions.json.

    Returns:
        The parsed dict, or a fresh default structure if the file is missing or corrupt.
    """
    if not path.exists():
        log.debug("[replacer] persona_versions.json not found — using defaults.")
        return {k: (v.copy() if isinstance(v, (dict, list)) else v)
                for k, v in _DEFAULT_VERSIONS.items()}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object at top level.")
        return data
    except (json.JSONDecodeError, ValueError, OSError) as exc:
        log.error("[replacer] Failed to load persona_versions.json: %s — using defaults.", exc)
        return {k: (v.copy() if isinstance(v, (dict, list)) else v)
                for k, v in _DEFAULT_VERSIONS.items()}


def save_persona_versions(
    versions: dict[str, Any],
    path: Path = PERSONA_VERSIONS_PATH,
) -> None:
    """Atomically save persona_versions.json with file locking.

    Args:
        versions: The full versions dict to persist.
        path:     Target path (defaults to PERSONA_VERSIONS_PATH).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.write_text("{}", encoding="utf-8")

    with path.open("r+", encoding="utf-8") as fh:
        with _FileLock(fh):
            fh.seek(0)
            json.dump(versions, fh, ensure_ascii=False, indent=2)
            fh.truncate()


# ══════════════════════════════════════════════════════════════════════════════
# 2. EXECUTE REPLACEMENT
# ══════════════════════════════════════════════════════════════════════════════

def execute_replacement(
    weak_persona_id: str,
    candidate: dict[str, Any],
    reason: str,
    personas_module: Any,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Replace a weak persona in memory and record the event to disk.

    This function:

    1. Removes ``weak_persona_id`` from the ``PERSONAS`` list in ``personas_module``.
    2. Appends the candidate persona (as a dict with ``id``, ``name``,
       ``system_prompt`` keys matching the existing PERSONAS schema) to that list.
    3. Loads ``persona_versions.json``, appends a replacement event, and saves it.
    4. Logs the event at INFO level with the standard ``[EVOLUTION]`` prefix.

    Args:
        weak_persona_id:  The persona ID being replaced.
        candidate:        Output of ``mutator.generate_replacement_persona()``.
        reason:           Human-readable reason string from the detector.
        personas_module:  The ``personas`` module object (contains ``PERSONAS`` list).
        logger:           Optional logger; defaults to this module's logger.

    Returns:
        The replacement event dict that was appended to persona_versions.json.

    Raises:
        ValueError: If ``weak_persona_id`` is not found in the active PERSONAS list.
    """
    _log = logger or log

    # ── 1. Update PERSONAS in memory ───────────────────────────────────────────
    personas_list: list[dict[str, Any]] = personas_module.PERSONAS
    old_persona = next((p for p in personas_list if p["id"] == weak_persona_id), None)
    if old_persona is None:
        raise ValueError(
            f"[replacer] Persona '{weak_persona_id}' not found in PERSONAS — cannot replace."
        )

    old_prompt = old_persona.get("system_prompt", "")
    old_index  = personas_list.index(old_persona)

    new_persona_entry: dict[str, Any] = {
        "id":            candidate["persona_id"],
        "name":          candidate["display_name"],
        "system_prompt": candidate["system_prompt"],
    }

    # Replace in place to preserve slot order
    personas_list[old_index] = new_persona_entry
    _log.info(
        "[EVOLUTION] Replaced '%s' → '%s' at slot %d",
        weak_persona_id, candidate["persona_id"], old_index,
    )

    # ── 2. Load, update, and save version history ──────────────────────────────
    versions = load_persona_versions()

    versions["version_counter"] = versions.get("version_counter", 0) + 1
    version_num = versions["version_counter"]

    # Rebuild active list from current in-memory PERSONAS
    versions["active_personas"] = [p["id"] for p in personas_list]

    event: dict[str, Any] = {
        "version":                   version_num,
        "timestamp":                 datetime.now(timezone.utc).isoformat(),
        "replaced_persona_id":       weak_persona_id,
        "replaced_persona_prompt":   old_prompt,
        "replacement_persona_id":    candidate["persona_id"],
        "replacement_persona_prompt": candidate["system_prompt"],
        "generated_by":              candidate.get("generated_by", "unknown"),
        "reason":                    reason,
        "avg_score_at_replacement":  None,   # caller may fill in
        "queries_evaluated":         None,   # caller may fill in
    }

    versions.setdefault("replacements", []).append(event)
    save_persona_versions(versions)

    # ── 3. Structured INFO log ─────────────────────────────────────────────────
    _log.info(
        "[EVOLUTION] Replaced '%s' → '%s' (v%d)\nReason: %s",
        weak_persona_id, candidate["persona_id"], version_num, reason,
    )

    return event


# ══════════════════════════════════════════════════════════════════════════════
# 3. READ HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def get_replacement_history(path: Path = PERSONA_VERSIONS_PATH) -> list[dict[str, Any]]:
    """Return the full list of replacement events from persona_versions.json.

    Args:
        path: Path to persona_versions.json.

    Returns:
        A list of replacement event dicts (may be empty).
    """
    versions = load_persona_versions(path)
    return versions.get("replacements", [])
