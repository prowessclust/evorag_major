"""
persona_evolution/tracker.py — Rolling Average Calculator

Reads score_store.json and computes per-persona rolling average scores
over a configurable sliding window of recent queries.

Public API
----------
    load_score_history(path) -> list[dict]
    compute_rolling_averages(history, window) -> dict[str, dict]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
ROLLING_WINDOW = 10            # number of recent queries to use for averaging
MIN_QUERIES_BEFORE_EVAL = 5   # require at least this many data points before evaluating


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════

def load_score_history(
    score_store_path: str | Path = "score_store.json",
) -> list[dict[str, Any]]:
    """Load all entries from score_store.json.

    Args:
        score_store_path: Path to the score store JSON file.

    Returns:
        A list of score entry dicts (may be empty if the file is missing or empty).
    """
    path = Path(score_store_path)
    if not path.exists():
        log.debug("score_store.json not found at %s — returning empty history.", path)
        return []
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            log.warning("score_store.json does not contain a JSON array — returning [].")
            return []
        return data
    except (json.JSONDecodeError, OSError) as exc:
        log.error("Failed to load score_store.json: %s", exc)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMPUTE ROLLING AVERAGES
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_averages(
    history: list[dict[str, Any]],
    window: int = ROLLING_WINDOW,
) -> dict[str, dict[str, Any]]:
    """Compute per-persona rolling average scores over the last *window* queries.

    Args:
        history: Full list of score entries from load_score_history().
        window:  How many recent entries to include in the rolling average.

    Returns:
        A dict keyed by persona_id with the following structure::

            {
                "analytical-critical": {
                    "avg_score": 6.2,
                    "query_count": 10,
                    "scores": [7.0, 6.0, 5.0, ...],  # last N raw avg_scores
                    "times_in_top_k": 3,
                    "fallback_count": 1,
                },
                ...
            }

        Only personas with at least MIN_QUERIES_BEFORE_EVAL entries are included.

    Notes:
        - ``avg_score`` is the mean of per-entry ``avg_score`` values in the window.
        - Entries where ``avg_score == 0.0`` are counted as ``fallback_count``
          (score was not genuinely computed — Ollama returned nothing useful).
        - ``times_in_top_k`` counts entries where ``made_top_k == True``.
    """
    # Collect per-persona time-ordered data from *all* history first,
    # then slice to the most recent `window` entries.
    raw_per_persona: dict[str, list[dict[str, Any]]] = {}

    for entry in history:
        persona_scores: dict[str, Any] = entry.get("persona_scores", {})
        for pid, ps in persona_scores.items():
            if not isinstance(ps, dict):
                continue
            raw_per_persona.setdefault(pid, []).append(ps)

    result: dict[str, dict[str, Any]] = {}
    for pid, records in raw_per_persona.items():
        total_count = len(records)
        if total_count < MIN_QUERIES_BEFORE_EVAL:
            log.debug(
                "[tracker] %s: only %d entries (need %d) — skipped.",
                pid, total_count, MIN_QUERIES_BEFORE_EVAL,
            )
            continue

        # Slice to the most recent `window` entries
        recent = records[-window:]

        scores: list[float] = []
        times_in_top_k = 0
        fallback_count = 0

        for rec in recent:
            raw_avg = rec.get("avg_score", 0.0)
            score_val = float(raw_avg) if raw_avg is not None else 0.0
            scores.append(score_val)

            if score_val == 0.0:
                fallback_count += 1
            if rec.get("made_top_k", False):
                times_in_top_k += 1

        avg = sum(scores) / len(scores) if scores else 0.0

        result[pid] = {
            "avg_score": round(avg, 4),
            "query_count": total_count,
            "scores": scores,
            "times_in_top_k": times_in_top_k,
            "fallback_count": fallback_count,
        }

    return result
