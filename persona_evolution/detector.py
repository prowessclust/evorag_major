"""
persona_evolution/detector.py — Underperformer Detection

Detects personas that are consistently underperforming and should be replaced.
Uses a threshold system with hysteresis to avoid premature replacement.

Public API
----------
    detect_underperformers(rolling_averages) -> list[dict]
    should_trigger_replacement(underperformers) -> bool
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# ── Replacement thresholds ────────────────────────────────────────────────────
REPLACEMENT_THRESHOLD      = 5.0   # avg_score below this → candidate for replacement
REPLACEMENT_CERTAINTY      = 0.7   # fraction of recent scores that must be below threshold
TOP_K_MIN_RATE             = 0.2   # persona must appear in top-K ≥ 20% of the time
MIN_QUERIES_BEFORE_REPLACE = 8     # must have at least this many queries before replacement


# ══════════════════════════════════════════════════════════════════════════════
# 1. DETECT UNDERPERFORMERS
# ══════════════════════════════════════════════════════════════════════════════

def detect_underperformers(
    rolling_averages: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return personas that qualify for replacement.

    A persona is flagged if ALL of the following are true:

    1. ``query_count >= MIN_QUERIES_BEFORE_REPLACE``
    2. ``avg_score < REPLACEMENT_THRESHOLD``
    3. More than ``REPLACEMENT_CERTAINTY`` fraction of recent scores are below threshold
    4. ``times_in_top_k / query_count < TOP_K_MIN_RATE``

    Args:
        rolling_averages: Output of ``tracker.compute_rolling_averages()``.

    Returns:
        A list of dicts, one per flagged persona::

            [
                {
                    "persona_id": "creative-lateral",
                    "avg_score": 4.1,
                    "top_k_rate": 0.1,
                    "query_count": 10,
                    "reason": "avg_score 4.1 < threshold 5.0, top_k_rate 10% < minimum 20%",
                },
                ...
            ]
    """
    flagged: list[dict[str, Any]] = []

    for pid, stats in rolling_averages.items():
        query_count   = stats.get("query_count", 0)
        avg_score     = stats.get("avg_score", 0.0)
        scores        = stats.get("scores", [])
        times_in_top_k = stats.get("times_in_top_k", 0)

        # Gate 1 — minimum data requirement
        if query_count < MIN_QUERIES_BEFORE_REPLACE:
            log.debug(
                "[detector] %s: query_count=%d < MIN=%d — skipped.",
                pid, query_count, MIN_QUERIES_BEFORE_REPLACE,
            )
            continue

        # Gate 2 — avg score threshold
        if avg_score >= REPLACEMENT_THRESHOLD:
            continue

        # Gate 3 — certainty: fraction of scores below threshold
        if scores:
            below_threshold = sum(1 for s in scores if s < REPLACEMENT_THRESHOLD)
            certainty = below_threshold / len(scores)
        else:
            certainty = 0.0

        if certainty < REPLACEMENT_CERTAINTY:
            log.debug(
                "[detector] %s: avg_score=%.2f < threshold but certainty=%.0f%% < %.0f%% — skipped.",
                pid, avg_score, certainty * 100, REPLACEMENT_CERTAINTY * 100,
            )
            continue

        # Gate 4 — top-K rate too low
        top_k_rate = times_in_top_k / query_count if query_count > 0 else 0.0
        if top_k_rate >= TOP_K_MIN_RATE:
            log.debug(
                "[detector] %s: avg_score low but top_k_rate=%.0f%% >= min — skipped.",
                pid, top_k_rate * 100,
            )
            continue

        reason = (
            f"avg_score {avg_score:.2f} < threshold {REPLACEMENT_THRESHOLD}, "
            f"top_k_rate {top_k_rate:.0%} < minimum {TOP_K_MIN_RATE:.0%}"
        )
        log.info("[detector] %s flagged for replacement: %s", pid, reason)
        flagged.append({
            "persona_id":   pid,
            "avg_score":    avg_score,
            "top_k_rate":   round(top_k_rate, 4),
            "query_count":  query_count,
            "reason":       reason,
        })

    return flagged


# ══════════════════════════════════════════════════════════════════════════════
# 2. TRIGGER DECISION
# ══════════════════════════════════════════════════════════════════════════════

def should_trigger_replacement(underperformers: list[dict[str, Any]]) -> bool:
    """Return True if a replacement should be triggered this cycle.

    Rules:
    - Only trigger if at least one underperformer is confirmed.
    - Never replace more than one persona per trigger cycle (caller must pick one).

    Args:
        underperformers: Output of ``detect_underperformers()``.

    Returns:
        ``True`` if a replacement should fire, ``False`` otherwise.
    """
    if not underperformers:
        log.debug("[detector] No underperformers detected — no replacement triggered.")
        return False

    log.info(
        "[detector] %d underperformer(s) detected — replacement will trigger for worst one.",
        len(underperformers),
    )
    return True
