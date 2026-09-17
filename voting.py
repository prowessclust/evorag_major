"""Peer scoring and consensus score computation for EvoRAG."""

import logging
from typing import Any, Optional

from config import PERSONAS, PERSONA_TIMEOUT_SECONDS, SCORING_TIMEOUT_SECONDS
from utils import call_ollama_async, race_ollama_with_gemini, safe_json_loads

log = logging.getLogger(__name__)

Score = dict[str, float]
ScoreMatrix = dict[str, dict[str, Optional[Score]]]

DIMENSIONS = ("factual_grounding", "reasoning_quality", "completeness")


def _persona_name_by_id() -> dict[str, str]:
    """Return persona display names keyed by persona id."""
    return {p["id"]: p["name"] for p in PERSONAS}


# Max words per persona response included in the scoring prompt.
# phi3:mini has a 4K token context window; 8 full responses (~10K chars) exceed it.
# 200 words per response keeps the total prompt well within the window.
_SCORING_RESPONSE_MAX_WORDS = 200


def _truncate_response(text: Optional[str], max_words: int = _SCORING_RESPONSE_MAX_WORDS) -> str:
    """Truncate a persona response to max_words for the scoring prompt."""
    if not text:
        return "[No response]"
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " [truncated]"


def _format_responses(responses: dict[str, Optional[str]]) -> str:
    """Format persona responses for the peer scoring prompt.

    Each response is truncated to _SCORING_RESPONSE_MAX_WORDS so the total
    prompt stays within phi3's 4K token context window.
    """
    blocks = []
    for persona_id, response in responses.items():
        blocks.append(f"{persona_id}:\n{_truncate_response(response)}")
    return "\n\n".join(blocks)


def build_scoring_prompt(
    scorer_id: str,
    query: str,
    responses: dict[str, Optional[str]],
) -> str:
    """Build the JSON-only prompt a persona uses to score peer responses."""
    names = _persona_name_by_id()
    # Build a full skeleton so the LLM knows every key it must return.
    skeleton_parts = []
    for pid in responses:
        if pid == scorer_id:
            skeleton_parts.append(f'    "{pid}": null')
        else:
            skeleton_parts.append(
                f'    "{pid}": {{"factual_grounding": <0-10>, "reasoning_quality": <0-10>, "completeness": <0-10>}}'
            )
    skeleton = "{\n  \"scores\": {\n" + ",\n".join(skeleton_parts) + "\n  }\n}"
    return f"""You are {names.get(scorer_id, scorer_id)}.

Below are responses to the question: "{query}"

Score each response EXCEPT YOUR OWN on three dimensions from 0 to 10:
- factual_grounding: How well does it use retrieved context and stay factual?
- reasoning_quality: Is the logic coherent and well-structured?
- completeness: Does it address what the question actually needs?

Your own response is labeled "{scorer_id}". Set its entry to null.

Responses:
{_format_responses(responses)}

Return ONLY valid JSON. No prose, no markdown, no code fences. Fill in numeric scores:
{skeleton}"""


def _coerce_score(value: Any) -> Optional[Score]:
    """Normalize a raw score object into three bounded numeric dimensions."""
    if not isinstance(value, dict):
        return None
    score: Score = {}
    for dimension in DIMENSIONS:
        try:
            number = float(value[dimension])
        except (KeyError, TypeError, ValueError):
            return None
        score[dimension] = max(0.0, min(10.0, number))
    return score


async def score_as_persona(
    scorer_id: str,
    query: str,
    responses: dict[str, Optional[str]],
) -> dict[str, Optional[Score]]:
    """Ask one persona to score all peer responses."""
    prompt = build_scoring_prompt(scorer_id, query, responses)
    # Fix 3: scoring prompts are 3-4× longer than generation prompts — use dedicated timeout
    raw, source = await race_ollama_with_gemini(
        call_ollama_async(prompt, timeout_seconds=SCORING_TIMEOUT_SECONDS),
        prompt,
        total_timeout_seconds=SCORING_TIMEOUT_SECONDS,
        log_label=f"[{scorer_id}] ",
    )

    # ── Diagnostic: log raw LLM output BEFORE any parsing ─────────────────────
    log.info("[%s] RAW SCORING OUTPUT (source=%s):\n%s", scorer_id, source, raw)

    parsed = safe_json_loads(raw)
    scores = parsed.get("scores") if parsed else None
    row: dict[str, Optional[Score]] = {pid: None for pid in responses}

    if not isinstance(scores, dict):
        log.warning("[%s] scoring response was not valid JSON scores.", scorer_id)
        return row

    for target_id in responses:
        if target_id == scorer_id:
            row[target_id] = None
            continue
        row[target_id] = _coerce_score(scores.get(target_id))
    return row


async def build_score_matrix(
    query: str,
    responses: dict[str, Optional[str]],
) -> ScoreMatrix:
    """Run all peer scoring calls sequentially and return an 8x8-style matrix.

    IMPORTANT: calls are serialised (not parallelised with asyncio.gather) because
    Ollama on CPU is compute-bound and cannot serve concurrent requests.  Firing all
    8 calls simultaneously overloads Ollama, causes all requests to fail, and
    produces an all-zero score matrix that breaks Phase 5 peer evaluation.
    Sequential execution mirrors the persona generation pattern in personas.py.
    """
    import time as _time
    scorer_ids = list(responses.keys())
    matrix: ScoreMatrix = {}
    for scorer_id in scorer_ids:
        t0 = _time.time()
        log.info("[SCORING] %s scoring peers...", scorer_id)
        row = await score_as_persona(scorer_id, query, responses)
        elapsed = round(_time.time() - t0, 1)
        valid_scores = sum(1 for v in row.values() if v is not None)
        log.info("[SCORING] %s done in %ss — %d/%d valid scores",
                 scorer_id, elapsed, valid_scores, len(row) - 1)
        matrix[scorer_id] = row
    log.info("SCORE MATRIX:\n%s", matrix)
    return matrix


def compute_final_scores(score_matrix: ScoreMatrix) -> dict[str, float]:
    """Compute final persona scores from all received peer score dimensions."""
    target_ids = list(next(iter(score_matrix.values()), {}).keys())
    final_scores = {persona_id: 0.0 for persona_id in target_ids}

    for target_id in target_ids:
        total = 0.0
        for scorer_row in score_matrix.values():
            score = scorer_row.get(target_id)
            if not score:
                continue
            total += sum(score[dimension] for dimension in DIMENSIONS) / len(DIMENSIONS)
        final_scores[target_id] = round(total, 4)
    return final_scores


def summarize_received_scores(score_matrix: ScoreMatrix) -> dict[str, dict[str, Any]]:
    """Summarize peer scores received by each persona for score storage."""
    target_ids = list(next(iter(score_matrix.values()), {}).keys())
    summaries: dict[str, dict[str, Any]] = {}

    for target_id in target_ids:
        composites: list[float] = []
        dimension_values = {dimension: [] for dimension in DIMENSIONS}
        for scorer_row in score_matrix.values():
            score = scorer_row.get(target_id)
            if not score:
                continue
            composites.append(sum(score.values()) / len(DIMENSIONS))
            for dimension in DIMENSIONS:
                dimension_values[dimension].append(score[dimension])

        avg = round(sum(composites) / len(composites), 4) if composites else 0.0
        dims = {
            dimension: round(sum(values) / len(values), 4) if values else 0.0
            for dimension, values in dimension_values.items()
        }
        # Fix 5: log aggregated values before persistence — proves path is reached
        log.info(
            "AGGREGATED %s avg=%s votes=%s dimensions=%s",
            target_id,
            avg,
            [round(v, 4) for v in composites],
            dims,
        )
        summaries[target_id] = {
            "avg_score": avg,
            "scores_received": [round(value, 4) for value in composites],
            "dimensions": dims,
        }
    return summaries
