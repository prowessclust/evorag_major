"""Peer scoring and consensus score computation for EvoRAG."""

import asyncio
import logging
from typing import Any, Optional

from config import PERSONAS, PERSONA_TIMEOUT_SECONDS
from utils import call_ollama_async, safe_json_loads

log = logging.getLogger(__name__)

Score = dict[str, float]
ScoreMatrix = dict[str, dict[str, Optional[Score]]]

DIMENSIONS = ("factual_grounding", "reasoning_quality", "completeness")


def _persona_name_by_id() -> dict[str, str]:
    """Return persona display names keyed by persona id."""
    return {p["id"]: p["name"] for p in PERSONAS}


def _format_responses(responses: dict[str, Optional[str]]) -> str:
    """Format persona responses for the peer scoring prompt."""
    blocks = []
    for persona_id, response in responses.items():
        blocks.append(f"{persona_id}:\n{response or '[No response]'}")
    return "\n\n".join(blocks)


def build_scoring_prompt(
    scorer_id: str,
    query: str,
    responses: dict[str, Optional[str]],
) -> str:
    """Build the JSON-only prompt a persona uses to score peer responses."""
    names = _persona_name_by_id()
    return f"""You are {names.get(scorer_id, scorer_id)}.

Below are responses to the question: "{query}"

Score each response EXCEPT YOUR OWN on three dimensions from 0 to 10:
- factual_grounding: How well does it use retrieved context and stay factual?
- reasoning_quality: Is the logic coherent and well-structured?
- completeness: Does it address what the question actually needs?

Your own response is labeled {scorer_id}. Do NOT score it; set it to null.

Responses:
{_format_responses(responses)}

Return ONLY a valid JSON object in this format:
{{
  "scores": {{
    "{scorer_id}": null
  }}
}}"""


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
    raw = await call_ollama_async(prompt, timeout_seconds=PERSONA_TIMEOUT_SECONDS)
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
    """Run all peer scoring calls in parallel and return an 8x8-style matrix."""
    scorer_ids = list(responses.keys())
    rows = await asyncio.gather(
        *[score_as_persona(scorer_id, query, responses) for scorer_id in scorer_ids]
    )
    return {scorer_id: row for scorer_id, row in zip(scorer_ids, rows)}


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

        summaries[target_id] = {
            "avg_score": round(sum(composites) / len(composites), 4) if composites else 0.0,
            "scores_received": [round(value, 4) for value in composites],
            "dimensions": {
                dimension: round(sum(values) / len(values), 4) if values else 0.0
                for dimension, values in dimension_values.items()
            },
        }
    return summaries
