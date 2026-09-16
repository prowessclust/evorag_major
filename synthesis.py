"""Top-K persona selection and final answer synthesis for EvoRAG."""

import logging
from typing import Optional

from config import CONSENSUS_TOP_K, PERSONA_TIMEOUT_SECONDS
from utils import call_ollama_async

log = logging.getLogger(__name__)


def select_top_personas(
    final_scores: dict[str, float],
    responses: dict[str, Optional[str]],
    k: int = CONSENSUS_TOP_K,
) -> list[str]:
    """Select the highest-scoring personas that produced non-empty responses."""
    eligible = [
        persona_id
        for persona_id, response in responses.items()
        if response and response.strip()
    ]
    ranked = sorted(eligible, key=lambda persona_id: final_scores.get(persona_id, 0.0), reverse=True)

    # Fix 6: detect degenerate case where all scoring calls failed
    if eligible and all(final_scores.get(pid, 0.0) == 0.0 for pid in eligible):
        log.warning(
            "ALL PEER SCORES ARE ZERO — scoring stage produced no valid votes. "
            "Winner selection is falling back to config insertion order, NOT genuine ranking. "
            "Check api_debug.log for RAW SCORING OUTPUT to diagnose the scoring failure."
        )

    return ranked[:k]


def build_synthesis_prompt(
    query: str,
    top_personas: list[str],
    responses: dict[str, Optional[str]],
    final_scores: dict[str, float],
) -> str:
    """Build the final neutral synthesis prompt from selected responses."""
    labels = ("A", "B", "C", "D", "E")
    blocks = []
    for label, persona_id in zip(labels, top_personas):
        blocks.append(
            f"Response {label} (persona: {persona_id}, score: {final_scores.get(persona_id, 0.0):.1f}):\n"
            f"{responses.get(persona_id) or ''}"
        )
    selected = "\n\n".join(blocks)
    return f"""You are a neutral analytical synthesizer.

Strong responses to the question "{query}" have been selected through peer evaluation.
Merge them into one coherent, well-structured answer.

Rules:
- Preserve the strongest insight from each response
- Resolve contradictions by noting the disagreement explicitly
- Do not add information that is not present in the selected responses
- Do not mention that this is a synthesis or that multiple perspectives were used
- Write as a single, natural, authoritative answer

{selected}

Synthesized answer:"""


async def synthesize_top_k(
    query: str,
    top_personas: list[str],
    responses: dict[str, Optional[str]],
    final_scores: dict[str, float],
) -> str:
    """Call Ollama once to synthesize selected persona responses."""
    if not top_personas:
        return "No persona produced enough information to synthesize a consensus answer."
    prompt = build_synthesis_prompt(query, top_personas, responses, final_scores)
    answer = await call_ollama_async(prompt, timeout_seconds=PERSONA_TIMEOUT_SECONDS)
    if answer:
        return answer
    log.warning("Synthesis call failed; falling back to highest-ranked persona response.")
    return responses.get(top_personas[0]) or ""
