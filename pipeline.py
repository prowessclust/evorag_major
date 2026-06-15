"""Full EvoRAG consensus pipeline orchestration."""

import logging
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from config import RETRIEVAL_TOP_K
from personas import run_all_personas
from score_store import append_score_entry
from synthesis import select_top_personas, synthesize_top_k
from utils import word_count
from voting import build_score_matrix, compute_final_scores, summarize_received_scores

log = logging.getLogger(__name__)


async def evorag_query(query: str, top_k: int = RETRIEVAL_TOP_K) -> dict[str, Any]:
    """Run retrieval, persona generation, peer voting, synthesis, and score logging."""
    query_id = str(uuid4())

    persona_result = await run_all_personas(query=query, top_k=top_k, persona_ids=None)
    responses: dict[str, Optional[str]] = {
        persona["id"]: persona.get("response") or None
        for persona in persona_result["personas"]
    }

    score_matrix = await build_score_matrix(query, responses)
    final_scores = compute_final_scores(score_matrix)
    top_personas = select_top_personas(final_scores, responses)
    final_answer = await synthesize_top_k(query, top_personas, responses, final_scores)

    received = summarize_received_scores(score_matrix)
    persona_scores = {}
    for persona_id, summary in received.items():
        persona_scores[persona_id] = {
            **summary,
            "made_top_k": persona_id in top_personas,
            "response_length": word_count(responses.get(persona_id)),
        }

    winner = top_personas[0] if top_personas else None
    score_entry = {
        "query_id": query_id,
        "query": query,
        "timestamp": datetime.utcnow().isoformat(),
        "persona_scores": persona_scores,
        "winner": winner,
        "synthesis_used": top_personas,
        "user_signal": None,
    }
    append_score_entry(score_entry)
    log.info("Consensus complete. winner=%s top_k=%s", winner, top_personas)

    return {
        "final_answer": final_answer,
        "top_personas": top_personas,
        "scores": final_scores,
        "all_responses": responses,
        "score_matrix": score_matrix,
        "query_id": query_id,
    }
