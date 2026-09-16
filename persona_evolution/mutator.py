"""
persona_evolution/mutator.py — Candidate Replacement Persona Generator

Generates a replacement persona when a weak one is detected.
Supports three strategies: predefined_pool, llm_generated, or hybrid.

Public API
----------
    generate_replacement_persona(
        weak_persona_id, weak_persona_prompt,
        existing_persona_ids, strategy
    ) -> dict
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx

from config import OLLAMA_BASE_URL, OLLAMA_MODEL

log = logging.getLogger(__name__)

# ── Strategy config ────────────────────────────────────────────────────────────
MUTATION_STRATEGY = "hybrid"  # "predefined_pool" | "llm_generated" | "hybrid"

# ── Predefined pool ────────────────────────────────────────────────────────────
REPLACEMENT_POOL: dict[str, str] = {
    "historical-comparativist": (
        "You are the Historical Comparativist. Your job is to find historical parallels "
        "and precedents for the current situation. When answering, ground every insight in "
        "a specific historical analogy or case study. Ask: has something like this happened "
        "before, and what was the outcome? Use only the provided context."
    ),
    "systems-thinker": (
        "You are the Systems Thinker. Your job is to map the feedback loops, second-order "
        "effects, and interdependencies in any situation. Avoid linear cause-effect thinking. "
        "Always ask: what are the downstream consequences, and what feeds back into the system? "
        "Use only the provided context."
    ),
    "contrarian-optimist": (
        "You are the Contrarian Optimist. Where others see risk, you find opportunity. "
        "Where the consensus is pessimistic, you find overlooked upsides. Your job is not "
        "to be blindly positive but to surface the strongest case that things will work out "
        "better than expected. Use only the provided context."
    ),
    "first-principles": (
        "You are the First Principles Reasoner. You strip away assumptions and rebuild "
        "reasoning from the ground up. Reject analogies and conventional wisdom. Start from "
        "the most basic truths available in the retrieved context and build upward. "
        "Use only the provided context."
    ),
    "ethical-evaluator": (
        "You are the Ethical Evaluator. Your job is to surface the moral dimensions of any "
        "situation — who benefits, who is harmed, whose interests are invisible in the mainstream "
        "analysis, and what values are implicitly embedded in the retrieved context. "
        "Use only the provided context."
    ),
}

# ── LLM meta-prompt ────────────────────────────────────────────────────────────
LLM_MUTATION_PROMPT = """\
You are designing an analytical persona for a multi-perspective reasoning system.

The system currently has these personas (do NOT duplicate any of these):
{existing_personas_list}

The persona being replaced was:
ID: {weak_persona_id}
Description: {weak_persona_prompt}

Design ONE new analytical persona that:
1. Is meaningfully different from all existing personas listed above
2. Fills a genuine analytical gap not covered by the existing set
3. Would produce different answers than the existing personas on the same question

Return ONLY valid JSON with this exact structure:
{{
    "persona_id": "slug-case-id-max-3-words",
    "display_name": "Display Name",
    "system_prompt": "You are the [Name]. Your job is..."
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# 1. PREDEFINED POOL STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

def _from_predefined_pool(existing_ids: set[str]) -> dict[str, Any] | None:
    """Pick the first pool entry not already in the active persona set.

    Returns:
        A candidate dict or ``None`` if the pool is exhausted.
    """
    for pid, prompt in REPLACEMENT_POOL.items():
        if pid not in existing_ids:
            display_name = pid.replace("-", " ").title()
            log.info("[mutator] Selected from predefined pool: %s", pid)
            return {
                "persona_id":    pid,
                "display_name":  display_name,
                "system_prompt": prompt,
                "generated_by":  "predefined_pool",
                "generated_at":  datetime.now(timezone.utc).isoformat(),
            }
    log.info("[mutator] Predefined pool exhausted — all entries are already active.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. LLM GENERATION STRATEGY
# ══════════════════════════════════════════════════════════════════════════════

def _from_llm(
    weak_persona_id: str,
    weak_persona_prompt: str,
    existing_persona_ids: list[str],
) -> dict[str, Any] | None:
    """Ask Ollama to generate a novel replacement persona.

    Returns:
        A candidate dict or ``None`` on parse/network failure.
    """
    existing_list = "\n".join(f"- {pid}" for pid in existing_persona_ids)
    meta_prompt = LLM_MUTATION_PROMPT.format(
        existing_personas_list=existing_list,
        weak_persona_id=weak_persona_id,
        weak_persona_prompt=weak_persona_prompt,
    )

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": meta_prompt,
        "stream": False,
        "options": {"temperature": 0.9, "num_predict": 400},
    }

    try:
        log.info("[mutator] Calling Ollama to generate LLM persona...")
        with httpx.Client(timeout=120) as client:
            resp = client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            resp.raise_for_status()
        raw = resp.json().get("response", "")
    except Exception as exc:
        log.error("[mutator] Ollama call failed: %s", exc)
        return None

    # Extract JSON block from response (LLMs sometimes wrap in markdown)
    json_match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
    if not json_match:
        log.warning("[mutator] LLM response did not contain parseable JSON: %s", raw[:300])
        return None

    try:
        parsed = json.loads(json_match.group(0))
        pid     = str(parsed.get("persona_id", "llm-generated")).strip()
        name    = str(parsed.get("display_name", pid.replace("-", " ").title())).strip()
        prompt  = str(parsed.get("system_prompt", "")).strip()
    except (json.JSONDecodeError, KeyError) as exc:
        log.warning("[mutator] Failed to parse LLM persona JSON: %s", exc)
        return None

    if not prompt:
        log.warning("[mutator] LLM returned empty system_prompt — discarding.")
        return None

    log.info("[mutator] LLM generated persona: %s", pid)
    return {
        "persona_id":    pid,
        "display_name":  name,
        "system_prompt": prompt,
        "generated_by":  "llm_generated",
        "generated_at":  datetime.now(timezone.utc).isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def generate_replacement_persona(
    weak_persona_id: str,
    weak_persona_prompt: str,
    existing_persona_ids: list[str],
    strategy: str = MUTATION_STRATEGY,
) -> dict[str, Any]:
    """Generate a candidate replacement persona.

    Strategy options:

    ``"predefined_pool"``
        Pick from REPLACEMENT_POOL, skipping any already active persona IDs.
        Falls back to ``llm_generated`` if pool is exhausted.

    ``"llm_generated"``
        Call Ollama with a meta-prompt to generate a novel persona.
        Falls back to ``predefined_pool`` if Ollama fails.

    ``"hybrid"`` *(default)*
        Try ``predefined_pool`` first; if exhausted, try ``llm_generated``.

    Args:
        weak_persona_id:      The ID of the persona being replaced.
        weak_persona_prompt:  The system prompt of the persona being replaced.
        existing_persona_ids: All currently active persona IDs.
        strategy:             One of ``"predefined_pool"``, ``"llm_generated"``, ``"hybrid"``.

    Returns:
        A candidate dict with keys: ``persona_id``, ``display_name``,
        ``system_prompt``, ``generated_by``, ``generated_at``.

    Raises:
        RuntimeError: If all strategies are exhausted and no candidate can be generated.
    """
    existing_set = set(existing_persona_ids)
    candidate: dict[str, Any] | None = None

    if strategy == "predefined_pool":
        candidate = _from_predefined_pool(existing_set)
        if candidate is None:
            log.info("[mutator] Pool exhausted — falling back to llm_generated.")
            candidate = _from_llm(weak_persona_id, weak_persona_prompt, existing_persona_ids)

    elif strategy == "llm_generated":
        candidate = _from_llm(weak_persona_id, weak_persona_prompt, existing_persona_ids)
        if candidate is None:
            log.info("[mutator] LLM generation failed — falling back to predefined_pool.")
            candidate = _from_predefined_pool(existing_set)

    else:  # "hybrid" — pool first, then LLM
        candidate = _from_predefined_pool(existing_set)
        if candidate is None:
            candidate = _from_llm(weak_persona_id, weak_persona_prompt, existing_persona_ids)

    if candidate is None:
        raise RuntimeError(
            f"[mutator] All mutation strategies exhausted for '{weak_persona_id}'. "
            "Cannot generate a replacement persona."
        )

    return candidate
