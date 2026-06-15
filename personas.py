"""
personas.py — EvoRAG Phase 4: Multi-Persona Engine

Public API
----------
    run_all_personas(query, top_k, persona_ids) -> Dict
        Returns: {query, sources, personas}
        Each persona dict: {id, name, response, error}

Strategy
--------
    1. Persona requests are serialised through an asyncio.Semaphore
       (OLLAMA_MAX_CONCURRENT=1 by default) so Ollama is never hammered.

       IMPORTANT: the Semaphore is created lazily inside run_all_personas(),
       NOT at module-import time.  If created at import time it binds to
       whatever event loop exists then; FastAPI/uvicorn replace that loop
       before the first request arrives, causing a silent RuntimeError that
       the except-block swallows — producing empty responses for every persona.

    2. After the initial pass, any persona with an empty response is retried
       sequentially (with a small delay) — fixes transient Ollama empties.
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional

import httpx

from config import (
    PERSONAS,
    RETRIEVAL_TOP_K,
    PERSONA_TIMEOUT_SECONDS,
    OLLAMA_MAX_CONCURRENT,
)
from embedder import release_model
from retriever import retrieve, build_prompt, ask_ollama_async

log = logging.getLogger(__name__)

MAX_RETRIES         = 2   # max retry attempts per empty-response persona
RETRY_DELAY_SECONDS = 2   # seconds between sequential retries

# Semaphore is created lazily on first call to run_all_personas() so that it
# always binds to FastAPI/uvicorn's running event loop, not the import-time loop.
_ollama_semaphore: Optional[asyncio.Semaphore] = None


def _preview(text: str, limit: int = 100) -> str:
    """Return a compact single-line preview for logging."""
    return text[:limit].strip().replace("\n", " ")


def _error_message(exc: Exception) -> str:
    """Extract a readable message from Ollama/HTTP failures."""
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        try:
            body = exc.response.json()
            if isinstance(body, dict) and body.get("error"):
                return str(body["error"])
        except ValueError:
            pass
        text = exc.response.text.strip()
        if text:
            return text[:500]
    return str(exc) or exc.__class__.__name__


# ══════════════════════════════════════════════════════════════════════════════
# 1. PERSONA PROMPT
# ══════════════════════════════════════════════════════════════════════════════

def _persona_prompt(persona: Dict, context_prompt: str) -> str:
    """Prepend the persona's system instruction to the shared RAG context prompt."""
    return f"{persona['system_prompt']}\n\n{context_prompt}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. SINGLE PERSONA RUN — async, safe (never raises)
# ══════════════════════════════════════════════════════════════════════════════

async def run_persona(
    persona: Dict,
    context_prompt: str,
    semaphore: asyncio.Semaphore,
) -> Dict:
    """
    Call Ollama for a single persona.

    The caller (run_all_personas) passes the semaphore explicitly so that it
    is always the one created on the running event loop — not a stale import-
    time object that would silently fail on uvicorn's loop.

    Returns:
        {"id": str, "name": str, "response": str, "error": str}
    """
    pid  = persona["id"]
    name = persona["name"]
    prompt = _persona_prompt(persona, context_prompt)

    log.info("[%s] Starting generation (persona=%s)...", pid, name)
    t0 = time.time()
    try:
        # Guard BOTH semaphore acquisition AND the Ollama HTTP call with the
        # same timeout so a persona can never block forever waiting for a slot.
        async def _acquire_and_call() -> str:
            async with semaphore:
                return await ask_ollama_async(prompt)

        response = await asyncio.wait_for(
            _acquire_and_call(),
            timeout=PERSONA_TIMEOUT_SECONDS,
        )
        response = (response or "").strip()
        elapsed = round(time.time() - t0, 1)

        if not response:
            msg = "Ollama returned an empty or whitespace-only response."
            log.warning(
                "[%s] %s elapsed=%ss persona=%s",
                pid, msg, elapsed, name,
            )
            return {"id": pid, "name": name, "response": "", "error": msg}

        log.info(
            "[%s] Done in %ss persona=%s length=%s preview='%s'",
            pid, elapsed, name, len(response), _preview(response),
        )
        return {"id": pid, "name": name, "response": response, "error": ""}
    except asyncio.TimeoutError:
        elapsed = round(time.time() - t0, 1)
        msg = f"Persona generation timed out after {PERSONA_TIMEOUT_SECONDS}s."
        log.error("[%s] %s elapsed=%ss persona=%s", pid, msg, elapsed, name)
        return {"id": pid, "name": name, "response": "", "error": msg}
    except Exception as exc:
        elapsed = round(time.time() - t0, 1)
        msg = _error_message(exc)
        log.exception(
            "[%s] Failed after %ss persona=%s: %s",
            pid, elapsed, name, msg,
        )
        return {"id": pid, "name": name, "response": "", "error": msg}


# ══════════════════════════════════════════════════════════════════════════════
# 3. ALL PERSONAS — retrieve once, run concurrently, retry empties
# ══════════════════════════════════════════════════════════════════════════════

async def run_all_personas(
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
    persona_ids: Optional[List[str]] = None,
) -> Dict:
    """
    Full multi-persona pipeline:
        1. Retrieve top-k chunks (once, shared)
        2. Build shared RAG context prompt
        3. Create a fresh asyncio.Semaphore on the *running* event loop
        4. Run persona calls sequentially (semaphore limit=1 by default)
        5. Retry any that returned empty responses (sequential, with delay)
        6. Return structured result dict

    Args:
        query:       User question.
        top_k:       Chunks to retrieve.
        persona_ids: Optional list of persona IDs to run. None = all 8.

    Returns:
        {"query": str, "sources": [...], "personas": [...]}
    """
    # Step 1 — Retrieve (shared)
    chunks = retrieve(query, top_k=top_k)
    sources = [
        {
            "title":  c.get("title", ""),
            "url":    c.get("url", ""),
            "source": c.get("source", ""),
            "score":  round(c.get("score", 0.0), 4),
        }
        for c in chunks
    ]
    if not chunks:
        log.warning("No chunks retrieved — personas will see empty context.")

    # Step 2 — Build shared context prompt
    context_prompt = build_prompt(query, chunks)

    # Free embedding model RAM before Ollama generation (phi3 needs ~3.5 GiB)
    release_model()

    # Step 3 — Select which personas to run
    if persona_ids:
        id_set = set(persona_ids)
        active = [p for p in PERSONAS if p["id"] in id_set]
        unknown = id_set - {p["id"] for p in active}
        if unknown:
            raise ValueError(
                f"Unknown persona_ids: {sorted(unknown)}. "
                f"Valid ids: {[p['id'] for p in PERSONAS]}"
            )
        if not active:
            raise ValueError("persona_ids matched no personas.")
    else:
        active = list(PERSONAS)

    log.info("Running %s personas for: '%s'", len(active), query[:60])

    # Step 4 — Create a fresh semaphore bound to the CURRENT running event loop.
    # This MUST happen here (inside an async function that is already running on
    # uvicorn's loop) — NOT at module-import time where the loop is different.
    semaphore = asyncio.Semaphore(max(1, OLLAMA_MAX_CONCURRENT))
    log.debug("Created Ollama semaphore (max_concurrent=%s) on loop %s",
              OLLAMA_MAX_CONCURRENT, id(asyncio.get_event_loop()))

    # Step 5 — Initial pass (sequential; semaphore inside run_persona guards Ollama)
    results: List[Dict] = []
    for persona in active:
        results.append(await run_persona(persona, context_prompt, semaphore))

    # Step 6 — Retry empty responses sequentially
    persona_by_id  = {p["id"]: p for p in active}
    result_by_id   = {r["id"]: i for i, r in enumerate(results)}

    for attempt in range(1, MAX_RETRIES + 1):
        empty = [r for r in results if not (r.get("response") or "").strip()]
        if not empty:
            break

        log.info(
            "Retry %s/%s: %s persona(s) still empty — retrying sequentially...",
            attempt, MAX_RETRIES, len(empty),
        )

        for r in empty:
            pid = r["id"]
            persona = persona_by_id.get(pid)
            if not persona:
                continue

            log.info("[%s] Retry attempt %s...", pid, attempt)
            await asyncio.sleep(RETRY_DELAY_SECONDS)
            retry = await run_persona(persona, context_prompt, semaphore)
            results[result_by_id[pid]] = retry

            if retry["response"].strip():
                log.info(
                    "[%s] Retry %s succeeded length=%s preview='%s'",
                    pid,
                    attempt,
                    len(retry["response"]),
                    _preview(retry["response"]),
                )
            else:
                log.warning(
                    "[%s] Retry %s still empty. error=%s",
                    pid,
                    attempt,
                    retry.get("error") or "(no error reported)",
                )

    # Summary log
    for result in results:
        response = (result.get("response") or "").strip()
        log.info(
            "Persona result before return: name=%s length=%s preview='%s' error=%s",
            result.get("name", ""),
            len(response),
            _preview(response),
            result.get("error") or "",
        )

    n_good  = sum(1 for r in results if (r.get("response") or "").strip())
    n_empty = len(results) - n_good
    if n_empty:
        log.warning(f"{n_empty}/{len(results)} persona(s) empty after {MAX_RETRIES} retries.")
    else:
        log.info(f"All {len(results)} personas returned non-empty responses.")

    return {"query": query, "sources": sources, "personas": results}
