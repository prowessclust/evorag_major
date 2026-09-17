"""Shared helpers for EvoRAG consensus orchestration."""

import asyncio
import json
import logging
import re
from typing import Any, Awaitable, Optional, TypeVar

import httpx

from config import (
    GEMINI_API_KEY,
    GEMINI_FALLBACK_SECONDS,
    GEMINI_MODEL,
    GEMINI_ONLY_MODE,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    PERSONA_TIMEOUT_SECONDS,
)

log = logging.getLogger(__name__)
T = TypeVar("T")


async def with_timeout(coro: Awaitable[T], timeout_seconds: int = PERSONA_TIMEOUT_SECONDS) -> Optional[T]:
    """Return a coroutine result or None if it fails or times out."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except (asyncio.TimeoutError, Exception) as exc:
        log.warning("Async call failed or timed out: %s", exc)
        return None


async def call_ollama_async(
    prompt: str,
    model: str = OLLAMA_MODEL,
    timeout_seconds: int = PERSONA_TIMEOUT_SECONDS,
) -> Optional[str]:
    """Call Ollama's non-streaming generation endpoint with a hard timeout."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "keep_alive": OLLAMA_KEEP_ALIVE}
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "").strip()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("Ollama call failed: %s", exc)
        return None


_GEMINI_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


async def call_gemini_async(
    prompt: str,
    model: str = GEMINI_MODEL,
    timeout_seconds: int = 90,
    max_attempts: int = 2,
    retry_delay_seconds: float = 3.0,
) -> Optional[str]:
    """Call Gemini's generateContent endpoint.

    Used as a hidden fallback when Ollama is too slow to respond — see
    GEMINI_FALLBACK_SECONDS in config.py. Never raises; returns None on any
    failure (missing key, network error, bad response) so callers can treat
    it exactly like a timed-out Ollama call.

    Transient errors (429 rate-limit, 5xx server overload) are retried a few
    times with a short delay — a single "high demand" blip from Google
    shouldn't waste the one shot this fallback gets against a slow persona.
    """
    if not GEMINI_API_KEY:
        log.warning("Gemini fallback skipped: GEMINI_API_KEY is not set.")
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                response = await client.post(url, params={"key": GEMINI_API_KEY}, json=payload)
                response.raise_for_status()
                data = response.json()
                candidates = data.get("candidates") or []
                if not candidates:
                    log.warning("Gemini fallback returned no candidates.")
                    return None
                parts = candidates[0].get("content", {}).get("parts") or []
                text = "".join(p.get("text", "") for p in parts).strip()
                return text or None
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in _GEMINI_RETRYABLE_STATUS and attempt < max_attempts:
                log.warning(
                    "Gemini fallback got %s (attempt %s/%s) — retrying in %ss...",
                    status, attempt, max_attempts, retry_delay_seconds,
                )
                await asyncio.sleep(retry_delay_seconds)
                continue
            log.warning("Gemini fallback call failed: %s", exc)
            return None
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            log.warning("Gemini fallback call failed: %s", exc)
            return None
    return None


async def race_ollama_with_gemini(
    ollama_coro: Awaitable[Optional[str]],
    prompt: str,
    total_timeout_seconds: int,
    fallback_after_seconds: int = GEMINI_FALLBACK_SECONDS,
    log_label: str = "",
) -> tuple[Optional[str], str]:
    """
    Await `ollama_coro`; if it hasn't finished within fallback_after_seconds,
    race a hidden Gemini call (using `prompt`) against the remaining timeout
    budget. Ollama keeps running in the background even after the fallback
    fires — it is never cancelled just for being slow; whichever of
    Ollama/Gemini finishes first with a usable answer wins, the loser is
    cancelled.

    Shared by personas.py (generation), voting.py (peer scoring), and
    synthesis.py (final synthesis) so every Ollama call in the consensus
    pipeline gets the same hidden speed guard.

    Returns (response_text_or_None, source) where source is "ollama",
    "gemini_fallback", or "timeout". Assumes ollama_coro follows
    call_ollama_async's contract of returning None (not raising) on failure.
    """
    if GEMINI_ONLY_MODE:
        ollama_coro.close()  # discard without running — Ollama is parked for now
        result = await call_gemini_async(prompt)
        return (result, "gemini_only") if result else (None, "gemini_only_failed")

    ollama_task = asyncio.ensure_future(ollama_coro)

    if not GEMINI_API_KEY:
        try:
            return await asyncio.wait_for(ollama_task, timeout=total_timeout_seconds), "ollama"
        except asyncio.TimeoutError:
            ollama_task.cancel()
            return None, "timeout"

    try:
        # asyncio.shield protects ollama_task from being cancelled when this
        # wait_for times out — it keeps running in the background below.
        response = await asyncio.wait_for(asyncio.shield(ollama_task), timeout=fallback_after_seconds)
        return response, "ollama"
    except asyncio.TimeoutError:
        pass  # fall through to the hidden Gemini race

    log.info(
        "%sOllama still running after %ss — starting hidden Gemini fallback",
        log_label, fallback_after_seconds,
    )
    gemini_task = asyncio.ensure_future(call_gemini_async(prompt))
    deadline = asyncio.get_event_loop().time() + max(1.0, total_timeout_seconds - fallback_after_seconds)
    pending = {ollama_task, gemini_task}

    # A failed/empty Gemini call finishes (with return_when=FIRST_COMPLETED)
    # long before Ollama does — that must NOT be treated as "nothing
    # finished". Loop so a fast Gemini failure just drops out of the race
    # and Ollama keeps getting awaited for whatever time budget remains.
    while pending:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        done, pending = await asyncio.wait(
            pending, timeout=remaining, return_when=asyncio.FIRST_COMPLETED,
        )

        if gemini_task in done:
            gemini_result = None
            try:
                gemini_result = gemini_task.result()
            except Exception as exc:
                log.warning("%sGemini fallback raised: %s", log_label, exc)
            if gemini_result:
                ollama_task.cancel()
                log.info("%sGemini fallback answered first", log_label)
                return gemini_result, "gemini_fallback"
            # Gemini failed/empty — fall through and keep waiting on Ollama.

        if ollama_task in done:
            try:
                return ollama_task.result(), "ollama"
            except Exception as exc:
                log.warning("%sOllama call raised: %s", log_label, exc)
                return None, "error"

    # Neither finished within the full timeout budget.
    ollama_task.cancel()
    gemini_task.cancel()
    return None, "timeout"


def safe_json_loads(raw: Optional[str]) -> Optional[dict[str, Any]]:
    """Parse a JSON object from model output, returning None on invalid JSON.

    Handles phi3's tendency to wrap JSON in markdown code fences:
        ```json { ... } ```   or   ``` { ... } ```
    The fence is stripped before any parse attempt.
    """
    if not raw:
        return None
    text = raw.strip()

    # Strip markdown code fences that phi3 adds despite instructions
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to extracting first {...} block from prose/mixed output
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def word_count(text: Optional[str]) -> int:
    """Count whitespace-delimited words in a possibly empty string."""
    return len((text or "").split())
