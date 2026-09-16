"""Shared helpers for EvoRAG consensus orchestration."""

import asyncio
import json
import logging
import re
from typing import Any, Awaitable, Optional, TypeVar

import httpx

from config import OLLAMA_BASE_URL, OLLAMA_KEEP_ALIVE, OLLAMA_MODEL, PERSONA_TIMEOUT_SECONDS

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
